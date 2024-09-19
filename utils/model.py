import torch
import torch.nn as nn

from keras.layers import Dense, Input, Dropout, Conv1D, Reshape, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.layers import Add, Activation, ZeroPadding2D, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2

from utils.self_attention import *
from utils.my_metric import *

try:
    from pytorch_pretrained_bert.modeling import PreTrainedBertModel as BertPreTrainedModel
except ImportError:
    from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from pytorch_pretrained_bert import BertModel
import math


dr = 0.12
l2c = 0.001

def LinearBlock(H1, H2, p):
    return nn.Sequential(
        nn.Linear(H1, H2),
        nn.BatchNorm1d(H2),
        nn.ReLU(),
        nn.Dropout(p))

def MLP(D, n, H, K, p):
    """
    MLP w batchnorm and dropout.

    Parameters
    ----------
    D : int, size of input layer
    n : int, number of hidden layers
    H : int, size of hidden layer
    K : int, size of output layer
    p : float, dropout probability
    """

    if n == 0:
        print("Defaulting to linear classifier/regressor")
        return nn.Linear(D, K)
    else:
        print("Using mlp with D=%d,H=%d,K=%d,n=%d"%(D, H, K, n))
        layers = [nn.BatchNorm1d(D),
                  LinearBlock(D, H, p)]
        for _ in range(n-1):
            layers.append(LinearBlock(H, H, p))
        layers.append(nn.Linear(H, K))
        return torch.nn.Sequential(*layers)


class BertPlusMLP(BertPreTrainedModel):
    """
    Bert model with MLP classifier/regressor head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel

    model_type : string
         specifies 'classifier' or 'regressor' model

    num_labels : int
        For a classifier, this is the number of distinct classes.
        For a regressor his will be 1.

    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code.

    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp.
    """

    def __init__(self, config,
                 model_type="classifier",
                 num_labels=2,
                 num_mlp_layers=2,
                 num_mlp_hiddens=500):

        super(BertPlusMLP, self).__init__(config)
        self.model_type = model_type
        self.num_labels = num_labels
        self.num_mlp_layers = num_mlp_layers
        self.num_mlp_hiddens = num_mlp_hiddens

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size

        self.mlp = MLP(D=self.input_dim,
                       n=self.num_mlp_layers,
                       H=self.num_mlp_hiddens,
                       K=self.num_labels,
                       p=config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, labels=None):

        last_layer, pooled_output = self.bert(input_ids,
                                              segment_ids,
                                              input_mask,
                                              output_all_encoded_layers=False)

        output = self.dropout(pooled_output)
        output = self.mlp(output)

        return output


def UniAMP(my_input_dim, args):
    ########################################################Feature########################################################

    input = Input(shape=(my_input_dim,), name='Protein')

    output = Dense(1024, activation='relu', kernel_initializer='glorot_normal', name='feature_1',
                kernel_regularizer=l2(l2c))(input)
    output = BatchNormalization(axis=-1)(output)
    output = Dropout(dr)(output)


    output = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='feature_2',
                kernel_regularizer=l2(l2c))(output)
    output = BatchNormalization(axis=-1)(output)
    output = Dropout(dr)(output)

    output = Reshape((8,32))(output)

    ##attention
    output_temp = output.get_shape().as_list()
    output_temp = output_temp[2]
    X = Self_Attention(output_temp)(output)
    output = BatchNormalization(axis=-1)(X)
    output = Dropout(0.1)(output)

    output = Flatten()(output)


    output = Dense(64, activation='relu', kernel_initializer='glorot_normal', name='feature_3',
                kernel_regularizer=l2(l2c))(output)
    output = BatchNormalization(axis=-1)(output)
    output = Dropout(dr)(output)

    pre_output = Reshape((1,64))(output)

    ##attention
    output = pre_output.get_shape().as_list()
    output = output[2]
    X = Self_Attention(output)(pre_output)
    pre_output = BatchNormalization(axis=-1)(X)
    pre_output = Dropout(0.1)(pre_output)

    pre_output = Flatten()(pre_output)

    pre_output = Dense(32, activation='relu', kernel_initializer='he_uniform', name='feature_4')(pre_output)
    pre_output = Dropout(dr)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.lr), metrics=['accuracy', get_precision, get_recall, get_F_score, get_MCC])
    # model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.lr), metrics=['accuracy', get_MCC, get_tp, get_fp, get_tn, get_fn])
    return model


