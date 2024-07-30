import argparse
import os
import json
from transformers import T5Tokenizer, T5EncoderModel

from utils.feature_extraction import *
from utils.my_metric import *
from utils.model import *
from utils.attention import Attention_layer
from utils.cal_protT5 import *

import torch
import torch.nn.functional as F
import pandas as pd
from Bio import SeqIO
from keras.models import load_model, Sequential
from tqdm import tqdm
import time
import subprocess


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_fasta(file_path):
    return {record.id: str(record.seq) for record in SeqIO.parse(file_path, 'fasta')}


def infer_LSTM(args):
    model = load_model(args.model_path,
                       custom_objects={'get_precision': get_precision,
                                       'get_recall': get_recall,
                                       'get_F_score': get_F_score,
                                       'get_MCC': get_MCC})
    sequences_dict = read_fasta(args.dataset_path)

    X_values = encode_sequence_to_vector(list(sequences_dict.values()), 50)
    Y_pred = model.predict(X_values)

    return [(key, sequences_dict[key], Y_pred[ind][0]) for ind, key in enumerate(sequences_dict.keys())]


def infer_ATT(args):
    model = load_model(args.model_path,
                       custom_objects={'Attention_layer': Attention_layer,
                                       'get_precision': get_precision,
                                       'get_recall': get_recall,
                                       'get_F_score': get_F_score,
                                       'get_MCC': get_MCC})
    sequences_dict = read_fasta(args.dataset_path)

    X_values = encode_sequence_to_vector(list(sequences_dict.values()), 50)
    Y_pred = model.predict(X_values)

    return [(key, sequences_dict[key], Y_pred[ind][0]) for ind, key in enumerate(sequences_dict.keys())]


def infer_BERT(args):
    model = torch.load(args.model_path)
    model.to(device)
    sequences_dict = read_fasta(args.dataset_path)

    X_values = np.array([' '.join(list(seq)) for seq in list(sequences_dict.values())])
    X = np.squeeze(X_values)

    X1, X2 = unpack_text_pairs(X)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = TextFeaturesDataset(X1, X2, None,
                                  'classifier',
                                  {0: 0, 1: 1},
                                  128,
                                  tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             args.batch_size,
                                             num_workers=5)

    model.eval()
    Y_pred = []
    batch_iter = tqdm(dataloader, desc="Inferring", leave=True)
    with torch.no_grad():
        for eval_steps, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            output = model(*batch)
            y_pred_prob = F.softmax(output).cpu().numpy()[:, 1].tolist()
            Y_pred += y_pred_prob

            
    return [(key, sequences_dict[key], Y_pred[ind]) for ind, key in enumerate(sequences_dict.keys())]


def infer_UniAMP(args):
    model = load_model(args.model_path,
                       custom_objects={'Self_Attention': Self_Attention,
                                       'get_precision': get_precision,
                                       'get_recall': get_recall,
                                       'get_F_score': get_F_score,
                                       'get_MCC': get_MCC})
    sequences_dict = read_fasta(args.dataset_path)
    # if args.feature == 'pca':
    #     X_values = feature_encode(list(sequences_dict.values()))
    # elif args.feature == 'unirep_protT5':
    if not os.path.exists(r'./data/temp'):
        os.makedirs(r'./data/temp')
    current_time = time.time()
    command = f"tape-embed unirep {args.dataset_path} ./data/temp/temp_input_{str(current_time)}.npz babbler-1900 --tokenizer unirep"
    subprocess.run(command, check=True, shell=True)
    data = np.load(rf'./data/temp/temp_input_{str(current_time)}.npz', allow_pickle=True)
    df = calculate_protT5(args.dataset_path)
    protT5_dict = {row['name'][1:]:row['protT5'] for _, row in df.iterrows()}
    X_values = np.array([np.concatenate((data[key].flatten()[0]['avg'], np.array(protT5_dict[key]))) for key in data.keys()])
    # else:
    #     raise ValueError('Model input error')

    Y_pred = model.predict(X_values)
    # if args.feature == 'pca':
    #     return [(key, sequences_dict[key], Y_pred[ind][0]) for ind, key in enumerate(sequences_dict.keys())]
    # else:
    result_dict = {key: Y_pred[ind][0] for ind, key in enumerate(data.keys())}
    # for key in sequences_dict:
    #     print(key)
    # for key in sequences_dict.keys():
    #     print(key)
    return [(key, sequences_dict[key], result_dict[key]) for key in sequences_dict.keys()]


# def infer_LSTM_feature(args):
#     model = load_model(args.model_path,
#                     custom_objects={'get_precision': get_precision,
#                                     'get_recall': get_recall,
#                                     'get_F_score': get_F_score,
#                                     'get_MCC': get_MCC})

#     sequences_dict = read_fasta(args.dataset_path)


#     X_values = encode_sequence_to_vector(list(sequences_dict.values()), 50)

#     layer_name = 'lstm_1'  
#     intermediate_layer_model = Model(inputs=model.input,
#                                     outputs=model.get_layer(layer_name).output)

#     Y_pred = intermediate_layer_model.predict(X_values)

#     print("Shape of output from LSTM layer:", Y_pred.shape)
#     print("Output values from LSTM layer:", Y_pred)
#     pd.DataFrame(Y_pred).to_csv(r'benchmark_dataset_LSTM.csv', index=False)