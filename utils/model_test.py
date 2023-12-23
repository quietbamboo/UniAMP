import argparse
import os

from utils.feature_extraction import *
from utils.my_metric import *
from utils.model import *
from utils.attention import Attention_layer

import torch
import torch.nn.functional as F
import pandas as pd
from keras.models import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_LSTM(args):
    model = load_model(args.model_path,
                       custom_objects={'get_precision': get_precision,
                                       'get_recall': get_recall,
                                       'get_F_score': get_F_score,
                                       'get_MCC': get_MCC})
    df = pd.read_csv(args.dataset_path)
    X_values = encode_sequence_to_vector(list(df['sequence']), 50)
    Y_values = np.array(list(df['class']), dtype=np.float)
    Y_pred = model.predict(X_values)

    fpr, tpr, thresholds = roc_curve(Y_values, Y_pred)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

    Y_pred = np.round(Y_pred)
    TP, FP, TN, FN = 0, 0, 0, 0
    for (y_true, y_pred) in zip(Y_values, Y_pred):
        if y_true == 1:
            if y_pred == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred == 1:
                FP += 1
            else:
                TN += 1
    return TP, FP, TN, FN


def test_ATT(args):
    model = load_model(args.model_path,
                       custom_objects={'Attention_layer': Attention_layer,
                                       'get_precision': get_precision,
                                       'get_recall': get_recall,
                                       'get_F_score': get_F_score,
                                       'get_MCC': get_MCC})
    df = pd.read_csv(args.dataset_path)
    X_values = encode_sequence_to_vector(list(df['sequence']), 50)
    Y_values = np.array(list(df['class']), dtype=np.float)
    Y_pred = model.predict(X_values)

    fpr, tpr, thresholds = roc_curve(Y_values, Y_pred)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

    Y_pred = np.round(Y_pred)
    TP, FP, TN, FN = 0, 0, 0, 0
    for (y_true, y_pred) in zip(Y_values, Y_pred):
        if y_true == 1:
            if y_pred == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred == 1:
                FP += 1
            else:
                TN += 1
    return TP, FP, TN, FN


def test_BERT(args):
    model = torch.load(args.model_path)
    model.to(device)

    df = pd.read_csv(args.dataset_path)
    X_values = np.array([' '.join(list(seq)) for seq in df['sequence']])
    Y_values = np.array(list(df['class']), dtype=np.float)

    X = np.squeeze(X_values)
    Y = np.squeeze(Y_values)

    X1, X2 = unpack_text_pairs(X)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = TextFeaturesDataset(X1, X2, Y,
                                  'classifier',
                                  {0: 0, 1: 1},
                                  128,
                                  tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             args.batch_size,
                                             num_workers=5)
    model.eval()
    TP = FP = TN = FN = 0
    Y_pred = []
    batch_iter = tqdm(dataloader, desc="Testing", leave=True)
    with torch.no_grad():
        for eval_steps, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            _, _, _, y_true = batch
            output = model(*batch[:-1])
            y_pred_prob = F.softmax(output).cpu().numpy()[:, 1].tolist()
            Y_pred += y_pred_prob
            _, y_pred = torch.max(output, 1)
            TP += torch.sum(y_pred & y_true)
            FP += torch.sum((y_pred == 1) & (y_true == 0))
            TN += torch.sum((y_pred == 0) & (y_true == 0))
            FN += torch.sum((y_pred == 0) & (y_true == 1))

    fpr, tpr, thresholds = roc_curve(Y_values, Y_pred)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

    return TP, FP, TN, FN


def test_UniAMP(args):
    model = load_model(args.model_path,
                       custom_objects={'Self_Attention': Self_Attention,
                                       'get_precision': get_precision,
                                       'get_recall': get_recall,
                                       'get_F_score': get_F_score,
                                       'get_MCC': get_MCC})
    if args.feature == 'pca':
        df = pd.read_csv(args.dataset_path)
        X_values = feature_encode(list(df['sequence']))
        Y_values = np.array(list(df['class']), dtype=np.float)
    elif args.feature == 'uni':
        data = np.load(args.dataset_path, allow_pickle=True)
        X_values = np.array([data[key].flatten()[0]['avg'] for key in data.keys()])
        Y_values = np.array([float(key[-1]) for key in data.keys()])
    else:
        raise ValueError('Feature input error')

    Y_pred = model.predict(X_values)

    fpr, tpr, thresholds = roc_curve(Y_values, Y_pred)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

    Y_pred = np.round(Y_pred)
    TP, FP, TN, FN = 0, 0, 0, 0
    for (y_true, y_pred) in zip(Y_values, Y_pred):
        if y_true == 1:
            if y_pred == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred == 1:
                FP += 1
            else:
                TN += 1
    return TP, FP, TN, FN


def test_sequences(args):
    model_ATT = load_model(r'/nfs/my/Huang/czx/amp_screening/models/model_ATT_aeruginosa.h5',
                           custom_objects={'Attention_layer': Attention_layer,
                                           'get_precision': get_precision,
                                           'get_recall': get_recall,
                                           'get_F_score': get_F_score,
                                           'get_MCC': get_MCC})
    model_LSTM = load_model(r'/nfs/my/Huang/czx/amp_screening/models/model_LSTM_aeruginosa.h5',
                            custom_objects={'Attention_layer': Attention_layer,
                                            'get_precision': get_precision,
                                            'get_recall': get_recall,
                                            'get_F_score': get_F_score,
                                            'get_MCC': get_MCC})
    model_BERT = torch.load(r'/nfs/my/Huang/czx/amp_screening/models/model_BERT_aeruginosa.bin')
    model_BERT.to(device)

    df = pd.read_csv(args.dataset_path)
    X_values = encode_sequence_to_vector(list(df['sequence']), 50)
    Y_values = np.array(list(df['class']), dtype=np.float)
    Y_pred_LSTM = model_LSTM.predict(X_values)
    Y_pred_LSTM = np.round(Y_pred_LSTM)
    Y_pred_ATT = model_ATT.predict(X_values)
    Y_pred_ATT = np.round(Y_pred_ATT)

    X_values = np.array([' '.join(list(seq)) for seq in df['sequence']])
    X = np.squeeze(X_values)
    Y = np.squeeze(Y_values)

    X1, X2 = unpack_text_pairs(X)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = TextFeaturesDataset(X1, X2, Y,
                                  'classifier',
                                  {0: 0, 1: 1},
                                  128,
                                  tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             args.batch_size,
                                             num_workers=5)
    model_BERT.eval()
    Y_pred_BERT = []
    batch_iter = tqdm(dataloader, desc="Testing", leave=True)
    with torch.no_grad():
        for eval_steps, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            _, _, _, y_true = batch
            output = model_BERT(*batch[:-1])
            _, y_pred = torch.max(output, 1)
            Y_pred_BERT += y_pred.cpu().numpy().tolist()

    print(len(Y_pred_LSTM), len(Y_pred_ATT), len(Y_pred_BERT), len(Y_values))
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(Y_pred_LSTM)):
        if Y_values[i] == 1:
            if Y_pred_LSTM[i] == 1 and Y_pred_ATT[i] == 1 and Y_pred_BERT[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if Y_pred_LSTM[i] == 1 and Y_pred_ATT[i] == 1 and Y_pred_BERT[i] == 1:
                FP += 1
            else:
                TN += 1
    return TP, FP, TN, FN
