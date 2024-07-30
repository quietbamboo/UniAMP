import argparse
import os
import json
from utils.feature_extraction import *
from utils.my_metric import *
from utils.model import *
from utils.data import *
from utils.attention import Attention_layer

import torch
import pandas as pd
from tqdm import tqdm
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Conv1D, Embedding, MaxPooling1D, Dropout, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_LSTM(args):
    model = Sequential()
    model.add(Embedding(input_dim=21, output_dim=128, input_length=50))
    model.add(Conv1D(filters=64, kernel_size=16, strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=5, padding='valid'))
    model.add(Dropout(0.1))
    model.add(LSTM(units=100, dropout=0.1, implementation=1))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    callbacks_list = [EarlyStopping(monitor='val_get_MCC', patience=args.patience, mode='max'),
                      ModelCheckpoint(
                      filepath=os.path.join(args.save_dir, 'model_LSTM.h5'),
                      monitor='val_get_MCC', mode='max', save_best_only=True),
                      CSVLogger(os.path.join(args.log_dir, 'log_LSTM.csv'),separator=',', append=False),
                      ]
    df = pd.read_csv(args.dataset_path)
    X_train = encode_sequence_to_vector(list(df['sequence']), 50)
    Y_train = np.array(list(df['class']), dtype=np.float)
    print(X_train.shape)
    print(Y_train.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=args.random_seed,
                                                      stratify=Y_train)
    
    model.compile(optimizer=Adam(args.lr), loss='binary_crossentropy',
                  metrics=['accuracy', get_precision, get_recall, get_F_score, get_MCC])
    if args.weight:
        class_weight = compute_class_weight('balanced', np.unique(Y_train), Y_train)
        class_weight = {i: class_weight[i] for i in range(2)}
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=train_args.batch_size,
                            class_weight=class_weight, callbacks=callbacks_list)
    else:
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=args.train_batch_size,
                            callbacks=callbacks_list)


def train_ATT(args):
    model = Sequential()
    model.add(Embedding(input_dim=21, output_dim=128, input_length=50))
    model.add(Conv1D(filters=64, kernel_size=16, strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=5, padding='valid'))
    model.add(Dropout(0.1))
    model.add(Attention_layer())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    callbacks_list = [EarlyStopping(monitor='val_get_MCC', patience=args.patience, mode='max'),
                      ModelCheckpoint(
                      filepath=os.path.join(args.save_dir, 'model_ATT.h5'),
                      monitor='val_get_MCC', mode='max', save_best_only=True),
                      CSVLogger(os.path.join(args.log_dir, 'log_ATT.csv'),separator=',', append=False),
                      ]
    df = pd.read_csv(args.dataset_path)
    X_train = encode_sequence_to_vector(list(df['sequence']), 50)
    Y_train = np.array(list(df['class']), dtype=np.float)
    print(X_train.shape)
    print(Y_train.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=args.random_seed,
                                                      stratify=Y_train)
    
    model.compile(optimizer=Adam(args.lr), loss='binary_crossentropy',
                  metrics=['accuracy', get_precision, get_recall, get_F_score, get_MCC])
    if args.weight:
        class_weight = compute_class_weight('balanced', np.unique(Y_train), Y_train)
        class_weight = {i: class_weight[i] for i in range(2)}
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=train_args.batch_size,
                            class_weight=class_weight, callbacks=callbacks_list)
    else:
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=args.train_batch_size,
                            callbacks=callbacks_list)



def train_BERT(args):
    model = BertPlusMLP.from_pretrained('bert-base-uncased',
                                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE\
                                        /'distributed_{}'.format(-1),
                                        num_labels=2,
                                        model_type='classifier',
                                        num_mlp_layers=2,
                                        num_mlp_hiddens=500)
    model.to(device)

    df = pd.read_csv(args.dataset_path)
    X_values = np.array([' '.join(list(seq)) for seq in df['sequence']])
    Y_values = np.array(list(df['class']), dtype=np.float)
    if args.weight:
        class_weight = compute_class_weight('balanced', np.unique(Y_values), Y_values)
        class_weight = torch.Tensor(class_weight)
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')
        criterion.to(device)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion.to(device)
    
    X = np.squeeze(X_values)
    Y = np.squeeze(Y_values)

    X1, X2 = unpack_text_pairs(X)
    train_dl, val_dl = get_dataloaders(X1, X2, Y, args)


    params = list(model.named_parameters())
    optimizer, num_opt_steps = get_optimizer(params, len(train_dl.dataset), args)
    max_MCC = 0
    patience = args.patience
    result = []
    for epoch in range(args.epochs):
        if patience == 0:
            break
        
        train_loss = []
        val_loss = []
        batch_iter = tqdm(train_dl, desc="Training", leave=True)
        model.train()
        for step, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            _, _, _, y_true = batch

            output = model(*batch[:-1])
            loss = criterion(output, y_true)
            loss = loss.mean()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())
            batch_iter.set_postfix(loss=np.mean(train_loss))

        batch_iter = tqdm(val_dl, desc="Testing", leave=True)
        loss = TP = FP = TN = FN = 0

        model.eval()
        for eval_steps, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            _, _, _, y_true = batch

            with torch.no_grad():
                output = model(*batch[:-1])
                loss = criterion(output, y_true)
            loss=loss.mean()
            val_loss.append(loss.item())
            _, y_pred = torch.max(output, 1)
            TP += torch.sum(y_pred & y_true)
            FP += torch.sum((y_pred == 1) & (y_true == 0))
            TN += torch.sum((y_pred == 0) & (y_true == 0))
            FN += torch.sum((y_pred == 0) & (y_true == 1))
        accuracy = (TP.item() + TN.item()) / (TP.item() + FP.item() + TN.item() +FN.item())
        precision = TP.item() / (TP.item() + FP.item()) if (TP.item() + FP.item()) != 0 else 0
        recall = TP.item() / (TP.item() + FN.item()) if (TP.item() + FN.item()) !=0 else 0
        f_score = 1.25 * precision * recall / (0.25 * precision + recall) if (precision + recall) != 0 else 0
        MCC = (TP.item()*TN.item() - FP.item()*FN.item())/((TP.item()+FP.item())*(TP.item()+FN.item())*(TN.item()+FP.item())*(TN.item()+FN.item()))**0.5 if ((TP.item()+FP.item())*(TP.item()+FN.item())*(TN.item()+FP.item())*(TN.item()+FN.item())) !=0 else 0

        if MCC > max_MCC:
            torch.save(model, os.path.join(args.save_dir, 'model_BERT.bin'))
            patience = args.patience
            max_MCC = MCC
            # torch.save(model, save_path+f'_epoch{epoch+1}.bin')
        else:
            patience -= 1

        print(f'epoch{epoch},train_loss={np.mean(train_loss):.4f},val_loss={np.mean(val_loss):.4f},'
              + f'val_acc={accuracy:.4f},val_pre={precision:.4f},val_rec={recall:.4f},val_f-score={f_score:.4f},val_MCC={MCC:.4f}')
        result.append([epoch, np.mean(val_loss), accuracy, precision, recall, f_score, MCC])
        pd.DataFrame(result, columns=['epoch', 'val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_f_score', 'val_MCC']).to_csv(
            os.path.join(args.log_dir, 'log_BERT.csv'), index=False)


def train_UniAMP(args):
    if args.feature == 'pca':
        model = UniAMP(402, args)
        df = pd.read_csv(args.dataset_path)
        X_train = feature_encode(list(df['sequence']))
        Y_train = np.array(list(df['class']), dtype=np.float)
        callbacks_list = [EarlyStopping(monitor='val_get_MCC', patience=args.patience, mode='max'),
                        ModelCheckpoint(
                        filepath=os.path.join(args.save_dir, 'model_UniAMP_pca.h5'),
                        monitor='val_get_MCC', mode='max', save_best_only=True),
                        CSVLogger(os.path.join(args.log_dir, 'log_UniAMP_pca.csv'),separator=',', append=False),
                        ]
    elif args.feature == 'comparison':
        temp_dict = {'pseaac': 24, 'ct': 343, 'ac': 35, 'ad1': 35, 'ad2': 35, 'ad3': 35}
        features = args.comparison.split('_')
        input_dim = sum([temp_dict[feature] for feature in features])
        print(features, input_dim)
        model = UniAMP(input_dim, args)
        df = pd.read_csv(args.dataset_path)
        X_train = feature_encode_comparison(list(df['sequence']), features)
        print(X_train.shape)
        Y_train = np.array(list(df['class']), dtype=np.float)
        callbacks_list = [EarlyStopping(monitor='val_get_MCC', patience=args.patience, mode='max'),
                        ModelCheckpoint(
                        filepath=os.path.join(args.save_dir, f'model_UniAMP_comparison_{args.comparison}.h5'),
                        monitor='val_get_MCC', mode='max', save_best_only=True),
                        CSVLogger(os.path.join(args.log_dir, f'log_UniAMP_comparison_{args.comparison}.csv'),separator=',', append=False),
                        ]
    elif args.feature == 'unirep_protT5':
        model = UniAMP(2924, args)
        # esm2_dict = load_esm2(args.dataset_path.replace('.csv', '_esm2.json'))
        protT5_dict = load_protT5(args.dataset_path.replace('.csv', '_protT5.json'))
        data = np.load(args.dataset_path.replace('.csv', '.npz'), allow_pickle=True)
        X_train = np.array([np.concatenate((data[key].flatten()[0]['avg'], np.array(protT5_dict[key]))) for key in data.keys()])
        Y_train = np.array([float(key[-1]) for key in data.keys()])
        callbacks_list = [EarlyStopping(monitor='val_get_MCC', patience=args.patience, mode='max'),
                            ModelCheckpoint(
                            filepath=os.path.join(args.save_dir, 'model_UniAMP_uni_protT5.h5'),
                            monitor='val_get_MCC', mode='max', save_best_only=False),
                            CSVLogger(os.path.join(args.log_dir, 'log_UniAMP_uni_protT5.csv'),separator=',', append=False),
                        ]
    else:
        raise ValueError('Feature input error')
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=123, stratify=Y_train)
    if args.weight:
        class_weight = compute_class_weight('balanced', np.unique(Y_train), Y_train)
        class_weight = {i: class_weight[i] for i in range(2)}
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=args.train_batch_size,
                            class_weight=class_weight, callbacks=callbacks_list)
    else:
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=args.train_batch_size,
                            callbacks=callbacks_list)



def train_MLP(args):
    # esm2_dict = load_esm2(args.dataset_path.replace('.csv', '_esm2.json'))
    protT5_dict = load_protT5(args.dataset_path.replace('.csv', '_protT5.json'))
    data = np.load(args.dataset_path.replace('.csv', '.npz'), allow_pickle=True)
    X_train = np.array([np.concatenate((data[key].flatten()[0]['avg'], np.array(protT5_dict[key]))) for key in data.keys()])
    Y_train = np.array([float(key[-1]) for key in data.keys()])

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=123, stratify=Y_train)
    train_dataset = MLPDataset(X_train, Y_train)
    val_dataset = MLPDataset(X_val, Y_val)
    train_dl = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True)

    model = MLP_test(input_size = 2924, hidden_sizes = [512, 256], output_size = 2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    result = []
    max_mcc = 0 
    patience = args.patience
    for epoch in range(args.epochs):
        if patience == 0:
            break
        batch_iter = tqdm(train_dl, desc="Training", leave=True)
        model.train()
        train_loss = []
        TP = FP = TN = FN = 0
        for step, (batch_x, batch_y) in enumerate(batch_iter):
            pred_y = model(batch_x.to(device))
            batch_y = batch_y.long().to(device)
            optimizer.zero_grad()
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()

            _, pred_y = torch.max(pred_y, 1)
            loss = loss.mean()
            train_loss.append(loss.item())
            TP += torch.sum(pred_y & batch_y)
            FP += torch.sum((pred_y == 1) & (batch_y == 0))
            TN += torch.sum((pred_y == 0) & (batch_y == 0))
            FN += torch.sum((pred_y == 0) & (batch_y == 1))
        train_accuracy = (TP.item() + TN.item()) / (TP.item() + FP.item() + TN.item() +FN.item())
        train_precision = TP.item() / (TP.item() + FP.item()) if (TP.item() + FP.item()) != 0 else 0
        train_recall = TP.item() / (TP.item() + FN.item()) if (TP.item() + FN.item()) !=0 else 0
        train_mcc = (TP.item()*TN.item() - FP.item()*FN.item())/((TP.item()+FP.item())*(TP.item()+FN.item())*(TN.item()+FP.item())*(TN.item()+FN.item()))**0.5 if ((TP.item()+FP.item())*(TP.item()+FN.item())*(TN.item()+FP.item())*(TN.item()+FN.item())) !=0 else 0

        batch_iter = tqdm(val_dl, desc="Testing", leave=True)
        model.eval()
        val_loss = []
        TP = FP = TN = FN = 0
        for step, (batch_x, batch_y) in enumerate(batch_iter):
            with torch.no_grad():
                batch_y = batch_y.long().to(device)
                pred_y = model(batch_x.to(device))
                loss = criterion(pred_y, batch_y)
                loss = loss.mean()
                val_loss.append(loss.item())
                _, pred_y = torch.max(pred_y, 1)
                TP += torch.sum(pred_y & batch_y)
                FP += torch.sum((pred_y == 1) & (batch_y == 0))
                TN += torch.sum((pred_y == 0) & (batch_y == 0))
                FN += torch.sum((pred_y == 0) & (batch_y == 1))
        val_accuracy = (TP.item() + TN.item()) / (TP.item() + FP.item() + TN.item() +FN.item())
        val_precision = TP.item() / (TP.item() + FP.item()) if (TP.item() + FP.item()) != 0 else 0
        val_recall = TP.item() / (TP.item() + FN.item()) if (TP.item() + FN.item()) !=0 else 0
        val_mcc = (TP.item()*TN.item() - FP.item()*FN.item())/((TP.item()+FP.item())*(TP.item()+FN.item())*(TN.item()+FP.item())*(TN.item()+FN.item()))**0.5 if ((TP.item()+FP.item())*(TP.item()+FN.item())*(TN.item()+FP.item())*(TN.item()+FN.item())) !=0 else 0
        
        print(f'Epoch {epoch}/{args.epochs}, Train Loss: {np.mean(train_loss)}, '
            f'Train Accuracy: {train_accuracy}, Train Precision: {train_precision}, '
            f'Train Recall: {train_recall}, Train MCC: {train_mcc}'
            f'Val Loss: {np.mean(val_loss)}, '
            f'Val Accuracy: {val_accuracy}, Val Precision: {val_precision}, '
            f'Val Recall: {val_recall}, Val MCC: {val_mcc}')

        if val_mcc >= max_mcc:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_MLP_uni_protT5.h5'))
            patience = args.patience
            max_mcc = val_mcc
        else:
            patience -= 1

        result.append([epoch, np.mean(train_loss), train_accuracy, train_precision, train_recall, train_mcc, 
                        np.mean(val_loss), val_accuracy, val_precision, val_recall, val_mcc])
        pd.DataFrame(result, columns=['epoch', 'train_loss', 'train_acc', 'train_pre', 'train_rec', 'train_mcc', 
                                'val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_mcc']).to_csv(os.path.join(args.log_dir, 'log_UniAMP_uni_protT5.csv'), index=False)
