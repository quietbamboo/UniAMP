import argparse

from utils.model_train import *


def train(args):
    if args.model == 'LSTM':
        train_LSTM(args)
    elif args.model == 'BERT':
        train_BERT(args)
    elif args.model == 'UniAMP':
        train_UniAMP(args)
    elif args.model == 'ATT':
        train_ATT(args)
    elif args.model == 'MLP':
        train_MLP(args)
    # elif args.model == 'KAN':
    #     train_KAN(args)
    else:
        raise ValueError('Model input error')
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='UniAMP',
                        help='The name of the model to be trained (LSTM/ATT/BERT/UniAMP).')
    parser.add_argument('-feature', type=str, default='unirep_protT5',
                        help='Methods for extracting sequence features (sequence/pca/unrep_protT5).')
    parser.add_argument('-dataset_path', type=str, default=r'./data/aeruginosa/training_dataset.csv',
                        help='Path of the dataset file (*.csv/*.npz/*.json).')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('-epochs', type=int, default=300,
                        help='Training epochs.')
    parser.add_argument('-patience', type=int, default=30,
                        help='Training patience.')
    parser.add_argument('-train_batch_size', type=int, default=256,
                        help='Training batch size.')
    parser.add_argument('-val_batch_size', type=int, default=256,
                        help='Validation batch size.')
    parser.add_argument('-save_dir', type=str, default=r'./models',
                        help='Training batch size.')
    parser.add_argument('-log_dir', type=str, default=r'./logs',
                        help='Training batch size.')
    parser.add_argument('-random_seed', type=int, default=123,
                        help='Random seed for splitting training and validation.')
    parser.add_argument('-val_pro', type=float, default=0.2,
                        help='Proportion of validation dataset.')
    parser.add_argument('-weight', action='store_true', default=False,
                        help='Whether to use the weight method.')
    parser.add_argument('-comparison', type=str, default='',
                        help='The features used to compare')
    train(parser.parse_args())
