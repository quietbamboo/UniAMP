import argparse

from utils.model_test import *


def test(args):
    if args.model == 'LSTM':
        result = test_LSTM(args)
        print(f'TP:{result[0]}, FP:{result[1]}, TN:{result[2]}, FN{result[3]}')
    elif args.model == 'BERT':
        result = test_BERT(args)
        print(f'TP:{result[0]}, FP:{result[1]}, TN:{result[2]}, FN{result[3]}')
    elif args.model == 'UniAMP':
        result = test_UniAMP(args)
        print(f'TP:{result[0]}, FP:{result[1]}, TN:{result[2]}, FN{result[3]}')
    elif args.model == 'ATT':
        result = test_ATT(args)
        print(f'TP:{result[0]}, FP:{result[1]}, TN:{result[2]}, FN{result[3]}')
    elif args.model == 'sequences':
        result = test_sequences(args)
        print(f'TP:{result[0]}, FP:{result[1]}, TN:{result[2]}, FN{result[3]}')
    elif args.model == 'MLP':
        result = test_MLP(args)
        print(f'TP:{result[0]}, FP:{result[1]}, TN:{result[2]}, FN{result[3]}')
    else:
        raise ValueError('Model input error')
    result = get_metrics(*result)
    print(f'Acc:{result[0]}, Pre:{result[1]}, Rec:{result[2]}, F_score:{result[3]}, MCC:{result[4]}')
    pass
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='UniAMP',
                        help='The name of the model to test (LSTM/ATT/BERT/UniAMP).')
    parser.add_argument('-model_path', type=str, default=r'./models/model_aeruginosa_UniAMP_uni_protT5.h5',
                        help='Path of the model file (*.h5/*.bin).')
    parser.add_argument('-dataset_path', type=str, default=r'./data/aeruginosa/benchmark_dataset.csv',
                        help='Path of the dataset file (*.csv/*.npz).')
    parser.add_argument('-batch_size', type=int, default=256,
                        help='Test batch size.')
    parser.add_argument('-feature', type=str, default='unirep_protT5',
                        help='Methods for extracting sequence features (sequence/pca/unrep_protT5).')
    parser.add_argument('-comparison', type=str, default='',
                        help='The features used to compare')

    test(parser.parse_args())
