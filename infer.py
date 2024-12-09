import argparse

import pandas as pd

from utils.model_infer import *


def infer(args):
    if args.model == 'LSTM':
        result = infer_LSTM(args)
    elif args.model == 'BERT':
        result = infer_BERT(args)
    elif args.model == 'UniAMP':
        result = infer_UniAMP(args)
    elif args.model == 'ATT':
        result = infer_ATT(args)
    else:
        raise ValueError('Model input error')
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    pd.DataFrame(result, columns=['seq_name', 'sequence', 'score']).to_csv(args.save_path, index=False)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='UniAMP',
                        help='The name of the model to infer (LSTM/ATT/BERT/UniAMP).')
    parser.add_argument('-model_path', type=str, default=r'./models/model_aeruginosa_UniAMP_uni_protT5.h5',
                        help='Path of the model file (*.h5/*.bin).')
    parser.add_argument('-dataset_path', type=str, default=r'./data/aeruginosa/benchmark_dataset.fasta',
                        help='Path of the dataset file (*.fasta).')
    parser.add_argument('-save_path', type=str, default=r'./data/results/results.csv',
                        help='Path of the save file (*.csv).')
    parser.add_argument('-batch_size', type=int, default=256,
                        help='Infer batch size.')
    parser.add_argument('-feature', type=str, default='uni',
                        help='Methods for extracting sequence features (sequence/pca/uni).')

    infer(parser.parse_args())
