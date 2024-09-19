import numpy as np
import pandas as pd
import torch
import esm_main.esm
import pickle, math
import os, sys
from tqdm import tqdm
from esm_main.esm.pretrained import esm2_t33_650M_UR50D

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
current_dir = os.path.dirname(sys.argv[0])


def run_calculate(data, batch_size=2):
    model, alphabet = esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    model = model.to(device)


    esm2_1280 = []
    data_batch = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    for data_batch_ in tqdm(data_batch):
        batch_labels, batch_strs, batch_tokens = batch_converter(data_batch_)

        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        esm2_vector = [token_representations[i, 1: tokens_len - 1].mean(0).cpu().detach().numpy() for i, tokens_len in
                     enumerate(batch_lens)]

        esm2_1280 += esm2_vector

    return esm2_1280



if __name__ == '__main__':
    sequence = 'MIENEAFLTAEQRIESLLDAMTLEEQVALLAGADFWTTVPIERLGIPAIKVSDGPNGARGGGSLVGGVKAASFPVGIALAASWDPALVKRVGQALAEEALSKGARVLLAPTVNIHRSTLNGRNFECYSEDPHLSARLAVAYIQGLQQNGVGATVKHFVGNESEFERMTISSEIDERALREIYLPPFEAAVKEAKTWALMSSYNKLNGTYVSERADMLLDLLKGEWGFDGVVMSDWFATHSTAPAQNGGLDLEMPGPSRFRG'
    esm2_vector = run_calculate('PNPG', sequence)
    print(esm2_vector)


