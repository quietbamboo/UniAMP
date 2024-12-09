import numpy as np
import pandas as pd
import torch
import esm
import pickle, os, math
import os
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

dirPath = os.getcwd()
os.chdir(dirPath)

df_avail = pd.read_pickle("/nfs/my/Xu/jicm/esm_main/dataset/brenda_df_mutant_seqs_lite.pkl")

# reading Sequence col
seq_data = [(protein_index, protein_sequence) for protein_index, protein_sequence in zip(df_avail['EC'], df_avail['Sequence'])]


all_file_names = os.listdir('/nfs/my/Xu/jicm/esm_main/results')
batch_size = 1

data_type_name = 'df_lite'

# """
# todo 
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model = model.to(device)
# """

for data, df_result in zip([seq_data], [df_avail]):
    data_batch = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    sequence_representations = []
    count = 0
    for data_batch_ in tqdm(data_batch):
        if data_type_name+'_'+str(count)+'.pkl' in all_file_names:
            count+=1
            continue

        torch.cuda.empty_cache()
        # # 通过写入二进制模式打开文件
        # pickle_file = open('/nfs/my/Xu/jicm/esm_main/mylist_epoch2.pkl', 'wb')

        # # 序列化对象，将对象obj保存到文件file中去
        # pickle.dump(data_batch_, pickle_file)
        # # 关闭文件
        # pickle_file.close()

        # """
        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        print(data_batch_[0][1])
        print(len(data_batch_[0][1]))
        if len(data_batch_[0][1]) > 400:
            continue
        batch_labels, batch_strs, batch_tokens = batch_converter(data_batch_)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)


        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

        temp_reps = [token_representations[i, 1 : tokens_len - 1].mean(0).cpu().detach().numpy() for i, tokens_len in enumerate(batch_lens)]
        df_data = pd.DataFrame(temp_reps)
        df_data.to_pickle('/nfs/my/Xu/jicm/esm_main/results/'+data_type_name+'_'+str(count)+'.pkl')
        # """

        # os.system('/home/coder/miniconda/envs/km_predict/bin/python /nfs/my/Xu/jicm/esm_main/calculate.py {} {}'.format(count, data_type_name))
        count+=1




