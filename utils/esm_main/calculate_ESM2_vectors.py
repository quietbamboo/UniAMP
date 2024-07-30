import numpy as np
import pandas as pd
import torch
import esm
import pickle

df_train_subs_avail = pd.read_pickle("./dataset/train_subs_avail.pkl")
df_test_subs_avail = pd.read_pickle("./dataset/test_subs_avail.pkl")

# reading Sequence col
train_data = [(protein_index, protein_sequence) for protein_index, protein_sequence in zip(df_train_subs_avail['EC'], df_train_subs_avail['Sequence'])]
test_data = [(protein_index, protein_sequence) for protein_index, protein_sequence in zip(df_test_subs_avail['EC'], df_test_subs_avail['Sequence'])]

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

for data, df_result in zip([train_data[:2], test_data[:2]], [df_train_subs_avail, df_test_subs_avail]):
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    df_result.insert(len(df_result), 'ESM_2_t33_650M_UR50D', np.array(sequence_representations))
    pass

