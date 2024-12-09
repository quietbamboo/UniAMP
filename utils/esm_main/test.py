import pandas as pd
import torch
import esm
import pickle

# df_train_subs_avail = pd.read_pickle("./dataSets/train_subs_avail.pkl")
# df_test_subs_avail = pd.read_pickle("./dataSets/test_subs_avail.pkl")


# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("Beta-glucosidase", "MIENEAFLTAEQRIESLLDAMTLEEQVALLAGADFWTTVPIERLGIPAIKVSDGPNGARGGGSLVGGVKAASFPVGIALAASWDPALVKRVGQALAEEALSKGARVLLAPTVNIHRSTLNGRNFECYSEDPHLSARLAVAYIQGLQQNGVGATVKHFVGNESEFERMTISSEIDERALREIYLPPFEAAVKEAKTWALMSSYNKLNGTYVSERADMLLDLLKGEWGFDGVVMSDWFATHSTAPAQNGGLDLEMPGPSRFRG"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    plt.title(seq)
    plt.savefig('attention_contacts.png')
    plt.show()

