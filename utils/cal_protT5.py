import torch
import re
import pandas as pd
import os, sys
# current_dir = os.path.dirname(sys.argv[0])
from tqdm import tqdm, trange
from transformers import T5Tokenizer, T5EncoderModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pre_models_path = r'./utils/protT5'

def run_calculate(df, batch_size, tokenizer, model):
    df_data = df.copy()
    all_names = df_data['name'].values.tolist()
    sequences = df_data['sequences']
    lengths = [len(seq) for seq in sequences]

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_preprocess = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    # batch
    for start_idx in trange(0, len(sequence_preprocess), batch_size):
        names = all_names[start_idx: start_idx+batch_size]
        batch_sequences = sequence_preprocess[start_idx: start_idx+batch_size]
        length_of_sequences = lengths[start_idx: start_idx+batch_size]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer(batch_sequences, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
        for idx_in_batch in range(len(batch_sequences)):
            name_ = names[idx_in_batch]
            embedding = embedding_repr.last_hidden_state[idx_in_batch, :length_of_sequences[idx_in_batch]].mean(dim=0)
            if df_data.loc[start_idx+idx_in_batch, 'name'] == name_:
                df_data.loc[start_idx+idx_in_batch, 'protT5'] = embedding.cpu().numpy()
                # print(df_data.loc[start_idx+idx_in_batch, 'protT5'].shape)

    return df_data

def calculate_protT5(input_file_path):
    
    protein_names, sequences = [], []
    with open(input_file_path, 'r') as FA:
        for count, value in enumerate(tqdm(FA.readlines())):
            line = value.strip('\n')
            if count % 2 == 0:
                protein_names.append(line)
                # print(line)
            else:
                if len(line) > 1000:
                    protein_names = protein_names[:-1]
                    continue
                sequences.append(line)

    df_result = pd.DataFrame({'name': protein_names, 'sequences': sequences})
    df_result['protT5'] = ''
    
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(pre_models_path, do_lower_case=False)
    # Load the model
    model = T5EncoderModel.from_pretrained(pre_models_path).to(device)

    # calculate
    df_result = run_calculate(df=df_result, batch_size=128, tokenizer=tokenizer, model=model)

    # if not os.path.exists(current_dir + '/result'):
    #     os.makedirs(current_dir + '/result')
    return df_result
    

if __name__ == '__main__':
    input_file_path =  r'../data/aeruginosa/benchmark_dataset.fasta'
    df_result = calculate_protT5(input_file_path)
    df_result.to_json(r'../data/aeruginosa/benchmark_dataset_protT5.json')

    