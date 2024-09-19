from utils.esm_main.run_calculate_esm2 import run_calculate
import pandas as pd
import os, sys
import numpy as np
# current_dir = os.path.dirname(sys.argv[0])
from tqdm import tqdm


protein_names, sequences = [], []
file_path =  r'../data/aeruginosa/benchmark_dataset.fasta'
with open(file_path,'r') as FA:
    for count, value in enumerate(tqdm(FA.readlines())):
        line=value.strip('\n')
        if count % 2 == 0:
            protein_names.append(line)
        else:
            if len(line) > 800:
                protein_names = protein_names[:-1]
                continue
            sequences.append(line)
            
df_lab = pd.DataFrame({'name': protein_names, 'sequences': sequences})


seq_data = [(protein_name_, protein_sequence_) for protein_name_, protein_sequence_ in zip(protein_names, sequences)]

esm2_1280 = run_calculate(seq_data, batch_size=2)
df_lab['ESM2'] = esm2_1280
df_lab.to_json(r'../data/aeruginosa/benchmark_dataset_esm2.json')
