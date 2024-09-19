from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from utils.data import TextFeaturesDataset
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import math
from decimal import *
from Bio import SeqIO

ORIGIN_DATA = {
    'A': [0.62, -0.5, 0.007187, 8.1, 0.046, 1.181, 27.5],
    'C': [0.29, -1.0, -0.036610, 5.5, 0.128, 1.461, 44.6],
    'D': [-0.90, 3.0, -0.023820, 13.0, 0.105, 1.587, 40.0],
    'E': [0.74, 3.0, 0.006802, 12.3, 0.151, 1.862, 62.0],
    'F': [1.19, -2.5, 0.037552, 5.2, 0.290, 2.228, 115.5],
    'G': [0.48, 0.0, 0.179052, 9.0, 0.000, 0.881, 0.0],
    'H': [-0.40, -0.5, -0.010690, 10.4, 0.230, 2.025, 79.0],
    'I': [1.38, -1.8, 0.021631, 5.2, 0.186, 1.810, 93.5],
    'K': [-1.50, 3.0, 0.017708, 11.3, 0.219, 2.258, 100.0],
    'L': [1.06, -1.8, 0.051672, 4.9, 0.186, 1.931, 93.5],
    'M': [0.64, -1.3, 0.002683, 5.7, 0.221, 2.034, 94.1],
    'N': [-0.78, 2.0, 0.005392, 11.6, 0.134, 1.655, 58.7],
    'P': [0.12, 0.0, 0.239531, 8.0, 0.131, 1.468, 41.9],
    'Q': [-0.85, 0.2, 0.049211, 10.5, 0.180, 1.932, 80.7],
    'R': [-2.53, 3.0, 0.043587, 10.5, 0.291, 2.560, 105.0],
    'S': [-0.18, 0.3, 0.004627, 9.2, 0.062, 1.298, 29.3],
    'T': [-0.05, -0.4, 0.003352, 8.6, 0.108, 1.525, 51.3],
    'V': [1.08, -1.5, 0.057004, 5.9, 0.140, 1.645, 71.5],
    'W': [0.81, -3.4, 0.037977, 5.4, 0.409, 2.663, 145.5],
    'Y': [0.26, -2.3, 117.3000, 6.2, 0.298, 2.368, 0.023599],
}

PCPNS = ['H1', 'H2', 'NCI', 'P1', 'P2', 'SASA', 'V']

# AAPCPVS: Physicochemical property values of amino acid
AAPCPVS = {
    'A': { 'H1': 0.62, 'H2':-0.5, 'NCI': 0.007187, 'P1': 8.1, 'P2':0.046, 'SASA':1.181, 'V': 27.5 },
    'C': { 'H1': 0.29, 'H2':-1.0, 'NCI':-0.036610, 'P1': 5.5, 'P2':0.128, 'SASA':1.461, 'V': 44.6 },
    'D': { 'H1':-0.90, 'H2': 3.0, 'NCI':-0.023820, 'P1':13.0, 'P2':0.105, 'SASA':1.587, 'V': 40.0 },
    'E': { 'H1': 0.74, 'H2': 3.0, 'NCI': 0.006802, 'P1':12.3, 'P2':0.151, 'SASA':1.862, 'V': 62.0 },
    'F': { 'H1': 1.19, 'H2':-2.5, 'NCI': 0.037552, 'P1': 5.2, 'P2':0.290, 'SASA':2.228, 'V':115.5 },
    'G': { 'H1': 0.48, 'H2': 0.0, 'NCI': 0.179052, 'P1': 9.0, 'P2':0.000, 'SASA':0.881, 'V':  0.0 },
    'H': { 'H1':-0.40, 'H2':-0.5, 'NCI':-0.010690, 'P1':10.4, 'P2':0.230, 'SASA':2.025, 'V': 79.0 },
    'I': { 'H1': 1.38, 'H2':-1.8, 'NCI': 0.021631, 'P1': 5.2, 'P2':0.186, 'SASA':1.810, 'V': 93.5 },
    'K': { 'H1':-1.50, 'H2': 3.0, 'NCI': 0.017708, 'P1':11.3, 'P2':0.219, 'SASA':2.258, 'V':100.0 },
    'L': { 'H1': 1.06, 'H2':-1.8, 'NCI': 0.051672, 'P1': 4.9, 'P2':0.186, 'SASA':1.931, 'V': 93.5 },
    'M': { 'H1': 0.64, 'H2':-1.3, 'NCI': 0.002683, 'P1': 5.7, 'P2':0.221, 'SASA':2.034, 'V': 94.1 },
    'N': { 'H1':-0.78, 'H2': 2.0, 'NCI': 0.005392, 'P1':11.6, 'P2':0.134, 'SASA':1.655, 'V': 58.7 },
    'P': { 'H1': 0.12, 'H2': 0.0, 'NCI': 0.239531, 'P1': 8.0, 'P2':0.131, 'SASA':1.468, 'V': 41.9 },
    'Q': { 'H1':-0.85, 'H2': 0.2, 'NCI': 0.049211, 'P1':10.5, 'P2':0.180, 'SASA':1.932, 'V': 80.7 },
    'R': { 'H1':-2.53, 'H2': 3.0, 'NCI': 0.043587, 'P1':10.5, 'P2':0.291, 'SASA':2.560, 'V':105.0 },
    'S': { 'H1':-0.18, 'H2': 0.3, 'NCI': 0.004627, 'P1': 9.2, 'P2':0.062, 'SASA':1.298, 'V': 29.3 },
    'T': { 'H1':-0.05, 'H2':-0.4, 'NCI': 0.003352, 'P1': 8.6, 'P2':0.108, 'SASA':1.525, 'V': 51.3 },
    'V': { 'H1': 1.08, 'H2':-1.5, 'NCI': 0.057004, 'P1': 5.9, 'P2':0.140, 'SASA':1.645, 'V': 71.5 },
    'W': { 'H1': 0.81, 'H2':-3.4, 'NCI': 0.037977, 'P1': 5.4, 'P2':0.409, 'SASA':2.663, 'V':145.5 },
    'Y': { 'H1': 0.26, 'H2':-2.3, 'NCI': 117.3000, 'P1': 6.2, 'P2':0.298, 'SASA':2.368, 'V':  0.023599 },
}


def encode_sequence_to_vector(sequences: List[str], length: int = 50) -> np.ndarray:
    dic = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4,
        'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12,
        'P': 13, 'Q': 14, 'R': 15, 'S': 16,
        'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    }

    sample_number = len(sequences)
    vectors = np.zeros((sample_number, length))

    for ind1, sequence in enumerate(sequences):
        len_seq = len(sequence)
        for ind2, c in enumerate(sequence):
            vectors[ind1, ind2 - len_seq] = dic[c]

    return vectors


def get_MBA(protein_sequence: str, lag: int) -> np.ndarray:
    # AA_index represents the letter of the amino acid
    AA_index = 'ACDEFGHIKLMNPQRSTVWY'
    # Remove the 'X' character from protein_sequence
    protein_sequence = protein_sequence.replace('X', '')

    # Calculate the length of protein_sequence
    L1 = len(protein_sequence)

    # Initialize an empty matrix to store descriptor values
    AA_num1 = []

    # Traverse each amino acid in protein_sequence
    for i in range(L1):
        # Find the index of the amino acid in AA_index
        index = AA_index.index(protein_sequence[i])

        # Extract the corresponding descriptor value list and add it to AA_num1
        descriptor_values = ORIGIN_DATA[AA_index[index]]
        AA_num1.append(descriptor_values)

    AA_num1 = np.array(AA_num1)

    MBA1 = np.zeros((lag, AA_num1.shape[1]))

    # Calculate MBA value
    for i in range(lag):
        sum_term = []
        for k in range(0, 7):
            p = 0
            for j in range(L1 - i):
                p = p + (AA_num1[j][k] * AA_num1[j + i][k])
            sum_term.append(p)
        sum_term = np.array(sum_term)

        MBA1[i, :] = (1 / (L1 - i)) * sum_term

    MBA = list(MBA1.T.flatten())
    return MBA


def get_MA(protein_sequence: str, lag: int) -> np.ndarray:
    AA_index = 'ACDEFGHIKLMNPQRSTVWY'
    protein_sequence = protein_sequence.replace('X', '')
    L1 = len(protein_sequence)
    AA_num1 = []

    for i in range(L1):
        index = AA_index.index(protein_sequence[i])
        descriptor_values = ORIGIN_DATA[AA_index[index]]
        AA_num1.append(descriptor_values)

    AA_num1 = np.array(AA_num1)
    mean_ = np.mean(AA_num1, axis=1)
    mean_ = np.reshape(mean_, (L1, 1))
    mean_AA_num1 = np.tile(mean_, (1, 7))
    AA_num = AA_num1 - mean_AA_num1

    MA_denominator = (1 / L1) * np.sum(AA_num ** 2)
    MA = np.zeros((lag, AA_num1.shape[1]))

    for i in range(lag):
        sum_term = []
        for k in range(0, 7):
            p = 0
            for j in range(L1 - i):
                p = p + (AA_num1[j][k] * AA_num1[j + i][k])
            sum_term.append(p)
        sum_term = np.array(sum_term)
        MA[i, :] = (1 / (L1 - i)) * sum_term
        MA[i, :] = MA[i, :] / MA_denominator
    MA = list(MA.T.flatten())
    return MA


def get_GA(protein_sequence: str, lag: int) -> np.ndarray:
    AA_index = 'ACDEFGHIKLMNPQRSTVWY'
    protein_sequence = protein_sequence.replace('X', '')
    L1 = len(protein_sequence)
    AA_num1 = []

    for i in range(L1):
        index = AA_index.index(protein_sequence[i])
        descriptor_values = ORIGIN_DATA[AA_index[index]]
        AA_num1.append(descriptor_values)

    AA_num1 = np.array(AA_num1)
    mean_ = np.mean(AA_num1, axis=1)
    mean_ = np.reshape(mean_, (L1, 1))
    mean_AA_num1 = np.tile(mean_, (1, 7))
    AA_num = AA_num1 - mean_AA_num1

    GA_denominator = (1 / L1) * np.sum(AA_num ** 2)
    GA = np.zeros((lag, AA_num1.shape[1]))

    for i in range(lag):
        sum_term = []
        for k in range(0, 7):
            p = 0
            for j in range(L1 - i):
                p = p + (AA_num[j][k] * AA_num[j + i][k])
            sum_term.append(p)
        sum_term = np.array(sum_term)
        GA[i, :] = 2 * (1 / (L1 - i)) * sum_term
        GA[i, :] = GA[i, :] / GA_denominator

    GA = list(GA.T.flatten())
    return GA


def get_PseAAC(sequence: str, lambd: int = 4) -> np.ndarray:
    sequence = sequence.replace('X', '')
    length = len(sequence)

    aa_order = 'ARNDCQEGHILKMFPSTWYV'
    num_aa = len(aa_order)

    # Properties of different amino acids (Hydrophobicity, Hydrophilicity, SideChainMass)
    aa_properties = {
        'A': [0.62, -0.5, 15], 'R': [-2.53, 3, 101],
        'N': [-0.78, 0.2, 58], 'D': [-0.9, 3, 59],
        'C': [0.29, -1, 47], 'E': [-0.74, 3, 73],
        'Q': [-0.85, 0.2, 72], 'G': [0.48, 0, 1],
        'H': [-0.4, -0.5, 82], 'I': [1.38, -1.8, 57],
        'L': [1.06, -1.8, 57], 'K': [-1.5, 3, 73],
        'M': [0.64, -1.3, 75], 'F': [1.19, -2.5, 91],
        'P': [0.12, 0, 42], 'S': [-0.18, 0.3, 31],
        'T': [-0.05, -0.4, 45], 'W': [0.81, -3.4, 130],
        'Y': [0.26, -2.3, 107], 'V': [1.08, -1.5, 43]
    }

    H1 = np.array([aa_properties[aa][0] for aa in aa_order])
    H2 = np.array([aa_properties[aa][1] for aa in aa_order])
    M = np.array([aa_properties[aa][2] for aa in aa_order])

    H1 = (H1 - np.mean(H1)) / np.std(H1, ddof=1)
    H2 = (H2 - np.mean(H2)) / np.std(H2, ddof=1)
    M = (M - np.mean(M)) / np.std(M, ddof=1)

    data = np.zeros(length)
    f = np.zeros(num_aa)

    for j in range(length):
        for k in range(num_aa):
            if sequence[j] == aa_order[k]:
                data[j] = k
                f[k] += 1

    Theta = np.zeros((lambd, length))
    H = np.vstack((H1, H2, M))

    for i in range(lambd):
        for j in range(length - i):
            Theta[i, j] = np.mean(np.mean((H[:, int(data[j])] - H[:, int(data[j + i])]) ** 2))

    theta = np.zeros(lambd)

    for j in range(lambd):
        theta[j] = np.mean(Theta[j, :length - j])

    f = f / length
    XC = f / (1 + 0.05 * np.sum(theta))

    XC2 = np.zeros(lambd)
    for i in range(lambd):
        XC2[i] = (0.05 * theta[i]) / (1 + 0.05 * np.sum(theta))

    return np.concatenate((XC, XC2))


def avg_sd(NUMBERS):
    AVG = sum(NUMBERS)/len(NUMBERS)
    TEM = [pow(NUMBER-AVG, 2) for NUMBER in NUMBERS]
    DEV = sum(TEM)/len(TEM)
    SD = math.sqrt(DEV)
    return (AVG, SD)

# PCPVS: Physicochemical property values
PCPVS = {'H1':[], 'H2':[], 'NCI':[], 'P1':[], 'P2':[], 'SASA':[], 'V':[]}
for AA, PCPS in AAPCPVS.items():
    for PCPN in PCPNS:
        PCPVS[PCPN].append(PCPS[PCPN])

# PCPASDS: Physicochemical property avg and sds
PCPASDS = {}
for PCP, VS in PCPVS.items():
    PCPASDS[PCP] = avg_sd(VS)

# NORMALIZED_AAPCPVS
NORMALIZED_AAPCPVS = {}
for AA, PCPS in AAPCPVS.items():
    NORMALIZED_PCPVS = {}
    for PCP, V in PCPS.items():
        NORMALIZED_PCPVS[PCP] = (V-PCPASDS[PCP][0])/PCPASDS[PCP][1]
    NORMALIZED_AAPCPVS[AA] = NORMALIZED_PCPVS

def pcp_value_of(AA, PCP):
    """Get physicochemical properties value of amino acid."""
    return NORMALIZED_AAPCPVS[AA][PCP];

def pcp_sequence_of(PS, PCP):
    """Make physicochemical properties sequence of protein sequence."""
    PCPS = []
    for I, CH in enumerate(PS):
        PCPS.append(pcp_value_of(CH, PCP))
    # Centralization
    AVG = sum(PCPS)/len(PCPS)
    for I, PCP in enumerate(PCPS):
        PCPS[I] = PCP - AVG
    return PCPS

def ac_values_of(PS, PCP, LAG):
    """Get ac values of protein sequence."""
    AVS = []
    PCPS = pcp_sequence_of(PS, PCP)
    for LG in range(1, LAG+1):
        SUM = 0
        for I in range(len(PCPS)-LG):
            SUM = SUM + PCPS[I]*PCPS[I+LG]
        SUM = SUM / (len(PCPS)-LG)
        AVS.append(SUM)
    return AVS

def all_ac_values_of(PS, LAG):
    """Get all ac values of protein sequence."""
    AAVS = []
    for PCP in PCPS:
        AVS = ac_values_of(PS, PCP, LAG)
        AAVS = AAVS + AVS
    return AAVS

def ac_code_of(PS):
    """Get ac code of protein sequence."""
    AC_Code = all_ac_values_of(PS, 5)
    # Normalizing AC_Code
    # MIN_CODE = min(AC_Code)
    # MAX_CODE = max(AC_Code)
    # AC_Code = [(N-MIN_CODE)*1.0/(MAX_CODE-MIN_CODE) for N in AC_Code]
    return AC_Code


def VS(rang):
    V = []
    for i in range(1, rang):
        for j in range(1, rang):
            for k in range(1, rang):
                tmp = "VS" + str(i) + str(j) + str(k)
                V.append(tmp)
    return V


# calculating conjoint triad for input sequence
def frequency(seq):
    frequency = []
    for i in range(0, (len(seq) - 3)):
        subSeq = seq[i:i + 3]
        tmp = "VS"
        for j in range(0, 3):
            if ((subSeq[j] == 'A') or (subSeq[j] == 'G') or (subSeq[j] == 'V')):
                tmp += "1"
            elif ((subSeq[j] == 'I') or (subSeq[j] == 'L') or (subSeq[j] == 'F') or (subSeq[j] == 'P')):
                tmp += "2"
            elif ((subSeq[j] == 'Y') or (subSeq[j] == 'M') or (subSeq[j] == 'T') or (subSeq[j] == 'S')):
                tmp += "3"
            elif ((subSeq[j] == 'H') or (subSeq[j] == 'N') or (subSeq[j] == 'Q') or (subSeq[j] == 'W')):
                tmp += "4"
            elif ((subSeq[j] == 'R') or (subSeq[j] == 'K')):
                tmp += "5"
            elif ((subSeq[j] == 'D') or (subSeq[j] == 'E')):
                tmp += "6"
            elif ((subSeq[j] == 'C')):
                tmp += "7"
        frequency.append(tmp)
    return frequency


# Creating frequency_dictionary, and calaculate frequency for eaech conjoint triad
def freq_dict(V, freq):
    frequency_dictionary = {}
    for i in range(0, len(V)):
        key = V[i]
        frequency_dictionary[key] = 0

    for i in range(0, len(freq)):
        frequency_dictionary[freq[i]] = frequency_dictionary[freq[i]] + 1

    # Normalization

    # Getting fmin & fmax
    fmax = int(0)
    fmin = int(0)
    for i in range(0, len(V)):
        key = V[i]
        if(frequency_dictionary[key] > fmax):
            fmax = frequency_dictionary[key]
        if(frequency_dictionary[key] < fmin):
            fmin = frequency_dictionary[key]

    # di = (fi - fmin) / fmax
    for i in range(0, len(V)):
        key = V[i]
        getcontext().prec = 3
        frequency_dictionary[key] = float("{0:.3f}".format(((frequency_dictionary[key] - fmin) / fmax)))

    return frequency_dictionary

def conjoint_triad(sequences: List[str]):
    # Creating vector space
    result = []
    v = VS(8)
    for i in range(0, len(sequences)):
        fi = frequency(sequences[i])
        freqDict = freq_dict(v, fi)
        result.append([freqDict[key] for key in v])
    return result


def Amino_acid_composition(sequences: List[str]):
    acid = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    result = []
    for sequence in sequences:
        dic = {x: 0 for x in acid}
        for c in sequence:
            dic[c] = dic.get(c, 0) + 1
        for key in acid:
            dic[key] /= len(sequence)
        result.append(list(dic.values()))
    return result

def ac_code(sequences: List[str]):
    result = []
    for sequence in sequences:
        # print(sequence)
        ac = ac_code_of(sequence)
        result.append(ac)
    return result


def feature_encode(sequences: List[str], lag: int=5, lambd: int=4):
    result = []
    # aac = Amino_acid_composition(sequences)
    ac = ac_code(sequences)
    ct = conjoint_triad(sequences)
    PseAAC = [list(get_PseAAC(seq, lambd)) for seq in sequences]
    # ad1 = [get_MBA(seq, lag) for seq in sequences]
    # ad2 = [get_MA(seq, lag) for seq in sequences]
    # ad3 = [get_GA(seq, lag) for seq in sequences]
    # print(len(PseAAC[0]), len(ct[0]), len(PseAAC[0]), len(ad1[0]), len(ad2[0]), len(ad3[0]), len(ac[0]))
    print('length check:', len(PseAAC) == len(ac) and len(PseAAC) == len(ct))
    if len(PseAAC) == len(ct) and len(PseAAC) == len(ac):
        for i in range(len(PseAAC)):
            result.append(PseAAC[i] + ct[i] + ac[i])
            # result.append(aac[i] + ct[i] + ad1[i])
    return np.array(result)

def feature_encode_comparison(sequences: List[str], features: List[str], lag: int=5, lambd: int=4):
    result = []
    features_vec = []
    # aac = Amino_acid_composition(sequences)
    for feature in features:
        if feature == 'pseaac':
            PseAAC = [list(get_PseAAC(seq, lambd)) for seq in sequences]
            features_vec.append(PseAAC)
        elif feature == 'ct':
            ct = conjoint_triad(sequences)
            features_vec.append(ct)
        elif feature == 'ac':
            ac = ac_code(sequences)
            features_vec.append(ac)
        elif feature == 'ad1':
            ad1 = [get_MBA(seq, lag) for seq in sequences]
            features_vec.append(ad1)
        elif feature == 'ad2':
            ad2 = [get_MA(seq, lag) for seq in sequences]
            features_vec.append(ad2)
        elif feature == 'ad3':
            ad3 = [get_GA(seq, lag) for seq in sequences]
            features_vec.append(ad3)
    # print(len(PseAAC[0]), len(ct[0]), len(PseAAC[0]), len(ad1[0]), len(ad2[0]), len(ad3[0]), len(ac[0]))
    # print('length check:', len(PseAAC) == len(ac) and len(PseAAC) == len(ct))
    # if len(PseAAC) == len(ct) and len(PseAAC) == len(ac):
    for i in range(len(features_vec[0])):
        sample_vec = []
        for feature in features_vec:
            sample_vec += feature[i]
        result.append(sample_vec)
    # print(len(result))
    # print(len(features_vec[0]))
    # pd.DataFrame(result).to_csv(r'/nfs/my/Huang/czx/amp_screening/data/results/encode.csv')
            # result.append(aac[i] + ct[i] + ad1[i])
    return np.array(result)


def unpack_text_pairs(X):
    """
    Unpack text pairs
    """
    if X.ndim == 1:
        texts_a = X
        texts_b = None
    else:
        texts_a = X[:, 0]
        texts_b = X[:, 1]

    return texts_a, texts_b


def get_dataloaders(X1, X2, y, args):
    """
    Get train and validation dataloaders.

    Parameters
    ----------
    X1 : list of strings
        text_a for input data
    X2 : list of strings
        text_b for input data text pairs
    y : list of string or list of floats)
        labels/targets for data
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = TextFeaturesDataset(X1, X2, y,
                                  'classifier',
                                  {0: 0, 1: 1},
                                  128,
                                  tokenizer)

    val_len = int(len(dataset) * args.val_pro)
    if val_len > 0:
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
        val_dl = DataLoader(val_ds, batch_size=args.val_batch_size,
                            num_workers=5, shuffle=True)
    else:
        val_dl = None
        train_ds = dataset

    train_sampler = RandomSampler(train_ds)

    train_dl = DataLoader(train_ds,
                        #   sampler=train_sampler,
                          batch_size=args.train_batch_size, num_workers=5,
                          drop_last=False,
                          shuffle=True)

    return train_dl, val_dl


def get_optimizer(params, len_train_data, args):
    """
    Get and prepare Bert Adam optimizer.

    Parameters
    ----------
    params :
        model parameters to be optimized
    len_train_data : int
        length of training data

    Returns
    -------
    optimizer : FusedAdam or BertAdam
        Optimizer for training model
    num_opt_steps : int
        number of optimization training steps
    """

    num_opt_steps = len_train_data / args.train_batch_size
    num_opt_steps = int(num_opt_steps) * args.epochs

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]

    optimizer = BertAdam(grouped_params,
                        lr=args.lr,
                        warmup=0.1,
                        t_total=num_opt_steps)

    return optimizer, num_opt_steps


##################################################


if __name__ == '__main__':
    # test()

    pass
