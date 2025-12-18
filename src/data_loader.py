import numpy as np
from datasets import load_dataset

# Configuration Globale
MAX_LEN = 128  # On reste raisonnable pour ton CPU (128 nucléotides)
RNA_MAP = {'<PAD>': 0, 'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5}
STRUCT_MAP = {'<PAD>': 0, '.': 1, '(': 2, ')': 3, '[': 4, ']': 5, '{': 6, '}': 7, '>': 8, '<': 9}
INV_STRUCT_MAP = {v: k for k, v in STRUCT_MAP.items()}

def encode_sequence(seq, map_dict, max_len):
    """Encode une séquence et ajoute le padding (les zéros)"""
    # 1. Encodage
    encoded = [map_dict.get(char, map_dict.get('N', 0)) for char in seq]
    
    # 2. Troncature (si trop long)
    encoded = encoded[:max_len]
    
    # 3. Padding (si trop court)
    padding = [0] * (max_len - len(encoded))
    encoded += padding
    
    return np.array(encoded, dtype=np.int32)

def get_bpRNA_data(split="train", limit=5000):
    """Charge les données proprement"""
    print(f"--- Chargement de {limit} séquences depuis HuggingFace ---")
    dataset = load_dataset("multimolecule/bprna", split=split, streaming=True)
    
    X_list = []
    Y_list = []
    count = 0
    
    for item in dataset:
        seq = item['sequence']
        struct = item['secondary_structure']
        
        # On ne prend que ce qui est utile
        if len(seq) <= MAX_LEN and len(seq) > 10:
            x_enc = encode_sequence(seq, RNA_MAP, MAX_LEN)
            y_enc = encode_sequence(struct, STRUCT_MAP, MAX_LEN)
            
            X_list.append(x_enc)
            Y_list.append(y_enc)
            count += 1
            
        if count >= limit:
            break
    
    print(f"--- Données prêtes : {len(X_list)} exemples ---")
    return np.array(X_list), np.array(Y_list)