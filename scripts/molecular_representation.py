import re #for regex
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import math

# ============================================================
# VOCABULARY
# ============================================================

SMILES_TOKENS = [
    '[PAD]', '[CLS]', '[UNK]', '[MASK]',   # special tokens
    'C', 'N', 'O', 'S', 'F', 'P',          # common atoms
    'Cl', 'Br', 'Si', 'Se', 'I', 'B',      # two-char + less common atoms
    'c', 'n', 'o', 's', 'p',               # aromatic atoms (lowercase)
    '=', '#', '-', '/', '\\',              # bond types
    '(', ')',                              # branches
    '[', ']',                              # bracket atoms
    '+', '-',                              # charges (inside brackets)
    '1', '2', '3', '4', '5', '6',          # ring closures
    '7', '8', '9', '%',                    # more ring closures
    '@', 'H',                              # chirality and hydrogen
]

#creates dictionary like: '[PAD]' : 0 ...
token_to_idx = {}
for idx, tok in enumerate(SMILES_TOKENS):
    token_to_idx[tok] = idx

#0: '[PAD]'...
idx_to_token = {}
for tok, idx in token_to_idx.items():
    idx_to_token[idx] = tok


PAD_IDX = token_to_idx['[PAD]']
CLS_IDX = token_to_idx['[CLS]']
UNK_IDX = token_to_idx['[UNK]']
MASK_IDX = token_to_idx['[MASK]']
VOCAB_SIZE = len(SMILES_TOKENS)

print(f"Vocabulary size: {VOCAB_SIZE}")

# ============================================================
# TOKENIZER
# ============================================================

#regex pattern: tries to match two-char atoms first (Cl, Br, Si, Se),
#then single-char atoms and symbols
SMILES_PATTERN = re.compile(
    r'\[.*?\]|'            # FIRST: anything in brackets as one token
    r'Cl|Br|Si|Se|'        # two-character atoms
    r'[BCFINOPSHbcnopsh]|' # single-character atoms
    r'[\[\]()=#@+\-/\\%]|' # remaining symbols
    r'[0-9]'               # digits
)
#re.compile(): compiles regular expression once and stores it as a regex object
#r in front means raw string

#SMILES string to token strings
def tokenize_smiles(smiles):
    """Convert a SMILES string to a list of token strings."""
    tokens = SMILES_PATTERN.findall(smiles)
    return tokens

#SMILES string -> fixed length list of integers
def encode(smiles, max_length=128):
    """
    Convert a SMILES string to a padded list of token indices.
    Prepends [CLS] token (used later to get the molecule-level representation).
    Truncates or pads to max_length.
    """
    tokens = tokenize_smiles(smiles)
    
    indices = [CLS_IDX]
    for t in tokens:
        if t in token_to_idx:
            indices.append(token_to_idx[t])
        else: 
            indices.append(UNK_IDX) #unknown,when token appears in the SMILES string but isn't in vocabulary

    #truncate if it's too long
    indices = indices[:max_length]

    #pad if it's too short
    padding_length = max_length - len(indices)
    indices = indices + [PAD_IDX] * padding_length

    return indices

def decode(indicies):
    """Convert a list of token indices back to a SMILES string."""
    tokens = []
    for i in indicies:
        if i not in (PAD_IDX, CLS_IDX):
            token = idx_to_token.get(i, '[UNK]')
            tokens.append(token)
    return ''.join(tokens)


# ============================================================
# TOKENIZER TESTS
# ============================================================

test_smiles = [
    'CCO',                          # ethanol
    'CC(=O)O',                      # acetic acid
    'c1ccccc1',                     # benzene
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',   # ibuprofen
    'ClC(Cl)Cl',                    # chloroform 
]

print("\nTokenization test:")
for smi in test_smiles:
    tokens = tokenize_smiles(smi)
    encoded = encode(smi, max_length=32)
    decoded = decode(encoded)
    print(f"\nSMILES:  {smi}")
    print(f"Tokens:  {tokens}")
    print(f"Encoded: {encoded[:len(tokens)+2]}...")
    print(f"Decoded: {decoded}")
    

# ============================================================
# DATASET LOADING
# ============================================================

df = pd.read_csv("/kaggle/input/datasets/simonacholakova/pubchem-data/pubchem_10m.csv")

print(df.columns)
print(df.head())

#tokenize dataset
df["tokens"] = df["smiles"].apply(tokenize_smiles)
print(df[["smiles", "tokens"]].head())

print(df["tokens"][1])

# ============================================================
# EMBEDDING LAYER
# ============================================================

class TokenEmbedding(nn.Module): #inherits from nn.Module
    def __init__(self, vocab_size, embed_dim): #constructor 
        super().__init__() #initializes the parent class (nn.Module)
        self.embedding = nn.Embedding(
            vocab_size, #number of unique tokens in the vocabulary
            embed_dim,  #size of each embedding vector
            padding_idx=PAD_IDX #just leave it as vector of zeros
        )
        #self.embedding: stores the layer as a member variable of the class
        #nn.Embedding: creates a lookup table

    def forward(self, input_ids):
        return self.embedding(input_ids) 
    #input_ids are the token ids
    #PyTorch automatically replaces each token ID with its vector

# ============================================================
# EMBEDDING TESTS
# ============================================================

EMBED_DIM = 128  #size of each token vector

token_embedding = TokenEmbedding(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)

# simulate a batch of 2 molecules, each with max_length=128 tokens
dummy_input = torch.tensor([
    encode('CCO'),           # ethanol
    encode('c1ccccc1'),      # benzene
])

output = token_embedding(dummy_input)
print(f"Input shape:  {dummy_input.shape}")  
print(f"Output shape: {output.shape}")  