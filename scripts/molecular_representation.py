import re #for regex
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

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

# -------------------------------------------------------
# 1. Positional Embedding
# -------------------------------------------------------

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_length=128):
        super().__init__()
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        #learnable position vectors, one per position (0 to max_length-1)
        #creates lookup table with 128 rows (one per position) and 128 columns (one number per dimension)
    
    def forward(self, x): #x is output from token embedding
        batch_size, seq_length, _ = x.shape
        #x shape: [batch_size, seq_length, embed_dim]

        positions = torch.arange(seq_length)
        #create position indices [0,1,2,...,seq_length-1]
        #position 0 is [CLS] token, position 1 is the first atom...

        pos_embeddings = self.position_embedding(positions)
        #look up position vectors
        #result matrix: [seq_length, embed_dim]
        #so each position returns its 128-number vector
        
        return x + pos_embeddings
        #add position embeddings to token embeddings
        #pos_embeddings gets broadcast across the batch dimension



# -------------------------------------------------------
# 2. Combined Embedding (Token + Positional)
# -------------------------------------------------------

class MoleculeEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length=128, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim) #token IDs to vectors
        self.positional_embedding = PositionalEmbedding(embed_dim, max_length) #position indicies to vectors and adds the token vectors
        self.dropout = nn.Dropout(dropout) #creates a dropout layer with 10% of values set to 0 randomly, prevents overfitting
        #during evaluation, dropout is automatically turned off

    def forward(self, input_ids):
        #input_ids shape: [batch_size, seq_length]

        #step 1: convert token IDs to vectors 
        token_embeds = self.token_embedding(input_ids)
        #shape: [batch_size, seq_length, embed_dim]

        #step 2: add positional info
        x = self.positional_embedding(token_embeds)
        #shape: [batch_size, seq_length, embed_dim]

        #step 3: dropout 
        x = self.dropout(x)

        return x
        

# -------------------------------------------------------
# 3. Apply to dataset
# -------------------------------------------------------

MAX_LENGTH = 128
EMBED_DIM = 128
DROPOUT = 0.1

#initialize the combined embedding
embedding_layer = MoleculeEmbedding(
    vocab_size = VOCAB_SIZE,
    embed_dim = EMBED_DIM,
    max_length = MAX_LENGTH,
    dropout = DROPOUT
)

#encode few molecules from dataset
sample_smiles = df['smiles'].head(4).tolist()
sample_encoded = torch.tensor([encode(smi, MAX_LENGTH) for smi in sample_smiles])
#shape: [4, 128]

print(f"Encoded input shape: {sample_encoded.shape}")

#pass through combined embedding
sample_embedded = embedding_layer(sample_encoded)
print(f"Embedded output shape: {sample_embedded.shape}")

# sanity check: [CLS] token is always at position 0
print(f"\n[CLS] token index in first molecule: {sample_encoded[0][0].item()} (should be {CLS_IDX})")
print(f"[CLS] embedding vector (first 5 values): {sample_embedded[0][0][:5].detach().numpy()}")


# ============================================================
# TRANSFORMER ENCODER
# ============================================================

# ============================================================
# TRANSFORMER ENCODER
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0 
        #embed_dim must be divisible by num_heads because we split we split embedding into num_heads pieces

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads #size of each head

        #linear layers to project input into Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim) #query projection
        self.key = nn.Linear(embed_dim, embed_dim) #key projection
        self.value = nn.Linear(embed_dim, embed_dim) #value projection
        self.output = nn.Linear(embed_dim, embed_dim) #output projection, final layer

        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim) #scaling factor to prevent large dot products 

        def forward(self, x, attention_mask=None):
        #x shape: [batch_size, seq_length, embed_dim]
            batch_size, seq_length, embed_dim = x.shape

        #Step 1: compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        #Step 2: split into multiple heads
        #[batch_size, seg_length, embed_dim] 
        #to [batch_size, seq_length, num_heads, head_dim]
        #to [batch_size, num_heads, seq_length, head_dim]
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        #Step 3: compute attention scores
        #shape: [batch_size, num_heads, seq_length, seq_length]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        #Step 4: apply attention mask, hide padding tokens
        #attention_mask: [batch_size, seq_length]
        #1 means real token, 0 means padding
        #padding positions have score -infinity, so after softmax they become 0 (no attention to padding)
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #shape: [batch_size, 1, 1, seq_length]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        #Step 5: softmax to get attention weights
        #converts scores to probabilities (0-1, sum to 1)
        attention_weights = torch.softmax(scores, dim = -1)
        attention_weights = self.dropout(attention_weights)

        #Step 6: weighted sum of values
        #shape: [batch_size, num_heads, seq_length, head_dim]
        attended = torch.matmul(attention_weights, V)

        #Step 7: concatenate heads back toogether
        #[batch_size, seq_length, num_heads, head_dim]
        #to [batch_size, seq_length, embed_dim]
        attended = attended.transpose(1,2).contigious()
        attended = attended.view(batch_size, seq_length, embed_dim)

        #Step 8: final linear projection
        output = self.output(attended)
        #shape: [batch_size, seq_length, embed_dim]
        return output
        