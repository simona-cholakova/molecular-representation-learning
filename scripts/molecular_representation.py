import re #for regex
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import os
import numpy as np
from sklearn.model_selection import train_test_split
import gc
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TOXRIC_DIR      = os.path.join(DATA_DIR, 'toxric')
ARTIFACTS_DIR   = os.path.join(BASE_DIR, 'artifacts')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
 
PUBCHEM_CSV        = os.path.join(DATA_DIR, 'pubchem_10m.csv')
SMILES_TXT         = os.path.join(ARTIFACTS_DIR, 'pubchem_smiles_10m.txt')
BPE_TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, 'bpe_tokenizer.json')
PRETRAINED_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, 'pretrained_encoder_10m.pt')
 
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ============================================================
# BPE TOKENIZER
# ============================================================

if os.path.exists(BPE_TOKENIZER_PATH):
    print(f"Found existing tokenizer at {BPE_TOKENIZER_PATH}, loading it...")
    tokenizer = Tokenizer.from_file(BPE_TOKENIZER_PATH)

else:
    print("No existing tokenizer found, training a new one...")

    print("Loading PubChem dataset...")
    df = pd.read_csv(PUBCHEM_CSV)
    print(f"Loaded {len(df):,} molecules")

    smiles_for_bpe = df['smiles'].dropna().tolist()[:5_000_000] #5M
    print(f"Using {len(smiles_for_bpe):,} molecules for BPE training")

    #free df from memory
    del df
    gc.collect()

    #write to file in chunks
    print("Writing SMILES to file...")

    chunk_size = 100_000
    with open(SMILES_TXT, 'w') as f:
        for i in range(0, len(smiles_for_bpe), chunk_size):
            chunk = smiles_for_bpe[i:i+chunk_size]
            f.write('\n'.join(chunk) + '\n')
            print(f"  Written {min(i+chunk_size, len(smiles_for_bpe)):,} / {len(smiles_for_bpe):,}")

    del smiles_for_bpe
    gc.collect()
    print("File written, memory freed")

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    special_tokens = ['[PAD]', '[CLS]', '[UNK]', '[MASK]']

    trainer = BpeTrainer(
        vocab_size     = 512,
        min_frequency  = 10,
        special_tokens = special_tokens,
        show_progress  = True
    )

    tokenizer.pre_tokenizer = Split(
        pattern=r'\[.*?\]|Cl|Br|Si|Se|[BCFINOPSHbcnopsh]|[\[\]()=#@+\-/\\%]|[0-9]',
        behavior='isolated'
    )

    print("Training BPE tokenizer...")
    tokenizer.train([SMILES_TXT], trainer)
    tokenizer.save(BPE_TOKENIZER_PATH)
    print(f"Saved to {BPE_TOKENIZER_PATH}")

print(f"BPE vocabulary size: {tokenizer.get_vocab_size()}")

# ============================================================
# BPE TOKENIZER SETUP 
# ============================================================

from tokenizers.processors import TemplateProcessing

#automatic [CLS] prepending
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A",
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]"))],
)

MAX_LENGTH = 128

#fixed-length padding/truncation
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]"),
    pad_token="[PAD]",
    length=MAX_LENGTH,
)
tokenizer.enable_truncation(max_length=MAX_LENGTH)

PAD_IDX = tokenizer.token_to_id("[PAD]")
CLS_IDX = tokenizer.token_to_id("[CLS]")
UNK_IDX = tokenizer.token_to_id("[UNK]")
MASK_IDX = tokenizer.token_to_id("[MASK]")
VOCAB_SIZE = tokenizer.get_vocab_size()

print(f"BPE Vocabulary size: {VOCAB_SIZE}")
print(f"PAD={PAD_IDX} CLS={CLS_IDX} UNK={UNK_IDX} MASK={MASK_IDX}")

def tokenize_smiles(smiles):
    """Convert a SMILES string to a list of token strings (for inspection)."""
    return tokenizer.encode(smiles).tokens

def encode(smiles, max_length=128):
    """
    Convert a SMILES string to a padded/truncated list of token indices.
    [CLS] is prepended automatically by the post-processor.
    NOTE: max_length here just needs to match MAX_LENGTH set above —
    the tokenizer object itself was configured with a fixed length.
    """
    return tokenizer.encode(smiles).ids

def decode(indices):
    """Convert a list of token indices back to a SMILES string."""
    return tokenizer.decode(indices, skip_special_tokens=True)

# ============================================================
# PUBCHEM DATASET LOADING
# ============================================================

df = pd.read_csv(PUBCHEM_CSV)

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

# -------------------------------------------------------
# Embedding 
# -------------------------------------------------------

class MoleculeEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim) #token IDs to vectors
        #self.positional_embedding = PositionalEmbedding(embed_dim, max_length) #position indicies to vectors and adds the token vectors
        self.dropout = nn.Dropout(dropout) #creates a dropout layer with 10% of values set to 0 randomly, prevents overfitting
        #during evaluation, dropout is automatically turned off

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)

        x = self.dropout(x)

        return x
        
MAX_LENGTH = 128
EMBED_DIM = 128
DROPOUT = 0.1

#initialize the combined embedding
embedding_layer = MoleculeEmbedding(
    vocab_size = VOCAB_SIZE,
    embed_dim = EMBED_DIM,
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
# RoPE positional embedding
# ============================================================

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary(x, cos, sin):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    x_rot = torch.stack(
        [
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ],
        dim=-1,
    )

    return x_rot.flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_length=128):
        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        positions = torch.arange(max_length).float()

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k):
        """
        q,k:
        [batch, heads, seq_len, head_dim]
        """

        seq_len = q.shape[2]

        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        return q, k

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
        self.rope = RotaryEmbedding(self.head_dim)

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
        Q, K = self.rope(Q, K)

        #Step 3: compute attention scores
        #shape: [batch_size, num_heads, seq_length, seq_length]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        #Step 4: apply attention mask, hide padding tokens
        #attention_mask: [batch_size, seq_length]
        #1 means real token, 0 means padding
        #padding positions have score -infinity, so after softmax they become 0 (no attention to padding)
        if attention_mask is not None:
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
        attended = attended.transpose(1,2).contiguous()
        attended = attended.view(batch_size, seq_length, embed_dim)

        #Step 8: final linear projection
        output = self.output(attended)
        #shape: [batch_size, seq_length, embed_dim]
        return output
        

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        #2 linear layers with relu in between
        #ff_dim is typically 4x embed_dim (expand then compress)
        self.linear1 = nn.Linear(embed_dim, ff_dim) #expand
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, embed_dim) #compress back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x shape: [batch_size, seq_length, embed_dim]
        x = self.linear1(x) #[batch_size, seq_length, ff_dim]
        x = self.relu(x) 
        x = self.dropout(x)
        x = self.linear2(x) #[batch_size, seq_length, embed_dim]
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        #2 sublayers
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        #layer normalization after each sublayer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        #x shape: [batch_size, seq_length, embed_dim]

        #Sublayer 1: multihead attention + residual + norm
        attended = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attended))
        #x + attended = residual connection
        #norm1 = layer normalization

        #Sublayer 2: feed-forward + residual + norm
        f_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(f_forward))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, 
                 num_layers, max_length=128, dropout=0.1):
        super().__init__()

        #embedding layer (token + positional)
        self.embedding = MoleculeEmbedding(vocab_size, embed_dim, dropout)

        #N encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        #input_ids shape: [batch_size, seq_length]

        #step 1: embed tokens + positions
        x = self.embedding(input_ids)
        #shape: [batch_size, seq_length, embed_dim]

        #step 2: pass through each encoder layer in sequence 
        for layer in self.layers:
            x = layer(x, attention_mask)
        #shape: [batch_size, seq_length, embed_dim]

        #step 3: extract [CLS] token vector (position 0)
        cls_output = x[:, 0, :]
        # shape: [batch_size, embed_dim]

        return cls_output
        #this single vector per molecule is what goes into
        #the prediction head for toxicity prediction
        

# ============================================================
# TRANSFORMER TEST
# ============================================================

EMBED_DIM  = 128
NUM_HEADS  = 8
FF_DIM     = 512
NUM_LAYERS = 4
MAX_LENGTH = 128
DROPOUT    = 0.1

#build the transformer
transformer = TransformerEncoder(
    vocab_size  = VOCAB_SIZE,
    embed_dim   = EMBED_DIM,
    num_heads   = NUM_HEADS,
    ff_dim      = FF_DIM,
    num_layers  = NUM_LAYERS,
    max_length  = MAX_LENGTH,
    dropout     = DROPOUT
)

#test with 4 molecules from dataset
sample_smiles = df['smiles'].head(4).tolist()
sample_ids    = torch.tensor([encode(smi, MAX_LENGTH) for smi in sample_smiles])
sample_mask   = (sample_ids != PAD_IDX).long()

#forward pass
cls_vectors = transformer(sample_ids, sample_mask)

print(f"Input shape:      {sample_ids.shape}")    #should be [4, 128]
print(f"CLS output shape: {cls_vectors.shape}")   #should be [4, 128]
print(f"\nFirst molecule CLS vector (first 5 values):")
print(cls_vectors[0][:5].detach().numpy())

#count parameters
total_params = sum(p.numel() for p in transformer.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")

# ============================================================
# PREDICTION HEAD
# ============================================================

class ToxicityPredictor(nn.Module):
    def __init__(self, embed_dim, num_tasks, dropout=0.1):
        #num_tasks is the number of values to be predicted (7 LD50)
        super().__init__()
        #takes the 128 number CLS vector and maps it to num_tasks outputs
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), #128->64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim //2, num_tasks) #64->7
        )
        #nn.Sequential runs the layers automatically one after another
    
    def forward(self, cls_vector):
        #cls_vector shape: [batch_size, embed_dim]
        return self.predictor(cls_vector)
        #output shape: [batch_size, num_tasks]

# ============================================================
# FULL MODEL (Transformer + Prediction Head)
# ============================================================

class MolecularToxicityModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
                num_layers, num_tasks, max_length=128, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size  = vocab_size,
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            ff_dim      = ff_dim,
            num_layers  = num_layers,
            max_length  = max_length,
            dropout     = dropout
        )
        self.predictor = ToxicityPredictor(embed_dim, num_tasks, dropout)

    def forward(self, input_ids, attention_mask=None):
        #step 1: get molecule representation from transformer
        cls_vector = self.encoder(input_ids, attention_mask)
        #shape: [batch_size, embed_dim]

        #step 2: predict toxicity values
        predictions = self.predictor(cls_vector)
        #shape: [batch_size, num_tasks]
        return predictions

# ============================================================
# TOXRIC DATASET
# ============================================================

class ToxicityDataset(Dataset):
    def __init__(self, smiles_list, labels, max_length=128):
        self.smiles_list = smiles_list
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        encoded = encode(smi, self.max_length)

        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = (input_ids != PAD_IDX).long()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


# # # ============================================================
# # # MLM PRETRAINING
# # # ============================================================

# class MLMDataset(Dataset):
#     """
#     Takes SMILES strings, tokenizes them, randomly masks 15% of tokens,
#     and returns:
#     - input_ids: tokenized SMILES with some tokens replaced by [MASK]
#     - attention_mask: 1 for real tokens, 0 for padding
#     - labels: original token ids (-100 for unmasked positions,
#               real token id for masked positions)
#     """
#     def __init__(self, smiles_list, max_length=128, mask_prob=0.15):
#         self.smiles_list = smiles_list
#         self.max_length = max_length
#         self.mask_prob = mask_prob
    
#     def __len__(self):
#         return len(self.smiles_list)
    
#     def __getitem__(self, idx):
#         smi = self.smiles_list[idx]
#         encoded = encode(smi, self.max_length)  #list of token indices, length 128
        
#         input_ids = encoded.copy()
#         labels = [-100] * self.max_length  #-100 = ignore in loss by default
        
#         for i in range(self.max_length):
#             token_id = input_ids[i]
            
#             #never mask special tokens or padding
#             if token_id in (PAD_IDX, CLS_IDX, UNK_IDX, MASK_IDX):
#                 continue
            
#             #randomly decide whether to mask this token
#             if torch.rand(1).item() < self.mask_prob:
#                 labels[i] = token_id  #remember the original token for the loss
                
#                 r = torch.rand(1).item()
#                 if r < 0.80:
#                     #80% replace with [MASK]
#                     input_ids[i] = MASK_IDX
#                 elif r < 0.90:
#                     #10% replace with a random token
#                     input_ids[i] = torch.randint(4, VOCAB_SIZE, (1,)).item()
#                     #start from 4 to skip special tokens
#                 #10% keep original token unchanged
#                 #but still predict it in the loss
        
#         return {
#             'input_ids': torch.tensor(input_ids, dtype=torch.long),
#             'attention_mask': torch.tensor(
#                 [1 if t != PAD_IDX else 0 for t in input_ids],
#                 dtype=torch.long
#             ),
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }


# class MLMHead(nn.Module):
#     """
#     Prediction head for MLM pretraining.
#     Takes the full sequence output from the transformer encoder
#     (not just CLS) and predicts the original token at each masked position.
#     """
#     def __init__(self, embed_dim, vocab_size):
#         super().__init__()
#         self.dense     = nn.Linear(embed_dim, embed_dim)
#         self.relu      = nn.ReLU()
#         self.norm      = nn.LayerNorm(embed_dim)
#         self.projector = nn.Linear(embed_dim, vocab_size)
    
#     def forward(self, x):
#         #x shape: [batch_size, seq_length, embed_dim]
#         x = self.dense(x)
#         x = self.relu(x)
#         x = self.norm(x)
#         x = self.projector(x)
#         #output shape: [batch_size, seq_length, vocab_size]
#         #for each position, a probability over every token in vocabulary
#         return x


# class TransformerEncoderForMLM(nn.Module):
#     """
#     Full encoder that returns ALL token outputs (not just CLS),
#     needed for MLM since we predict at every masked position.
#     """
#     def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
#                  num_layers, dropout=0.1):
#         super().__init__()
#         self.embedding = MoleculeEmbedding(
#             vocab_size, embed_dim, dropout)
#         self.layers = nn.ModuleList([
#             EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
#             for _ in range(num_layers)
#         ])
    
#     def forward(self, input_ids, attention_mask=None):
#         x = self.embedding(input_ids)
#         for layer in self.layers:
#             x = layer(x, attention_mask)
#         return x  #[batch_size, seq_length, embed_dim]—ALL positions

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


# class MLMModel(nn.Module):
#     """
#     Full MLM pretraining model:
#     encoder (learns representations) + MLM head (predicts masked tokens)
#     After pretraining, only the encoder is kept for fine-tuning.
#     """
#     def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
#                  num_layers, max_length=128, dropout=0.1):
#         super().__init__()
#         self.encoder = TransformerEncoderForMLM(
#             vocab_size, embed_dim, num_heads, ff_dim,
#             num_layers, dropout
#         )
#         self.mlm_head = MLMHead(embed_dim, vocab_size)
    
#     def forward(self, input_ids, attention_mask=None):
#         #get representations for all token positions
#         x = self.encoder(input_ids, attention_mask)
#         #predict token at each position
#         logits = self.mlm_head(x)
#         return logits
#         #shape: [batch_size, seq_length, vocab_size]


# # ============================================================
# # PRETRAINING SETUP
# # ============================================================

# PRETRAIN_SAMPLE = 10_000_000  

# pubchem_smiles = df['smiles'].dropna().tolist()

# #shuffle and take sample
# import random
# random.seed(42)
# random.shuffle(pubchem_smiles)
# pubchem_smiles = pubchem_smiles[:PRETRAIN_SAMPLE]

# print(f"Pretraining on {len(pubchem_smiles):,} molecules")

# #split into train/val
# n_pretrain = int(0.95 * len(pubchem_smiles))
# pretrain_smiles = pubchem_smiles[:n_pretrain]
# preval_smiles   = pubchem_smiles[n_pretrain:]

# pretrain_dataset = MLMDataset(pretrain_smiles, max_length=MAX_LENGTH)
# preval_dataset   = MLMDataset(preval_smiles,   max_length=MAX_LENGTH)

# pretrain_loader = DataLoader(
#     pretrain_dataset, batch_size=128, shuffle=True,  num_workers=4)
# preval_loader   = DataLoader(
#     preval_dataset,   batch_size=128, shuffle=False, num_workers=4)

# print(f"Pretrain batches: {len(pretrain_loader):,}")
# print(f"Preval batches:   {len(preval_loader):,}")

# # ============================================================
# # PRETRAINING LOOP
# # ============================================================

# PRETRAIN_EPOCHS = 10
# PRETRAIN_LR     = 1e-4

# mlm_model = MLMModel(
#     vocab_size = VOCAB_SIZE,
#     embed_dim  = EMBED_DIM,
#     num_heads  = NUM_HEADS,
#     ff_dim     = FF_DIM,
#     num_layers = NUM_LAYERS,
#     max_length = MAX_LENGTH,
#     dropout    = DROPOUT
# )

# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     mlm_model = nn.DataParallel(mlm_model)

# mlm_model = mlm_model.to(device)

# pretrain_optimizer = torch.optim.Adam(
#     mlm_model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)

# #CrossEntropyLoss with ignore_index=-100
# #so positions where labels=-100 (unmasked) don't contribute to loss
# criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)

# best_preval_loss = float('inf')

# for epoch in range(PRETRAIN_EPOCHS):
    
#     #train 
#     mlm_model.train()
#     train_loss  = 0.0
#     train_acc   = 0.0
#     train_count = 0
    
#     for batch in pretrain_loader:
#         input_ids      = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels         = batch['labels'].to(device)
        
#         pretrain_optimizer.zero_grad()
        
#         logits = mlm_model(input_ids, attention_mask)
#         #logits shape: [batch, seq_len, vocab_size]
#         #labels shape: [batch, seq_len]
        
#         #reshape for CrossEntropyLoss:
#         #expects [N, C] predictions and [N] targets
#         loss = criterion_mlm(
#             logits.view(-1, VOCAB_SIZE),
#             labels.view(-1)
#         )
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), 1.0)
#         pretrain_optimizer.step()
        
#         #track accuracy on masked tokens only
#         masked_positions = labels.view(-1) != -100
#         if masked_positions.sum() > 0:
#             preds   = logits.view(-1, VOCAB_SIZE).argmax(dim=-1)
#             correct = (preds[masked_positions] == 
#                       labels.view(-1)[masked_positions]).sum().item()
#             train_acc   += correct
#             train_count += masked_positions.sum().item()
        
#         train_loss += loss.item()
    
#     train_loss /= len(pretrain_loader)
#     train_acc   = train_acc / train_count if train_count > 0 else 0
    
#     #validate 
#     mlm_model.eval()
#     val_loss  = 0.0
#     val_acc   = 0.0
#     val_count = 0
    
#     with torch.no_grad():
#         for batch in preval_loader:
#             input_ids      = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels         = batch['labels'].to(device)
            
#             logits = mlm_model(input_ids, attention_mask)
#             loss   = criterion_mlm(
#                 logits.view(-1, VOCAB_SIZE),
#                 labels.view(-1)
#             )
#             val_loss += loss.item()
            
#             masked_positions = labels.view(-1) != -100
#             if masked_positions.sum() > 0:
#                 preds   = logits.view(-1, VOCAB_SIZE).argmax(dim=-1)
#                 correct = (preds[masked_positions] == 
#                           labels.view(-1)[masked_positions]).sum().item()
#                 val_acc   += correct
#                 val_count += masked_positions.sum().item()
    
#     val_loss /= len(preval_loader)
#     val_acc   = val_acc / val_count if val_count > 0 else 0
    
#     if val_loss < best_preval_loss:
#         best_preval_loss = val_loss
#         #save pretrained encoder weights
#         encoder_to_save = mlm_model.module.encoder if isinstance(mlm_model, nn.DataParallel) else mlm_model.encoder
#         torch.save(encoder_to_save.state_dict(), PRETRAINED_ENCODER_PATH)
#         torch.save(encoder_to_save.state_dict(), f'{CHECKPOINTS_DIR}/pretrained_encoder_epoch{epoch+1}.pt')
#         print(f"  → saved to {PRETRAINED_ENCODER_PATH}")
    
#     print(f"Epoch {epoch+1:3d}/{PRETRAIN_EPOCHS}  "
#           f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  "
#           f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

# print(f"\nPretraining complete. Best val loss: {best_preval_loss:.4f}")
# print(f"Pretrained encoder saved to: {PRETRAINED_ENCODER_PATH}")


# # ============================================================
# # USING PRETRAINED ENCODER'S WEIGHTS FOR TOXICITY PREDICTION
# # ============================================================

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# def train_and_evaluate(endpoint_name, csv_path, epochs=50, use_pretrained=False):
#     print(f"\n{'='*60}")
#     print(f"Endpoint: {endpoint_name}")
#     if use_pretrained:
#         print(f"Mode: PRETRAINED encoder")
#     else:
#         print(f"Mode: FROM SCRATCH")
#     print(f"{'='*60}")
    
#     #load data 
#     df_endpoint = pd.read_csv(csv_path)
    
#     target_col = [c for c in df_endpoint.columns if 'LD50' in c or 'LDLo' in c][0]
#     smiles_col = 'Canonical SMILES' if 'Canonical SMILES' in df_endpoint.columns else 'SMILES'
    
#     df_endpoint = df_endpoint.dropna(subset=[smiles_col, target_col])
#     print(f"Compounds: {len(df_endpoint)}")
    
#     smiles = df_endpoint[smiles_col].tolist()
#     labels = df_endpoint[target_col].values
    
#     #split 
#     idx = np.arange(len(smiles))
#     idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=42)
#     idx_val, idx_test   = train_test_split(idx_temp, test_size=0.5, random_state=42)
    
#     #normalize 
#     train_labels = labels[idx_train]
#     label_mean   = train_labels.mean()
#     label_std    = train_labels.std()
#     labels_norm  = (labels - label_mean) / label_std
    
#     #datasets
#     train_dataset = ToxicityDataset(
#         [smiles[i] for i in idx_train], labels_norm[idx_train])
#     val_dataset   = ToxicityDataset(
#         [smiles[i] for i in idx_val],   labels_norm[idx_val])
#     test_dataset  = ToxicityDataset(
#         [smiles[i] for i in idx_test],  labels_norm[idx_test])
    
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
#     test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    
#     #model 
#     model = MolecularToxicityModel(
#         vocab_size = VOCAB_SIZE,
#         embed_dim  = EMBED_DIM,
#         num_heads  = NUM_HEADS,
#         ff_dim     = FF_DIM,
#         num_layers = NUM_LAYERS,
#         num_tasks  = 1,
#         max_length = MAX_LENGTH,
#         dropout    = DROPOUT
#     ).to(device)
    
#     #load pretrained encoder weights if requested
#     PRETRAINED_PATH = PRETRAINED_ENCODER_PATH

#     if use_pretrained:
#         if not os.path.exists(PRETRAINED_PATH):
#             print(f"  ERROR: {PRETRAINED_PATH} not found!")
#             return None, None
#         model.encoder.load_state_dict(
#             torch.load(PRETRAINED_PATH, map_location=device)
#     )
#         print("Loaded pretrained encoder weights successfully")
    
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=1e-4, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', patience=3, factor=0.5)
#     criterion = nn.MSELoss()
    
#     best_val_loss    = float('inf')
#     best_model_state = None
    
#     #training 
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         for batch in train_loader:
#             input_ids      = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels_batch   = batch['label'].to(device)
            
#             optimizer.zero_grad()
#             predictions = model(input_ids, attention_mask).squeeze(-1)
#             loss = criterion(predictions, labels_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)
        
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids      = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 labels_batch   = batch['label'].to(device)
#                 predictions = model(input_ids, attention_mask).squeeze(-1)
#                 val_loss += criterion(predictions, labels_batch).item()
#         val_loss /= len(val_loader)
#         scheduler.step(val_loss)
        
#         if val_loss < best_val_loss:
#             best_val_loss    = val_loss
#             best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
#         if (epoch + 1) % 10 == 0:
#             print(f"  Epoch {epoch+1:3d}/{epochs}  "
#                   f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")
    
#     #test
#     model.load_state_dict(best_model_state)
#     model.eval()
    
#     all_preds, all_labels_list = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids      = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels_batch   = batch['label'].to(device)
#             predictions = model(input_ids, attention_mask).squeeze(-1)
#             all_preds.append(predictions.cpu().numpy())
#             all_labels_list.append(labels_batch.cpu().numpy())
    
#     all_preds       = np.concatenate(all_preds)       * label_std + label_mean
#     all_labels_list = np.concatenate(all_labels_list) * label_std + label_mean
    
#     rmse = np.sqrt(np.mean((all_preds - all_labels_list) ** 2))
#     r2   = 1 - np.sum((all_labels_list - all_preds) ** 2) / \
#                np.sum((all_labels_list - all_labels_list.mean()) ** 2)
    
#     print(f"\n  RMSE: {rmse:.4f}  R2: {r2:.4f}")
#     return rmse, r2


# # ============================================================
# # FROM SCRATCH + PRETRAINED
# # ============================================================

# endpoint_files = {
#     'mouse_oral_LD50':            'Acute Toxicity_mouse_oral_LD50 (1).csv',
#     'mouse_subcutaneous_LD50':    'Acute Toxicity_mouse_subcutaneous_LD50 (1).csv',
#     'mouse_intraperitoneal_LD50': 'Acute Toxicity_mouse_intraperitoneal_LD50 (1).csv',
#     'mouse_intravenous_LD50':     'Acute Toxicity_mouse_intravenous_LD50 (1).csv',
#     'rat_oral_LD50':              'Acute Toxicity_rat_oral_LD50 (1).csv',
#     'rat_intravenous_LD50':       'Acute Toxicity_rat_intravenous_LD50 (1).csv',
#     'rat_intraperitoneal_LD50':   'Acute Toxicity_rat_intraperitoneal_LD50 (1).csv',
# }

# print(f"pretrained_encoder.pt exists: {os.path.exists(PRETRAINED_ENCODER_PATH)}")

# #from scratch
# print("\n" + "="*60)
# print("PHASE 1: FROM SCRATCH")
# print("="*60)

# scratch_results = {}
# for endpoint_name, csv_file in endpoint_files.items():
#     rmse, r2 = train_and_evaluate(
#         endpoint_name,
#         os.path.join(TOXRIC_DIR, csv_file),
#         epochs=50,
#         use_pretrained=False
#     )
#     scratch_results[endpoint_name] = {'rmse': rmse, 'r2': r2}

# #PHASE 1: FROM SCRATCH (already ran, hardcoded results)
# scratch_results = {
#     'mouse_oral_LD50':            {'rmse': 0.5312, 'r2': 0.2458},
#     'mouse_subcutaneous_LD50':    {'rmse': 0.7332, 'r2': 0.3530},
#     'mouse_intraperitoneal_LD50': {'rmse': 0.5553, 'r2': 0.3593},
#     'mouse_intravenous_LD50':     {'rmse': 0.5649, 'r2': 0.3782},
#     'rat_oral_LD50':              {'rmse': 0.6914, 'r2': 0.4227},
#     'rat_intravenous_LD50':       {'rmse': 0.8062, 'r2': 0.3774},
#     'rat_intraperitoneal_LD50':   {'rmse': 0.7705, 'r2': 0.2171},
# }

# #pretrained encoder 
# print("\n" + "="*60)
# print("PHASE 2: WITH PRETRAINED ENCODER")
# print("="*60)

# pretrained_results = {}
# for endpoint_name, csv_file in endpoint_files.items():
#     rmse, r2 = train_and_evaluate(
#         endpoint_name,
#         os.path.join(TOXRIC_DIR, csv_file),
#         epochs=50,
#         use_pretrained=True
#     )
#     pretrained_results[endpoint_name] = {'rmse': rmse, 'r2': r2}

# # ============================================================
# # COMPARISON TABLE
# # ============================================================

# print(f"\n{'='*75}")
# print(f"FINAL RESULTS: From Scratch vs Pretrained vs TOXRIC Benchmarks")
# print(f"{'='*75}")
# print(f"{'Endpoint':35s} {'Scratch':>10s} {'Pretrained':>12s} {'Diff':>8s} {'Better?':>8s}")
# print(f"{'-'*75}")

# for endpoint in scratch_results:
#     r_scratch    = scratch_results[endpoint]['rmse']
#     r_pretrained = pretrained_results[endpoint]['rmse']
#     diff         = r_pretrained - r_scratch
#     better       = '✓ YES' if diff < 0 else '✗ NO'
#     print(f"{endpoint:35s} {r_scratch:10.4f} {r_pretrained:12.4f} "
#           f"{diff:+8.4f} {better:>8s}")

