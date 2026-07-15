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
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tokenizers.processors import TemplateProcessing
import json
import joblib
from sklearn.model_selection import RandomizedSearchCV


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TOXRIC_DIR = os.path.join(DATA_DIR, 'toxric')
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")
 
PUBCHEM_CSV = os.path.join(DATA_DIR, 'pubchem_10m.csv')
SMILES_TXT = os.path.join(ARTIFACTS_DIR, 'pubchem_smiles_10m.txt')
BPE_TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, 'bpe_tokenizer.json')
 
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("PROJECT_DIR:", PROJECT_DIR)
print("DATA_DIR:", DATA_DIR)
print("ARTIFACTS_DIR:", ARTIFACTS_DIR)
print("CHECKPOINTS_DIR:", CHECKPOINTS_DIR)

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

    smiles_for_bpe = df['smiles'].dropna().tolist()[:3_000_000] #3M
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
        min_frequency  = 2,
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
    max_length here needs to match MAX_LENGTH set above, but
    the tokenizer object itself was configured with a fixed length.
    """
    return tokenizer.encode(smiles).ids

def decode(indices):
    """Convert a list of token indices back to a SMILES string."""
    return tokenizer.decode(indices, skip_special_tokens=True)

# ============================================================
# EMBEDDING LAYER
# ============================================================

class TokenEmbedding(nn.Module): 
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
        self.rope = RotaryEmbedding(self.head_dim) #positional embedding

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

    def forward(self, input_ids, attention_mask=None, return_sequence=False):
        #input_ids shape: [batch_size, seq_length]

        #step 1: embed tokens + positions
        x = self.embedding(input_ids)
        #shape: [batch_size, seq_length, embed_dim]

        #step 2: pass through each encoder layer in sequence 
        for layer in self.layers:
            x = layer(x, attention_mask)
        #shape: [batch_size, seq_length, embed_dim]

        if return_sequence:
            return x
            #full per-token output, needed for MLM pretraining

        #step 3: extract [CLS] token vector (position 0)
        cls_output = x[:, 0, :]
        # shape: [batch_size, embed_dim]

        return cls_output
        #this single vector per molecule is what goes into
        #the prediction head for toxicity prediction
        

# ============================================================
# TRANSFORMER VALUES
# ============================================================

EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 6
MAX_LENGTH = 128
DROPOUT = 0.15

RUN_CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, "dim256_newtok")
os.makedirs(RUN_CHECKPOINTS_DIR, exist_ok=True)

# ============================================================
# MLM PRETRAINING
# ============================================================

def mask_tokens(input_ids, mlm_probability=0.15):
    """
    Standard BERT-style masking:
      - 15% of non-special tokens are chosen
      - of those: 80% -> [MASK], 10% -> random token, 10% -> unchanged
    Returns masked_input_ids, labels (labels = -100 for unmasked positions,
    ignored by CrossEntropyLoss)
    """
    input_ids = input_ids.clone()
    labels = input_ids.clone()

    probability_matrix = torch.full(
        labels.shape,
        mlm_probability,
        device=input_ids.device
    )
    #never mask PAD or CLS
    special_mask = (input_ids == PAD_IDX) | (input_ids == CLS_IDX)
    probability_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100 #ignore_index for loss

    #80% -> [MASK]
    indices_replaced = (
        torch.bernoulli(
            torch.full(labels.shape, 0.8, device=input_ids.device)
        ).bool() & masked_indices
    )    
    input_ids[indices_replaced] = MASK_IDX

    #10% random token (of the remaining 20%, half = 10% overall)
    indices_random = (
        torch.bernoulli(
            torch.full(labels.shape, 0.5, device=input_ids.device)
        ).bool() & masked_indices & ~indices_replaced
    )
    random_tokens = torch.randint(
        VOCAB_SIZE,
        labels.shape,
        dtype=torch.long,
        device=input_ids.device
    )
    input_ids[indices_random] = random_tokens[indices_random]

    #remaining 10% left unchanged

    return input_ids, labels


class MLMHead(nn.Module):
    """Predicts the original token id at each masked position."""
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, hidden_states):
        #hidden_states: [batch, seq_len, embed_dim]
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.norm(x)
        return self.decoder(x) #[batch, seq_len, vocab_size]


class UnlabeledSMILESDataset(Dataset):
    """Wraps raw SMILES strings (no labels) for MLM pretraining."""
    def __init__(self, smiles_list, max_length=128):
        self.smiles_list = smiles_list
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        ids = encode(self.smiles_list[idx], self.max_length)
        return torch.tensor(ids, dtype=torch.long)


PRETRAINED_ENCODER_PATH = os.path.join(RUN_CHECKPOINTS_DIR, 'pretrained_encoder_mlm.pt')


def pretrain_encoder_mlm(smiles_list, device, num_epochs=5, batch_size=256, lr=1e-4):
    """
    Pretrains a TransformerEncoder with a masked-language-modeling
    objective on unlabeled SMILES strings. Returns the trained encoder
    (without the MLM head, which is only needed during pretraining).
    """
    dataset = UnlabeledSMILESDataset(smiles_list, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2)

    encoder = TransformerEncoder(
        vocab_size = VOCAB_SIZE,
        embed_dim  = EMBED_DIM,
        num_heads  = NUM_HEADS,
        ff_dim     = FF_DIM,
        num_layers = NUM_LAYERS,
        max_length = MAX_LENGTH,
        dropout    = DROPOUT
    ).to(device)
    mlm_head = MLMHead(EMBED_DIM, VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(mlm_head.parameters()),
        lr=lr, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(num_epochs):
        encoder.train()
        mlm_head.train()
        total_loss = 0.0
        num_seen = 0

        for step, batch_ids in enumerate(loader):
            batch_ids = batch_ids.to(device)
            attention_mask = (batch_ids != PAD_IDX).long()

            masked_ids, labels = mask_tokens(batch_ids)
            masked_ids = masked_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            hidden = encoder(masked_ids, attention_mask, return_sequence=True)
            logits = mlm_head(hidden) #[batch, seq_len, vocab_size]

            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(mlm_head.parameters()), max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item() * batch_ids.size(0)
            num_seen += batch_ids.size(0)

            if step % 200 == 0:
                print(f"  epoch {epoch+1} step {step} loss={loss.item():.4f}")

        print(f"[MLM pretrain] epoch {epoch+1}/{num_epochs} avg_loss={total_loss/num_seen:.4f}")

    return encoder


def load_pretrained_encoder(model, path, device):
    if not os.path.exists(path):
        print(f"WARNING: no pretrained encoder found at {path}, training encoder from scratch")
        return model

    state_dict = torch.load(path, map_location=device)
    model.encoder.load_state_dict(state_dict)
    print(f"Loaded pretrained encoder from {path}")
    return model

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

# ============================================================
# SUPERVISED ENCODER TRAINING (encoder + predictor, end-to-end)
# ============================================================

def train_encoder_supervised(train_smiles, train_labels, val_smiles, val_labels,
                              device, num_epochs=50, batch_size=64,
                              lr=1e-4, patience=8):
    """
    Training MolecularToxicityModel (encoder + prediction head)
    on the toxicity regression task. Returns the model
    """
    train_dataset = ToxicityDataset(train_smiles, train_labels, max_length=MAX_LENGTH)
    val_dataset = ToxicityDataset(val_smiles, val_labels, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = MolecularToxicityModel(
        vocab_size = VOCAB_SIZE,
        embed_dim = EMBED_DIM,
        num_heads = NUM_HEADS,
        ff_dim  = FF_DIM,
        num_layers = NUM_LAYERS,
        num_tasks = 1, #single LD50 target per endpoint
        max_length = MAX_LENGTH,
        dropout = DROPOUT
    ).to(device)

    model = load_pretrained_encoder(model, PRETRAINED_ENCODER_PATH, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        #train 
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device).unsqueeze(1)  # [B, 1]

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss  = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
        train_loss /= len(train_dataset)

        #validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['label'].to(device).unsqueeze(1)

                preds = model(input_ids, attention_mask)
                loss  = criterion(preds, labels)
                val_loss += loss.item() * input_ids.size(0)
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        
        #early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    #restore best weights
    model.load_state_dict(best_state)
    print(f"Best val_loss: {best_val_loss:.4f}")

    return model 


def evaluate_full_model(model, test_smiles, test_labels, device, batch_size=256):
    """
    Evaluate the full encoder+predictor model directly,
    as a baseline to compare against encoder features + XGBoost.
    """
    model.eval()
    all_preds = []

    for i in range(0, len(test_smiles), batch_size):
        batch = test_smiles[i:i+batch_size]
        input_ids = torch.tensor(
            [encode(smi) for smi in batch], dtype=torch.long
        ).to(device)
        attention_mask = (input_ids != PAD_IDX).long()

        with torch.no_grad():
            preds = model(input_ids, attention_mask)

        all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    rmse = np.sqrt(mean_squared_error(test_labels, all_preds))
    r2 = r2_score(test_labels, all_preds)
    return rmse, r2

# ============================================================
# XGBOOST WITH TRANSFORMER REPRESENTATIONS
# ============================================================

def extract_cls_vectors(smiles_list, encoder, device, batch_size=256):
    encoder.eval()
    all_cls = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]

        input_ids = torch.tensor(
            [encode(smi) for smi in batch],
            dtype=torch.long
        ).to(device)

        attention_mask = (input_ids != PAD_IDX).long()

        with torch.no_grad():
            cls = encoder(input_ids, attention_mask)

        all_cls.append(cls.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  {min(i+batch_size, len(smiles_list)):,} / {len(smiles_list):,}")

    return np.concatenate(all_cls, axis=0)

def train_xgboost_with_transformer(endpoint_name, csv_path, device):

    print(f"\n{'='*60}")
    print(f"XGBoost + Transformer Encoder")
    print(f"Endpoint: {endpoint_name}")
    print(f"{'='*60}")

    #load data
    df_endpoint = pd.read_csv(csv_path)
    target_col  = [c for c in df_endpoint.columns
                   if 'LD50' in c or 'LDLo' in c][0]
    smiles_col  = ('Canonical SMILES'
                   if 'Canonical SMILES' in df_endpoint.columns
                   else 'SMILES')
    df_endpoint = df_endpoint.dropna(subset=[smiles_col, target_col])
    print(f"Compounds: {len(df_endpoint)}")

    smiles = df_endpoint[smiles_col].tolist()
    labels = df_endpoint[target_col].values

    #split 
    idx = np.arange(len(smiles))
    idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=42)
    idx_val,   idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

    train_smiles = [smiles[i] for i in idx_train]
    val_smiles   = [smiles[i] for i in idx_val]
    test_smiles  = [smiles[i] for i in idx_test]
    train_labels = labels[idx_train]
    val_labels   = labels[idx_val]
    test_labels  = labels[idx_test]

    #train encoder 
    print(f"\nTraining transformer encoder on {endpoint_name}...")
    full_model = train_encoder_supervised(
        train_smiles, train_labels,
        val_smiles,   val_labels,
        device = device
    )
    encoder = full_model.encoder

    #baseline: how good is the transformer's own prediction, unaided?
    baseline_rmse, baseline_r2 = evaluate_full_model(full_model, test_smiles, test_labels, device)
    print(f"\nBaseline (transformer alone): RMSE={baseline_rmse:.4f}, R2={baseline_r2:.4f}")

    #save the trained encoder 
    endpoint_encoder_path = os.path.join(
        RUN_CHECKPOINTS_DIR, f'encoder_{endpoint_name}.pt'
    )
    torch.save(encoder.state_dict(), endpoint_encoder_path)
    print(f"Saved trained encoder to {endpoint_encoder_path}")

    #extract CLS vectors using the now-trained encoder
    print(f"\nExtracting CLS vectors...")
    print(f"  Train ({len(train_smiles):,}):")
    X_train = extract_cls_vectors(train_smiles, encoder, device)
    print(f"  Val ({len(val_smiles):,}):")
    X_val   = extract_cls_vectors(val_smiles,   encoder, device)
    print(f"  Test ({len(test_smiles):,}):")
    X_test  = extract_cls_vectors(test_smiles,  encoder, device)

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([train_labels, val_labels], axis=0)

    print(f"\nShapes — trainval: {X_trainval.shape}, test: {X_test.shape}")

    #GridSearchCV
    print("\nRunning GridSearchCV (5-fold CV)...")
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    xgb_base = XGBRegressor(random_state=42, n_jobs=4, verbosity=0, eval_metric='rmse')
    grid_search = GridSearchCV(
        estimator=xgb_base, param_grid=param_grid, cv=5,
        scoring='neg_root_mean_squared_error', n_jobs=4, verbose=2, refit=True
    )
    grid_search.fit(X_trainval, y_trainval)

    print(f"\nBest parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

    preds = grid_search.best_estimator_.predict(X_test)
    rmse = np.sqrt(mean_squared_error(test_labels, preds))
    r2 = r2_score(test_labels, preds)

    #save XGBoost model
    xgb_path = os.path.join(RUN_CHECKPOINTS_DIR, f'xgb_{endpoint_name}.joblib')
    joblib.dump(grid_search.best_estimator_, xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")

    #save best hyperparameters + metrics 
    params_path = os.path.join(RUN_CHECKPOINTS_DIR, f'xgb_{endpoint_name}_params.json')
    
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': grid_search.best_params_,
            'best_cv_rmse': -grid_search.best_score_,
            'test_rmse': rmse,
            'test_r2': r2,
        }, f, indent=2)
    print(f"Saved hyperparameters to {params_path}")

    print(f"\n{'='*40}")
    print(f"TEST RESULTS — {endpoint_name}")
    print(f"{'='*40}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return rmse, r2, baseline_rmse, baseline_r2, grid_search.best_estimator_, grid_search.best_params_


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #run MLM pretraining once if we don't already have weights saved
    if not os.path.exists(PRETRAINED_ENCODER_PATH):
        print("\nNo pretrained encoder found — running MLM pretraining on PubChem...")
        pretrain_df = pd.read_csv(PUBCHEM_CSV)
        pretrain_smiles = pretrain_df['smiles'].dropna().tolist()[:5_000_000]
        del pretrain_df
        gc.collect()

        pretrained_encoder = pretrain_encoder_mlm(pretrain_smiles, device)
        torch.save(pretrained_encoder.state_dict(), PRETRAINED_ENCODER_PATH)
        print(f"Saved pretrained encoder to {PRETRAINED_ENCODER_PATH}")

        del pretrain_smiles, pretrained_encoder
        gc.collect()
    else:
        print(f"\nFound existing pretrained encoder at {PRETRAINED_ENCODER_PATH}, skipping pretraining")

    endpoints = {
        'mouse_oral_LD50':            'Acute Toxicity_mouse_oral_LD50.csv',
        'mouse_subcutaneous_LD50':    'Acute Toxicity_mouse_subcutaneous_LD50.csv',
        'mouse_intraperitoneal_LD50': 'Acute Toxicity_mouse_intraperitoneal_LD50.csv',
        'mouse_intravenous_LD50':     'Acute Toxicity_mouse_intravenous_LD50.csv',
        'rat_oral_LD50':              'Acute Toxicity_rat_oral_LD50.csv',
        'rat_intravenous_LD50':       'Acute Toxicity_rat_intravenous_LD50.csv',
        'rat_intraperitoneal_LD50':   'Acute Toxicity_rat_intraperitoneal_LD50.csv',
    }

    results = {}

    for endpoint_name, filename in endpoints.items():
        csv_path = os.path.join(TOXRIC_DIR, filename)

        rmse, r2, baseline_rmse, baseline_r2, best_model, best_params = train_xgboost_with_transformer(
            endpoint_name = endpoint_name,
            csv_path = csv_path,
            device = device
        )

        results[endpoint_name] = {
            'rmse': rmse, 'r2': r2,
            'baseline_rmse': baseline_rmse, 'baseline_r2': baseline_r2
        }

    #summary table
    print(f"\n{'='*60}")
    print("SUMMARY — XGBoost + Transformer")
    print(f"{'='*60}")
    print(f"{'Endpoint':<40}{'RMSE':<10}{'R2'}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<40}{res['rmse']:<10.4f}{res['r2']:.4f}")