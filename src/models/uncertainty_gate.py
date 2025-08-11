import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyARDM(nn.Module):
    def __init__(self, vocab_size=64, max_seq_len=16, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 4, batch_first=True), 
            num_layers=2
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Simple overwrite gate
        self.overwrite_gate = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.embedding(x)
        hidden = self.transformer(x)
        logits = self.output(hidden)
        overwrite_probs = self.overwrite_gate(hidden).squeeze(-1)
        return logits, overwrite_probs, hidden
