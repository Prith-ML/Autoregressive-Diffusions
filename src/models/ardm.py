import torch
import torch.nn as nn
import torch.nn.functional as F

class ARDM(nn.Module):
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
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x
