import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from transformers import AutoModelForSeq2SeqLM
except Exception:
    AutoModelForSeq2SeqLM = None

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


class BARTWithOverwriteGate(nn.Module):
    """Wrapper around a pretrained seq2seq denoiser (e.g., BART) with an overwrite gate head.

    Usage:
      - Call set_encoder_inputs(input_ids, attention_mask) for the source sequence
      - Call forward(decoder_input_ids) to get logits, overwrite_probs, hidden
    """

    def __init__(self, model_name: str = 'facebook/bart-base'):
        super().__init__()
        if AutoModelForSeq2SeqLM is None:
            raise ImportError("transformers not installed; please install to use BARTWithOverwriteGate")
        self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hidden_dim = self.seq2seq.config.d_model
        self.overwrite_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._encoder_outputs = None
        # Expose decoder start/pad IDs for proper BOS selection in decoding
        self.decoder_start_token_id = getattr(self.seq2seq.config, 'decoder_start_token_id', None)
        self.pad_token_id = getattr(self.seq2seq.config, 'pad_token_id', None)
        self.eos_token_id = getattr(self.seq2seq.config, 'eos_token_id', None)

    @torch.no_grad()
    def set_encoder_inputs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        encoder = self.seq2seq.get_encoder()
        self._encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Save original inputs for generate()
        self._last_source_input_ids = input_ids
        self._last_source_attention_mask = attention_mask

    def forward(self, decoder_input_ids: torch.Tensor):
        if self._encoder_outputs is None:
            raise RuntimeError("Encoder inputs not set. Call set_encoder_inputs(...) before decoding.")
        outputs = self.seq2seq(
            encoder_outputs=self._encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        logits = outputs.logits  # [B,L,V]
        hidden = outputs.decoder_hidden_states[-1]  # [B,L,H]
        overwrite_probs = self.overwrite_gate(hidden).squeeze(-1)  # [B,L]
        return logits, overwrite_probs, hidden
