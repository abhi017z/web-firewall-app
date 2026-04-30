from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class WAFTransformer(nn.Module):
    """Transformer-based autoencoder for sequence reconstruction."""

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, ff_dim: int = 512, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.positional = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.size()
        positions = torch.arange(0, seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.embed(input_ids) + self.positional(positions)
        key_padding_mask = attention_mask == 0
        memory = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # Teacher forcing: use input as target query (could shift right)
        decoded = self.decoder(x, memory, tgt_key_padding_mask=key_padding_mask, memory_key_padding_mask=key_padding_mask)
        logits = self.proj(decoded)
        return logits

    @torch.no_grad()
    def get_reconstruction_error(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        # Cross-entropy per position
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        bsz, seqlen, vocab = logits.size()
        loss = loss_fn(logits.view(bsz * seqlen, vocab), input_ids.view(-1))
        loss = loss.view(bsz, seqlen)
        # Mask pad positions
        mask = attention_mask.float()
        masked = (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return masked


class AnomalyDetector:
    """Wraps a model and exposes thresholded anomaly detection."""

    def __init__(self, model: WAFTransformer, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        err = self.model.get_reconstruction_error(input_ids, attention_mask)
        is_anom = err > self.threshold
        return is_anom, err


__all__ = ["WAFTransformer", "AnomalyDetector"]


