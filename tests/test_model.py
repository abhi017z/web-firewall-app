from __future__ import annotations

import torch

from src.models.transformer_model import WAFTransformer


def test_forward_shapes():
    model = WAFTransformer(vocab_size=100, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.1, max_len=16)
    input_ids = torch.randint(0, 100, (2, 16))
    mask = torch.ones_like(input_ids)
    logits = model(input_ids, mask)
    assert logits.shape == (2, 16, 100)


