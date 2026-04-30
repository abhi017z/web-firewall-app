from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[List[int]], attention_masks: List[List[int]]):
        self.sequences = sequences
        self.attention_masks = attention_masks

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sequences)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.attention_masks[idx], dtype=torch.long),
        )


def load_training_data(data_path: str) -> Dataset:
    """Load processed data from a jsonl directory with input_ids and attention_mask."""
    sequences: List[List[int]] = []
    masks: List[List[int]] = []
    if os.path.isdir(data_path):
        files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".jsonl")]
    else:
        files = [data_path]
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sequences.append(list(map(int, obj["input_ids"])))
                masks.append(list(map(int, obj["attention_mask"])))
    return SequenceDataset(sequences, masks)


def _loss_fn(logits: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, vocab = logits.size()
    loss = nn.functional.cross_entropy(logits.view(bsz * seqlen, vocab), target_ids.view(-1), reduction="none")
    loss = loss.view(bsz, seqlen)
    mask = attention_mask.float()
    masked = (loss * mask).sum() / (mask.sum() + 1e-8)
    return masked


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_val = float("inf")
    patience = 5
    stale = 0
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for input_ids, attn in train_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            logits = model(input_ids, attn)
            loss = _loss_fn(logits, input_ids, attn)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for input_ids, attn in val_loader:
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                logits = model(input_ids, attn)
                val_total += _loss_fn(logits, input_ids, attn).item()

        if val_total < best_val:
            best_val = val_total
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for input_ids, attn in test_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            logits = model(input_ids, attn)
            total += _loss_fn(logits, input_ids, attn).item()
            n += 1
    return {"loss": total / max(n, 1)}


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])  # type: ignore[index]
    optimizer.load_state_dict(state["optimizer"])  # type: ignore[index]
    return int(state.get("epoch", 0))


__all__ = [
    "SequenceDataset",
    "load_training_data",
    "train_model",
    "evaluate_model",
    "save_checkpoint",
    "load_checkpoint",
]


