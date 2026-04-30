from __future__ import annotations

import os
from typing import Dict, List

import torch

from .transformer_model import WAFTransformer


class InferenceEngine:
    """Caches model in memory and provides fast prediction APIs."""

    def __init__(self, model_path: str, threshold: float) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.model: WAFTransformer | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        if self.model is not None:
            return
        # Minimal config for demonstration; replace with config-driven initialization
        ckpt = torch.load(self.model_path, map_location=self.device)
        meta = ckpt.get("meta", {"vocab_size": 10000, "embed_dim": 256, "num_heads": 8, "num_layers": 4, "max_len": 512})
        model = WAFTransformer(
            vocab_size=int(meta.get("vocab_size", 10000)),
            embed_dim=int(meta.get("embed_dim", 256)),
            num_heads=int(meta.get("num_heads", 8)),
            num_layers=int(meta.get("num_layers", 4)),
            ff_dim=int(meta.get("ff_dim", 512)),
            dropout=float(meta.get("dropout", 0.1)),
            max_len=int(meta.get("max_len", 512)),
        )
        model.load_state_dict(ckpt["model"])  # type: ignore[index]
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_single(self, input_ids: List[int], attention_mask: List[int]) -> Dict[str, float | bool]:
        assert self.model is not None, "Model not loaded"
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(ids, mask)
        score = float(errors[0].item())
        is_anomaly = score > self.threshold
        confidence = min(max((score - self.threshold) / max(self.threshold, 1e-6), 0.0), 1.0)
        return {"is_anomaly": is_anomaly, "anomaly_score": score, "confidence": confidence}

    def predict_batch(self, batch_inputs: List[List[int]], batch_masks: List[List[int]]) -> List[Dict[str, float | bool]]:
        assert self.model is not None, "Model not loaded"
        ids = torch.tensor(batch_inputs, dtype=torch.long, device=self.device)
        mask = torch.tensor(batch_masks, dtype=torch.long, device=self.device)
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(ids, mask)
        out: List[Dict[str, float | bool]] = []
        for e in errors.tolist():
            score = float(e)
            is_anomaly = score > self.threshold
            confidence = min(max((score - self.threshold) / max(self.threshold, 1e-6), 0.0), 1.0)
            out.append({"is_anomaly": is_anomaly, "anomaly_score": score, "confidence": confidence})
        return out

    def update_threshold(self, new_threshold: float) -> None:
        self.threshold = float(new_threshold)


__all__ = ["InferenceEngine"]


