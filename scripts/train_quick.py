from __future__ import annotations

"""
Quick training script:
- Loads JSONL dataset from data/train/train.jsonl (created by scripts/generate_benign.py)
- Trains a small Transformer autoencoder for a few epochs
- Saves checkpoint to models/checkpoints/best.pt

Usage:
  PYTHONPATH=. python scripts/train_quick.py --epochs 3 --batch 64
"""

import argparse
import json
import os
import random
import urllib.parse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.transformer_model import WAFTransformer
from src.models.train import SequenceDataset, train_model
from src.preprocessing.normalizer import normalize_params, normalize_path, replace_dynamic_values
from src.preprocessing.tokenizer import HTTPRequestTokenizer


def _read_jsonl_objects(paths: Sequence[str]) -> List[dict]:
    objects: List[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_dir():
            candidates = sorted(path.glob("*.jsonl"))
        else:
            candidates = [path]
        for candidate in candidates:
            with candidate.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        objects.append(obj)
    return objects


def _compose_request_text(record: dict) -> str:
    request = record.get("request")
    if isinstance(request, str):
        return request.strip()
    if isinstance(request, dict):
        payload = request
    elif any(key in record for key in ("method", "path", "query_params", "body")):
        payload = record
    else:
        return ""

    method = str(payload.get("method") or "GET").upper()
    path = normalize_path(str(payload.get("path") or "/"))

    query_params = payload.get("query_params") or payload.get("query") or {}
    if isinstance(query_params, str):
        query_params = dict(urllib.parse.parse_qsl(query_params, keep_blank_values=True))
    if not isinstance(query_params, dict):
        query_params = {}
    normalized_params = normalize_params({str(k): str(v) for k, v in query_params.items()})

    composed = f"{method} {path}"
    if normalized_params:
        query = "&".join(f"{key}={value}" for key, value in normalized_params.items())
        composed += f"?{query}"

    body = payload.get("body") or ""
    if body:
        if not isinstance(body, str):
            body = json.dumps(body, sort_keys=True, ensure_ascii=False)
        composed += " BODY:" + replace_dynamic_values(body)

    return composed


def _load_raw_corpus(paths: Sequence[str]) -> List[str]:
    corpus: List[str] = []
    for record in _read_jsonl_objects(paths):
        text = _compose_request_text(record)
        if text:
            corpus.append(text)
    return corpus


def _load_encoded_jsonl(path: str) -> SequenceDataset:
    sequences: List[List[int]] = []
    masks: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sequences.append(obj["input_ids"])  # type: ignore[index]
            masks.append(obj["attention_mask"])  # type: ignore[index]
    return SequenceDataset(sequences, masks)


def _split_dataset(items: Sequence[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    if not items:
        return [], []
    indices = list(range(len(items)))
    random.shuffle(indices)
    split = max(1, int(train_ratio * len(indices)))
    train_items = [items[i] for i in indices[:split]]
    val_items = [items[i] for i in indices[split:]]
    if not val_items:
        val_items = train_items[-1:]
    return train_items, val_items


def _encode_corpus(tokenizer: HTTPRequestTokenizer, corpus: Sequence[str], max_len: int) -> SequenceDataset:
    sequences: List[List[int]] = []
    masks: List[List[int]] = []
    for sample in corpus:
        enc = tokenizer.encode(sample, max_length=max_len)
        sequences.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
    return SequenceDataset(sequences, masks)


def _collect_scores(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> List[float]:
    scores: List[float] = []
    model.eval()
    with torch.no_grad():
        for input_ids, attn in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            batch_scores = model.get_reconstruction_error(input_ids, attn)
            scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())
    return scores


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train/train.jsonl", help="Encoded JSONL fallback dataset")
    parser.add_argument(
        "--raw-data",
        default="data/training/mega_benign_1500.jsonl,data/training/benign_large.jsonl,data/training/benign_requests.jsonl",
        help="Comma-separated raw benign JSONL files to use when available",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=5000)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()

    raw_paths = [p.strip() for p in args.raw_data.split(",") if p.strip()]
    corpus = _load_raw_corpus(raw_paths)

    if corpus:
        train_corpus, val_corpus = _split_dataset(corpus)
        tokenizer = HTTPRequestTokenizer(vocab_size=args.vocab)
        tokenizer.build_vocab(train_corpus)
        train_ds = _encode_corpus(tokenizer, train_corpus, args.maxlen)
        val_ds = _encode_corpus(tokenizer, val_corpus, args.maxlen)

        os.makedirs("models/checkpoints", exist_ok=True)
        tokenizer.save_vocab("models/checkpoints/vocab.json")

        with open(args.data, "w", encoding="utf-8") as f:
            for sample in corpus:
                enc = tokenizer.encode(sample, max_length=args.maxlen)
                f.write(json.dumps(enc) + "\n")
        vocab_size = len(tokenizer.token_to_id)
    else:
        ds = _load_encoded_jsonl(args.data)
        n = len(ds)
        idx = list(range(n))
        random.shuffle(idx)
        split = max(1, int(0.8 * n))
        train_idx, val_idx = idx[:split], idx[split:]

        train_ds = SequenceDataset([ds.sequences[i] for i in train_idx], [ds.attention_masks[i] for i in train_idx])
        val_ds = SequenceDataset([ds.sequences[i] for i in val_idx], [ds.attention_masks[i] for i in val_idx])
        vocab_size = args.vocab

    if len(train_ds) == 0:
        raise RuntimeError("No training samples found")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = WAFTransformer(
        vocab_size=vocab_size,
        embed_dim=args.embed,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_dim=args.ff,
        dropout=0.1,
        max_len=args.maxlen,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, epochs=args.epochs, device=device)

    val_scores = _collect_scores(model, val_loader, device)
    calibrated_threshold = _percentile(val_scores, args.threshold_percentile)
    if calibrated_threshold <= 0:
        calibrated_threshold = 0.75

    os.makedirs("models/checkpoints", exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "meta": {
            "vocab_size": vocab_size,
            "embed_dim": args.embed,
            "num_heads": args.heads,
            "num_layers": args.layers,
            "ff_dim": args.ff,
            "dropout": 0.1,
            "max_len": args.maxlen,
            "threshold": calibrated_threshold,
            "threshold_percentile": args.threshold_percentile,
        },
    }
    torch.save(ckpt, "models/checkpoints/best.pt")
    print("Saved models/checkpoints/best.pt")
    print(f"Calibrated threshold: {calibrated_threshold:.4f}")


if __name__ == "__main__":
    main()



