from __future__ import annotations

import json
import os
import random
from typing import List

from src.preprocessing.tokenizer import HTTPRequestTokenizer


ROUTES = [
    ("GET", "/", {}),
    ("GET", "/products", {"page": "1", "sort": "asc"}),
    ("GET", "/products/123", {}),
    ("POST", "/login", {}),
    ("GET", "/api/users/42/profile", {}),
]


def synthesize_requests(n: int) -> List[str]:
    out: List[str] = []
    for _ in range(n):
        method, path, params = random.choice(ROUTES)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        composed = f"{method} {path}"
        if query:
            composed += f"?{query}"
        out.append(composed)
    return out


def main() -> None:
    os.makedirs("data/train", exist_ok=True)
    tok = HTTPRequestTokenizer(vocab_size=5000)
    train_reqs = synthesize_requests(10000)
    tok.build_vocab(train_reqs)
    max_len = 128
    with open("data/train/train.jsonl", "w", encoding="utf-8") as f:
        for r in train_reqs:
            enc = tok.encode(r, max_length=max_len)
            f.write(json.dumps(enc) + "\n")
    tok.save_vocab("models/checkpoints/vocab.json")


if __name__ == "__main__":
    main()


