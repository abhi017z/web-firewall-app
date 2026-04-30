from __future__ import annotations

"""
Generate benign traffic by hitting a base URL (the deployed (WAR) app) and
writing normalized/tokenized samples for training.

Usage:
  PYTHONPATH=. python scripts/simulate_benign_from_url.py --base http://localhost:8081 --out data/train/train.jsonl --count 1000
"""

import argparse
import json
import random
import time
from typing import List

import requests

from src.preprocessing.tokenizer import HTTPRequestTokenizer


ROUTES = [
    ("GET", "/", {}),
    ("GET", "/products", {"page": "1", "sort": "asc"}),
    ("GET", "/products/123", {}),
    ("GET", "/search", {"q": "shoes"}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="data/train/train.jsonl")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--vocab", default="models/checkpoints/vocab.json")
    args = ap.parse_args()

    tok = HTTPRequestTokenizer(vocab_size=10000)
    corpus: List[str] = []
    for _ in range(args.count):
        method, path, params = random.choice(ROUTES)
        url = args.base + path
        try:
            if method == "GET":
                requests.get(url, params=params, timeout=1.0)
            else:
                requests.post(url, json=params, timeout=1.0)
        except Exception:
            pass
        query = "&".join(f"{k}={v}" for k, v in params.items())
        composed = f"{method} {path}"
        if query:
            composed += f"?{query}"
        corpus.append(composed)
        time.sleep(0.01)

    tok.build_vocab(corpus)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in corpus:
            enc = tok.encode(r, max_length=args.maxlen)
            f.write(json.dumps(enc) + "\n")
    tok.save_vocab(args.vocab)
    print(f"Wrote {args.out} and vocab {args.vocab}")


if __name__ == "__main__":
    main()


