from __future__ import annotations

"""
End-to-end ingestion + preparation script for WAR app access logs.

Usage examples:
  # Batch ingest from a directory of log files, parse/normalize/tokenize, write JSONL
  PYTHONPATH=. python scripts/ingest_and_prepare.py \
    --logs data/raw_logs \
    --out data/train/train.jsonl \
    --vocab models/checkpoints/vocab.json \
    --maxlen 128

  # Ingest a single file
  PYTHONPATH=. python scripts/ingest_and_prepare.py --logs /path/to/access.log --out data/train/train.jsonl
"""

import argparse
import json
import os
from typing import Dict, List

from src.ingestion.batch_ingestion import batch_ingest_logs
from src.preprocessing.parser import parse_request
from src.preprocessing.normalizer import normalize_path, normalize_params, replace_dynamic_values
from src.preprocessing.tokenizer import HTTPRequestTokenizer


def iterate_structured(path: str):
    # Use batch_ingest_logs to create a temp JSONL if input is raw logs
    ext = os.path.splitext(path)[1].lower()
    if os.path.isdir(path) or ext not in {".jsonl", ".json"}:
        tmp = "data/processed/_tmp_ingested.jsonl"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        batch_ingest_logs(path, tmp)
        src = tmp
    else:
        src = path
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_requests_for_tokenizer(logs_iter) -> List[str]:
    corpus: List[str] = []
    for entry in logs_iter:
        req = parse_request(entry)
        # Normalize path and params; keep attack patterns
        npath = normalize_path(req.path)
        nparams = normalize_params(req.query_params)
        composed = f"{req.method} {npath}"
        if nparams:
            query = "&".join(f"{k}={v}" for k, v in nparams.items())
            composed += f"?{query}"
        if req.body:
            composed += " BODY:" + replace_dynamic_values(req.body)
        corpus.append(composed)
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="Path to raw logs (file or dir) or JSONL of structured logs")
    ap.add_argument("--out", required=True, help="Output JSONL with encoded samples")
    ap.add_argument("--vocab", default="models/checkpoints/vocab.json")
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--maxlen", type=int, default=128)
    args = ap.parse_args()

    logs_iter = iterate_structured(args.logs)
    # We need two passes: build corpus, then tokenize
    corpus = build_requests_for_tokenizer(logs_iter)
    tok = HTTPRequestTokenizer(vocab_size=args.vocab_size)
    tok.build_vocab(corpus)
    os.makedirs(os.path.dirname(args.vocab) or ".", exist_ok=True)
    tok.save_vocab(args.vocab)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for r in corpus:
            enc = tok.encode(r, max_length=args.maxlen)
            out.write(json.dumps(enc) + "\n")
    print(f"Wrote: {args.out}")
    print(f"Saved vocab: {args.vocab}")


if __name__ == "__main__":
    main()


