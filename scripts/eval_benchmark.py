from __future__ import annotations

"""
Evaluate WAF detection repeatedly and save results to CSV to estimate
false positives (benign flagged) and false negatives (attacks missed).

Usage examples:
  PYTHONPATH=. python scripts/eval_benchmark.py \
    --base http://localhost:8000 \
    --out results.csv \
    --iters 50

Notes:
  - Assumes detection API running at --base
  - Uses tests/attack_payloads.txt for attack queries
  - Uses a small built-in set of benign queries/paths
  - CSV columns: ts,label,method,path,query,score,is_anomaly
  - Prints aggregate rates at the end
"""

import argparse
import csv
import json
import os
import time
from typing import List, Tuple

import requests


BENIGN_TERMS = [
    "hello", "home", "products", "profile", "about", "contact",
    "search", "welcome", "shop", "cart", "status", "help", "docs",
]
BENIGN_PATHS = [
    ("GET", "/", {}),
    ("GET", "/search", {}),
    ("GET", "/products", {"page": "1", "sort": "asc"}),
    ("GET", "/api/users/1/profile", {}),
]


def load_attacks(path: str) -> List[str]:
    payloads: List[str] = []
    if not os.path.exists(path):
        return payloads
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            payloads.append(s)
    return payloads


def detect(base: str, method: str, path: str, query: dict, api_key: str) -> Tuple[float, bool]:
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    body = {"method": method, "path": path, "query_params": query, "headers": {}, "body": ""}
    r = requests.post(f"{base}/detect", headers=headers, data=json.dumps(body), timeout=5)
    r.raise_for_status()
    obj = r.json()
    return float(obj["anomaly_score"]), bool(obj["is_anomaly"])  # type: ignore[index]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--api_key", default="dev-key")
    ap.add_argument("--attacks", default="tests/attack_payloads.txt")
    args = ap.parse_args()

    attacks = load_attacks(args.attacks)
    if not attacks:
        print("No attack payloads found; continue with benign only")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "label", "method", "path", "query", "anomaly_score", "is_anomaly"])

        fp = 0  # benign flagged
        fn = 0  # attack missed
        benign_total = 0
        attack_total = 0

        for _ in range(args.iters):
            # Benign runs
            for method, path, base_q in BENIGN_PATHS:
                q = dict(base_q)
                if path == "/search":
                    q["q"] = BENIGN_TERMS[_ % len(BENIGN_TERMS)]
                score, is_anom = detect(args.base, method, path, q, args.api_key)
                benign_total += 1
                if is_anom:
                    fp += 1
                writer.writerow([time.time(), "benign", method, path, json.dumps(q), score, is_anom])

            # Attack runs
            for payload in attacks[:10]:  # cap to keep runtime reasonable
                score, is_anom = detect(args.base, "GET", "/search", {"q": payload}, args.api_key)
                attack_total += 1
                if not is_anom:
                    fn += 1
                writer.writerow([time.time(), "attack", "GET", "/search", json.dumps({"q": payload}), score, is_anom])

    # Print aggregates
    print("Saved:", args.out)
    if benign_total:
        print(f"False Positive Rate: {fp}/{benign_total} = {fp/benign_total:.3f}")
    if attack_total:
        print(f"False Negative Rate: {fn}/{attack_total} = {fn/attack_total:.3f}")


if __name__ == "__main__":
    main()


