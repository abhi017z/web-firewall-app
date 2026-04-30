from __future__ import annotations

import time
import requests


def run_attack_simulation(base_url: str, payloads: list[str]):
    results = {"detected": 0, "missed": 0}
    headers = {"X-API-Key": "dev-key"}
    for payload in payloads:
        requests.get(f"{base_url}/search", params={"q": payload}, headers=headers)
        requests.post(f"{base_url}/login", json={"username": payload, "password": "test"}, headers=headers)
        time.sleep(0.05)
    return results


if __name__ == "__main__":
    with open("tests/attack_payloads.txt", "r", encoding="utf-8") as f:
        payloads = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    print(run_attack_simulation("http://localhost:8000", payloads))


