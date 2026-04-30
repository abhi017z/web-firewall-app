from __future__ import annotations

import time
from typing import Dict

import httpx


def tail_apache_logs(log_path: str, api_endpoint: str, rate_limit_hz: float = 50.0) -> None:
    sleep_interval = 1.0 / max(rate_limit_hz, 1.0)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, 2)
        client = httpx.Client(timeout=2.0)
        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.25)
                f.seek(where)
            else:
                payload = _parse_common_log(line)
                if payload is None:
                    continue
                try:
                    client.post(api_endpoint, json=payload, headers={"X-API-Key": "dev-key"})
                except Exception:
                    pass
                time.sleep(sleep_interval)


def _parse_common_log(line: str):
    try:
        first = line.split('"')
        if len(first) < 2:
            return None
        req = first[1]
        parts = req.split()
        method = parts[0] if parts else "GET"
        path = parts[1] if len(parts) > 1 else "/"
        q: Dict[str, str] = {}
        if "?" in path:
            p, qs = path.split("?", 1)
            path = p
            for kv in qs.split("&"):
                if not kv:
                    continue
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    q[k] = v
                else:
                    q[kv] = ""
        return {"method": method, "path": path, "query_params": q, "headers": {}, "body": ""}
    except Exception:
        return None


__all__ = ["tail_apache_logs"]


