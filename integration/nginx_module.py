from __future__ import annotations

import json
import time
from typing import Callable

import httpx


def tail_nginx_logs(log_path: str, api_endpoint: str, rate_limit_hz: float = 50.0) -> None:
    """Tail an Nginx log file and mirror requests to the detection API."""
    sleep_interval = 1.0 / max(rate_limit_hz, 1.0)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        # Seek to end
        f.seek(0, 2)
        client = httpx.Client(timeout=2.0)
        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.25)
                f.seek(where)
            else:
                payload = _line_to_detection_payload(line)
                if payload is None:
                    continue
                try:
                    client.post(api_endpoint, json=payload, headers={"X-API-Key": "dev-key"})
                except Exception:
                    pass
                time.sleep(sleep_interval)


def _line_to_detection_payload(line: str):
    # Very light parser matching src.ingestion behavior; expects combined format
    try:
        # Detect JSON logs first
        if line.strip().startswith("{"):
            obj = json.loads(line)
            req = obj.get("request", "GET /")
        else:
            # request is inside quotes: "METHOD PATH HTTP/x.y"
            start = line.find('"')
            end = line.find('"', start + 1)
            req = line[start + 1 : end] if start != -1 and end != -1 else "GET /"
        parts = req.split()
        method = parts[0] if parts else "GET"
        path = parts[1] if len(parts) > 1 else "/"
        q = {}
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


__all__ = ["tail_nginx_logs"]


