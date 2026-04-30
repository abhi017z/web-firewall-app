"""
Batch log ingestion utilities for Apache/Nginx access logs.

Features:
- Detect log format (combined vs JSON)
- Parse key fields from each line
- Chunked reading for large files and directories
- Output structured JSONL or CSV

All functions include type hints and docstrings.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Iterator, List, Optional


# Regex for Apache/Nginx Combined Log Format with request and user-agent etc.
_COMBINED_REGEX = re.compile(
    r"^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+\"(?P<request>[^\"]*)\"\s+"
    r"(?P<status>\d{3})\s+(?P<size>\S+)\s+\"(?P<referer>[^\"]*)\"\s+\"(?P<agent>[^\"]*)\"(?:\s+\"(?P<body>[^\"]*)\")?\s*$"
)


@dataclass
class ParsedLog:
    """Structured representation of a parsed log line."""

    timestamp: str
    ip: str
    method: str
    path: str
    query: str
    user_agent: str
    status: int
    response_size: int
    referer: str
    body: str


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def detect_log_format(sample_lines: List[str]) -> str:
    """Detect log format from sample lines.

    Returns one of: "json", "combined".
    """
    json_hits = 0
    combined_hits = 0
    for line in sample_lines:
        s = line.strip()
        if not s:
            continue
        # JSON lines start with { ... }
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                # Heuristic: presence of common fields
                if any(k in obj for k in ("ip", "timestamp", "request", "status")):
                    json_hits += 1
                    continue
            except Exception:
                pass
        if _COMBINED_REGEX.match(s):
            combined_hits += 1
    return "json" if json_hits >= combined_hits else "combined"


def _iter_paths(log_path: str) -> Iterator[str]:
    if os.path.isdir(log_path):
        for root, _dirs, files in os.walk(log_path):
            for f in files:
                yield os.path.join(root, f)
    else:
        yield log_path


def _parse_request_line(request_line: str) -> tuple[str, str, str]:
    method, path, protocol = "", "", ""
    try:
        parts = request_line.split()
        if len(parts) >= 1:
            method = parts[0]
        if len(parts) >= 2:
            path = parts[1]
        if len(parts) >= 3:
            protocol = parts[2]
    except Exception:
        pass
    return method, path, protocol


def parse_log_line(line: str) -> Dict[str, object]:
    """Parse a single log line (combined or json) into a dict.

    Extracts: timestamp, ip, method, path, query, user-agent, status, response_size, referer, body
    """
    s = line.strip()
    if not s:
        return {}

    # Try JSON first
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            request_line = obj.get("request") or obj.get("req") or ""
            method, path_with_query, _ = _parse_request_line(request_line)
            path, query = _split_query(path_with_query)
            return asdict(
                ParsedLog(
                    timestamp=str(obj.get("timestamp") or obj.get("time") or ""),
                    ip=str(obj.get("ip") or obj.get("remote_addr") or ""),
                    method=method,
                    path=path,
                    query=query,
                    user_agent=str(obj.get("user_agent") or obj.get("agent") or ""),
                    status=_safe_int(str(obj.get("status") or 0)),
                    response_size=_safe_int(str(obj.get("body_bytes_sent") or obj.get("size") or 0)),
                    referer=str(obj.get("referer") or obj.get("http_referer") or ""),
                    body=str(obj.get("request_body") or obj.get("body") or ""),
                )
            )
        except Exception:
            # Fall through to combined regex
            pass

    m = _COMBINED_REGEX.match(s)
    if not m:
        return {}
    ip = m.group("ip")
    timestamp = m.group("time")
    request_line = m.group("request") or ""
    method, path_with_query, _ = _parse_request_line(request_line)
    path, query = _split_query(path_with_query)
    status = _safe_int(m.group("status"))
    size_raw = m.group("size")
    response_size = _safe_int("0" if size_raw == "-" else size_raw)
    referer = m.group("referer") or ""
    agent = m.group("agent") or ""
    body = m.group("body") or ""

    return asdict(
        ParsedLog(
            timestamp=timestamp,
            ip=ip,
            method=method,
            path=path,
            query=query,
            user_agent=agent,
            status=status,
            response_size=response_size,
            referer=referer,
            body=body,
        )
    )


def _split_query(path_with_query: str) -> tuple[str, str]:
    if not path_with_query:
        return "", ""
    if "?" in path_with_query:
        p, q = path_with_query.split("?", 1)
        return p, q
    return path_with_query, ""


def _iter_lines(paths: Iterable[str], chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Yield lines efficiently from potentially large files."""
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                buf = io.StringIO()
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    buf.write(data)
                    buf.seek(0)
                    for line in buf:
                        yield line
                    # Keep tail if last line not ended
                    tail = buf.read()
                    buf.close()
                    buf = io.StringIO()
                    buf.write(tail)
                # Flush remaining
                buf.seek(0)
                for line in buf:
                    yield line
        except FileNotFoundError:
            continue


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _writer(output_path: str):
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".jsonl", ".json"):
        return _jsonl_writer(output_path)
    if ext == ".csv":
        return _csv_writer(output_path)
    # Default to jsonl
    return _jsonl_writer(output_path)


def _jsonl_writer(output_path: str):
    _ensure_parent(output_path)
    f = open(output_path, "w", encoding="utf-8")

    def write_row(row: Dict[str, object]) -> None:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close() -> None:
        f.close()

    return write_row, close


def _csv_writer(output_path: str):
    _ensure_parent(output_path)
    f = open(output_path, "w", encoding="utf-8", newline="")
    fieldnames = [
        "timestamp",
        "ip",
        "method",
        "path",
        "query",
        "user_agent",
        "status",
        "response_size",
        "referer",
        "body",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    def write_row(row: Dict[str, object]) -> None:
        writer.writerow({k: row.get(k) for k in fieldnames})

    def close() -> None:
        f.close()

    return write_row, close


def batch_ingest_logs(log_path: str, output_path: str) -> None:
    """Read Apache/Nginx access logs from a path and write structured output.

    - Supports file or directory input
    - Detects format from first 50 non-empty lines
    - Streams through input to limit memory usage
    - Output format inferred by output extension (jsonl/csv)
    """
    paths = list(_iter_paths(log_path))
    # Sample lines for detection
    sample: List[str] = []
    for line in _iter_lines(paths):
        if line.strip():
            sample.append(line)
        if len(sample) >= 50:
            break
    fmt = detect_log_format(sample)

    write_row, close = _writer(output_path)
    try:
        for line in _iter_lines(paths):
            if not line.strip():
                continue
            row = parse_log_line(line) if fmt == "combined" else parse_log_line(line)
            if not row:
                continue
            write_row(row)
    finally:
        close()


__all__ = [
    "parse_log_line",
    "batch_ingest_logs",
    "detect_log_format",
]


