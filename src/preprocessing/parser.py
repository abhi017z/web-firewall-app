"""
HTTP request parser utilities.

Parses log-entry dicts into structured HTTP request fields and helpers for
headers, query parameters, and URL decoding.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional


_COOKIE_SPLIT = re.compile(r";\s*")


@dataclass
class ParsedRequest:
    """Structured HTTP request parsed from a log entry."""

    method: str
    path: str
    query_params: Dict[str, str]
    headers: Dict[str, str]
    cookies: Dict[str, str]
    body: str
    status: int
    response_size: int


def decode_url(encoded_url: str) -> str:
    """Decode percent-encoded URL safely (leaves invalid sequences intact)."""
    if not encoded_url:
        return ""
    try:
        return urllib.parse.unquote_plus(encoded_url)
    except Exception:
        return encoded_url


def parse_query_params(query_string: str) -> Dict[str, str]:
    """Parse query string into a flat dict, keeping last value on duplicates."""
    params: Dict[str, str] = {}
    if not query_string:
        return params
    for key, values in urllib.parse.parse_qs(query_string, keep_blank_values=True).items():
        if not values:
            continue
        params[decode_url(key)] = decode_url(values[-1])
    return params


def extract_headers(raw_headers: str) -> Dict[str, str]:
    """Extract headers from a raw header string if present.

    Accepts CRLF or LF separated "Key: Value" lines.
    """
    headers: Dict[str, str] = {}
    if not raw_headers:
        return headers
    for line in raw_headers.splitlines():
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


def _extract_cookies_from_headers(headers: Dict[str, str]) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    cookie_header = headers.get("Cookie") or headers.get("cookie")
    if not cookie_header:
        return cookies
    for pair in _COOKIE_SPLIT.split(cookie_header):
        if not pair or "=" not in pair:
            continue
        name, value = pair.split("=", 1)
        cookies[name.strip()] = value.strip()
    return cookies


def parse_request(log_entry: Dict[str, object]) -> ParsedRequest:
    """Parse a normalized log_entry into a ParsedRequest.

    Expected fields in log_entry: method, path, query, headers(optional), user_agent(optional), body, status, response_size
    """
    method = str(log_entry.get("method") or "").upper()
    raw_path = str(log_entry.get("path") or "")
    path = decode_url(raw_path)
    query_string = str(log_entry.get("query") or "")
    query_params = parse_query_params(query_string)

    # Construct headers dict combining provided headers and common fields
    headers: Dict[str, str] = {}
    provided_headers = log_entry.get("headers")
    if isinstance(provided_headers, dict):
        headers.update({str(k): str(v) for k, v in provided_headers.items()})
    user_agent = str(log_entry.get("user_agent") or "")
    if user_agent and "User-Agent" not in headers and "user-agent" not in headers:
        headers["User-Agent"] = user_agent
    referer = str(log_entry.get("referer") or "")
    if referer and "Referer" not in headers and "referer" not in headers:
        headers["Referer"] = referer

    cookies = _extract_cookies_from_headers(headers)
    body = str(log_entry.get("body") or "")
    status = int(log_entry.get("status") or 0)
    response_size = int(log_entry.get("response_size") or 0)

    return ParsedRequest(
        method=method,
        path=path,
        query_params=query_params,
        headers=headers,
        cookies=cookies,
        body=body,
        status=status,
        response_size=response_size,
    )


__all__ = [
    "ParsedRequest",
    "parse_request",
    "extract_headers",
    "parse_query_params",
    "decode_url",
]


