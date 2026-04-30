"""
Normalization utilities for HTTP requests that preserve attack patterns while
replacing dynamic values with placeholders.
"""

from __future__ import annotations

import re
from typing import Dict


# Patterns for dynamic values
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
_ID_RE = re.compile(r"(?<=/)(\d{3,})(?=/|$)")  # numeric ids in path segments
_SESSION_RE = re.compile(r"\b(session|token|auth|sid|sessionid|jwt)=([A-Za-z0-9._-]+)", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?\b")

# Attack patterns to preserve (ensure not replaced or lowercased away)
_ATTACK_KEYWORDS = [
    "select", "union", "insert", "update", "delete", "drop", "or", "and",
    "script", "onerror", "onload", "alert", "img", "iframe",
    "../", "..\\", "%2f", "%2e%2e%2f", "../../", "..%2F..%2F",
]


def preserve_attack_patterns(text: str) -> str:
    """Return text with canonical forms for known attack tokens preserved.

    Converts known encodings to a standard representation without removing them.
    """
    if not text:
        return ""
    canonical = text
    # Normalize common encodings for path traversal to a canonical form
    canonical = canonical.replace("%2F", "/").replace("%2f", "/")
    canonical = canonical.replace("%2E", ".").replace("%2e", ".")
    return canonical


def normalize_path(path: str) -> str:
    """Lowercase path, collapse redundant slashes, keep traversal tokens."""
    if not path:
        return ""
    p = preserve_attack_patterns(path)
    # Lowercase only path components; query handled separately
    p = p.lower()
    # Collapse multiple slashes
    p = re.sub(r"/+", "/", p)
    # Replace numeric IDs in path segments
    p = _ID_RE.sub("{USER_ID}", p)
    # Replace IP-like segments
    p = _IP_RE.sub("{IP}", p)
    # Replace UUIDs / long hex
    p = _UUID_RE.sub("{UUID}", p)
    p = _HEX_RE.sub("{HASH}", p)
    return p


def normalize_params(params: Dict[str, str]) -> Dict[str, str]:
    """Normalize query parameters: lowercase keys, replace dynamic values, sort keys."""
    normalized: Dict[str, str] = {}
    for key in sorted(params.keys(), key=lambda k: k.lower()):
        value = params[key]
        k = key.lower()
        normalized[k] = replace_dynamic_values(value)
    return normalized


def replace_dynamic_values(text: str) -> str:
    """Replace dynamic tokens (IDs, IPs, timestamps, sessions, hashes) with placeholders."""
    if not text:
        return ""
    t = preserve_attack_patterns(text)
    t = _TIMESTAMP_RE.sub("{TIMESTAMP}", t)
    t = _IP_RE.sub("{IP}", t)
    # Session-like tokens in key=value pairs
    t = _SESSION_RE.sub(lambda m: f"{m.group(1)}={{SESSION}}", t)
    t = _UUID_RE.sub("{UUID}", t)
    # Long hex strings often hashes
    t = _HEX_RE.sub("{HASH}", t)
    return t


__all__ = [
    "normalize_path",
    "normalize_params",
    "replace_dynamic_values",
    "preserve_attack_patterns",
]


