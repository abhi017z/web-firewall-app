"""
Tokenizer for HTTP requests supporting simple word-level tokenization with
special tokens and attention mask creation.

This is a lightweight tokenizer suitable for initial experiments; can be
replaced by BPE if needed.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
}


_TOKEN_SPLIT = re.compile(r"([/?=&:#.;,\[\]{}()<>\-_'\"|`~!@^*+\\])")


class HTTPRequestTokenizer:
    """Simple tokenizer with a learned vocabulary.

    Args:
        vocab_size: Target maximum vocabulary size including special tokens.
    """

    def __init__(self, vocab_size: int = 10000) -> None:
        self.vocab_size = max(vocab_size, len(SPECIAL_TOKENS))
        self.token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {i: t for t, i in SPECIAL_TOKENS.items()}

    def build_vocab(self, requests: List[str]) -> None:
        """Build vocabulary from training requests by frequency."""
        freq: Dict[str, int] = {}
        for req in requests:
            for tok in self._basic_tokenize(req):
                if tok.strip() == "":
                    continue
                freq[tok] = freq.get(tok, 0) + 1
        # Reserve ids for special tokens
        next_id = len(SPECIAL_TOKENS)
        for token, _count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if token in self.token_to_id:
                continue
            if next_id >= self.vocab_size:
                break
            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token
            next_id += 1

    def _basic_tokenize(self, request: str) -> List[str]:
        if not request:
            return []
        # Split by defined punctuation while keeping tokens
        parts: List[str] = []
        for piece in _TOKEN_SPLIT.split(request):
            if piece == "":
                continue
            if piece.isspace():
                continue
            parts.append(piece)
        return parts

    def tokenize(self, request: str) -> List[int]:
        """Tokenize into token IDs without adding special tokens."""
        tokens = self._basic_tokenize(request)
        unk_id = SPECIAL_TOKENS["[UNK]"]
        return [self.token_to_id.get(tok, unk_id) for tok in tokens]

    def encode(self, request: str, max_length: int) -> Dict[str, List[int]]:
        """Encode request with [CLS] and [SEP], pad/truncate to max_length.

        Returns dict with keys: input_ids, attention_mask.
        """
        cls_id = SPECIAL_TOKENS["[CLS]"]
        sep_id = SPECIAL_TOKENS["[SEP]"]
        pad_id = SPECIAL_TOKENS["[PAD]"]
        token_ids = self.tokenize(request)
        # +2 for CLS and SEP
        token_ids = [cls_id] + token_ids[: max_length - 2] + [sep_id]
        attention_mask = [1] * len(token_ids)
        # Pad
        if len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a string (best-effort)."""
        tokens: List[str] = []
        for tid in token_ids:
            if tid in (SPECIAL_TOKENS["[PAD]"], SPECIAL_TOKENS["[CLS]"], SPECIAL_TOKENS["[SEP]"]):
                continue
            tokens.append(self.id_to_token.get(tid, ""))
        # Heuristic join without adding spaces around punctuation tokens
        out: List[str] = []
        for t in tokens:
            if not out:
                out.append(t)
                continue
            if _TOKEN_SPLIT.fullmatch(t):
                out.append(t)
            elif _TOKEN_SPLIT.fullmatch(out[-1]):
                out.append(t)
            else:
                out.append(" " + t)
        return "".join(out)

    def save_vocab(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id, "vocab_size": self.vocab_size}, f)

    def load_vocab(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = int(data.get("vocab_size", len(SPECIAL_TOKENS)))
        token_to_id = dict(data.get("token_to_id", {}))
        # Ensure special tokens exist at required IDs
        token_to_id.update(SPECIAL_TOKENS)
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}


__all__ = [
    "HTTPRequestTokenizer",
]


