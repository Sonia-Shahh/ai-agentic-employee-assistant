"""Token-aware text chunking utilities."""
from __future__ import annotations

from typing import Iterable, List

import tiktoken


def normalize_text(text: str) -> str:
    """Collapse excessive whitespace and trim."""
    return " ".join(text.split())


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str = "cl100k_base",
) -> List[str]:
    """Chunk text into overlapping segments respecting token limits."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_value = encoding.decode(chunk_tokens)
        chunks.append(normalize_text(chunk_text_value))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return [chunk for chunk in chunks if chunk]


def chunk_documents(texts: Iterable[str], max_tokens: int, overlap_tokens: int) -> List[List[str]]:
    return [chunk_text(text, max_tokens, overlap_tokens) for text in texts]
