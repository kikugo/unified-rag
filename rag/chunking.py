"""Stable chunk ids and fixed-window page chunking.

A chunk id is a hash of where the chunk sits (document, page, character offset,
kind), never of a random uuid or an insertion counter. Rebuilding the index from
the same corpus text therefore gives the same ids, which the golden set relies
on: if ids moved on re-indexing, every gold_chunk_id would silently point at
nothing.
"""
from __future__ import annotations

import hashlib
from typing import Any

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 40


def make_chunk_id(doc_id: str, page: int, char_offset: int, kind: str = "text") -> str:
    """16 hex chars of sha256 over the chunk's position.

    `kind` keeps a figure and a text window that start at the same offset of the
    same section from colliding ("text", or "figure-<n>" with the figure's
    ordinal in its document, since consecutive figures share an offset).
    """
    key = f"{doc_id}|{page}|{char_offset}|{kind}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _snap_back(text: str, end: int, floor: int) -> int:
    """Move `end` back to just after the last whitespace in (floor, end]."""
    cut = text.rfind(" ", floor, end)
    return cut + 1 if cut > floor else end


def _snap_forward(text: str, start: int) -> int:
    """Move `start` forward past the word it lands inside."""
    if start <= 0 or text[start - 1].isspace():
        return start
    nxt = text.find(" ", start)
    return len(text) if nxt == -1 else nxt + 1


def chunk_page(
    doc_id: str,
    page: int,
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split one page (or section) into overlapping windows on word boundaries.

    `char_offset` is the window's start in `text`, so `text[offset:offset+len]`
    always reproduces the chunk. Windows shorter than MIN_CHUNK_CHARS are dropped
    unless they are the whole page.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[dict[str, Any]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            end = _snap_back(text, end, start + size - overlap)
        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        offset = start + lead
        if body and (len(body) >= MIN_CHUNK_CHARS or not chunks and end == length):
            chunks.append({
                "chunk_id": make_chunk_id(doc_id, page, offset),
                "doc_id": doc_id,
                "page": page,
                "char_offset": offset,
                "kind": "text",
                "text": body,
            })
        if end >= length:
            break
        nxt = _snap_forward(text, end - overlap)
        start = nxt if nxt > start else end
    return chunks
