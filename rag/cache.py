"""On-disk cache for embeddings and LLM drafts.

Keyed by a hash of everything that changes the output (model, dimension, task
type, content), so rebuilding the index or rerunning an eval reuses earlier
calls. That makes a rebuild free, and it makes it deterministic: the same chunk
gets the same vector every time, so retrieval order cannot drift between runs.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "rag_cache.sqlite"


def cache_key(*parts: str | bytes | int) -> str:
    digest = hashlib.sha256()
    for part in parts:
        data = part if isinstance(part, bytes) else str(part).encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class DiskCache:
    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value BLOB)")
        self._db.commit()

    def _get(self, key: str) -> bytes | None:
        row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _put(self, key: str, value: bytes) -> None:
        self._db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))
        self._db.commit()

    def get_vector(self, key: str) -> np.ndarray | None:
        raw = self._get(key)
        return np.frombuffer(raw, dtype=np.float32).copy() if raw is not None else None

    def put_vector(self, key: str, vector) -> None:
        self._put(key, np.asarray(vector, dtype=np.float32).tobytes())

    def get_text(self, key: str) -> str | None:
        raw = self._get(key)
        return raw.decode("utf-8") if raw is not None else None

    def put_text(self, key: str, text: str) -> None:
        self._put(key, text.encode("utf-8"))

    def __contains__(self, key: str) -> bool:
        return self._get(key) is not None

    def close(self) -> None:
        self._db.close()
