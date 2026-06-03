from __future__ import annotations

import hashlib
import mimetypes
import uuid
from pathlib import Path


STORAGE_ROOT = Path(".unified_rag_store")
CORPUS_ID_PATH = STORAGE_ROOT / "corpus_id"


def load_or_create_corpus_id() -> str:
    """Return the persisted corpus id, creating and saving one on first use.

    Persisting it (instead of minting a fresh uuid each session) keeps a single
    stable corpus across restarts, so the index can be rehydrated and old rows
    are not orphaned every time the app reloads.
    """
    try:
        if CORPUS_ID_PATH.exists():
            existing = CORPUS_ID_PATH.read_text(encoding="utf-8").strip()
            if existing:
                return existing
    except Exception:
        pass
    corpus_id = str(uuid.uuid4())
    set_corpus_id(corpus_id)
    return corpus_id


def set_corpus_id(corpus_id: str):
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    CORPUS_ID_PATH.write_text(corpus_id, encoding="utf-8")


def _safe_extension(mime_type: str, source_name: str | None = None) -> str:
    if source_name:
        suffix = Path(source_name).suffix.lower()
        if suffix:
            return suffix
    guessed = mimetypes.guess_extension(mime_type or "")
    if guessed:
        return guessed
    return ".bin"


def persist_blob(data: bytes, mime_type: str, source_name: str | None = None) -> str:
    """Store blob content under a deterministic hash path and return that path."""
    digest = hashlib.sha256(data).hexdigest()
    ext = _safe_extension(mime_type, source_name)
    blob_dir = STORAGE_ROOT / digest[:2]
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"{digest}{ext}"
    if not blob_path.exists():
        blob_path.write_bytes(data)
    return str(blob_path)


def read_blob(path: str | None) -> bytes:
    if not path:
        return b""
    try:
        return Path(path).read_bytes()
    except Exception:
        return b""


def delete_blob(path: str | None):
    if not path:
        return
    try:
        target = Path(path)
        if target.exists() and target.is_file():
            target.unlink()
            parent = target.parent
            if parent != STORAGE_ROOT and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
    except Exception:
        return
