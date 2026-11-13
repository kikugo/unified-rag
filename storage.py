from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

STORAGE_ROOT = Path(".unified_rag_store")
CORPUS_ID_PATH = STORAGE_ROOT / "corpus_id"
# Demo sessions each get their own corpus; this file records when each was last
# used so idle ones can be swept. Blobs for those corpora live under
# STORAGE_ROOT / "corpora" / <corpus_id>, so deleting one never touches another
# visitor's files even when both uploaded the same bytes.
CORPORA_DIRNAME = "corpora"
CORPORA_INDEX_NAME = "corpora.json"
_corpora_lock = threading.Lock()


def new_corpus_id() -> str:
    return str(uuid.uuid4())


def _corpora_index() -> Path:
    return STORAGE_ROOT / CORPORA_INDEX_NAME


def _read_corpora() -> dict[str, float]:
    path = _corpora_index()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


def _write_corpora(data: dict[str, float]):
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(STORAGE_ROOT), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(tmp, _corpora_index())
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def touch_corpus(corpus_id: str, now: float | None = None, min_interval: float = 300.0) -> bool:
    """Mark a session corpus as in use. Writes at most once per `min_interval`
    seconds per corpus so a Streamlit rerun does not rewrite the file. Returns
    True when it wrote."""
    now = time.time() if now is None else now
    with _corpora_lock:
        data = _read_corpora()
        if now - data.get(corpus_id, 0.0) < min_interval:
            return False
        data[corpus_id] = now
        _write_corpora(data)
        return True


def stale_corpora(max_age_seconds: float, now: float | None = None) -> list[str]:
    """Session corpora not touched for longer than `max_age_seconds`."""
    now = time.time() if now is None else now
    with _corpora_lock:
        return [cid for cid, seen in _read_corpora().items() if now - seen > max_age_seconds]


def forget_corpus(corpus_id: str):
    """Delete a session corpus's blobs and drop it from the activity index."""
    with _corpora_lock:
        data = _read_corpora()
        data.pop(corpus_id, None)
        _write_corpora(data)
    shutil.rmtree(corpus_dir(corpus_id), ignore_errors=True)


def corpus_dir(corpus_id: str) -> Path:
    # uuid.UUID() rejects anything that could escape the folder.
    return STORAGE_ROOT / CORPORA_DIRNAME / str(uuid.UUID(corpus_id))


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


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of a blob — the dedup key for uploaded files."""
    return hashlib.sha256(data).hexdigest()


def persist_blob(data: bytes, mime_type: str, source_name: str | None = None,
                 corpus_id: str | None = None) -> str:
    """Store blob content under a deterministic hash path and return that path.

    With `corpus_id`, the blob goes under that corpus's own folder, so two demo
    sessions uploading the same file get separate copies.
    """
    digest = hash_bytes(data)
    ext = _safe_extension(mime_type, source_name)
    base = corpus_dir(corpus_id) if corpus_id else STORAGE_ROOT
    blob_dir = base / digest[:2]
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
            if (parent != STORAGE_ROOT and parent.parent != STORAGE_ROOT / CORPORA_DIRNAME
                    and parent.exists() and not any(parent.iterdir())):
                parent.rmdir()
    except Exception:
        return
