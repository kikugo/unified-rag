from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(".unified_rag_store/source_registry.json")


def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"sources": []}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return {"sources": []}


def _save_registry(payload: dict[str, Any]):
    """Write the registry atomically so a crash mid-write can't truncate it."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(REGISTRY_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, REGISTRY_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def upsert_source(entry: dict[str, Any]):
    payload = _load_registry()
    sources = payload.get("sources", [])
    source_id = entry.get("source_id")
    updated = False
    for idx, source in enumerate(sources):
        if source.get("source_id") == source_id:
            sources[idx] = entry
            updated = True
            break
    if not updated:
        sources.append(entry)
    payload["sources"] = sources
    _save_registry(payload)


def delete_source(source_id: str):
    payload = _load_registry()
    payload["sources"] = [s for s in payload.get("sources", []) if s.get("source_id") != source_id]
    _save_registry(payload)


def list_sources(corpus_id: str | None = None) -> list[dict[str, Any]]:
    payload = _load_registry()
    sources = payload.get("sources", [])
    if corpus_id is None:
        return sources
    return [s for s in sources if s.get("corpus_id") == corpus_id]
