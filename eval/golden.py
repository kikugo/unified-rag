"""The golden set's schema, targets and loader.

`eval/golden.jsonl` holds hand-reviewed questions. Rows are only ever appended
by `eval.review`; nothing rewrites one after review. Every later metric reads
the set through `load_golden`, so a malformed row fails loudly here instead of
quietly skewing a score.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"

TARGET_MIX = {
    "factual": 40,
    "keyword": 20,
    "paraphrase": 20,
    "multihop": 10,
    "unanswerable": 10,
}
TYPES = tuple(TARGET_MIX)
FIELDS = ("id", "question", "gold_chunk_ids", "gold_answer", "type", "answerable")


class GoldenSetError(ValueError):
    pass


def validate_row(row: dict[str, Any]) -> None:
    missing = [f for f in FIELDS if f not in row]
    if missing:
        raise GoldenSetError(f"{row.get('id', '?')}: missing {missing}")
    rid = row["id"]
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise GoldenSetError(f"{rid}: empty question")
    if row["type"] not in TYPES:
        raise GoldenSetError(f"{rid}: unknown type {row['type']!r}")
    ids = row["gold_chunk_ids"]
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise GoldenSetError(f"{rid}: gold_chunk_ids must be a list of chunk ids")
    if len(ids) != len(set(ids)):
        raise GoldenSetError(f"{rid}: duplicate gold_chunk_ids")
    if row["answerable"] is not bool(ids):
        raise GoldenSetError(f"{rid}: answerable must be true exactly when gold_chunk_ids is non-empty")
    if (row["type"] == "unanswerable") == row["answerable"]:
        raise GoldenSetError(f"{rid}: only unanswerable rows may have answerable=false")
    if row["answerable"] and not str(row["gold_answer"]).strip():
        raise GoldenSetError(f"{rid}: answerable row needs a gold_answer")
    if row["type"] == "multihop" and len(ids) < 2:
        raise GoldenSetError(f"{rid}: multihop needs at least two gold chunks")


def load_golden(path: str | Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"line {lineno}: not JSON ({exc})") from exc
        validate_row(row)
        if row["id"] in seen:
            raise GoldenSetError(f"duplicate id {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    return rows


def mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {t: 0 for t in TYPES}
    for row in rows:
        counts[row["type"]] += 1
    return counts
