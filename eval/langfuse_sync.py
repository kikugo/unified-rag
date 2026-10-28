"""Upload eval/golden.jsonl to Langfuse as a dataset, so traced eval runs can link to it.

    uv run python -m eval.langfuse_sync

Each golden row becomes one dataset item, keyed by a stable id derived from the
golden id, so running this again updates items instead of duplicating them.
The golden file stays the source of truth; nothing is ever read back from
Langfuse into it.
"""
from __future__ import annotations

import sys
from typing import Any

from eval.golden import golden_hash, load_golden
from rag import tracing

DATASET = "b1-golden"


def item_id(golden_id: str) -> str:
    # Item ids are unique per Langfuse project, so prefix them with the dataset.
    return f"{DATASET}-{golden_id}"


def dataset_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item_id(row["id"]),
            "input": {"question": row["question"]},
            "expected_output": {"answer": row["gold_answer"], "gold_chunk_ids": row["gold_chunk_ids"]},
            "metadata": {"golden_id": row["id"], "type": row["type"], "answerable": row["answerable"]},
        }
        for row in rows
    ]


def sync(client, rows: list[dict[str, Any]], golden_sha: str) -> int:
    client.create_dataset(
        name=DATASET,
        description="Unified RAG eval bench golden set (eval/golden.jsonl). Edit the file, not this dataset.",
        metadata={"golden_sha256_16": golden_sha},
    )
    items = dataset_items(rows)
    for item in items:
        client.create_dataset_item(dataset_name=DATASET, **item)
    client.flush()
    return len(items)


def main(argv: list[str] | None = None) -> int:
    tracing.require_keys()
    n = sync(tracing.client(), load_golden(), golden_hash())
    print(f"synced {n} items to Langfuse dataset {DATASET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
