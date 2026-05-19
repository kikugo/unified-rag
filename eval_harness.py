from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_gold_set(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def recall_at_k(result_ids: list[str], expected_ids: list[str], k: int) -> float:
    if k <= 0 or not expected_ids:
        return 0.0
    top_k = set(result_ids[:k])
    expected = set(expected_ids)
    hits = len(top_k & expected)
    return hits / len(expected)


def evaluate_queries(
    cases: list[dict[str, Any]],
    search_fn,
    k_values: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    metrics = {f"recall@{k}": [] for k in k_values}
    details = []
    for case in cases:
        query = case["query"]
        expected_ids = case["expected_ids"]
        results = search_fn(query=query, top_k=max(k_values))
        result_ids = [r.get("id", "") for r in results]
        row = {"query": query, "expected_ids": expected_ids, "result_ids": result_ids}
        for k in k_values:
            score = recall_at_k(result_ids, expected_ids, k)
            metrics[f"recall@{k}"].append(score)
            row[f"recall@{k}"] = score
        details.append(row)

    summary = {name: (sum(values) / len(values) if values else 0.0) for name, values in metrics.items()}
    return {"summary": summary, "details": details}
