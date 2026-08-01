from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rag_core import bm25_rank, reciprocal_rank_fusion


def load_gold_set(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def make_bm25_search_fn(corpus: list[dict[str, Any]]):
    """Build an offline, BM25-only search_fn over a fixed corpus.

    Corpus items need `id` and `searchable_text`. The returned function matches
    the signature evaluate_queries expects (search_fn(query, top_k)) and needs
    no Gemini key or live index, so it can run the gold set in CI.
    """
    def search_fn(query: str, top_k: int) -> list[dict[str, Any]]:
        ranked = bm25_rank(query, corpus)[:top_k]
        return [{"id": doc_id} for doc_id, _ in ranked]

    return search_fn


def make_dense_search_fn(corpus: list[dict[str, Any]], embeddings: dict[str, Any]):
    """Build an offline dense search_fn over cached embeddings.

    Vectors are read from a committed JSON cache rather than the API, so the
    ablation reruns in CI with no key and gives the same answer every time.
    Corpus items need `id`; `embeddings` holds {"docs": {id: vec}, "queries": {q: vec}}.
    """
    doc_vectors = embeddings["docs"]
    query_vectors = embeddings["queries"]

    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search_fn(query: str, top_k: int) -> list[dict[str, Any]]:
        # A missing vector must raise rather than score zero. Silently ranking an
        # unembedded query last would corrupt the ablation without failing.
        if query not in query_vectors:
            raise KeyError(f"no cached embedding for query: {query!r}")
        qv = query_vectors[query]
        scored = []
        for doc in corpus:
            doc_id = doc["id"]
            if doc_id not in doc_vectors:
                raise KeyError(f"no cached embedding for doc: {doc_id!r}")
            scored.append((doc_id, _cosine(qv, doc_vectors[doc_id])))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [{"id": doc_id} for doc_id, _ in scored[:top_k]]

    return search_fn


def make_hybrid_search_fn(corpus: list[dict[str, Any]], embeddings: dict[str, Any]):
    """Dense and BM25 rankings fused with reciprocal rank fusion.

    Reuses `rag_core.reciprocal_rank_fusion`, the same function the app itself
    calls, so the ablation measures the shipped fusion rather than a copy of it.
    """
    dense = make_dense_search_fn(corpus, embeddings)

    def search_fn(query: str, top_k: int) -> list[dict[str, Any]]:
        pool = max(top_k * 4, top_k)
        dense_ids = [r["id"] for r in dense(query=query, top_k=pool)]
        sparse_ids = [doc_id for doc_id, _ in bm25_rank(query, corpus)[:pool]]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])[:top_k]
        return [{"id": doc_id} for doc_id in fused]

    return search_fn


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
