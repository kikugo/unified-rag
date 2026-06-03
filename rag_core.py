from __future__ import annotations

from copy import copy
import math
import re
from typing import Any


LOCAL_BACKEND = "local"
MANAGED_BACKEND = "managed"
TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def is_local_source(source: dict[str, Any]) -> bool:
    return source.get("backend", LOCAL_BACKEND) == LOCAL_BACKEND and bool(source.get("id"))


def source_chroma_ids(source: dict[str, Any]) -> list[str]:
    if source.get("chroma_ids"):
        return list(source["chroma_ids"])
    if source.get("id"):
        return [source["id"]]
    return []


def pop_source_and_embedding(
    sources: list[dict[str, Any]],
    embeddings: list[Any],
    source_index: int,
) -> tuple[dict[str, Any], list[str]]:
    """Remove one UI source and its matching local embedding, if it has one."""
    if source_index < 0 or source_index >= len(sources):
        raise IndexError("source_index out of range")

    source = sources[source_index]
    local_embedding_index = sum(1 for item in sources[:source_index] if is_local_source(item))
    removed = sources.pop(source_index)

    if is_local_source(source) and local_embedding_index < len(embeddings):
        embeddings.pop(local_embedding_index)

    return removed, source_chroma_ids(removed)


def select_context_results(results: list[dict[str, Any]], max_results: int = 5) -> list[dict[str, Any]]:
    selected = []
    for idx, result in enumerate(results[:max(0, max_results)], start=1):
        item = copy(result)
        item["citation_id"] = f"S{idx}"
        selected.append(item)
    return selected


def build_local_answer_prompt(question: str, selected_results: list[dict[str, Any]]) -> str:
    lines = [
        "Use the retrieved sources below to answer the user's question.",
        "Ground the answer only in these sources. When you use evidence, cite the source IDs like [S1].",
        "If the sources do not contain the answer, say that clearly.",
        "",
        "Retrieved sources:",
    ]

    for result in selected_results:
        score = result.get("score")
        score_text = f", score={score:.4f}" if isinstance(score, (int, float)) else ""
        lines.append(
            f"[{result['citation_id']}] {result.get('name', 'Unknown source')} "
            f"(type={result.get('type', 'unknown')}{score_text})"
        )

    lines.extend(["", f"Question: {question}"])
    return "\n".join(lines)


def tokenize_for_search(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def bm25_rank(query: str, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
    tokenized_docs = [tokenize_for_search(doc.get("searchable_text", "")) for doc in docs]
    query_terms = tokenize_for_search(query)
    if not query_terms or not tokenized_docs:
        return []

    doc_count = len(tokenized_docs)
    avg_len = sum(len(tokens) for tokens in tokenized_docs) / doc_count if doc_count else 0
    doc_freqs: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    scored = []
    for doc, tokens in zip(docs, tokenized_docs):
        if not doc.get("id") or not tokens:
            continue
        term_counts: dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        score = 0.0
        doc_len = len(tokens)
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freqs.get(term, 0)
            idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1 - b + b * (doc_len / avg_len if avg_len else 0))
            score += idf * ((tf * (k1 + 1)) / denom)
        if score > 0:
            scored.append((doc["id"], score))

    return sorted(scored, key=lambda item: item[1], reverse=True)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    seen_order = 0
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in first_seen:
                first_seen[doc_id] = seen_order
                seen_order += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)

    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))
