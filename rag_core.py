from __future__ import annotations

from copy import copy
import math
import re
import time
from typing import Any, Callable


LOCAL_BACKEND = "local"
MANAGED_BACKEND = "managed"
TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def is_rate_limit_error(error: Exception) -> bool:
    """True when a Gemini error looks like a rate-limit / quota rejection."""
    msg = str(error).lower()
    return any(token in msg for token in ("429", "resource_exhausted", "rate limit", "quota"))


def call_with_retry(fn: Callable, attempts: int = 3, base_delay: float = 1.0, sleep=time.sleep):
    """Call fn(), retrying on rate-limit errors with exponential backoff.

    Non-rate-limit errors raise immediately; rate-limit errors raise after the
    final attempt. `sleep` is injectable so tests can run without real delays.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as error:
            if attempt == attempts - 1 or not is_rate_limit_error(error):
                raise
            sleep(base_delay * (2 ** attempt))


def is_local_source(source: dict[str, Any]) -> bool:
    return source.get("backend", LOCAL_BACKEND) == LOCAL_BACKEND and bool(source.get("id"))


def source_chroma_ids(source: dict[str, Any]) -> list[str]:
    if source.get("chroma_ids"):
        return list(source["chroma_ids"])
    if source.get("id"):
        return [source["id"]]
    return []


def source_artifacts(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-chunk blob records for a source, for cleanup on delete.

    Falls back to the row's own paths for legacy single-chunk rows that predate
    artifact tracking.
    """
    if source.get("artifacts"):
        return list(source["artifacts"])
    paths = {k: source.get(k) for k in ("file_path", "preview_path", "video_path") if source.get(k)}
    return [paths] if paths else []


def pop_source_by_id(
    sources: list[dict[str, Any]],
    source_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Remove a source row by `source_id`, returning (removed_row, chroma_ids)."""
    for idx, source in enumerate(sources):
        if source.get("source_id") == source_id:
            removed = sources.pop(idx)
            return removed, source_chroma_ids(removed)
    return {}, []


def build_rehydrated_sources(registry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild the in-session sidebar source list from persisted registry rows.

    Only local rows are restored (managed sources live in Google's store and are
    not re-listed here). The registry stores `doc_id`; the sidebar expects `id`,
    so the value is mapped across.
    """
    restored = []
    for row in registry_rows:
        if row.get("backend", LOCAL_BACKEND) != LOCAL_BACKEND:
            continue
        doc_id = row.get("doc_id") or (row.get("chroma_ids") or [None])[0]
        if not doc_id:
            continue
        restored.append({
            "source_id": row.get("source_id"),
            "id": doc_id,
            "name": row.get("name", "Unknown"),
            "type": row.get("type", "unknown"),
            "mime": row.get("mime", "application/octet-stream"),
            "file_path": row.get("file_path"),
            "preview_path": row.get("preview_path"),
            "video_path": row.get("video_path"),
            "backend": LOCAL_BACKEND,
            "searchable_text": row.get("searchable_text", ""),
            "content_hash": row.get("content_hash", ""),
            "chroma_ids": row.get("chroma_ids") or [doc_id],
            "artifacts": row.get("artifacts") or [],
            "page_range": row.get("page_range"),
            "caption": row.get("caption"),
            "timestamp_start_sec": row.get("timestamp_start_sec"),
            "timestamp_end_sec": row.get("timestamp_end_sec"),
        })
    return restored


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
