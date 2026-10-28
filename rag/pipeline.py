"""Retrieval and answering behind one Config, callable without Streamlit.

The app, `python -m rag.cli` and the eval runner all go through `retrieve`, so
an ablation row measures the code the app ships, not a copy of it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from collections.abc import Callable

import numpy as np
from google.genai import types

from rag.tracing import generation, record_generation, stage
from rag_core import (
    bm25_rank,
    build_local_answer_prompt,
    call_with_retry,
    reciprocal_rank_fusion_scores,
    select_context_results,
)

MODES = ("dense", "bm25", "hybrid_rrf")
MAX_HISTORY = 6  # last 3 exchanges
HYDE_PROMPT = "Write a short, factual passage that would answer this question:\n{question}"


@dataclass(frozen=True)
class Config:
    mode: str = "hybrid_rrf"
    rerank: bool = False
    hyde: bool = False
    top_k: int = 5
    context_k: int = 3
    candidate_k: int = 20  # per-arm pool fused by RRF, and the pool the reranker sees
    rrf_k: int = 60

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 1 <= self.context_k <= self.top_k:
            raise ValueError("context_k must be between 1 and top_k")


@dataclass
class Hit:
    chunk_id: str
    score: float  # score of the stage that set the final order
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_score: float | None = None  # cosine similarity
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...
    def hyde_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class ChromaStore:
    """A Chroma collection, optionally filtered by a `where` clause.

    The BM25 side scans every chunk's `searchable_text`, so the full listing is
    fetched once per store and reused. The app builds a fresh store per search
    because its corpus changes between queries; the CLI keeps one.

    `exact=True` replaces Chroma's approximate HNSW search with an exact cosine
    scan over the stored vectors. HNSW occasionally drops a true neighbour from
    the candidate pool, and not the same one every process, so a benchmark rerun
    could rank differently with nothing changed. The eval index is small enough
    that the exact scan costs a few milliseconds.
    """

    def __init__(self, collection, where: dict | None = None, exact: bool = False):
        self.collection = collection
        self.where = where
        self.exact = exact
        self._metas: dict[str, dict[str, Any]] | None = None
        self._matrix: tuple[list[str], np.ndarray] | None = None

    def _corpus(self) -> dict[str, dict[str, Any]]:
        if self._metas is None:
            got = self.collection.get(where=self.where, include=["metadatas"])
            self._metas = {i: (m or {}) for i, m in zip(got.get("ids") or [], got.get("metadatas") or [])}
        return self._metas

    def count(self) -> int:
        return len(self._corpus())

    def metadata(self, chunk_id: str) -> dict[str, Any]:
        return self._corpus().get(chunk_id, {})

    def text(self, chunk_id: str) -> str:
        meta = self.metadata(chunk_id)
        return meta.get("searchable_text", meta.get("name", ""))

    def sparse_docs(self) -> list[dict[str, Any]]:
        return [{"id": i, "searchable_text": self.text(i)} for i in self._corpus()]

    def _normalized_vectors(self) -> tuple[list[str], np.ndarray]:
        if self._matrix is None:
            got = self.collection.get(where=self.where, include=["embeddings"])
            ids = list(got.get("ids") or [])
            matrix = np.asarray(got.get("embeddings"), dtype=np.float64).reshape(len(ids), -1)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._matrix = ids, matrix / np.where(norms == 0, 1.0, norms)
        return self._matrix

    def dense(self, vector: np.ndarray, n: int) -> list[tuple[str, float]]:
        n = min(n, self.count())
        if n <= 0:
            return []
        if self.exact:
            ids, matrix = self._normalized_vectors()
            q = np.asarray(vector, dtype=np.float64)
            sims = matrix @ (q / (np.linalg.norm(q) or 1.0))
            # ties broken by chunk id, so the order never depends on storage order
            order = np.lexsort((np.asarray(ids), -sims))[:n]
            return [(ids[j], float(sims[j])) for j in order]
        res = self.collection.query(
            query_embeddings=[np.asarray(vector).tolist()],
            n_results=n,
            where=self.where,
            include=["distances"],
        )
        ids = res["ids"][0] if res.get("ids") else []
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        # Chroma returns cosine distance; similarity is 1 - distance.
        return [(i, 1.0 - d) for i, d in zip(ids, dists)]


def hyde_vector(
    question: str,
    draft_fn: Callable[[str], str],
    embed_fn: Callable[[str], np.ndarray],
) -> np.ndarray | None:
    """HyDE: average the query vector with the vector of a drafted answer.

    Falls back to the query vector alone when the draft or its embedding fails,
    and returns None only when nothing could be embedded.
    """
    try:
        hypothetical = (draft_fn(question) or "").strip()
    except Exception:
        hypothetical = ""
    vectors = []
    for text in (question, hypothetical):
        if not text:
            continue
        try:
            vectors.append(np.asarray(embed_fn(text), dtype=float))
        except Exception:
            continue
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def retrieve(
    question: str | None,
    cfg: Config,
    store: ChromaStore,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    query_emb: np.ndarray | None = None,
    timings: dict[str, float] | None = None,
) -> list[Hit]:
    """Top `cfg.top_k` chunks for a question under one config.

    `query_emb` skips query embedding (the app's image search passes one). With
    no question text the BM25 arm has nothing to match, so hybrid falls back to
    the dense order, which is what the app always did for image queries.

    Pass a dict as `timings` to get milliseconds per stage (embed, dense, bm25,
    fuse, rerank) for the stages this config runs; the eval runner uses it.
    With LANGFUSE_TRACING=1 the same stages become spans: `retrieve` (with
    embed, dense, bm25 and fuse inside it), then `rerank`.
    """
    if store.count() == 0:
        return []
    stage_ms = timings if timings is not None else {}
    with stage("retrieve", mode=cfg.mode, hyde=cfg.hyde, candidate_k=cfg.candidate_k):
        hits = _first_stage(question, cfg, store, embedder, query_emb, stage_ms)
    if cfg.rerank and hits and question:
        if reranker is None:
            raise ValueError("cfg.rerank is set but no reranker was given")
        with stage("rerank", candidates=min(len(hits), cfg.candidate_k)):
            hits = _rerank(question, cfg, store, reranker, hits, stage_ms)
    return hits[:cfg.top_k]


def _timed(stage_ms: dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    with stage(name):
        start = time.perf_counter()
        out = fn()
        stage_ms[name] = (time.perf_counter() - start) * 1000
    return out


def _first_stage(question, cfg: Config, store: ChromaStore, embedder, query_emb, stage_ms) -> list[Hit]:
    pool = max(cfg.candidate_k, cfg.top_k)

    dense: list[tuple[str, float]] = []
    if cfg.mode in ("dense", "hybrid_rrf"):
        vector = query_emb
        if vector is None:
            if embedder is None or not question:
                raise ValueError("dense retrieval needs an embedder and a question, or query_emb")
            vector = _timed(stage_ms, "embed",
                            lambda: embedder.hyde_query(question) if cfg.hyde else embedder.embed_query(question))
        dense = _timed(stage_ms, "dense", lambda: store.dense(vector, pool))

    sparse: list[tuple[str, float]] = []
    if cfg.mode in ("bm25", "hybrid_rrf") and question:
        sparse = _timed(stage_ms, "bm25", lambda: bm25_rank(question, store.sparse_docs())[:pool])

    dense_by_id = dict(dense)
    sparse_by_id = dict(sparse)
    if cfg.mode == "dense":
        ranked = dense
    elif cfg.mode == "bm25":
        ranked = sparse
    else:
        ranked = _timed(stage_ms, "fuse", lambda: reciprocal_rank_fusion_scores(
            [[i for i, _ in dense], [i for i, _ in sparse]], k=cfg.rrf_k)[:pool])

    return [
        Hit(
            chunk_id=chunk_id,
            score=score,
            metadata=store.metadata(chunk_id),
            dense_score=dense_by_id.get(chunk_id),
            bm25_score=sparse_by_id.get(chunk_id),
            rrf_score=score if cfg.mode == "hybrid_rrf" else None,
        )
        for chunk_id, score in ranked
    ]


def _rerank(question: str, cfg: Config, store: ChromaStore, reranker, hits: list[Hit], stage_ms) -> list[Hit]:
    start = time.perf_counter()
    candidates = hits[:cfg.candidate_k]
    scores = reranker.score(question, [store.text(h.chunk_id) for h in candidates])
    for hit, s in zip(candidates, scores):
        hit.rerank_score = float(s)
        hit.score = float(s)
    # sorted() is stable, so equal rerank scores keep the first-stage order.
    ranked = sorted(candidates, key=lambda h: -h.score)
    stage_ms["rerank"] = (time.perf_counter() - start) * 1000
    return ranked


def build_answer_contents(
    question: str,
    selected_results: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
) -> list[types.Content]:
    """Prior turns, then each cited source, then the grounded prompt.

    A source is sent as its bytes when it has them (the app's PDFs, images,
    frames) and as text otherwise (benchmark text chunks).
    """
    contents: list[types.Content] = []
    for msg in (history or [])[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    parts = []
    for result in selected_results:
        parts.append(types.Part(text=f"Source [{result['citation_id']}] content follows."))
        if result.get("bytes"):
            parts.append(types.Part.from_bytes(
                data=result["bytes"],
                mime_type=result.get("mime", "application/octet-stream"),
            ))
        elif result.get("text"):
            parts.append(types.Part(text=result["text"]))
    parts.append(types.Part(text=build_local_answer_prompt(question, selected_results)))
    contents.append(types.Content(role="user", parts=parts))
    return contents


def hits_to_results(hits: list[Hit], load_bytes: Callable[[dict[str, Any]], tuple[bytes, str] | None] | None = None
                    ) -> list[dict[str, Any]]:
    """Hits as the result dicts the prompt builder and the app expect."""
    results = []
    for hit in hits:
        meta = hit.metadata
        row = {
            "id": hit.chunk_id,
            "name": meta.get("name") or meta.get("title", "Unknown source"),
            "type": meta.get("type") or meta.get("kind", "text"),
            "score": hit.score,
            "text": meta.get("text") or meta.get("searchable_text", ""),
            "page": meta.get("page"),
        }
        if load_bytes:
            loaded = load_bytes(meta)
            if loaded:
                row["bytes"], row["mime"] = loaded
        results.append(row)
    return results


def answer(
    question: str,
    hits: list[Hit],
    cfg: Config,
    client,
    model: str,
    load_bytes: Callable[[dict[str, Any]], tuple[bytes, str] | None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Non-streaming grounded answer over the top `cfg.context_k` hits."""
    contents, gen_config, selected = answer_request(question, hits, cfg, load_bytes)
    with generation("generate", model=model, input=question) as gen:
        resp = call_with_retry(lambda: client.models.generate_content(
            model=model, contents=contents, config=gen_config,
        ))
        text = (resp.text or "").strip()
        record_generation(gen, text, resp.usage_metadata)
    return text, selected


def answer_request(
    question: str,
    hits: list[Hit],
    cfg: Config,
    load_bytes: Callable[[dict[str, Any]], tuple[bytes, str] | None] | None = None,
) -> tuple[list[types.Content], types.GenerateContentConfig, list[dict[str, Any]]]:
    """The exact request `answer` sends, so the eval runner can time and cache it."""
    selected = select_context_results(hits_to_results(hits, load_bytes), max_results=cfg.context_k)
    return build_answer_contents(question, selected), types.GenerateContentConfig(temperature=0), selected
