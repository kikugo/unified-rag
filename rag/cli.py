"""Ask the benchmark index a question from the terminal.

    uv run python -m rag.cli "question" --mode hybrid_rrf --rerank
    uv run python -m rag.cli "question" --mode dense --hyde --answer

Prints the ranked chunk ids with their scores. Build the index first with
`python -m rag.index build`.
"""
from __future__ import annotations

import argparse
import sys

from rag.cache import DiskCache
from rag.gemini_client import GeminiEmbedder
from rag.index import DEFAULT_DIM, INDEX_DIR, open_store, read_figure
from rag.models import ANSWER_MODEL
from rag.pipeline import MODES, Config, answer, retrieve
from rag.rerank import CrossEncoderReranker


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


def _load_bytes(meta: dict) -> tuple[bytes, str] | None:
    return read_figure(meta) if meta.get("image") else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.cli")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rrf")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--hyde", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--answer", action="store_true", help="also generate a cited answer")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    args = parser.parse_args(argv)

    cfg = Config(mode=args.mode, rerank=args.rerank, hyde=args.hyde,
                 top_k=args.top_k, context_k=min(args.context_k, args.top_k))
    store = open_store(args.index_dir, args.dim)
    embedder = GeminiEmbedder(DiskCache(), dim=args.dim)
    reranker = CrossEncoderReranker() if cfg.rerank else None
    hits = retrieve(args.question, cfg, store, embedder, reranker)

    print(f"# {cfg}")
    print(f"{'rank':>4}  {'chunk_id':16}  {'score':>8}  {'dense':>7}  {'bm25':>7}  {'rrf':>7}  {'rerank':>7}  doc_id / page / kind")
    for rank, hit in enumerate(hits, start=1):
        meta = hit.metadata
        print(f"{rank:>4}  {hit.chunk_id:16}  {hit.score:8.4f}  {_fmt(hit.dense_score):>7}  "
              f"{_fmt(hit.bm25_score):>7}  {_fmt(hit.rrf_score):>7}  {_fmt(hit.rerank_score):>7}  "
              f"{meta.get('doc_id')} / {meta.get('page')} / {meta.get('kind')}")

    if args.answer:
        text, selected = answer(args.question, hits, cfg, embedder.client, ANSWER_MODEL, _load_bytes)
        print()
        print(text)
        for row in selected:
            print(f"[{row['citation_id']}] {row['id']}  {row['name']} (page/section {row['page']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
