"""Build, delete and check the benchmark index.

    uv run python -m rag.index build [--yes]
    uv run python -m rag.index delete
    uv run python -m rag.index verify-stable

The index is a local Chroma store in .eval_index/ built from the committed
corpus text, separate from the app's ./chroma_db so the two never mix.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sys
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient

from media_ingest import normalize_image
from rag.cache import DiskCache
from rag.chunking import chunk_page, make_chunk_id
from rag.corpus import CORPUS_DIR, load_manifest
from rag.gemini_client import GeminiEmbedder
from rag.models import EMBEDDING_MODEL
from rag.pipeline import ChromaStore, Config, retrieve
from rag.rerank import CrossEncoderReranker

INDEX_DIR = Path(__file__).resolve().parent.parent / ".eval_index"
DEFAULT_DIM = 3072
UPSERT_BATCH = 200
# Rough Gemini token cost of one image part, for the pre-build estimate only.
IMAGE_TOKEN_ESTIMATE = 258

# Fixed probes for verify-stable: an exact keyword, a paraphrase, and a figure.
STABILITY_QUERIES = [
    "Article 5 prohibited AI practices",
    "how can lenders decide whether someone is likely to repay a loan",
    "chart of transistor counts over time",
]


def collection_name(dim: int) -> str:
    return f"bench_{dim}"


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, Any]]:
    """Every text window and figure in the corpus, in a fixed order.

    `searchable_text` (what gets embedded and BM25-scored) prefixes the document
    title and section heading so a chunk that never names its subject can still
    be found. The id is built from the position only, never from that prefix.
    """
    manifest = load_manifest(corpus_dir)
    titles = {d["doc_id"]: d["title"] for d in manifest["documents"]}
    chunks: list[dict[str, Any]] = []
    for doc in manifest["documents"]:
        doc_id = doc["doc_id"]
        path = corpus_dir / "docs" / f"{doc_id}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for chunk in chunk_page(doc_id, row["page"], row["text"]):
                chunk["title"] = titles[doc_id]
                chunk["heading"] = row["heading"]
                chunk["searchable_text"] = f"{titles[doc_id]} | {row['heading']}\n{chunk['text']}"
                chunks.append(chunk)
    figures_path = corpus_dir / "figures.jsonl"
    if figures_path.exists():
        for line in figures_path.read_text(encoding="utf-8").splitlines():
            fig = json.loads(line)
            doc_id = fig["doc_id"]
            chunks.append({
                # Figures placed back to back share an offset, so the hash key
                # also carries the figure's ordinal in its document.
                "chunk_id": make_chunk_id(doc_id, fig["page"], fig["char_offset"],
                                          kind=f"figure-{fig['figure_index']}"),
                "doc_id": doc_id,
                "page": fig["page"],
                "char_offset": fig["char_offset"],
                "kind": "figure",
                "text": fig["caption"],
                "title": titles[doc_id],
                "heading": f"figure {fig['figure_index']}",
                "searchable_text": f"{titles[doc_id]} | figure\n{fig['caption']}",
                "image": fig["image"],
            })
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate chunk ids in the corpus")
    return chunks


def read_figure(chunk: dict[str, Any], corpus_dir: Path = CORPUS_DIR) -> tuple[bytes, str]:
    """Figure bytes re-encoded the way the app normalizes uploads.

    The extension's mime type is passed as the fallback: Pillow refuses a few
    valid PNGs with oversized text chunks, and normalize_image then returns the
    original bytes with whatever mime it was given.
    """
    path = corpus_dir / chunk["image"]
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return normalize_image(path.read_bytes(), mime)


def estimate(chunks: list[dict[str, Any]], embedder) -> dict[str, int]:
    """Count only what is not cached yet, so a rebuild estimates zero."""
    texts = [c for c in chunks if c["kind"] == "text"
             and not embedder.is_cached("RETRIEVAL_DOCUMENT", c["searchable_text"])]
    figures = [c for c in chunks if c["kind"] == "figure"]
    return {
        "chunks": len(chunks),
        "uncached_text_chunks": len(texts),
        "figures": len(figures),
        "estimated_tokens": sum(len(c["searchable_text"]) for c in texts) // 4 + len(figures) * IMAGE_TOKEN_ESTIMATE,
    }


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = {k: chunk[k] for k in ("doc_id", "page", "char_offset", "kind", "title", "heading", "searchable_text")}
    meta["text"] = chunk["text"]
    if chunk.get("image"):
        meta["image"] = chunk["image"]
    return meta


def open_collection(index_dir: Path = INDEX_DIR, dim: int = DEFAULT_DIM, create: bool = False):
    client = chromadb.PersistentClient(path=str(index_dir))
    if create:
        return client.get_or_create_collection(
            name=collection_name(dim),
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL, "dim": dim},
        )
    return client.get_collection(collection_name(dim))


def open_store(index_dir: Path = INDEX_DIR, dim: int = DEFAULT_DIM, exact: bool = False) -> ChromaStore:
    return ChromaStore(open_collection(index_dir, dim), exact=exact)


def build_index(chunks: list[dict[str, Any]], embedder, index_dir: Path = INDEX_DIR,
                dim: int = DEFAULT_DIM, figure_reader=read_figure) -> int:
    """Embed and upsert every chunk. Order is fixed by `load_chunks`."""
    collection = open_collection(index_dir, dim, create=True)
    text_chunks = [c for c in chunks if c["kind"] == "text"]
    vectors = dict(zip(
        (c["chunk_id"] for c in text_chunks),
        embedder.embed_documents([c["searchable_text"] for c in text_chunks]),
    ))
    for chunk in chunks:
        if chunk["kind"] == "figure":
            image, mime = figure_reader(chunk)
            vectors[chunk["chunk_id"]] = embedder.embed_figure(image, mime, chunk["text"])
    for start in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[start:start + UPSERT_BATCH]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[vectors[c["chunk_id"]].tolist() for c in batch],
            metadatas=[_metadata(c) for c in batch],
        )
    return collection.count()


def delete_index(index_dir: Path = INDEX_DIR) -> None:
    # Chroma keeps one client per path for the life of the process. Without
    # clearing it, a rebuild in the same process writes through the handle of
    # the deleted store and fails with "attempt to write a readonly database".
    SharedSystemClient.clear_system_cache()
    if index_dir.exists():
        shutil.rmtree(index_dir)


def stability_configs() -> list[Config]:
    return [Config(mode=m, rerank=r) for m in ("dense", "bm25", "hybrid_rrf") for r in (False, True)]


def snapshot(store: ChromaStore, embedder, reranker, queries: list[str]) -> dict[str, list[str]]:
    out = {}
    for cfg in stability_configs():
        for q in queries:
            hits = retrieve(q, cfg, store, embedder, reranker if cfg.rerank else None)
            out[f"{cfg.mode}{'+rerank' if cfg.rerank else ''} | {q}"] = [h.chunk_id for h in hits]
    return out


def _confirm(est: dict[str, int], assume_yes: bool) -> bool:
    print(f"{est['chunks']} chunks ({est['figures']} figures), "
          f"{est['uncached_text_chunks']} text chunks not cached, ~{est['estimated_tokens']:,} tokens to embed")
    if assume_yes or (est["uncached_text_chunks"] == 0 and est["figures"] == 0):
        return True
    return input("embed now? [y/N] ").strip().lower() == "y"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.index")
    parser.add_argument("command", choices=["build", "delete", "verify-stable"])
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args(argv)

    if args.command == "delete":
        delete_index(args.index_dir)
        print(f"deleted {args.index_dir}")
        return 0

    embedder = GeminiEmbedder(DiskCache(), dim=args.dim)
    chunks = load_chunks()
    if args.command == "build":
        if not _confirm(estimate(chunks, embedder), args.yes):
            return 1
        count = build_index(chunks, embedder, args.index_dir, args.dim)
        print(f"indexed {count} chunks into {args.index_dir} ({embedder.calls} API calls)")
        return 0

    # verify-stable: snapshot, delete, rebuild, snapshot again, compare.
    reranker = CrossEncoderReranker()
    before = snapshot(open_store(args.index_dir, args.dim), embedder, reranker, STABILITY_QUERIES)
    delete_index(args.index_dir)
    print(f"deleted {args.index_dir}, rebuilding")
    build_index(chunks, embedder, args.index_dir, args.dim)
    after = snapshot(open_store(args.index_dir, args.dim), embedder, reranker, STABILITY_QUERIES)
    changed = [key for key in before if before[key] != after.get(key)]
    for key in before:
        status = "CHANGED" if key in changed else "same"
        print(f"{status:7} {key}: {' '.join(after.get(key, []))}")
    print(f"{len(before) - len(changed)}/{len(before)} rankings identical after delete and rebuild")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
