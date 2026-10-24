"""Delete-and-rebuild stability on a real Chroma store, with no API calls."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag.cache import DiskCache, cache_key
from rag.index import build_index, delete_index, load_chunks, open_store
from rag.pipeline import Config, retrieve

DIM = 16


def _hash_vector(text: str) -> np.ndarray:
    """Bag-of-words hashed into DIM buckets: deterministic and loosely semantic."""
    vec = np.zeros(DIM, dtype=np.float32)
    for word in text.lower().split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    return vec if vec.any() else np.ones(DIM, dtype=np.float32)


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def is_cached(self, task, content):
        return False

    def embed_documents(self, texts):
        self.calls += 1
        return [_hash_vector(t) for t in texts]

    def embed_query(self, text):
        return _hash_vector(text)

    def hyde_query(self, text):
        return _hash_vector(text)

    def embed_figure(self, image, mime, caption):
        return _hash_vector(caption)


def _write_corpus(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    manifest = {"documents": [
        {"doc_id": "law", "title": "Example Act"},
        {"doc_id": "wiki", "title": "Example article"},
    ]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    words = " ".join(f"provision{i} applies to providers" for i in range(400))
    rows = {
        "law": [{"page": 1, "heading": "page 1", "text": "Article 5 prohibits social scoring. " + words},
                {"page": 2, "heading": "page 2", "text": "Annex III lists high-risk uses. " + words}],
        "wiki": [{"page": 0, "heading": "Introduction", "text": "Moore's law tracks transistor counts. " + words}],
    }
    for doc_id, doc_rows in rows.items():
        lines = "".join(json.dumps({"doc_id": doc_id, **r}) + "\n" for r in doc_rows)
        (root / "docs" / f"{doc_id}.jsonl").write_text(lines, encoding="utf-8")
    fig = {"doc_id": "wiki", "figure_index": 0, "page": 0, "char_offset": 0,
           "caption": "Transistor count chart", "image": "raw/figures/wiki-0.png"}
    fig2 = {**fig, "figure_index": 1, "caption": "A second chart at the same spot"}
    (root / "figures.jsonl").write_text(json.dumps(fig) + "\n" + json.dumps(fig2) + "\n", encoding="utf-8")


class IndexRebuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.corpus = root / "corpus"
        self.index_dir = root / "index"
        _write_corpus(self.corpus)

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self):
        chunks = load_chunks(self.corpus)
        build_index(chunks, FakeEmbedder(), self.index_dir, DIM, figure_reader=lambda c: (b"", "image/png"))
        return chunks

    def _rankings(self):
        store = open_store(self.index_dir, DIM)
        out = {}
        for mode in ("dense", "bm25", "hybrid_rrf"):
            for q in ("Article 5 social scoring", "transistor chart", "providers"):
                out[(mode, q)] = [h.chunk_id for h in retrieve(q, Config(mode=mode), store, FakeEmbedder())]
        return out

    def test_delete_and_rebuild_gives_identical_ids_and_rankings(self):
        chunks = self._build()
        ids_before = {d["id"] for d in open_store(self.index_dir, DIM).sparse_docs()}
        self.assertEqual(ids_before, {c["chunk_id"] for c in chunks})
        before = self._rankings()
        delete_index(self.index_dir)
        self.assertFalse(self.index_dir.exists())
        self._build()
        ids_after = {d["id"] for d in open_store(self.index_dir, DIM).sparse_docs()}
        self.assertEqual(ids_before, ids_after)
        self.assertEqual(before, self._rankings())

    def test_chunks_are_text_windows_plus_figures_with_unique_ids(self):
        chunks = load_chunks(self.corpus)
        kinds = [c["kind"] for c in chunks]
        self.assertEqual(kinds.count("figure"), 2)
        self.assertGreater(kinds.count("text"), 3)
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        self.assertTrue(all(c["searchable_text"].startswith(c["title"]) for c in chunks))

    def test_keyword_query_finds_its_page(self):
        self._build()
        hits = retrieve("Annex III high-risk", Config(mode="bm25", top_k=1, context_k=1), open_store(self.index_dir, DIM))
        self.assertEqual((hits[0].metadata["doc_id"], hits[0].metadata["page"]), ("law", 2))


class DiskCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = DiskCache(Path(self.tmp.name) / "c.sqlite")

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_vector_and_text_round_trip(self):
        self.cache.put_vector("k", [0.5, -1.25])
        np.testing.assert_array_equal(self.cache.get_vector("k"), np.array([0.5, -1.25], dtype=np.float32))
        self.cache.put_text("t", "draft passage")
        self.assertEqual(self.cache.get_text("t"), "draft passage")
        self.assertIsNone(self.cache.get_vector("missing"))
        self.assertIn("k", self.cache)

    def test_keys_separate_every_part(self):
        self.assertNotEqual(cache_key("m", 3072, "Q", "x"), cache_key("m", 768, "Q", "x"))
        self.assertNotEqual(cache_key("ab", "c"), cache_key("a", "bc"))
        self.assertEqual(cache_key("m", 1, b"\x00"), cache_key("m", 1, b"\x00"))


if __name__ == "__main__":
    unittest.main()
