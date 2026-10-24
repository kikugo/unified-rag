import unittest

import numpy as np

from rag.pipeline import (
    ChromaStore,
    Config,
    build_answer_contents,
    hits_to_results,
    hyde_vector,
    retrieve,
)
from rag_core import reciprocal_rank_fusion, select_context_results


class FakeCollection:
    """Just enough of a Chroma collection: get() and an exact cosine query()."""

    def __init__(self, docs: dict[str, tuple[list[float], str]]):
        self.docs = docs

    def get(self, where=None, include=None):
        ids = list(self.docs)
        return {"ids": ids, "metadatas": [{"searchable_text": self.docs[i][1]} for i in ids]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        q = np.asarray(query_embeddings[0], dtype=float)
        scored = []
        for doc_id, (vec, _) in self.docs.items():
            v = np.asarray(vec, dtype=float)
            sim = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))
            scored.append((doc_id, 1.0 - sim))
        scored.sort(key=lambda pair: pair[1])
        scored = scored[:n_results]
        return {"ids": [[i for i, _ in scored]], "distances": [[d for _, d in scored]]}


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.hyde_calls = 0

    def embed_query(self, text):
        return np.asarray(self.vectors[text], dtype=float)

    def hyde_query(self, text):
        self.hyde_calls += 1
        return np.asarray(self.vectors["hyde:" + text], dtype=float)


class ReverseReranker:
    """Scores texts so the reranked order is the reverse of the input order."""

    def score(self, query, texts):
        return [float(i) for i in range(len(texts))]


# "a" is the dense winner, "c" holds the exact keyword, "b" sits between.
DOCS = {
    "a": ([1.0, 0.0, 0.0], "the regulation on machine systems"),
    "b": ([0.7, 0.7, 0.0], "general purpose models and transparency"),
    "c": ([0.0, 1.0, 0.0], "Article 5 bans social scoring under Article 5"),
    "d": ([0.0, 0.0, 1.0], "unrelated text about weather"),
}
QUESTION = "Article 5 social scoring"


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.store = ChromaStore(FakeCollection(DOCS))
        self.embedder = FakeEmbedder({
            QUESTION: [1.0, 0.1, 0.0],
            "hyde:" + QUESTION: [0.0, 1.0, 0.0],
        })

    def _ids(self, cfg, **kw):
        return [h.chunk_id for h in retrieve(QUESTION, cfg, self.store, self.embedder, **kw)]

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            Config(mode="sparse")
        with self.assertRaises(ValueError):
            Config(top_k=3, context_k=4)
        with self.assertRaises(ValueError):
            Config(top_k=0, context_k=0)

    def test_dense_orders_by_cosine(self):
        hits = retrieve(QUESTION, Config(mode="dense", top_k=4), self.store, self.embedder)
        self.assertEqual([h.chunk_id for h in hits], ["a", "b", "c", "d"])
        self.assertAlmostEqual(hits[0].score, hits[0].dense_score)
        self.assertGreater(hits[0].dense_score, hits[1].dense_score)

    def test_bm25_puts_keyword_doc_first_and_skips_non_matches(self):
        hits = retrieve(QUESTION, Config(mode="bm25", top_k=4), self.store, self.embedder)
        self.assertEqual(hits[0].chunk_id, "c")
        self.assertNotIn("d", [h.chunk_id for h in hits])
        self.assertIsNone(hits[0].dense_score)

    def test_hybrid_matches_shared_rrf(self):
        cfg = Config(mode="hybrid_rrf", top_k=4)
        dense = self._ids(Config(mode="dense", top_k=4))
        sparse = self._ids(Config(mode="bm25", top_k=4))
        hits = retrieve(QUESTION, cfg, self.store, self.embedder)
        self.assertEqual([h.chunk_id for h in hits], reciprocal_rank_fusion([dense, sparse])[:4])
        self.assertAlmostEqual(hits[0].rrf_score, hits[0].score)

    def test_rerank_reorders_the_candidate_pool(self):
        base = self._ids(Config(mode="dense", top_k=4))
        reranked = self._ids(Config(mode="dense", rerank=True, top_k=4), reranker=ReverseReranker())
        self.assertEqual(reranked, list(reversed(base)))

    def test_rerank_without_reranker_is_an_error(self):
        with self.assertRaises(ValueError):
            retrieve(QUESTION, Config(rerank=True), self.store, self.embedder)

    def test_hyde_uses_the_hyde_vector(self):
        ids = self._ids(Config(mode="dense", hyde=True, top_k=1, context_k=1))
        self.assertEqual(self.embedder.hyde_calls, 1)
        self.assertEqual(ids, ["c"])

    def test_query_emb_without_text_falls_back_to_dense_order(self):
        # The app's image search passes a vector and no question.
        hits = retrieve(None, Config(top_k=2, context_k=2), self.store, query_emb=np.array([0.0, 0.0, 1.0]))
        self.assertEqual(hits[0].chunk_id, "d")

    def test_top_k_truncates(self):
        self.assertEqual(len(self._ids(Config(mode="dense", top_k=2, context_k=1))), 2)

    def test_empty_store_returns_nothing(self):
        empty = ChromaStore(FakeCollection({}))
        self.assertEqual(retrieve(QUESTION, Config(), empty, self.embedder), [])


class HydeVectorTest(unittest.TestCase):
    def test_averages_query_and_draft(self):
        vec = hyde_vector("q", lambda q: "draft", lambda t: [2.0, 0.0] if t == "q" else [0.0, 2.0])
        np.testing.assert_allclose(vec, [1.0, 1.0])

    def test_falls_back_to_query_when_draft_fails(self):
        def boom(_):
            raise RuntimeError("quota")
        np.testing.assert_allclose(hyde_vector("q", boom, lambda t: [3.0]), [3.0])

    def test_none_when_nothing_embeds(self):
        def boom(_):
            raise RuntimeError("down")
        self.assertIsNone(hyde_vector("q", lambda q: "d", boom))


class AnswerContentsTest(unittest.TestCase):
    def test_text_chunks_are_sent_as_text_and_context_k_limits_sources(self):
        embedder = FakeEmbedder({QUESTION: [1.0, 0.1, 0.0]})
        hits = retrieve(QUESTION, Config(mode="dense", top_k=3), ChromaStore(FakeCollection(DOCS)), embedder)
        selected = select_context_results(hits_to_results(hits), max_results=2)
        contents = build_answer_contents(QUESTION, selected, history=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(contents), 2)
        texts = [p.text for p in contents[-1].parts if p.text]
        self.assertIn("the regulation on machine systems", texts)
        self.assertNotIn("Article 5 bans social scoring under Article 5", texts, "third hit is past context_k")
        self.assertIn("[S2]", texts[-1])
        self.assertNotIn("[S3]", texts[-1])

    def test_bytes_take_priority_over_text(self):
        selected = [{"citation_id": "S1", "name": "chart.png", "type": "image",
                     "bytes": b"\x89PNG", "mime": "image/png", "text": "caption"}]
        parts = build_answer_contents("q", selected)[0].parts
        self.assertTrue(any(p.inline_data for p in parts))
        self.assertNotIn("caption", [p.text for p in parts])


if __name__ == "__main__":
    unittest.main()
