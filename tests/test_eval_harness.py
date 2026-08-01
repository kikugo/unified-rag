import json
import tempfile
import unittest
from pathlib import Path

from eval_harness import (
    evaluate_queries,
    load_gold_set,
    make_bm25_search_fn,
    make_dense_search_fn,
    make_hybrid_search_fn,
    recall_at_k,
)

DATA_DIR = Path(__file__).parent / "data"


class EvalHarnessTest(unittest.TestCase):
    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["b"], 2), 1.0)
        self.assertEqual(recall_at_k(["a", "b", "c"], ["d"], 3), 0.0)

    def test_evaluate_queries(self):
        def fake_search(query: str, top_k: int):
            if query == "q1":
                return [{"id": "a"}, {"id": "b"}]
            return [{"id": "x"}, {"id": "y"}]

        report = evaluate_queries(
            [
                {"query": "q1", "expected_ids": ["b"]},
                {"query": "q2", "expected_ids": ["z"]},
            ],
            search_fn=fake_search,
            k_values=(1, 2),
        )
        self.assertIn("summary", report)
        self.assertIn("details", report)
        self.assertEqual(len(report["details"]), 2)
        self.assertGreaterEqual(report["summary"]["recall@2"], report["summary"]["recall@1"])

    def test_load_gold_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.json"
            payload = [{"query": "q1", "expected_ids": ["a"]}]
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_gold_set(path)
            self.assertEqual(loaded[0]["expected_ids"], ["a"])

    def test_seeded_gold_set_retrieves_every_expected_doc(self):
        corpus = load_gold_set(DATA_DIR / "eval_corpus.json")
        gold = load_gold_set(DATA_DIR / "eval_gold_set.json")

        # Every gold doc id must exist in the seeded corpus, or the set is stale.
        corpus_ids = {doc["id"] for doc in corpus}
        for case in gold:
            for expected in case["expected_ids"]:
                self.assertIn(expected, corpus_ids, f"gold id {expected} missing from corpus")

        report = evaluate_queries(gold, search_fn=make_bm25_search_fn(corpus), k_values=(3, 5))
        # Regression gate: BM25 must surface every gold doc within the top 3.
        self.assertEqual(report["summary"]["recall@3"], 1.0)
        self.assertEqual(report["summary"]["recall@5"], 1.0)
        for row in report["details"]:
            self.assertEqual(row["recall@3"], 1.0, f"regressed on query: {row['query']}")


class DenseAndHybridArmsTest(unittest.TestCase):
    CORPUS = [
        {"id": "a", "searchable_text": "alpha"},
        {"id": "b", "searchable_text": "beta"},
    ]
    EMBEDDINGS = {
        "docs": {"a": [1.0, 0.0], "b": [0.0, 1.0]},
        "queries": {"alpha": [0.9, 0.1], "beta": [0.1, 0.9]},
    }

    def test_dense_search_fn_ranks_by_cosine_similarity(self):
        fn = make_dense_search_fn(self.CORPUS, self.EMBEDDINGS)
        self.assertEqual([r["id"] for r in fn(query="alpha", top_k=2)], ["a", "b"])
        self.assertEqual([r["id"] for r in fn(query="beta", top_k=2)], ["b", "a"])

    def test_dense_search_fn_respects_top_k(self):
        fn = make_dense_search_fn(self.CORPUS, self.EMBEDDINGS)
        self.assertEqual(len(fn(query="alpha", top_k=1)), 1)

    def test_dense_search_fn_raises_on_missing_query_vector(self):
        fn = make_dense_search_fn(self.CORPUS, self.EMBEDDINGS)
        # a silent miss would score every document zero and quietly corrupt an
        # ablation, so this has to be loud
        with self.assertRaises(KeyError):
            fn(query="never-embedded", top_k=2)

    def test_hybrid_search_fn_returns_ids_from_both_rankings(self):
        fn = make_hybrid_search_fn(self.CORPUS, self.EMBEDDINGS)
        results = fn(query="alpha", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("a", [r["id"] for r in results])


if __name__ == "__main__":
    unittest.main()
