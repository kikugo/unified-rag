import json
import tempfile
import unittest
from pathlib import Path

from eval_harness import (
    evaluate_queries,
    load_gold_set,
    make_bm25_search_fn,
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


if __name__ == "__main__":
    unittest.main()
