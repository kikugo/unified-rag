import json
import tempfile
import unittest
from pathlib import Path

from eval_harness import evaluate_queries, load_gold_set, recall_at_k


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


if __name__ == "__main__":
    unittest.main()
