"""The README tables are a pure function of the results files."""
import json
import tempfile
import unittest
from pathlib import Path

from eval.report import END, START, ReportError, load_results, main, render, splice


def _result(hits, faith, golden="abc"):
    rows = []
    for i, (h, f) in enumerate(zip(hits, faith), start=1):
        rows.append({"id": f"g{i:03d}", "type": "keyword" if i % 2 else "paraphrase", "answerable": True,
                     "hit@5": h, "faithfulness": f})
    by_type = {"keyword": {"n": 2, "hit@5": 1.0, "mrr@10": 0.75}, "paraphrase": {"n": 2, "hit@5": 0.5, "mrr@10": 0.5}}
    latency = {"p50": 100.0, "p95": 1234.5}
    totals = {"n": 4, "n_answerable": 4, "hit@1": 0.5, "hit@5": sum(hits) / 4, "hit@5_ci95": [0.25, 1.0],
              "mrr@10": 0.625, "multihop_recall@5": None, "citation_correctness": 0.8, "faithfulness": sum(faith) / 4,
              "faithfulness_ci95": [0.5, 1.0], "answer_relevancy": 0.9, "refusal_accuracy": None,
              "false_refusal_rate": 0.0, "cost_usd_per_100_queries": 0.29,
              "latency_ms": {k: latency for k in ("embed", "retrieve", "rerank", "generate", "total")}}
    meta = {"golden_sha256_16": golden, "judge": {"model": "gpt-4.1-mini", "version": "2025-04-14", "temperature": 0},
            "config": {"generator": "gemini-3.6-flash"}, "prices_checked": "2026-10-26"}
    return {"meta": meta, "totals": totals, "by_type": by_type, "rows": rows}


README_TEXT = f"# Title\n\nIntro stays.\n\n{START}\nold table\n{END}\n\nOutro stays.\n"


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.results = self.dir / "results"
        self.results.mkdir()
        self._write("hybrid_rrf_rerank", _result([1, 1, 0, 1], [1.0, 1.0, 0.5, 1.0]))
        self._write("bm25", _result([1, 0, 0, 1], [1.0, 0.5, 0.5, 1.0]))
        self.readme = self.dir / "README.md"
        self.readme.write_text(README_TEXT)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, data):
        (self.results / f"{name}.json").write_text(json.dumps(data))

    def _run(self, *extra):
        return main(["--results", str(self.results), "--readme", str(self.readme), *extra])

    def test_tables_and_order(self):
        text = render(load_results(self.results))
        self.assertTrue(text.startswith(START) and text.endswith(END))
        self.assertLess(text.index("| BM25 |"), text.index("| Hybrid + rerank |"))  # fixed config order
        self.assertIn("| BM25 | 0.500 | 0.500 [0.250, 1.000] | -0.250 [", text)
        self.assertIn("| Hybrid + rerank | 0.500 | 0.750 [0.250, 1.000] | baseline |", text)
        self.assertIn("1,234", text)
        no_rerank = _result([1, 1, 1, 1], [1.0] * 4)
        no_rerank["meta"]["config"]["rerank"] = False
        self.assertIn("| 1,234 | – | 1,234 |", render({"dense": no_rerank}))
        self.assertIn("$0.29", text)
        self.assertIn("1.000 / 0.750", text)  # keyword hit@5 / MRR@10
        self.assertIn("`gpt-4.1-mini` version `2025-04-14`", text)

    def test_splice_keeps_surrounding_text_and_is_stable(self):
        self.assertEqual(self._run(), 0)
        first = self.readme.read_text()
        self.assertTrue(first.startswith("# Title\n\nIntro stays.\n\n"))
        self.assertTrue(first.endswith("\n\nOutro stays.\n"))
        self.assertNotIn("old table", first)
        self.assertEqual(self._run("--check"), 0)
        self._run()
        self.assertEqual(self.readme.read_text(), first, "regenerating must be byte-identical")

    def test_check_fails_when_out_of_date(self):
        self.assertEqual(self._run("--check"), 1)
        self.assertEqual(self.readme.read_text(), README_TEXT, "--check never writes")

    def test_partial_results_are_ignored(self):
        self._write("dense.partial", _result([0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]))
        self.assertEqual(set(load_results(self.results)), {"hybrid_rrf_rerank", "bm25"})

    def test_incomparable_results_are_refused(self):
        self._write("dense", _result([1, 1, 1, 1], [1.0] * 4, golden="other"))
        with self.assertRaises(ReportError):
            load_results(self.results)

    def test_missing_markers_is_an_error(self):
        with self.assertRaises(ReportError):
            splice("no markers here", "x")


if __name__ == "__main__":
    unittest.main()
