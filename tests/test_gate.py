"""The CI gate's thresholds, refusals and summary."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from eval.gate import (
    FAITH_DROP,
    FALSE_REFUSAL_RISE,
    HIT5_DROP,
    SMOKE_PATH,
    SMOKE_QUOTA,
    GateError,
    changed_rows,
    compare,
    load_smoke_ids,
    main,
    make_subset,
    render_summary,
)
from eval.golden import load_golden


def _run(hit5=0.8, faith=0.9, false_refusal=0.1, context_k=3, ids=("g001", "g002")):
    rows = [{"id": i, "type": "factual", "answerable": True, "hit@5": 1.0, "refused": False, "faithfulness": 1.0}
            for i in ids]
    return {
        "meta": {"golden_sha256_16": "abc", "judge": {"model": "gpt-4.1-mini", "version": "2025-04-14"},
                 "config": {"name": "hybrid_rrf_rerank", "mode": "hybrid_rrf", "rerank": True, "hyde": False,
                            "context_k": context_k}},
        "totals": {"hit@5": hit5, "faithfulness": faith, "false_refusal_rate": false_refusal},
        "rows": rows,
    }


def _verdicts(checks):
    return {c["metric"]: c["passed"] for c in checks}


class ThresholdTest(unittest.TestCase):
    def test_unchanged_passes(self):
        self.assertTrue(all(_verdicts(compare(_run(), _run())).values()))

    def test_each_limit_is_inclusive_and_one_step_past_fails(self):
        base = _run()
        at_limit = _run(hit5=0.8 - HIT5_DROP, faith=0.9 - FAITH_DROP, false_refusal=0.1 + FALSE_REFUSAL_RISE)
        self.assertTrue(all(_verdicts(compare(base, at_limit)).values()))
        self.assertFalse(_verdicts(compare(base, _run(hit5=0.8 - HIT5_DROP - 0.01)))["hit@5"])
        self.assertFalse(_verdicts(compare(base, _run(faith=0.9 - FAITH_DROP - 0.01)))["faithfulness"])
        self.assertFalse(_verdicts(compare(base, _run(false_refusal=0.1 + FALSE_REFUSAL_RISE + 0.01)))
                         ["false_refusal_rate"])

    def test_improvements_never_fail(self):
        better = _run(hit5=1.0, faith=1.0, false_refusal=0.0)
        self.assertTrue(all(_verdicts(compare(_run(), better)).values()))

    def test_incomparable_runs_are_refused(self):
        with self.assertRaises(GateError):
            compare(_run(), _run(ids=("g001", "g003")))
        other_judge = _run()
        other_judge["meta"]["judge"]["version"] = "2026-01-01"
        with self.assertRaises(GateError):
            compare(_run(), other_judge)
        no_judge = _run()
        no_judge["totals"]["faithfulness"] = None
        with self.assertRaises(GateError):
            compare(_run(), no_judge)


class SummaryTest(unittest.TestCase):
    def test_summary_lists_checks_and_changed_questions(self):
        base, cur = _run(), _run(false_refusal=0.5, context_k=1)
        cur["rows"][1] = {**cur["rows"][1], "refused": True, "faithfulness": None}
        text = render_summary(compare(base, cur), changed_rows(base, cur), cur)
        self.assertIn("## Eval gate: FAILED", text)
        self.assertIn("context_k 1", text)
        self.assertIn("| false_refusal_rate | 0.100 | 0.500 | +0.400 | rise ≤ 0.10 | **fail** |", text)
        self.assertIn("| g002 | factual |  | no → yes | 1.000 → – |", text)

    def test_main_exit_code_and_summary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, cur, summary = Path(tmp, "b.json"), Path(tmp, "c.json"), Path(tmp, "summary.md")
            base.write_text(json.dumps(_run()))
            cur.write_text(json.dumps(_run(hit5=0.5)))
            self.assertEqual(main(["--baseline", str(base), "--current", str(cur), "--summary", str(summary)]), 1)
            self.assertIn("**fail**", summary.read_text())
            self.assertEqual(main(["--baseline", str(base), "--current", str(base)]), 0)


class SubsetTest(unittest.TestCase):
    def test_subset_is_deterministic_and_keeps_quotas(self):
        rows = load_golden()
        ids = make_subset(rows)
        self.assertEqual(ids, make_subset(copy.deepcopy(rows)))
        by_id = {r["id"]: r["type"] for r in rows}
        for kind, k in SMOKE_QUOTA.items():
            self.assertEqual(sum(by_id[i] == kind for i in ids), k)

    def test_committed_subset_matches_the_generator(self):
        self.assertTrue(SMOKE_PATH.exists())
        self.assertEqual(load_smoke_ids(), make_subset(load_golden()))

    def test_too_few_rows_is_an_error(self):
        with self.assertRaises(GateError):
            make_subset([{"id": "g001", "type": "factual"}])


class CommittedBaselineTest(unittest.TestCase):
    """The committed baseline is a smoke run of the gated config on the committed subset."""

    def test_baseline_covers_the_smoke_subset(self):
        baseline = json.loads(Path("results/baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(r["id"] for r in baseline["rows"]), sorted(load_smoke_ids()))
        self.assertEqual(baseline["meta"]["config"]["name"], "hybrid_rrf_rerank")
        self.assertTrue(all(_verdicts(compare(baseline, baseline)).values()))


if __name__ == "__main__":
    unittest.main()
