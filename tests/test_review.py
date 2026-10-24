import json
import tempfile
import unittest
from pathlib import Path

from eval.golden import load_golden
from eval.review import Reviewer, highlight, next_golden_id

CHUNKS = {
    "a1": {"chunk_id": "a1", "title": "Act", "heading": "page 1", "text": "Article 5 bans social scoring."},
    "b1": {"chunk_id": "b1", "title": "Wiki", "heading": "Penalties", "text": "Fines reach 7% of turnover."},
}


def _cand(cid, kind="factual", gold=("a1",), answer="Social scoring", sources=("a1",)):
    return {"candidate_id": cid, "type": kind, "question": f"Question {cid}?", "answer": answer,
            "gold_chunk_ids": list(gold), "source_chunk_ids": list(sources),
            "evidence": [{"source": 1, "quote": "bans social scoring"}], "flags": [], "drafter_model": "m"}


class ScriptedInput:
    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, prompt):
        return self.answers.pop(0)


class ReviewerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.golden = root / "golden.jsonl"
        self.log = root / "log.jsonl"
        self.out: list[str] = []
        self.previews: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def _reviewer(self, cands, answers, targets=None):
        def preview(q):
            self.previews.append(q)
            return [{"chunk_id": "a1", "doc_id": "act", "page": 1, "text": "Article 5 bans social scoring."}]
        return Reviewer(cands, CHUNKS, self.golden, self.log, ScriptedInput(answers), self.out.append,
                        preview, targets or {"factual": 5, "keyword": 5, "paraphrase": 5, "multihop": 5,
                                             "unanswerable": 5})

    def test_keep_edit_drop_then_resume(self):
        cands = [_cand("c001"), _cand("c002"), _cand("c003"), _cand("c004")]
        # keep c001; edit c002's question; drop c003; quit at c004
        kept = self._reviewer(cands, ["k", "e", "Edited question?", "", "", "d", "q"]).run()
        self.assertEqual(kept, 2)
        rows = load_golden(self.golden)
        self.assertEqual([r["id"] for r in rows], ["g001", "g002"])
        self.assertEqual(rows[1]["question"], "Edited question?")
        self.assertEqual(rows[1]["source_candidate"], "c002")

        before = self.golden.read_text(encoding="utf-8")
        self._reviewer(cands, ["k"]).run()  # resumes at c004
        after = self.golden.read_text(encoding="utf-8")
        self.assertTrue(after.startswith(before), "existing rows must never be rewritten")
        self.assertEqual([r["source_candidate"] for r in load_golden(self.golden)], ["c001", "c002", "c004"])
        decisions = [json.loads(line)["decision"] for line in self.log.read_text().splitlines()]
        self.assertEqual(decisions, ["keep", "edit", "drop", "keep"])

    def test_skip_is_not_recorded(self):
        self._reviewer([_cand("c001")], ["s"]).run()
        self.assertFalse(self.log.exists())
        self.assertEqual(load_golden(self.golden), [])

    def test_preview_only_for_unanswerable(self):
        cands = [_cand("c001"), _cand("c002", kind="unanswerable", gold=(), answer="")]
        self._reviewer(cands, ["k", "k"]).run()
        self.assertEqual(self.previews, ["Question c002?"])
        rows = load_golden(self.golden)
        self.assertEqual((rows[1]["answerable"], rows[1]["gold_chunk_ids"]), (False, []))

    def test_invalid_keep_is_refused_until_dropped(self):
        # A factual candidate whose quote was not found has no gold ids.
        self._reviewer([_cand("c001", gold=())], ["k", "d"]).run()
        self.assertEqual(load_golden(self.golden), [])
        self.assertTrue(any("can't keep" in line for line in self.out))

    def test_types_at_target_are_not_offered(self):
        cands = [_cand("c001"), _cand("c002"), _cand("c003", kind="keyword")]
        targets = {"factual": 1, "keyword": 1, "paraphrase": 0, "multihop": 0, "unanswerable": 0}
        self._reviewer(cands, ["k", "k"], targets).run()
        self.assertEqual([r["source_candidate"] for r in load_golden(self.golden)], ["c001", "c003"])

    def test_edit_to_unanswerable_clears_gold(self):
        self._reviewer([_cand("c001")], ["e", "", "", "unanswerable"]).run()
        row = load_golden(self.golden)[0]
        self.assertEqual((row["type"], row["gold_chunk_ids"], row["gold_answer"]), ("unanswerable", [], ""))


class HelpersTest(unittest.TestCase):
    def test_next_id_and_highlight(self):
        self.assertEqual(next_golden_id([]), "g001")
        self.assertEqual(next_golden_id([{"id": "g009"}, {"id": "g002"}]), "g010")
        self.assertIn("\033[33m", highlight("Article 5 bans social scoring.", ["bans social"]))


if __name__ == "__main__":
    unittest.main()
