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


class _ReviewBase(unittest.TestCase):
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


class ReviewerTest(_ReviewBase):
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


class OfflineReviewTest(_ReviewBase):
    def test_export_packet_includes_sources_and_unanswerable_preview(self):
        cands = [_cand("c001"), _cand("c002", kind="unanswerable", gold=(), answer="")]
        path = Path(self.tmp.name) / "packet.jsonl"
        self.assertEqual(self._reviewer(cands, []).export_packet(path), 2)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[0]["sources"][0]["text"], CHUNKS["a1"]["text"])
        self.assertNotIn("retrieval_preview", rows[0])
        self.assertEqual(rows[1]["retrieval_preview"][0]["chunk_id"], "a1")

    def _suggest(self, cands, answers, suggestions):
        r = self._reviewer(cands, answers)
        r.suggestions = suggestions
        return r

    def test_accepting_suggestions_keep_edit_drop(self):
        cands = [_cand("c001"), _cand("c002"), _cand("c003")]
        sugs = {
            "c001": {"decision": "keep", "reviewer": "haiku", "note": "fine"},
            "c002": {"decision": "edit", "question": "Better question?", "reviewer": "haiku"},
            "c003": {"decision": "drop", "reviewer": "haiku", "note": "ambiguous"},
        }
        self._suggest(cands, ["a", "a", "a"], sugs).run()
        rows = load_golden(self.golden)
        self.assertEqual([r["question"] for r in rows], ["Question c001?", "Better question?"])
        log = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([e["decision"] for e in log], ["keep", "edit", "drop"])
        self.assertTrue(all(e["accepted"] and e["reviewer"] == "human" for e in log))
        self.assertTrue(any("Suggested: EDIT" in line for line in self.out))

    def test_overriding_a_suggestion_is_recorded(self):
        sugs = {"c001": {"decision": "drop", "reviewer": "haiku", "note": "meh"}}
        self._suggest([_cand("c001")], ["k"], sugs).run()
        log = json.loads(self.log.read_text().splitlines()[0])
        self.assertEqual((log["decision"], log["suggestion"], log["accepted"]), ("keep", "drop", False))
        self.assertEqual(len(load_golden(self.golden)), 1)

    def test_edit_prefills_with_the_suggestion(self):
        sugs = {"c001": {"decision": "edit", "question": "Suggested wording?", "reviewer": "haiku"}}
        # Enter on every prompt keeps the suggested values
        self._suggest([_cand("c001")], ["e", "", "", ""], sugs).run()
        self.assertEqual(load_golden(self.golden)[0]["question"], "Suggested wording?")

    def test_second_opinion_disagreement_is_shown_and_can_be_accepted(self):
        sugs = {
            "c001": {"decision": "keep", "reviewer": "haiku",
                     "second": {"decision": "edit", "question": "Sonnet wording?", "reviewer": "sonnet"}},
            "c002": {"decision": "keep", "reviewer": "haiku",
                     "second": {"decision": "drop", "reviewer": "sonnet", "note": "unsupported"}},
            "c003": {"decision": "keep", "reviewer": "haiku", "second": {"decision": "keep", "reviewer": "sonnet"}},
        }
        self._suggest([_cand("c001"), _cand("c002"), _cand("c003")], ["b", "b", "a"], sugs).run()
        self.assertEqual([r["question"] for r in load_golden(self.golden)], ["Sonnet wording?", "Question c003?"])
        log = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([(e["decision"], e["accepted"], e["accepted_second"]) for e in log],
                         [("edit", False, True), ("drop", False, True), ("keep", True, False)])
        disagreements = [line for line in self.out if "DISAGREES" in line]
        self.assertEqual(len(disagreements), 2, "only the two differing opinions are marked")

    def test_b_without_a_second_opinion_is_not_a_choice(self):
        sugs = {"c001": {"decision": "keep", "reviewer": "haiku"}}
        self._suggest([_cand("c001")], ["b", "a"], sugs).run()
        self.assertEqual(len(load_golden(self.golden)), 1)

    def test_g_adds_confirmed_alternates_as_extra_gold(self):
        cand = {**_cand("c001"), "alt_gold_candidates": ["b1", "zz"]}
        self._reviewer([cand], ["g", "1", "k"]).run()
        row = load_golden(self.golden)[0]
        self.assertEqual(row["gold_chunk_ids"], ["a1", "b1"])
        log = json.loads(self.log.read_text().splitlines()[0])
        self.assertEqual(log["added_gold"], ["b1"])
        self.assertTrue(any("Also possibly answered by" in line for line in self.out))

    def test_alternates_are_ignored_unless_added(self):
        cand = {**_cand("c001"), "alt_gold_candidates": ["b1"]}
        self._reviewer([cand], ["k"]).run()
        self.assertEqual(load_golden(self.golden)[0]["gold_chunk_ids"], ["a1"])

    def test_accept_without_a_suggestion_is_not_a_choice(self):
        self._reviewer([_cand("c001")], ["a", "k"]).run()
        self.assertEqual(len(load_golden(self.golden)), 1)
        self.assertTrue(any(line.startswith("type k, e") for line in self.out))


class HelpersTest(unittest.TestCase):
    def test_next_id_and_highlight(self):
        self.assertEqual(next_golden_id([]), "g001")
        self.assertEqual(next_golden_id([{"id": "g009"}, {"id": "g002"}]), "g010")
        self.assertIn("\033[33m", highlight("Article 5 bans social scoring.", ["bans social"]))


if __name__ == "__main__":
    unittest.main()
