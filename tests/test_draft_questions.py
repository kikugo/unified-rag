import json
import tempfile
import unittest
from pathlib import Path

from eval.draft_questions import (
    DRAFT_COUNTS,
    EXTRA_MULTIHOP,
    DuplicateIndex,
    finalize,
    gold_ids_for_quote,
    identifiers,
    overlap_score,
    parse_draft,
    read_candidates,
    sample_candidates,
    write_candidates,
)
from rag.chunking import chunk_page
from rag.index import load_chunks


def _page():
    """One page long enough for several overlapping windows."""
    filler = " ".join(f"filler{i}" for i in range(300))
    text = f"Article 5 bans social scoring by public authorities. {filler} The fine is 7% of turnover. {filler}"
    chunks = chunk_page("doc", 1, text)
    for c in chunks:
        c.update(title="Doc", heading="page 1", searchable_text=c["text"])
    return text, chunks


class SamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks()

    def test_sample_is_deterministic_and_hits_counts(self):
        a = sample_candidates(self.chunks, seed=7)
        b = sample_candidates(self.chunks, seed=7)
        self.assertEqual([(s["type"], [c["chunk_id"] for c in s["sources"]]) for s in a],
                         [(s["type"], [c["chunk_id"] for c in s["sources"]]) for s in b])
        counts = {}
        for s in a:
            counts[s["type"]] = counts.get(s["type"], 0) + 1
        self.assertEqual(counts, {**DRAFT_COUNTS, "multihop": DRAFT_COUNTS["multihop"] + EXTRA_MULTIHOP})

    def test_extra_multihop_batch_leaves_earlier_candidates_unchanged(self):
        def key(specs):
            return [(s["candidate_id"], s["type"], [c["chunk_id"] for c in s["sources"]]) for s in specs]
        without = sample_candidates(self.chunks, seed=7, extra_multihop=0)
        with_extra = sample_candidates(self.chunks, seed=7)
        self.assertEqual(key(with_extra)[:len(without)], key(without))
        self.assertTrue(all(s["type"] == "multihop" for s in with_extra[len(without):]))

    def test_no_chunk_is_sampled_twice_and_multihop_spans_documents(self):
        specs = sample_candidates(self.chunks, seed=7)
        ids = [c["chunk_id"] for s in specs for c in s["sources"]]
        self.assertEqual(len(ids), len(set(ids)))
        for s in specs:
            if s["type"] == "multihop":
                self.assertEqual(len(s["sources"]), 2)
                self.assertNotEqual(s["sources"][0]["doc_id"], s["sources"][1]["doc_id"])

    def test_keyword_sources_carry_an_identifier(self):
        for s in sample_candidates(self.chunks, seed=7):
            if s["type"] == "keyword":
                self.assertTrue(identifiers(s["sources"][0]["text"]))


class EvidenceTest(unittest.TestCase):
    def test_quote_in_one_window(self):
        text, chunks = _page()
        ids = gold_ids_for_quote("Article 5 bans social scoring by public authorities.", chunks[0], text, chunks)
        self.assertEqual(ids, [chunks[0]["chunk_id"]])

    def test_quote_in_the_overlap_gets_both_windows(self):
        text, chunks = _page()
        first, second = chunks[0], chunks[1]
        overlap_start = second["char_offset"]
        overlap_end = first["char_offset"] + len(first["text"])
        quote = text[overlap_start:overlap_end].split(" ", 1)[1].rsplit(" ", 1)[0]
        ids = gold_ids_for_quote(quote, first, text, chunks)
        self.assertEqual(set(ids), {first["chunk_id"], second["chunk_id"]})

    def test_quote_straddling_a_boundary_gets_every_touched_window(self):
        text, chunks = _page()
        # Longer than the overlap: starts before window 2 and ends after window 1,
        # so no single window holds all of it.
        end_of_first = chunks[0]["char_offset"] + len(chunks[0]["text"])
        quote = text[chunks[1]["char_offset"] - 30:end_of_first + 30].strip()
        ids = gold_ids_for_quote(quote, chunks[0], text, chunks)
        self.assertIn(chunks[0]["chunk_id"], ids)
        self.assertIn(chunks[1]["chunk_id"], ids)

    def test_whitespace_differences_are_tolerated_and_missing_quote_is_empty(self):
        text, chunks = _page()
        self.assertTrue(gold_ids_for_quote("Article 5  bans\nsocial scoring", chunks[0], text, chunks))
        self.assertEqual(gold_ids_for_quote("Article 6 is invented", chunks[0], text, chunks), [])

    def test_repeated_sentence_resolves_to_the_source_window(self):
        filler = " ".join(f"filler{i}" for i in range(300))
        text = f"The measures in Article 32 apply. {filler} The measures in Article 32 apply. {filler}"
        chunks = chunk_page("doc", 1, text)
        later = next(c for c in chunks if c["char_offset"] > 0 and "Article 32" in c["text"])
        ids = gold_ids_for_quote("The measures in Article 32 apply.", later, text, chunks)
        self.assertIn(later["chunk_id"], ids)
        self.assertNotIn(chunks[0]["chunk_id"], ids)

    def test_figure_quote_maps_to_the_figure(self):
        fig = {"kind": "figure", "chunk_id": "fig1"}
        self.assertEqual(gold_ids_for_quote("any caption", fig, "", []), ["fig1"])


class FinalizeTest(unittest.TestCase):
    def setUp(self):
        self.text, self.chunks = _page()
        self.pages = {("doc", 1): self.text}
        self.by_page = {("doc", 1): self.chunks}

    def _spec(self, kind, sources=None):
        return {"candidate_id": "c001", "type": kind, "sources": sources or [self.chunks[0]]}

    def test_good_factual_has_gold_and_no_flags(self):
        draft = {"question": "What does Article 5 ban?", "answer": "Social scoring",
                 "evidence": [{"source": 1, "quote": "Article 5 bans social scoring by public authorities."}]}
        row = finalize(self._spec("factual"), draft, self.pages, self.by_page)
        self.assertEqual(row["gold_chunk_ids"], [self.chunks[0]["chunk_id"]])
        self.assertEqual(row["flags"], [])

    def test_flags(self):
        bad_quote = {"question": "q?", "answer": "a", "evidence": [{"source": 1, "quote": "not there"}]}
        self.assertIn("quote_not_found(source 1)",
                      finalize(self._spec("factual"), bad_quote, self.pages, self.by_page)["flags"])
        para = {"question": "Which practices does Article 5 ban for public authorities?", "answer": "a",
                "evidence": [{"source": 1, "quote": "Article 5 bans social scoring by public authorities."}]}
        flags = finalize(self._spec("paraphrase"), para, self.pages, self.by_page)["flags"]
        self.assertTrue(any(f.startswith("paraphrase_overlap") for f in flags))
        kw = {"question": "What is banned?", "answer": "a",
              "evidence": [{"source": 1, "quote": "Article 5 bans social scoring by public authorities."}]}
        self.assertIn("keyword_identifier_missing",
                      finalize(self._spec("keyword"), kw, self.pages, self.by_page)["flags"])

    def test_multihop_needs_both_sources(self):
        draft = {"question": "q?", "answer": "a",
                 "evidence": [{"source": 1, "quote": "Article 5 bans social scoring by public authorities."}]}
        row = finalize(self._spec("multihop", [self.chunks[0], self.chunks[-1]]), draft, self.pages, self.by_page)
        self.assertIn("multihop_missing_a_source", row["flags"])

    def test_unanswerable_has_no_gold_or_answer(self):
        draft = {"question": "What did the authors eat?", "answer": "pasta", "evidence": []}
        row = finalize(self._spec("unanswerable"), draft, self.pages, self.by_page)
        self.assertEqual((row["gold_chunk_ids"], row["answer"]), ([], ""))
        self.assertIn("unanswerable_has_answer", row["flags"])


class DuplicateTest(unittest.TestCase):
    QUOTE = "The provider shall ensure that all appropriate corrective action is taken in respect of the AI systems."

    def _chunk(self, cid, text):
        return {"chunk_id": cid, "doc_id": "d", "page": 1, "char_offset": 0, "kind": "text", "text": text}

    def test_exact_and_near_copies_are_separated(self):
        index = DuplicateIndex([
            self._chunk("gold", "Intro. " + self.QUOTE),
            self._chunk("copy", "Recital text. " + self.QUOTE + " More."),
            self._chunk("near", self.QUOTE.replace("provider", "operator")),
            self._chunk("other", "Unrelated passage about transistor counts and Moore's law."),
        ])
        exact, near = index.find(self.QUOTE, exclude={"gold"})
        self.assertEqual((exact, near), (["copy"], ["near"]))

    def test_finalize_adds_exact_copies_to_gold_and_offers_near_ones(self):
        gold_chunk = self._chunk("gold", "Intro. " + self.QUOTE)
        copy_chunk = {**self._chunk("copy", "Recital. " + self.QUOTE), "page": 2}
        near_chunk = {**self._chunk("near", self.QUOTE.replace("provider", "operator")), "page": 3}
        index = DuplicateIndex([gold_chunk, copy_chunk, near_chunk])
        spec = {"candidate_id": "c001", "type": "factual", "sources": [gold_chunk]}
        draft = {"question": "Who must take corrective action?", "answer": "The provider",
                 "evidence": [{"source": 1, "quote": self.QUOTE}]}
        row = finalize(spec, draft, {("d", 1): gold_chunk["text"]}, {("d", 1): [gold_chunk]}, index)
        self.assertEqual(row["gold_chunk_ids"], ["gold", "copy"])
        self.assertEqual(row["alt_gold_candidates"], ["near"])
        self.assertIn("gold_has_exact_copy", row["flags"])


class HelpersTest(unittest.TestCase):
    def test_overlap_score(self):
        self.assertEqual(overlap_score("social scoring ban", "Article 5 bans social scoring"), 2 / 3)
        self.assertEqual(overlap_score("", "anything"), 0.0)

    def test_parse_draft_rejects_bad_json(self):
        for raw in ("not json", "[]", '{"question": ""}', '{"question": "q", "evidence": "x"}',
                    '{"question": "q", "evidence": [{"source": 1}]}'):
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                parse_draft(raw)
        ok = parse_draft('{"question": " q? ", "answer": "a", "evidence": [{"source": 2, "quote": " x "}]}')
        self.assertEqual(ok, {"question": "q?", "answer": "a", "evidence": [{"source": 2, "quote": "x"}]})

    def test_csv_round_trip(self):
        rows = [{"candidate_id": "c001", "type": "factual", "question": 'He said "hi", then left',
                 "answer": "a, b", "gold_chunk_ids": ["x", "y"], "alt_gold_candidates": ["z"],
                 "source_chunk_ids": ["x"],
                 "evidence": [{"source": 1, "quote": "line\nbreak"}], "flags": [], "drafter_model": "m"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            write_candidates(rows, path)
            self.assertEqual(read_candidates(path), rows)


if __name__ == "__main__":
    unittest.main()
