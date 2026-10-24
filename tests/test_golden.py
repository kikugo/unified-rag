"""Done-when checks for the golden set, plus the schema the loader enforces."""
import json
import tempfile
import unittest
from pathlib import Path

from eval.golden import GOLDEN_PATH, TARGET_MIX, GoldenSetError, load_golden, mix, validate_row
from rag.index import INDEX_DIR, load_chunks, open_store


def _row(**overrides):
    row = {"id": "g001", "question": "What does Article 5 ban?", "gold_chunk_ids": ["abc"],
           "gold_answer": "Social scoring.", "type": "factual", "answerable": True}
    row.update(overrides)
    return row


class GoldenSchemaTest(unittest.TestCase):
    def test_valid_rows_pass(self):
        validate_row(_row())
        validate_row(_row(type="unanswerable", gold_chunk_ids=[], gold_answer="", answerable=False))
        validate_row(_row(type="multihop", gold_chunk_ids=["a", "b"]))

    def test_bad_rows_fail(self):
        bad = [
            _row(type="trivia"),
            _row(question="  "),
            _row(gold_chunk_ids=[]),                                  # answerable without gold
            _row(type="unanswerable", gold_chunk_ids=[], answerable=True),
            _row(type="unanswerable", gold_chunk_ids=["a"], answerable=True),
            _row(type="multihop", gold_chunk_ids=["a"]),
            _row(gold_chunk_ids=["a", "a"]),
            _row(gold_answer=""),
        ]
        for row in bad:
            with self.assertRaises(GoldenSetError, msg=str(row)):
                validate_row(row)
        row = _row()
        del row["type"]
        with self.assertRaises(GoldenSetError):
            validate_row(row)

    def test_loader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.jsonl"
            path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")
            with self.assertRaises(GoldenSetError):
                load_golden(path)

    def test_missing_file_is_empty(self):
        self.assertEqual(load_golden(Path("/nonexistent/golden.jsonl")), [])


class GoldenSetTest(unittest.TestCase):
    """Runs against the committed eval/golden.jsonl."""

    @classmethod
    def setUpClass(cls):
        cls.rows = load_golden(GOLDEN_PATH)

    def test_every_gold_chunk_id_exists_in_the_corpus(self):
        # Built from the committed corpus text, so this runs in CI with no API key.
        known = {c["chunk_id"] for c in load_chunks()}
        missing = [(r["id"], cid) for r in self.rows for cid in r["gold_chunk_ids"] if cid not in known]
        self.assertEqual(missing, [])

    def test_every_gold_chunk_id_exists_in_the_built_index(self):
        if not INDEX_DIR.exists():
            self.skipTest("no .eval_index built")
        if not self.rows:
            self.skipTest("golden set is empty")
        store = open_store()
        missing = [(r["id"], cid) for r in self.rows for cid in r["gold_chunk_ids"] if not store.metadata(cid)]
        self.assertEqual(missing, [])

    def test_mix_meets_targets_once_complete(self):
        if len(self.rows) < sum(TARGET_MIX.values()):
            self.skipTest(f"golden set has {len(self.rows)} rows, review not finished")
        counts = mix(self.rows)
        for kind, target in TARGET_MIX.items():
            self.assertGreaterEqual(counts[kind], target, kind)


if __name__ == "__main__":
    unittest.main()
