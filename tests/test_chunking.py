import unittest

from rag.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_page, make_chunk_id


def _text(words: int) -> str:
    return " ".join(f"word{i}" for i in range(words))


class ChunkIdTest(unittest.TestCase):
    def test_same_position_gives_same_id(self):
        self.assertEqual(make_chunk_id("doc", 3, 120), make_chunk_id("doc", 3, 120))

    def test_id_changes_with_each_field(self):
        base = make_chunk_id("doc", 3, 120)
        self.assertNotEqual(base, make_chunk_id("doc2", 3, 120))
        self.assertNotEqual(base, make_chunk_id("doc", 4, 120))
        self.assertNotEqual(base, make_chunk_id("doc", 3, 121))
        self.assertNotEqual(base, make_chunk_id("doc", 3, 120, kind="figure"))

    def test_id_is_short_hex(self):
        chunk_id = make_chunk_id("doc", 0, 0)
        self.assertEqual(len(chunk_id), 16)
        int(chunk_id, 16)


class ChunkPageTest(unittest.TestCase):
    def test_rechunking_is_deterministic(self):
        text = _text(800)
        first = [c["chunk_id"] for c in chunk_page("d", 1, text)]
        second = [c["chunk_id"] for c in chunk_page("d", 1, text)]
        self.assertEqual(first, second)
        self.assertGreater(len(first), 3)

    def test_offset_reproduces_chunk_text(self):
        text = "  leading space. " + _text(600)
        for chunk in chunk_page("d", 1, text):
            start = chunk["char_offset"]
            self.assertEqual(text[start:start + len(chunk["text"])], chunk["text"])
            self.assertEqual(chunk["chunk_id"], make_chunk_id("d", 1, start))

    def test_windows_respect_size_and_overlap_on_word_boundaries(self):
        text = _text(800)
        chunks = chunk_page("d", 1, text)
        for prev, cur in zip(chunks, chunks[1:]):
            self.assertLessEqual(len(prev["text"]), CHUNK_SIZE)
            prev_end = prev["char_offset"] + len(prev["text"])
            self.assertLess(cur["char_offset"], prev_end, "consecutive windows must overlap")
            self.assertLessEqual(prev_end - cur["char_offset"], CHUNK_OVERLAP)
            self.assertTrue(text[cur["char_offset"] - 1].isspace())
        self.assertEqual(chunks[-1]["char_offset"] + len(chunks[-1]["text"]), len(text))

    def test_every_word_is_covered(self):
        text = _text(500)
        covered = set()
        for chunk in chunk_page("d", 1, text):
            covered.update(chunk["text"].split())
        self.assertEqual(covered, set(text.split()))

    def test_short_page_is_one_chunk_and_empty_page_is_none(self):
        self.assertEqual(len(chunk_page("d", 1, "Short page.")), 1)
        self.assertEqual(chunk_page("d", 1, "   "), [])

    def test_overlap_must_be_smaller_than_size(self):
        with self.assertRaises(ValueError):
            chunk_page("d", 1, "text", size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
