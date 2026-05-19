import unittest

try:
    import fitz
except ImportError:  # pragma: no cover - optional in minimal dev envs
    fitz = None

from media_ingest import chunk_pdf, extract_pdf_text


@unittest.skipUnless(fitz is not None, "PyMuPDF is required for PDF ingest tests")
class MediaPdfTest(unittest.TestCase):
    def _make_pdf(self, pages: int) -> bytes:
        doc = fitz.open()
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {i + 1} keyword{i + 1}")
        data = doc.tobytes()
        doc.close()
        return data

    def test_extract_pdf_text(self):
        pdf = self._make_pdf(2)
        text = extract_pdf_text(pdf)
        self.assertIn("keyword1", text)
        self.assertIn("keyword2", text)

    def test_chunk_pdf_overlap_covers_all_pages(self):
        pdf = self._make_pdf(7)
        chunks = chunk_pdf(pdf, chunk_size=4, overlap_pages=1)
        self.assertGreaterEqual(len(chunks), 2)
        labels = [label for _, label, _ in chunks]
        self.assertTrue(any("1" in label for label in labels))
        self.assertTrue("pages" in labels[-1])


if __name__ == "__main__":
    unittest.main()
