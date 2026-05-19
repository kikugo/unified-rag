import unittest

from media_ingest import build_image_searchable_text, parse_caption_map


class MediaImageTest(unittest.TestCase):
    def test_parse_caption_map(self):
        raw = "a.png|Revenue chart\nb.jpg|Headcount trend"
        mapping = parse_caption_map(raw)
        self.assertEqual(mapping["a.png"], "Revenue chart")
        self.assertEqual(mapping["b.jpg"], "Headcount trend")

    def test_build_image_searchable_text(self):
        text = build_image_searchable_text("chart.png", "Q4 revenue by region")
        self.assertIn("chart.png", text)
        self.assertIn("Q4 revenue", text)


if __name__ == "__main__":
    unittest.main()
