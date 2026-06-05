import io
import unittest

from PIL import Image

from media_ingest import build_image_searchable_text, normalize_image, parse_caption_map


def _encode(mode: str, color, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (40, 30), color).save(buf, format=fmt)
    return buf.getvalue()


class MediaImageTest(unittest.TestCase):
    def test_parse_caption_map(self):
        raw = "a.png|Revenue chart\nb.jpg|Headcount trend"
        mapping = parse_caption_map(raw)
        self.assertEqual(mapping["a.png"], "Revenue chart")
        self.assertEqual(mapping["b.jpg"], "Headcount trend")

    def test_normalize_opaque_image_becomes_jpeg(self):
        data, mime = normalize_image(_encode("RGB", (120, 30, 30), "PNG"), "image/png")
        self.assertEqual(mime, "image/jpeg")
        self.assertEqual(Image.open(io.BytesIO(data)).format, "JPEG")

    def test_normalize_transparent_image_stays_png_with_alpha(self):
        data, mime = normalize_image(_encode("RGBA", (0, 0, 0, 0), "PNG"), "image/png")
        self.assertEqual(mime, "image/png")
        self.assertEqual(Image.open(io.BytesIO(data)).mode, "RGBA")

    def test_normalize_bad_bytes_falls_back(self):
        self.assertEqual(normalize_image(b"not-an-image", "image/png"), (b"not-an-image", "image/png"))

    def test_build_image_searchable_text(self):
        text = build_image_searchable_text("chart.png", "Q4 revenue by region")
        self.assertIn("chart.png", text)
        self.assertIn("Q4 revenue", text)


if __name__ == "__main__":
    unittest.main()
