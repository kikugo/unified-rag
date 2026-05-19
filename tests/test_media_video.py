import unittest

from media_ingest import choose_frame_interval, format_timestamp


class MediaVideoTest(unittest.TestCase):
    def test_choose_frame_interval_short_video(self):
        self.assertEqual(choose_frame_interval(60), 5)

    def test_choose_frame_interval_medium_video(self):
        self.assertEqual(choose_frame_interval(300), 10)

    def test_choose_frame_interval_long_video(self):
        self.assertEqual(choose_frame_interval(1200), 20)

    def test_choose_frame_interval_unknown_duration(self):
        self.assertEqual(choose_frame_interval(None), 5)

    def test_format_timestamp_for_frame_labels(self):
        self.assertEqual(format_timestamp(125), "2:05")


if __name__ == "__main__":
    unittest.main()
