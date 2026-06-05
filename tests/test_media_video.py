import unittest

from media_ingest import choose_frame_interval, format_timestamp, transcript_for_timestamp


class MediaVideoTest(unittest.TestCase):
    def test_transcript_for_timestamp_picks_covering_window(self):
        windows = [(0.0, 2.0, "hello there"), (2.0, 4.0, "second part")]
        self.assertEqual(transcript_for_timestamp(0.5, windows), "hello there")
        self.assertEqual(transcript_for_timestamp(2.0, windows), "second part")

    def test_transcript_for_timestamp_past_end_uses_last_window(self):
        windows = [(0.0, 2.0, "only window")]
        self.assertEqual(transcript_for_timestamp(5.0, windows), "only window")

    def test_transcript_for_timestamp_empty_windows(self):
        self.assertEqual(transcript_for_timestamp(1.0, []), "")

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
