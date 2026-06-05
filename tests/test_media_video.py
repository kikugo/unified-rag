import io
import unittest

from PIL import Image

from media_ingest import (
    choose_frame_interval,
    format_timestamp,
    select_scene_frames,
    transcript_for_timestamp,
)


def _solid_png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


class MediaVideoTest(unittest.TestCase):
    def test_select_scene_frames_drops_near_duplicates(self):
        black, white = _solid_png((0, 0, 0)), _solid_png((255, 255, 255))
        frames = [(black, 0.0), (black, 5.0), (white, 10.0), (white, 15.0), (black, 20.0)]
        kept = select_scene_frames(frames)
        # one frame per scene: black @0, white @10, black @20
        self.assertEqual([ts for _, ts in kept], [0.0, 10.0, 20.0])

    def test_select_scene_frames_keeps_first_and_all_distinct(self):
        frames = [(_solid_png((10, 10, 10)), 0.0), (_solid_png((200, 50, 50)), 3.0)]
        self.assertEqual(len(select_scene_frames(frames)), 2)
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
