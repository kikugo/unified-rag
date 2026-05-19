import unittest

from media_ingest import attach_transcript_metadata, build_audio_searchable_text, build_video_frame_searchable_text


class MediaTranscriptTest(unittest.TestCase):
    def test_attach_transcript_updates_searchable_text(self):
        meta = attach_transcript_metadata(
            {"searchable_text": "clip.wav audio 0:00 1:20"},
            transcript="revenue grew twenty percent",
        )
        self.assertEqual(meta["transcript"], "revenue grew twenty percent")
        self.assertIn("revenue grew", meta["searchable_text"])

    def test_build_audio_searchable_text_includes_transcript(self):
        text = build_audio_searchable_text("clip.wav", 0.0, 80.0, "hello world")
        self.assertIn("clip.wav", text)
        self.assertIn("hello world", text)

    def test_build_video_frame_searchable_text_includes_transcript(self):
        text = build_video_frame_searchable_text("demo.mp4", "1:05", "speaker mentions launch")
        self.assertIn("demo.mp4", text)
        self.assertIn("speaker mentions launch", text)


if __name__ == "__main__":
    unittest.main()
