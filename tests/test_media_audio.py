import io
import struct
import unittest
import wave

from media_ingest import chunk_audio_bytes, format_timestamp


class MediaAudioTest(unittest.TestCase):
    def _make_wav(self, duration_sec: float, framerate: int = 8000) -> bytes:
        nframes = int(duration_sec * framerate)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(struct.pack("<" + "h" * nframes, *([0] * nframes)))
        return buf.getvalue()

    def test_short_wav_returns_single_window(self):
        audio = self._make_wav(10.0)
        chunks = chunk_audio_bytes(audio, "audio/wav", chunk_seconds=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1], 0.0)
        self.assertAlmostEqual(chunks[0][2], 10.0, places=1)

    def test_long_wav_splits_with_timestamps(self):
        audio = self._make_wav(170.0)
        chunks = chunk_audio_bytes(audio, "audio/wav", chunk_seconds=80)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0][1], 0.0)
        self.assertGreater(chunks[1][1], 0.0)
        self.assertLessEqual(chunks[0][2], 80.0)

    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(75), "1:15")


if __name__ == "__main__":
    unittest.main()
