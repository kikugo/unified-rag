import io
import struct
import unittest
import wave

try:
    import av
except ImportError:  # pragma: no cover - optional in minimal dev envs
    av = None

from media_ingest import chunk_audio_bytes, format_timestamp, proportional_audio_windows


def _make_mp3(seconds: float, rate: int = 44100, freq: int = 440) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp3")
    stream = container.add_stream("libmp3lame", rate=rate)
    stream.layout = "mono"
    samples_per_frame = 1152
    total = int(seconds * rate)
    pts = 0
    pos = 0
    while pos < total:
        n = min(samples_per_frame, total - pos)
        arr = (0.3 * np.sin(2 * np.pi * freq * (np.arange(pos, pos + n) / rate)) * 32767).astype("int16").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
        frame.rate = rate
        frame.pts = pts
        pts += n
        for packet in stream.encode(frame):
            container.mux(packet)
        pos += n
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue()


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
        self.assertEqual(chunks[0][3], "audio/wav")

    def test_long_wav_splits_with_timestamps(self):
        audio = self._make_wav(170.0)
        chunks = chunk_audio_bytes(audio, "audio/wav", chunk_seconds=80)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0][1], 0.0)
        self.assertGreater(chunks[1][1], 0.0)
        self.assertLessEqual(chunks[0][2], 80.0)

    def test_wav_overlap_produces_overlapping_windows(self):
        audio = self._make_wav(10.0)
        chunks = chunk_audio_bytes(audio, "audio/wav", chunk_seconds=4, overlap_sec=1)
        self.assertGreaterEqual(len(chunks), 3)
        # each window starts before the previous one ends — i.e. they overlap
        for (_, _, end_prev, _), (_, start_next, _, _) in zip(chunks, chunks[1:]):
            self.assertLess(start_next, end_prev)

    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(75), "1:15")

    def test_proportional_windows_cover_duration_with_monotonic_bytes(self):
        windows = proportional_audio_windows(1000, 170.0, 80)
        self.assertEqual([(round(s, 1), round(e, 1)) for _, _, s, e in windows],
                         [(0.0, 80.0), (80.0, 160.0), (160.0, 170.0)])
        # byte offsets are non-decreasing and stay within the blob
        starts = [w[0] for w in windows]
        ends = [w[1] for w in windows]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(ends[-1], 1000)

    def test_proportional_windows_zero_duration_returns_whole_blob(self):
        self.assertEqual(proportional_audio_windows(500, 0.0, 80), [(0, 500, 0.0, 0.0)])


@unittest.skipUnless(av is not None, "PyAV is required for MP3 decode tests")
class MediaMp3Test(unittest.TestCase):
    def test_mp3_decodes_to_valid_wav_chunks_with_timestamps(self):
        mp3 = _make_mp3(5.0)
        chunks = chunk_audio_bytes(mp3, "audio/mp3", chunk_seconds=2)
        self.assertGreaterEqual(len(chunks), 3)
        # Every chunk is a valid, independently readable WAV (the whole point).
        for chunk_bytes, start_sec, end_sec, chunk_mime in chunks:
            self.assertEqual(chunk_mime, "audio/wav")
            with wave.open(io.BytesIO(chunk_bytes)) as wf:
                self.assertGreater(wf.getnframes(), 0)
            self.assertLessEqual(start_sec, end_sec)
        self.assertEqual(chunks[0][1], 0.0)
        self.assertAlmostEqual(chunks[-1][2], 5.0, delta=0.5)

    def test_short_mp3_is_a_single_wav_window(self):
        mp3 = _make_mp3(1.5)
        chunks = chunk_audio_bytes(mp3, "audio/mp3", chunk_seconds=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][3], "audio/wav")
        self.assertAlmostEqual(chunks[0][2], 1.5, delta=0.3)


if __name__ == "__main__":
    unittest.main()
