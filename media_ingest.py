from __future__ import annotations

import io
import os
import tempfile
import wave

import numpy as np
from PIL import Image

AUDIO_MAX_SECONDS = 80
VIDEO_MAX_SECONDS = 80
FRAME_INTERVAL_SEC = 5
IMAGE_MAX_DIM = 4096
SCENE_CHANGE_THRESHOLD = 0.12


def parse_caption_map(raw: str) -> dict[str, str]:
    mapping = {}
    for line in (raw or "").splitlines():
        if "|" not in line:
            continue
        name, caption = line.split("|", 1)
        name = name.strip()
        caption = caption.strip()
        if name and caption:
            mapping[name] = caption
    return mapping


def build_image_searchable_text(filename: str, caption: str = "") -> str:
    return f"{filename} {caption or ''}".strip()


def attach_transcript_metadata(
    base_meta: dict,
    transcript: str = "",
) -> dict:
    meta = dict(base_meta)
    meta["transcript"] = transcript or ""
    if transcript:
        existing = meta.get("searchable_text", "")
        meta["searchable_text"] = f"{existing}\n{transcript}".strip()
    return meta


def extract_pdf_text(pdf_bytes: bytes) -> str:
    import fitz

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return ""


def chunk_pdf(
    pdf_bytes: bytes,
    chunk_size: int = 4,
    overlap_pages: int = 1,
) -> list[tuple[bytes, str, bytes]]:
    """Split a PDF into overlapping page windows using PyMuPDF."""
    import fitz

    src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = src_doc.page_count
    chunks = []

    step = max(1, chunk_size - overlap_pages)
    for start in range(0, total_pages, step):
        end = min(start + chunk_size, total_pages)

        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(src_doc, from_page=start, to_page=end - 1)
        chunk_bytes = chunk_doc.tobytes()

        page = chunk_doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        preview_png = pix.tobytes(output="png")
        chunk_doc.close()

        if total_pages > chunk_size:
            label = f"pages {start + 1}\u2013{end}"
        else:
            label = "page 1" if total_pages == 1 else f"pages 1\u2013{total_pages}"

        chunks.append((chunk_bytes, label, preview_png))
        if end == total_pages:
            break

    src_doc.close()
    return chunks


def format_timestamp(sec: float) -> str:
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes}:{seconds:02d}"


def get_video_duration_seconds(video_bytes: bytes) -> float | None:
    import av

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        with av.open(tmp_path) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            return None
    except Exception:
        return None
    finally:
        os.remove(tmp_path)


def choose_frame_interval(duration_sec: float | None) -> int:
    if not duration_sec:
        return FRAME_INTERVAL_SEC
    if duration_sec <= 120:
        return 5
    if duration_sec <= 600:
        return 10
    return 20


def extract_video_frames(
    video_bytes: bytes,
    interval_sec: int = FRAME_INTERVAL_SEC,
) -> list[tuple[bytes, float]]:
    import av

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name
    try:
        container = av.open(tmp_path)
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if not video_stream:
            return []
        fps = float(video_stream.average_rate or 25)
        interval_sec = max(1, interval_sec)
        last_yielded_sec = -interval_sec
        frames = []
        for i, frame in enumerate(container.decode(video_stream)):
            ts = float(frame.pts * video_stream.time_base) if frame.pts else i / fps
            if ts - last_yielded_sec >= interval_sec:
                img = frame.to_image()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                frames.append((buf.getvalue(), ts))
                last_yielded_sec = ts
        container.close()
        return frames
    except Exception:
        return []
    finally:
        os.remove(tmp_path)


def frame_signature(image_bytes: bytes, size: int = 32) -> np.ndarray:
    """A small grayscale fingerprint of a frame for cheap visual comparison."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((size, size))
    return np.asarray(img, dtype="float32").flatten() / 255.0


def is_scene_change(sig_a: np.ndarray, sig_b: np.ndarray, threshold: float = SCENE_CHANGE_THRESHOLD) -> bool:
    """True when two frame signatures differ enough to count as a new scene."""
    return float(np.mean(np.abs(sig_a - sig_b))) >= threshold


def select_scene_frames(
    frames: list[tuple[bytes, float]], threshold: float = SCENE_CHANGE_THRESHOLD
) -> list[tuple[bytes, float]]:
    """Keep one frame per visual scene, dropping near-duplicates.

    A slide held across several sampling intervals collapses to a single frame,
    so a presentation video indexes one frame per slide instead of many copies —
    better recall of distinct content and fewer embedding calls.
    """
    kept = []
    last_sig = None
    for image_bytes, ts in frames:
        try:
            sig = frame_signature(image_bytes)
        except Exception:
            kept.append((image_bytes, ts))
            continue
        if last_sig is None or is_scene_change(last_sig, sig, threshold):
            kept.append((image_bytes, ts))
            last_sig = sig
    return kept


def transcript_for_timestamp(ts: float, windows: list[tuple[float, float, str]]) -> str:
    """Return the transcript of the window covering `ts`.

    `windows` is [(start_sec, end_sec, transcript)]. A frame at `ts` inherits the
    transcript of the audio window it falls in; a frame past the last window keeps
    the last window's transcript so trailing frames stay searchable.
    """
    for start, end, transcript in windows:
        if start <= ts < end:
            return transcript
    if windows and ts >= windows[-1][1]:
        return windows[-1][2]
    return ""


def build_video_frame_searchable_text(filename: str, timestamp_label: str, transcript: str = "") -> str:
    base = f"{filename} frame timestamp {timestamp_label}"
    if transcript:
        return f"{base}\n{transcript}".strip()
    return base


def build_audio_searchable_text(filename: str, start_sec: float, end_sec: float, transcript: str = "") -> str:
    start = format_timestamp(start_sec)
    end = format_timestamp(end_sec)
    base = f"{filename} audio {start} {end}"
    if transcript:
        return f"{base}\n{transcript}".strip()
    return base


def resize_image_if_needed(image_bytes: bytes, mime_type: str) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if max(w, h) <= IMAGE_MAX_DIM:
            return image_bytes
        ratio = IMAGE_MAX_DIM / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "JPEG" if mime_type in ("image/jpeg", "image/jpg") else "PNG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    except Exception:
        return image_bytes


def get_audio_duration_seconds(audio_bytes: bytes, suffix: str = ".mp3") -> float | None:
    import av

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with av.open(tmp_path) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            return None
    except Exception:
        return None
    finally:
        os.remove(tmp_path)


def decode_audio_to_wav(audio_bytes: bytes, suffix: str = ".mp3", rate: int = 16000) -> bytes | None:
    """Decode PyAV-readable audio to a mono 16-bit PCM WAV blob, or None on failure.

    Lets MP3 (and other compressed formats) be chunked into valid, independently
    embeddable WAV windows instead of raw byte slices that only the first chunk
    can decode.
    """
    import av
    from av.audio.resampler import AudioResampler

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        container = av.open(tmp_path)
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            container.close()
            return None
        resampler = AudioResampler(format="s16", layout="mono", rate=rate)
        pcm = bytearray()
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                pcm.extend(resampled.to_ndarray().astype("int16").tobytes())
        for resampled in resampler.resample(None):  # flush
            pcm.extend(resampled.to_ndarray().astype("int16").tobytes())
        container.close()
        out = io.BytesIO()
        with wave.open(out, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(bytes(pcm))
        return out.getvalue()
    except Exception:
        return None
    finally:
        os.remove(tmp_path)


def _chunk_wav_bytes(
    wav_bytes: bytes, chunk_seconds: float, overlap_sec: float = 0.0
) -> list[tuple[bytes, float, float]]:
    """Slice a WAV blob into (bytes, start_sec, end_sec) windows by frame position.

    `overlap_sec` carries a tail of the previous window into the next so a phrase
    split across a boundary still lands whole in at least one chunk.
    """
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        framerate = wf.getframerate()
        total_frames = wf.getnframes()
        params = wf.getparams()
        duration = total_frames / framerate
        if duration <= chunk_seconds:
            return [(wav_bytes, 0.0, duration)]
        step = max(0.001, chunk_seconds - max(0.0, overlap_sec))
        windows = []
        start_sec = 0.0
        while start_sec < duration:
            end_sec = min(start_sec + chunk_seconds, duration)
            start_frame = int(start_sec * framerate)
            end_frame = int(end_sec * framerate)
            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setparams(params)
                out.writeframes(frames)
            windows.append((buf.getvalue(), start_sec, end_sec))
            if end_sec >= duration:
                break
            start_sec += step
        return windows


def proportional_audio_windows(
    total_len: int, duration: float, chunk_seconds: float
) -> list[tuple[int, int, float, float]]:
    """Byte-range windows over a blob, paired with their time span.

    Returns (start_byte, end_byte, start_sec, end_sec) tuples. Used for formats
    we can time but not cheaply decode, so byte position tracks elapsed time.
    """
    if duration <= 0:
        return [(0, total_len, 0.0, 0.0)]
    windows = []
    start_sec = 0.0
    while start_sec < duration:
        end_sec = min(start_sec + chunk_seconds, duration)
        start_byte = int((start_sec / duration) * total_len)
        end_byte = int((end_sec / duration) * total_len)
        windows.append((start_byte, end_byte, start_sec, end_sec))
        start_sec += chunk_seconds
    return windows


def chunk_audio_bytes(
    audio_bytes: bytes,
    mime_type: str,
    chunk_seconds: int = AUDIO_MAX_SECONDS,
    overlap_sec: float = 0.0,
) -> list[tuple[bytes, float, float, str]]:
    """Chunk audio into windows, returning (bytes, start_sec, end_sec, mime).

    WAV is sliced exactly. MP3 is decoded to WAV first (PyAV) so every chunk is
    a valid, independently embeddable clip with accurate timestamps. If decoding
    fails, we fall back to byte-proportional slices anchored to the real duration,
    keeping the original mime so playback still works. `overlap_sec` overlaps
    decoded windows so phrases split across a boundary still land whole somewhere.
    """
    if mime_type == "audio/wav":
        try:
            return [(b, s, e, "audio/wav") for b, s, e in _chunk_wav_bytes(audio_bytes, chunk_seconds, overlap_sec)]
        except Exception:
            return [(audio_bytes, 0.0, float(chunk_seconds), "audio/wav")]

    decoded = decode_audio_to_wav(audio_bytes, suffix=".mp3")
    if decoded is not None:
        try:
            return [(b, s, e, "audio/wav") for b, s, e in _chunk_wav_bytes(decoded, chunk_seconds, overlap_sec)]
        except Exception:
            pass

    duration = get_audio_duration_seconds(audio_bytes)
    if duration is None:
        duration = len(audio_bytes) / 16_000  # rough fallback when PyAV can't read it
    if duration <= chunk_seconds:
        return [(audio_bytes, 0.0, duration, mime_type)]
    return [
        (audio_bytes[start_byte:end_byte], start_sec, end_sec, mime_type)
        for start_byte, end_byte, start_sec, end_sec in proportional_audio_windows(
            len(audio_bytes), duration, chunk_seconds
        )
    ]
