from __future__ import annotations

import io
import os
import tempfile
import wave

from PIL import Image

AUDIO_MAX_SECONDS = 80
VIDEO_MAX_SECONDS = 80
FRAME_INTERVAL_SEC = 5
IMAGE_MAX_DIM = 4096

def format_timestamp(sec: float) -> str:
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes}:{seconds:02d}"


def chunk_audio_bytes(
    audio_bytes: bytes,
    mime_type: str,
    chunk_seconds: int = AUDIO_MAX_SECONDS,
) -> list[tuple[bytes, float, float]]:
    """Chunk audio into windows with timestamps; MP3 uses proportional chunking fallback."""
    if mime_type == "audio/wav":
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                framerate = wf.getframerate()
                total_frames = wf.getnframes()
                params = wf.getparams()
                duration = total_frames / framerate
                if duration <= chunk_seconds:
                    return [(audio_bytes, 0.0, duration)]
                windows = []
                start_sec = 0.0
                while start_sec < duration:
                    start_frame = int(start_sec * framerate)
                    end_frame = int(min((start_sec + chunk_seconds), duration) * framerate)
                    wf.setpos(start_frame)
                    frames = wf.readframes(end_frame - start_frame)
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as out:
                        out.setparams(params)
                        out.writeframes(frames)
                    windows.append(
                        (buf.getvalue(), start_sec, min(start_sec + chunk_seconds, duration))
                    )
                    start_sec += chunk_seconds
                return windows
        except Exception:
            return [(audio_bytes, 0.0, float(chunk_seconds))]
    estimated_duration = len(audio_bytes) / 16_000
    if estimated_duration <= chunk_seconds:
        return [(audio_bytes, 0.0, estimated_duration)]
    windows = []
    start_sec = 0.0
    while start_sec < estimated_duration:
        end_sec = min(start_sec + chunk_seconds, estimated_duration)
        start_byte = int((start_sec / estimated_duration) * len(audio_bytes))
        end_byte = int((end_sec / estimated_duration) * len(audio_bytes))
        windows.append((audio_bytes[start_byte:end_byte], start_sec, end_sec))
        start_sec += chunk_seconds
    return windows


def build_audio_searchable_text(filename: str, start_sec: float, end_sec: float, transcript: str = "") -> str:
    start = format_timestamp(start_sec)
    end = format_timestamp(end_sec)
    base = f"{filename} audio {start} {end}"
    if transcript:
        return f"{base}\n{transcript}".strip()
    return base

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


def build_video_frame_searchable_text(filename: str, timestamp_label: str, transcript: str = "") -> str:
    base = f"{filename} frame timestamp {timestamp_label}"
    if transcript:
        return f"{base}\n{transcript}".strip()
    return base

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
