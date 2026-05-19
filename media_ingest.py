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
