"""Cost and safety guards for the public demo.

No Streamlit import, so the tests can exercise all of it directly. The app wires
these in: a process-wide daily call budget for the shared demo key, a client
wrapper that charges that budget on every Gemini request, upload checks that run
before any decoding, a safe way to put text inside an inline <script>, and
error messages a visitor can act on instead of raw API errors.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from media_ingest import get_audio_duration_seconds, get_video_duration_seconds
from rag_core import is_rate_limit_error, is_transient_error

log = logging.getLogger("unified_rag")

MAX_QUESTION_CHARS = 2000
ANSWER_MAX_OUTPUT_TOKENS = 2048
UTILITY_MAX_OUTPUT_TOKENS = 1024


class DemoBudgetExceeded(Exception):
    """Raised before a demo-key request once today's call budget is spent.

    The message avoids the words the retry helper treats as transient
    ("quota", "429", "rate limit"), so a spent budget is never retried.
    """

    def __init__(self):
        super().__init__("demo daily call budget spent")


class DailyBudget:
    """Counts Gemini requests made on the demo key per UTC day, across sessions.

    One instance per process (the app holds it in st.cache_resource). The count
    is written to a small JSON file so a container restart in the same day does
    not hand out a fresh budget.
    """

    def __init__(self, limit: int, path: Path | None = None,
                 today: Callable[[], str] | None = None):
        self.limit = limit
        self.path = path
        self._today = today or (lambda: dt.datetime.now(dt.UTC).date().isoformat())
        self._lock = threading.Lock()
        self._day, self._used = self._load()

    def _load(self) -> tuple[str, int]:
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return str(data.get("day", "")), int(data.get("used", 0))
            except Exception:
                pass
        return self._today(), 0

    def _save(self):
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"day": self._day, "used": self._used}, handle)
            os.replace(tmp, self.path)
        except Exception:
            log.exception("could not save the demo budget")

    def _roll(self):
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)

    def spend(self, calls: int = 1):
        """Take `calls` from today's budget, or raise DemoBudgetExceeded."""
        with self._lock:
            self._roll()
            if self._used + calls > self.limit:
                raise DemoBudgetExceeded()
            self._used += calls
            self._save()


class _Charged:
    """Proxy for a client sub-API that spends the budget before chosen methods."""

    def __init__(self, inner: Any, budget: DailyBudget, charged: set[str]):
        self._inner = inner
        self._budget = budget
        self._charged = charged

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if name not in self._charged or not callable(attr):
            return attr

        def call(*args, **kwargs):
            self._budget.spend()
            return attr(*args, **kwargs)

        return call


class BudgetedClient:
    """Wraps a genai.Client so every paid request on it spends the daily budget.

    Covers generation, streaming, embedding and the File Search calls the app
    makes. Anything else passes straight through.
    """

    _MODELS = {"generate_content", "generate_content_stream", "embed_content"}
    _STORES = {"create", "upload_to_file_search_store"}

    def __init__(self, client: Any, budget: DailyBudget):
        self._client = client
        self.budget = budget
        self.models = _Charged(client.models, budget, self._MODELS)
        self.file_search_stores = _Charged(client.file_search_stores, budget, self._STORES)

    def __getattr__(self, name: str):
        return getattr(self._client, name)


def js_string_literal(text: str) -> str:
    """Encode text as a JavaScript string literal that is safe inside <script>.

    json.dumps alone leaves `</script>` and `<!--` intact, so answer text could
    close the script element and inject markup. Escaping <, > and & as \\u
    sequences keeps the literal's value identical while the HTML parser never
    sees a tag. U+2028/2029 are escaped too for older JS engines (ensure_ascii
    already does, this keeps it explicit).
    """
    return (json.dumps(text, ensure_ascii=True)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
            .replace(" ", "\\u2028")
            .replace(" ", "\\u2029"))


def friendly_error(error: Exception, using_demo_key: bool = True) -> str:
    """A message a visitor can act on. The raw error goes to the server log."""
    log.warning("gemini call failed: %r", error)
    if isinstance(error, DemoBudgetExceeded):
        return ("The shared demo key has used today's budget. Paste your own Google API key "
                "in the sidebar to keep going, or come back tomorrow.")
    msg = str(error).lower()
    if (any(t in msg for t in ("api key not valid", "api_key_invalid", "permission_denied",
                               "unauthenticated"))
            or re.search(r"\b40[13]\b", msg)):
        return ("Google rejected the API key. Check the key in the sidebar."
                if not using_demo_key else
                "The demo key was rejected. Paste your own Google API key in the sidebar.")
    if is_rate_limit_error(error):
        return ("The shared demo key is rate limited right now. Wait a minute and try again, "
                "or paste your own key in the sidebar." if using_demo_key else
                "Your key hit Gemini's rate limit. Wait a minute and try again.")
    if is_transient_error(error):
        return "Gemini is busy right now. Try again in a moment."
    return "Something went wrong talking to Gemini. Try again, and reload the page if it keeps happening."


@dataclass(frozen=True)
class UploadLimits:
    max_mb: int
    max_pdf_pages: int
    max_audio_seconds: int
    max_video_seconds: int
    max_image_pixels: int


DEMO_LIMITS = UploadLimits(max_mb=20, max_pdf_pages=30, max_audio_seconds=10 * 60,
                           max_video_seconds=5 * 60, max_image_pixels=40_000_000)
# A local run keeps generous limits, but still refuses decompression bombs.
LOCAL_LIMITS = UploadLimits(max_mb=200, max_pdf_pages=500, max_audio_seconds=2 * 60 * 60,
                            max_video_seconds=60 * 60, max_image_pixels=90_000_000)

AUDIO_MIMES = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav"}
VIDEO_MIMES = {"video/mp4": ".mp4", "video/quicktime": ".mov"}


def _minutes(seconds: int) -> str:
    return f"{seconds // 60} minutes"


def image_too_large(data: bytes, max_pixels: int) -> str | None:
    """Read only the image header and refuse anything over the pixel cap."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
    except Exception:
        return "it is not an image this app can read"
    if w * h > max_pixels:
        return f"it is {w}x{h} pixels, over the {max_pixels // 1_000_000} megapixel limit"
    return None


def check_upload(data: bytes, mime: str, limits: UploadLimits) -> str | None:
    """Why an upload is refused, or None if it may be processed.

    Runs before any page rendering, audio decoding or frame extraction, so an
    oversized file costs a header read, not a decode.
    """
    if len(data) > limits.max_mb * 1024 * 1024:
        return f"it is over {limits.max_mb} MB"
    mime = mime or ""
    if mime == "application/pdf":
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                pages = doc.page_count
        except Exception:
            return "it is not a PDF this app can read"
        if pages > limits.max_pdf_pages:
            return f"it has {pages} pages, over the {limits.max_pdf_pages} page limit"
        return None
    if mime in AUDIO_MIMES:
        duration = get_audio_duration_seconds(data, suffix=AUDIO_MIMES[mime])
        if duration is not None and duration > limits.max_audio_seconds:
            return f"it is longer than {_minutes(limits.max_audio_seconds)}"
        return None
    if mime in VIDEO_MIMES:
        duration = get_video_duration_seconds(data)
        if duration is not None and duration > limits.max_video_seconds:
            return f"it is longer than {_minutes(limits.max_video_seconds)}"
        return None
    if mime.startswith("image/"):
        return image_too_large(data, limits.max_image_pixels)
    return "this file type is not supported"


def clamp_question(text: str, limit: int = MAX_QUESTION_CHARS) -> str:
    """Trim a question to the length cap (chat_input enforces it too)."""
    return (text or "")[:limit]
