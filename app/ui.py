"""Small pure helpers for the Streamlit page, kept here so they can be tested without a browser."""

import re

from rag_core import CITATION_RE

_MD_SPECIAL = re.compile(r"([\\`*_\[\]#|~<>])")

# Accepted upload types, shared by the sidebar uploader and the one on the empty library.
UPLOAD_TYPES = ["png", "jpg", "jpeg", "pdf", "mp3", "wav", "mp4", "mov"]

# A question answered once against the six sample charts, shown on the empty demo library
# so a visitor sees a cited answer before spending anything. Recorded on 2026-11-14 with
# gemini-3.5-flash-lite; the figures match the Netflix chart (APAC +23% Y/Y, 32% margin).
SAMPLE_QUESTION = "Which Netflix region grew fastest, and what was the operating margin?"
SAMPLE_ANSWER = (
    "Based on the provided Netflix income statement image [S1]:\n\n"
    "- **Fastest-growing region:** APAC grew the fastest at +23% Y/Y (compared to UCAN at "
    "+9%, EMEA at +15%, and LATAM at +8%).\n"
    "- **Operating margin:** Netflix's overall operating margin was 32% (with a +4pp Y/Y "
    "change) [S1]."
)
SAMPLE_SOURCES = ["Netflix Q4 2024", "Tesla Q4 2024", "Nike Q4 2024"]


def md_escape(text: str) -> str:
    """Escape a file name so underscores or brackets in it don't turn into markdown."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def cited_ids(text: str) -> set[int]:
    """The source numbers an answer cites: "[S1, S3]" -> {1, 3}."""
    ids = set()
    for match in CITATION_RE.finditer(text or ""):
        for part in match.group(1).split(","):
            digits = part.strip().lstrip("S")
            if digits.isdigit():
                ids.add(int(digits))
    return ids


def quota_text(remaining: int, limit: int) -> str:
    """The demo meter's label, e.g. "Shared demo key: 987 of 1,000 calls left today"."""
    remaining = max(0, min(remaining, limit))
    if remaining == 0:
        return "Shared demo key: today's calls are used up. It resets at 00:00 UTC."
    return f"Shared demo key: {remaining:,} of {limit:,} calls left today"


def quota_fraction(remaining: int, limit: int) -> float:
    """Share of today's demo budget left, clamped to 0..1 for st.progress."""
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, remaining / limit))


def ingest_fraction(done_files: int, total_files: int, step: int = 0, steps: int = 0) -> float:
    """Overall progress across files, counting the steps of the file in progress."""
    if total_files <= 0:
        return 1.0
    within = step / steps if steps > 0 else 0.0
    return max(0.0, min(1.0, (done_files + within) / total_files))


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"
