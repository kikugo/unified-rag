"""Gemini API helpers: embedding, routing, captioning, and transcription.

These wrap the Google GenAI client so `app.py` stays focused on UI and flow. The
client is a per-session singleton, set once via `set_client`, so call sites stay
simple. Functions read the active embedding dimension from Streamlit session
state and report errors through Streamlit, so they belong to the app layer
rather than the pure helper modules.
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

from rag_core import call_with_retry
from media_ingest import chunk_audio_bytes, decode_audio_to_wav, resize_image_if_needed

_client: genai.Client | None = None


def set_client(client: genai.Client) -> None:
    """Register the active Gemini client for this session."""
    global _client
    _client = client


def route_query(question: str) -> str:
    """Classify whether a question is best answered by the local vector store or
    the Google Managed RAG backend, using a fast Gemini Flash call.

    Returns 'local' or 'managed'.
    - 'local'   → best for specific visual content, charts, images, audio, video clips
    - 'managed' → best for broad text summarisation, multi-document synthesis,
                  or anything needing full-text search across large corpora
    """
    prompt = (
        "You are a query router for a multimodal RAG application. "
        "Given a user question, decide which backend is best:\n"
        "- 'local': precise lookup in locally embedded images, PDF pages, audio, or video frame timestamps\n"
        "- 'managed': broad text search / summarisation across large text documents\n\n"
        f"User question: {question}\n\n"
        "Reply with exactly one word: local OR managed"
    )
    try:
        resp = call_with_retry(lambda: _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        ))
        answer = resp.text.strip().lower()
        return "managed" if "managed" in answer else "local"
    except Exception:
        return "local"  # safe fallback


def auto_caption_image(image_bytes: bytes, mime_type: str) -> str:
    try:
        resp = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text="Write a concise search caption for this image in one sentence."),
            ],
        )
        return (resp.text or "").strip()
    except Exception:
        return ""


def transcribe_audio_bytes(audio_bytes: bytes, mime_type: str) -> str:
    """Best-effort speech-to-text for one audio chunk via Gemini.

    Returns the transcript, or "" when there is no clear speech or on error. The
    NO_SPEECH sentinel guards against the model echoing the prompt back on silent
    or music-only audio. Audio-only for now; video stays frame-indexed.
    """
    prompt = (
        "You are a speech-to-text transcriber. Transcribe the spoken words in this "
        "audio verbatim. Output only the transcript text and nothing else. "
        "If there is no clear speech, output exactly: NO_SPEECH"
    )
    try:
        resp = call_with_retry(lambda: _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                types.Part(text=prompt),
            ],
        ))
        text = (resp.text or "").strip()
        if not text or text.upper().startswith("NO_SPEECH"):
            return ""
        return text
    except Exception:
        return ""


def hyde_query_embedding(query: str, dim: int) -> np.ndarray | None:
    """HyDE: draft a hypothetical answer, embed it, and average with the query.

    Retrieval then matches passages by meaning, not just shared keywords. Falls
    back to the plain query embedding if the draft or its embedding fails.
    """
    try:
        draft = call_with_retry(lambda: _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Write a short, factual passage that would answer this question:\n{query}",
        ))
        hypothetical = (draft.text or "").strip()
    except Exception:
        hypothetical = ""

    vectors = []
    for text in (query, hypothetical):
        if not text:
            continue
        try:
            result = call_with_retry(lambda t=text: _client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=t,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dim),
            ))
            vectors.append(np.array(result.embeddings[0].values))
        except Exception:
            continue
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def transcribe_video_audio(video_bytes: bytes) -> list[tuple[float, float, str]]:
    """Transcribe a video's audio track, chunked into timestamped windows.

    Returns [(start_sec, end_sec, transcript)] for windows that contain speech,
    so each frame can inherit the transcript covering its timestamp. Empty when
    the video has no audio track or nothing was transcribed.
    """
    wav = decode_audio_to_wav(video_bytes, suffix=".mp4")
    if not wav:
        return []
    windows = []
    for chunk_bytes, start_sec, end_sec, chunk_mime in chunk_audio_bytes(wav, "audio/wav"):
        transcript = transcribe_audio_bytes(chunk_bytes, chunk_mime)
        if transcript:
            windows.append((start_sec, end_sec, transcript))
    return windows


def embed_image(image_bytes: bytes, mime_type: str = "image/png") -> np.ndarray | None:
    """Embed an image using Gemini Embedding 2."""
    image_bytes = resize_image_if_needed(image_bytes, mime_type)
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = call_with_retry(lambda: _client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
            config=types.EmbedContentConfig(output_dimensionality=dim),
        ))
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None


def embed_image_with_caption(image_bytes: bytes, caption: str, mime_type: str = "image/png") -> np.ndarray | None:
    """Embed an image interleaved with a text caption as one unified vector.
    Produces richer embeddings than image-only when caption provides useful context.
    """
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = call_with_retry(lambda: _client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=types.Content(
                parts=[
                    types.Part(text=caption),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ]
            ),
            config=types.EmbedContentConfig(output_dimensionality=dim),
        ))
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None


def embed_pdf(pdf_bytes: bytes) -> list[np.ndarray]:
    """Embed a single PDF chunk using Gemini Embedding 2."""
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = call_with_retry(lambda: _client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            ],
            config=types.EmbedContentConfig(output_dimensionality=dim),
        ))
        return [np.array(e.values) for e in result.embeddings]
    except Exception as e:
        st.error(f"PDF embedding error: {e}")
        return []


def embed_audio(audio_bytes: bytes, mime_type: str) -> np.ndarray | None:
    """Embed an audio file using Gemini Embedding 2."""
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = call_with_retry(lambda: _client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            ],
            config=types.EmbedContentConfig(output_dimensionality=dim),
        ))
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Audio embedding error: {e}")
        return None
