"""Gemini calls for the CLI and eval code, without Streamlit.

`app/gemini.py` reads the embedding dimension from session state and reports
errors with `st.error`, so it cannot run outside the app. These wrappers take
everything as arguments, raise on failure instead of returning None, and go
through the disk cache.
"""
from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.cache import DiskCache, cache_key
from rag.models import EMBEDDING_MODEL, UTILITY_MODEL
from rag.pipeline import HYDE_PROMPT, hyde_vector
from rag_core import call_with_retry

EMBED_BATCH_SIZE = 50
# A full index build crosses the per-minute embedding quota. Backing off
# 5, 10, 20, 40, 80 seconds outlasts a one-minute window instead of failing.
QUOTA_RETRY = {"attempts": 6, "base_delay": 5.0}


def make_client() -> genai.Client:
    """Client from GOOGLE_API_KEY, reading .env when the variable is unset."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("set GOOGLE_API_KEY (or put it in .env) to call Gemini")
    return genai.Client(api_key=api_key)


class GeminiEmbedder:
    """Embeds documents, queries and figures, caching every vector on disk.

    The client is created on first use, so a fully cached rebuild or query never
    needs an API key.
    """

    def __init__(self, cache: DiskCache, dim: int = 3072, model: str = EMBEDDING_MODEL,
                 client: genai.Client | None = None):
        self.cache = cache
        self.dim = dim
        self.model = model
        self._client = client
        self.calls = 0

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = make_client()
        return self._client

    def _key(self, task: str, content: str | bytes) -> str:
        return cache_key(self.model, self.dim, task, content)

    def is_cached(self, task: str, content: str | bytes) -> bool:
        return self._key(task, content) in self.cache

    def _embed_contents(self, contents: list[types.Content], task: str | None) -> list[np.ndarray]:
        config = types.EmbedContentConfig(output_dimensionality=self.dim, task_type=task)
        result = call_with_retry(lambda: self.client.models.embed_content(
            model=self.model, contents=contents, config=config,
        ), **QUOTA_RETRY)
        self.calls += 1
        vectors = [np.array(e.values, dtype=np.float32) for e in result.embeddings]
        # One Content per input should give one vector per input. If the model
        # ever aggregates them instead, fail rather than misassign vectors.
        if len(vectors) != len(contents):
            raise RuntimeError(f"asked for {len(contents)} embeddings, got {len(vectors)}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        task = "RETRIEVAL_DOCUMENT"
        out: list[np.ndarray | None] = [self.cache.get_vector(self._key(task, t)) for t in texts]
        missing = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[start:start + EMBED_BATCH_SIZE]
            contents = [types.Content(parts=[types.Part(text=texts[i])]) for i in batch]
            for i, vec in zip(batch, self._embed_contents(contents, task)):
                self.cache.put_vector(self._key(task, texts[i]), vec)
                out[i] = vec
        return out  # type: ignore[return-value]

    def embed_query(self, text: str) -> np.ndarray:
        task = "RETRIEVAL_QUERY"
        key = self._key(task, text)
        cached = self.cache.get_vector(key)
        if cached is not None:
            return cached
        vec = self._embed_contents([types.Content(parts=[types.Part(text=text)])], task)[0]
        self.cache.put_vector(key, vec)
        return vec

    def embed_figure(self, image_bytes: bytes, mime: str, caption: str) -> np.ndarray:
        """Image and caption as one interleaved vector, the same way the app embeds
        captioned images (no task type)."""
        key = self._key("FIGURE", caption.encode("utf-8") + b"\0" + image_bytes)
        cached = self.cache.get_vector(key)
        if cached is not None:
            return cached
        content = types.Content(parts=[
            types.Part(text=caption),
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
        ])
        vec = self._embed_contents([content], None)[0]
        self.cache.put_vector(key, vec)
        return vec

    def draft_hypothetical(self, question: str) -> str:
        key = cache_key(UTILITY_MODEL, "HYDE", question)
        cached = self.cache.get_text(key)
        if cached is not None:
            return cached
        resp = call_with_retry(lambda: self.client.models.generate_content(
            model=UTILITY_MODEL,
            contents=HYDE_PROMPT.format(question=question),
            config=types.GenerateContentConfig(temperature=0),
        ))
        self.calls += 1
        draft = (resp.text or "").strip()
        self.cache.put_text(key, draft)
        return draft

    def hyde_query(self, question: str) -> np.ndarray:
        vec = hyde_vector(question, self.draft_hypothetical, self.embed_query)
        if vec is None:
            raise RuntimeError("HyDE produced no vector")
        return vec
