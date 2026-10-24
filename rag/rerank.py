"""Cross-encoder reranking on CPU through ONNX.

fastembed runs the model on onnxruntime, which chromadb already pulls in, so
reranking adds no torch install. Usage follows the fastembed README section
"Rerank documents with TextCrossEncoder" (context7 /qdrant/fastembed).
"""
from __future__ import annotations

from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.models import RERANK_MODEL

MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "fastembed"


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL, cache_dir: str | Path = MODEL_CACHE_DIR):
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance logit per text, higher is more relevant."""
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, texts)]
