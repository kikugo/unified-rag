"""Tracing is invisible when off and records the right spans when on."""
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from test_pipeline import DOCS, QUESTION, FakeCollection, FakeEmbedder, ReverseReranker

from rag import tracing
from rag.pipeline import ChromaStore, Config, retrieve


class RecordingClient:
    """Stands in for the Langfuse client: records each observation and its nesting depth."""

    def __init__(self):
        self.spans = []
        self.depth = 0
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, as_type, name, **kwargs):
        record = {"type": as_type, "name": name, "depth": self.depth, **kwargs}
        self.spans.append(record)
        self.depth += 1
        try:
            yield SimpleNamespace(update=lambda **kw: record.update(kw))
        finally:
            self.depth -= 1

    def flush(self):
        self.flushed = True


class _Base(unittest.TestCase):
    def setUp(self):
        self.store = ChromaStore(FakeCollection(DOCS))
        self.embedder = FakeEmbedder({QUESTION: [1.0, 0.1, 0.0], "hyde:" + QUESTION: [0.0, 1.0, 0.0]})
        self.client = RecordingClient()
        tracing.use_client(self.client)

    def tearDown(self):
        tracing.use_client(None)

    def _ids(self, cfg):
        return [h.chunk_id for h in retrieve(QUESTION, cfg, self.store, self.embedder, ReverseReranker())]


class TracingOffTest(_Base):
    def test_off_records_nothing_and_ranks_the_same(self):
        with mock.patch.dict(os.environ, {tracing.FLAG: "0"}):
            ids = self._ids(Config(mode="hybrid_rrf", rerank=True, top_k=4))
            with tracing.stage("x") as span, tracing.generation("g", model="m") as gen:
                tracing.record_generation(gen, "text")
            with tracing.attributes(["a"]):
                pass
            tracing.flush()
        self.assertEqual(self.client.spans, [])
        self.assertIsNone(span)
        self.assertFalse(self.client.flushed)
        with mock.patch.dict(os.environ, {tracing.FLAG: "1"}):
            self.assertEqual(self._ids(Config(mode="hybrid_rrf", rerank=True, top_k=4)), ids)


class TracingOnTest(_Base):
    def test_retrieve_opens_stage_spans_in_order(self):
        with mock.patch.dict(os.environ, {tracing.FLAG: "1"}):
            self._ids(Config(mode="hybrid_rrf", rerank=True, top_k=4))
        self.assertEqual([(s["name"], s["depth"]) for s in self.client.spans],
                         [("retrieve", 0), ("embed", 1), ("dense", 1), ("bm25", 1), ("fuse", 1), ("rerank", 0)])
        self.assertEqual(self.client.spans[0]["metadata"]["mode"], "hybrid_rrf")

    def test_bm25_only_has_no_embed_span(self):
        with mock.patch.dict(os.environ, {tracing.FLAG: "1"}):
            self._ids(Config(mode="bm25", top_k=4))
        self.assertEqual([s["name"] for s in self.client.spans], ["retrieve", "bm25"])

    def test_generation_records_model_output_and_usage(self):
        usage = SimpleNamespace(prompt_token_count=120, candidates_token_count=30, thoughts_token_count=10)
        with mock.patch.dict(os.environ, {tracing.FLAG: "1"}):
            with tracing.generation("generate", model="gemini-3.6-flash", input="Q?") as gen:
                tracing.record_generation(gen, "A.", usage)
            with tracing.generation("generate", model="m") as gen:
                tracing.record_generation(gen, "B.", input_tokens=5, output_tokens=2)
            tracing.flush()
        first, second = self.client.spans
        self.assertEqual((first["type"], first["model"], first["output"]), ("generation", "gemini-3.6-flash", "A."))
        self.assertEqual(first["usage_details"], {"input": 120, "output": 40})  # thinking counts as output
        self.assertEqual(second["usage_details"], {"input": 5, "output": 2})
        self.assertTrue(self.client.flushed)

    def test_require_keys_names_what_is_missing(self):
        env = {k: "" for k in tracing.KEY_VARS}
        with (mock.patch.dict(os.environ, env), mock.patch.object(tracing, "load_dotenv"),
              self.assertRaises(RuntimeError) as err):
            tracing.require_keys()
        self.assertIn("LANGFUSE_SECRET_KEY", str(err.exception))


if __name__ == "__main__":
    unittest.main()
