"""The eval runner end to end with fakes: no index, no Gemini, no Azure."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval.judge import FoundryJudge, is_refusal, score_generation
from eval.run import RunConfig, Services, estimate, load_prices, load_run_config, run
from rag.cache import DiskCache
from rag.pipeline import ChromaStore

DOCS = {
    "art5": ([1.0, 0.0], "Article 5 prohibits social scoring by public authorities."),
    "fines": ([0.0, 1.0], "Fines for prohibited practices reach 7% of worldwide turnover."),
    "other": ([0.7, 0.7], "The regulation entered into force in 2024."),
}
VECTORS = {
    "What does Article 5 prohibit?": [1.0, 0.1],
    "How high are the fines?": [0.1, 1.0],
    "Who won the 1998 World Cup?": [0.5, 0.5],
}
ROWS = [
    {"id": "g001", "question": "What does Article 5 prohibit?", "gold_chunk_ids": ["art5"],
     "gold_answer": "Social scoring.", "type": "factual", "answerable": True},
    {"id": "g002", "question": "How high are the fines?", "gold_chunk_ids": ["fines"],
     "gold_answer": "7% of turnover.", "type": "keyword", "answerable": True},
    {"id": "g003", "question": "Who won the 1998 World Cup?", "gold_chunk_ids": [],
     "gold_answer": "", "type": "unanswerable", "answerable": False},
]
ANSWERS = {
    "What does Article 5 prohibit?": "It prohibits social scoring [S1].",
    "How high are the fines?": "Up to 7% of turnover [S1] [S9].",
    "Who won the 1998 World Cup?": "The sources do not say.",
}


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    def get(self, where=None, include=None):
        ids = list(self.docs)
        return {"ids": ids, "metadatas": [{"searchable_text": self.docs[i][1], "title": i} for i in ids]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        q = np.asarray(query_embeddings[0], dtype=float)
        scored = sorted(
            ((i, 1.0 - float(q @ np.asarray(v) / (np.linalg.norm(q) * np.linalg.norm(v))))
             for i, (v, _) in self.docs.items()),
            key=lambda p: p[1])[:n_results]
        return {"ids": [[i for i, _ in scored]], "distances": [[d for _, d in scored]]}


class FakeEmbedder:
    def embed_query(self, text):
        return np.asarray(VECTORS[text], dtype=float)

    def hyde_query(self, text):
        return self.embed_query(text)


class FakeChat:
    """Stands in for the OpenAI client. One JSON reply satisfies every DeepEval schema."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, temperature, **kwargs):
        self.calls += 1
        prompt = messages[0]["content"]
        answer = prompt.rsplit("ANSWER:", 1)[-1]
        reply = {"truths": ["t"], "claims": ["c"], "statements": ["s"],
                 "verdicts": [{"verdict": "yes", "reason": "supported"}], "reason": "looks right",
                 "refused": "do not say" in answer}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        )


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = DiskCache(Path(self.tmp.name) / "cache.sqlite")
        self.chat = FakeChat()
        self.generated = 0
        self.measured = 0

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def _services(self, judge=True):
        def generate(model, contents, gen_config):
            self.generated += 1
            question = contents[-1].parts[-1].text.rsplit("Question: ", 1)[1]
            return {"text": ANSWERS[question], "latency_ms": 500.0 + self.generated,
                    "input_tokens": 1000, "output_tokens": 50}

        def measure(question, hyde):
            self.measured += 1
            return {"embed_ms": 80.0, "embed_tokens": 8, "hyde_input_tokens": 0, "hyde_output_tokens": 0}

        return Services(
            store=ChromaStore(FakeCollection(DOCS)), embedder=FakeEmbedder(), reranker=None,
            judge=FoundryJudge(self.cache, deployment="gpt-4.1-mini", client=self.chat) if judge else None,
            cache=self.cache, generate=generate, measure_query=measure,
        )


class RunnerTest(_Base):
    cfg = RunConfig(name="test", mode="hybrid_rrf", context_k=2)

    def test_end_to_end_rows_and_totals(self):
        result = run(ROWS, self.cfg, self._services(), load_prices())
        rows = {r["id"]: r for r in result["rows"]}
        self.assertEqual(set(result), {"meta", "totals", "by_type", "rows"})

        self.assertEqual(rows["g001"]["hit@1"], 1.0)
        self.assertEqual(rows["g001"]["citation_correctness"], 1.0)
        # [S9] does not exist, so half of g002's citations are wrong
        self.assertEqual(rows["g002"]["citation_correctness"], 0.5)
        self.assertNotIn("hit@5", rows["g003"])  # no retrieval metrics without gold

        totals = result["totals"]
        self.assertEqual((totals["n"], totals["n_answerable"]), (3, 2))
        self.assertEqual(totals["hit@5"], 1.0)
        self.assertEqual(totals["citation_correctness"], 0.75)
        self.assertEqual(totals["no_citation_rate"], 0.0)
        self.assertEqual(totals["refusal_accuracy"], 1.0)
        self.assertEqual(totals["false_refusal_rate"], 0.0)
        self.assertEqual(totals["faithfulness"], 1.0)
        self.assertEqual(totals["n_generation_judged"], 2)
        self.assertNotIn("faithfulness", rows["g003"])  # unanswerable rows only get the refusal check
        self.assertEqual(set(result["by_type"]), {"factual", "keyword", "unanswerable"})

        latency = rows["g001"]["latency_ms"]
        self.assertEqual(latency["embed"], 80.0)
        self.assertAlmostEqual(latency["total"], latency["retrieve"] + latency["rerank"] + latency["generate"])
        self.assertAlmostEqual(totals["cost_usd_per_100_queries"], totals["cost_usd_per_query"] * 100)
        self.assertGreater(totals["judge_cost_usd_total"], 0)
        self.assertEqual(result["meta"]["paid_calls_this_run"], {"answer": 3, "judge_row": 3, "query": 3})

    def test_rerun_is_free_and_keeps_recorded_latency(self):
        svc = self._services()
        first = run(ROWS, self.cfg, svc, load_prices())
        calls = (self.generated, self.measured, self.chat.calls)
        self.assertEqual(estimate(ROWS, self.cfg, self._services(), load_prices())["calls"],
                         {"query": 0, "answer": 0, "judge_row": 0})
        second = run(ROWS, self.cfg, self._services(), load_prices())
        self.assertEqual((self.generated, self.measured, self.chat.calls), calls)
        self.assertEqual(second["meta"]["paid_calls_this_run"], {"answer": 0, "judge_row": 0, "query": 0})
        for a, b in zip(first["rows"], second["rows"]):
            self.assertEqual(a["latency_ms"]["generate"], b["latency_ms"]["generate"])
            self.assertEqual(a["answer"], b["answer"])
            self.assertEqual(a.get("faithfulness"), b.get("faithfulness"))
        self.assertEqual(first["totals"]["judge_cost_usd_total"], second["totals"]["judge_cost_usd_total"])

    def test_estimate_counts_uncached_calls(self):
        est = estimate(ROWS, self.cfg, self._services(), load_prices())
        self.assertEqual(est["calls"], {"query": 3, "answer": 3, "judge_row": 3})
        self.assertGreater(est["usd"], 0)

    def test_no_judge_and_bm25_skip_paid_steps(self):
        result = run(ROWS, RunConfig(name="b", mode="bm25"), self._services(judge=False), load_prices())
        self.assertEqual(self.measured, 0)  # bm25 embeds nothing
        self.assertNotIn("faithfulness", result["totals"])
        self.assertEqual(result["rows"][0]["latency_ms"].get("embed"), None)

    def test_golden_rows_are_not_modified(self):
        before = json.dumps(ROWS, sort_keys=True)
        run(ROWS, self.cfg, self._services(), load_prices())
        self.assertEqual(json.dumps(ROWS, sort_keys=True), before)


class JudgeTest(_Base):
    def test_deepeval_metrics_through_the_cached_judge(self):
        judge = FoundryJudge(self.cache, deployment="gpt-4.1-mini", client=self.chat)
        scores = score_generation(judge, "Q?", "Social scoring [S1].", ["Article 5 prohibits social scoring."])
        self.assertEqual(scores["faithfulness"], 1.0)
        self.assertEqual(scores["answer_relevancy"], 1.0)
        self.assertEqual(scores["faithfulness_reason"], "looks right")
        calls = self.chat.calls
        self.assertGreater(calls, 0)
        self.assertEqual(judge.take_usage()["input_tokens"], 100 * calls)

        again = score_generation(judge, "Q?", "Social scoring [S1].", ["Article 5 prohibits social scoring."])
        self.assertEqual(again, scores)
        self.assertEqual(self.chat.calls, calls, "a repeated judgement must come from the cache")

    def test_refusal(self):
        judge = FoundryJudge(self.cache, deployment="gpt-4.1-mini", client=self.chat)
        self.assertTrue(is_refusal(judge, "Who won?", "The sources do not say."))
        self.assertFalse(is_refusal(judge, "Who won?", "France."))


class ConfigTest(unittest.TestCase):
    def test_shipped_configs_load(self):
        for name in ("dense", "bm25", "hybrid_rrf", "hybrid_rrf_rerank", "hybrid_rrf_rerank_hyde",
                     "hybrid_rrf_rerank_ctx1"):
            cfg = load_run_config(name)
            self.assertEqual(cfg.pipeline_config().top_k, 10)
        self.assertEqual(load_run_config("hybrid_rrf_rerank_ctx1").context_k, 1)

    def test_unknown_config_or_key_fails(self):
        with self.assertRaises(SystemExit):
            load_run_config("nope")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.json").write_text('{"mode": "dense", "topk": 3}')
            with self.assertRaises(SystemExit):
                load_run_config("bad", Path(tmp))

    def test_every_model_has_a_price(self):
        prices = load_prices()["models"]
        for model in ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-embedding-2-preview", "gpt-4.1-mini"):
            self.assertIn(model, prices)


if __name__ == "__main__":
    unittest.main()
