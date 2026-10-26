"""The generation judge: one pinned model outside the generator's family.

The generator is Gemini, so the judge is `gpt-4.1-mini` (version 2025-04-14) on
an Azure AI Foundry deployment, called keyless through Entra (`az login`) at
temperature 0. DeepEval's FaithfulnessMetric and AnswerRelevancyMetric run on
top of it through the `DeepEvalBaseLLM` interface. Every prompt/response pair is
cached on disk, so rerunning an eval makes no judge calls and returns the same
scores.

Endpoint and deployment come from JUDGE_ENDPOINT and JUDGE_DEPLOYMENT, so a
clone can point the judge at its own deployment.
"""
from __future__ import annotations

import json
import os
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from openai import OpenAI

from rag.cache import DiskCache, cache_key

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_VERSION = "2025-04-14"
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"

REFUSAL_PROMPT = """You are checking answers from a question-answering system that must answer only from
retrieved sources. Decide whether the ANSWER declines to answer the QUESTION because the sources
do not contain the information (for example "the sources do not say", "I cannot find this").
An answer that gives the requested information, even partly or with caveats, is not a refusal.

QUESTION: {question}

ANSWER: {answer}

Reply with JSON only: {{"refused": true}} or {{"refused": false}}."""


def make_judge_client(endpoint: str | None = None) -> OpenAI:
    """OpenAI client on the Foundry v1 endpoint, authenticated with an Entra token."""
    load_dotenv()
    endpoint = endpoint or os.getenv("JUDGE_ENDPOINT")
    if not endpoint:
        raise RuntimeError("set JUDGE_ENDPOINT to the Foundry resource endpoint to run the judge")
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), TOKEN_SCOPE)
    return OpenAI(base_url=f"{endpoint.rstrip('/')}/openai/v1/", api_key=token_provider)


class FoundryJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by a Foundry chat deployment, cached on disk.

    The client is created on first uncached call, so a fully cached rerun needs
    no Azure login.
    """

    def __init__(self, cache: DiskCache, deployment: str | None = None, client: OpenAI | None = None):
        self.cache = cache
        self.deployment = deployment or os.getenv("JUDGE_DEPLOYMENT") or JUDGE_MODEL
        self._client = client
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        super().__init__(self.deployment)

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return f"{self.deployment} (Foundry)"

    def supports_temperature(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_judge_client()
        return self._client

    def take_usage(self) -> dict[str, int]:
        """Tokens spent on uncached calls since the last take, then reset."""
        usage, self.usage = self.usage, {"input_tokens": 0, "output_tokens": 0}
        return usage

    def complete(self, prompt: str, json_mode: bool) -> str:
        key = cache_key("judge", self.deployment, JUDGE_VERSION, int(json_mode), prompt)
        cached = self.cache.get_text(key)
        if cached is not None:
            return cached
        kwargs: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        self.calls += 1
        if resp.usage:
            self.usage["input_tokens"] += resp.usage.prompt_tokens
            self.usage["output_tokens"] += resp.usage.completion_tokens
        self.cache.put_text(key, text)
        return text

    def generate(self, prompt: str, schema=None, **kwargs):
        text = self.complete(prompt, json_mode=schema is not None)
        return schema.model_validate_json(text) if schema is not None else text

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema)


def is_refusal(judge: FoundryJudge, question: str, answer: str) -> bool:
    text = judge.complete(REFUSAL_PROMPT.format(question=question, answer=answer), json_mode=True)
    return bool(json.loads(text).get("refused"))


def score_generation(judge: FoundryJudge, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
    """Faithfulness and answer relevancy for one answer, with the judge's reasons."""
    case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
    out: dict[str, Any] = {}
    for name, metric in (
        ("faithfulness", FaithfulnessMetric(model=judge, eval_mode="llm", async_mode=False)),
        ("answer_relevancy", AnswerRelevancyMetric(model=judge, eval_mode="llm", async_mode=False)),
    ):
        metric.measure(case, _show_indicator=False)
        out[name] = metric.score
        out[f"{name}_reason"] = metric.reason
    return out
