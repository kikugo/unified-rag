"""Run the golden set through one retrieval config and write results/<name>.json.

    uv run python -m eval.run --config hybrid_rrf_rerank
    uv run python -m eval.run --config dense --limit 20 --no-judge

Each row goes through `rag.pipeline.retrieve` and the request `rag.pipeline.answer`
sends, so the numbers measure the code the app runs. Retrieval is scored at depth
10 (MRR@10); the answer sees the top `context_k` hits.

Every paid call is cached in `.cache/`: answers, judge verdicts and query
measurements. Cached records keep the latency and tokens of the original call,
so a rerun costs nothing and still reports real latency, not a cache hit's 0 ms.
Local stages (vector search, BM25, fusion, reranking) are timed live.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from google.genai import types
from langfuse import Evaluation

from eval.golden import TYPES, golden_hash, load_golden
from eval.judge import (
    JUDGE_MODEL,
    JUDGE_VERSION,
    FoundryJudge,
    is_refusal,
    score_generation,
)
from eval.langfuse_sync import DATASET, item_id
from eval.metrics import (
    bootstrap_ci,
    citation_correctness,
    cited_ids,
    hit_at_k,
    mean,
    mrr_at_k,
    percentile,
    recall_at_k,
)
from rag import tracing
from rag.cache import DiskCache, cache_key
from rag.gemini_client import GeminiEmbedder
from rag.index import DEFAULT_DIM, open_store, read_figure
from rag.models import ANSWER_MODEL, EMBEDDING_MODEL, UTILITY_MODEL
from rag.pipeline import HYDE_PROMPT, Config, Hit, answer_request, retrieve
from rag.rerank import CrossEncoderReranker
from rag_core import call_with_retry

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
CONFIG_DIR = EVAL_DIR / "configs"
PRICES_PATH = EVAL_DIR / "prices.json"
RESULTS_DIR = ROOT / "results"
TRACED_DIR = RESULTS_DIR / "traced"
RETRIEVAL_DEPTH = 10
LOCAL_STAGES = ("dense", "bm25", "fuse")
# Rough judge spend per judged answer, only for the pre-run estimate:
# faithfulness and relevancy prompts plus the refusal check.
EST_JUDGE_TOKENS = {"input": 7000, "output": 900}
EST_ANSWER_TOKENS = {"input": 1500, "output": 400}


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: str = "hybrid_rrf"
    rerank: bool = False
    hyde: bool = False
    context_k: int = 3
    candidate_k: int = 20
    rrf_k: int = 60
    generator: str = ANSWER_MODEL

    def pipeline_config(self) -> Config:
        # top_k only sets how many ranked ids are returned: the RRF pool and the
        # reranker's pool are candidate_k either way, so depth 10 changes no ranking.
        return Config(mode=self.mode, rerank=self.rerank, hyde=self.hyde, top_k=RETRIEVAL_DEPTH,
                      context_k=self.context_k, candidate_k=self.candidate_k, rrf_k=self.rrf_k)


def load_run_config(name: str, config_dir: Path = CONFIG_DIR) -> RunConfig:
    path = config_dir / f"{name}.json"
    if not path.exists():
        known = sorted(p.stem for p in config_dir.glob("*.json"))
        raise SystemExit(f"no config {name!r}; known: {', '.join(known)}")
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(RunConfig)} - {"name"}
    unknown = set(data) - allowed
    if unknown:
        raise SystemExit(f"{path.name}: unknown keys {sorted(unknown)}")
    cfg = RunConfig(name=name, **data)
    cfg.pipeline_config()  # validates mode and context_k
    return cfg


def load_prices(path: Path = PRICES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def token_cost(prices: dict[str, Any], model: str, input_tokens: int, output_tokens: int = 0) -> float:
    rate = prices["models"][model]
    return (input_tokens * rate["input_per_1m"] + output_tokens * rate["output_per_1m"]) / 1_000_000


@dataclass
class Services:
    """Everything the runner calls out to, so tests can swap in fakes."""

    store: Any
    embedder: Any
    reranker: Any
    judge: FoundryJudge | None
    cache: DiskCache
    generate: Callable[[str, list[types.Content], types.GenerateContentConfig], dict[str, Any]]
    measure_query: Callable[[str, bool], dict[str, Any]]
    load_bytes: Callable[[dict[str, Any]], tuple[bytes, str] | None] | None = None
    paid_calls: dict[str, int] = field(default_factory=lambda: {"answer": 0, "judge_row": 0, "query": 0})


# --- real services ------------------------------------------------------------

def gemini_generate(client) -> Callable[[str, list[types.Content], types.GenerateContentConfig], dict[str, Any]]:
    def generate(model, contents, gen_config):
        with tracing.generation("generate", model=model) as gen:
            start = time.perf_counter()
            resp = call_with_retry(lambda: client.models.generate_content(
                model=model, contents=contents, config=gen_config))
            latency = (time.perf_counter() - start) * 1000
            usage = resp.usage_metadata
            out = {
                "text": (resp.text or "").strip(),
                "latency_ms": latency,
                "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
                # thinking tokens are billed as output
                "output_tokens": ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
            }
            tracing.record_generation(gen, out["text"], input_tokens=out["input_tokens"],
                                      output_tokens=out["output_tokens"])
        return out
    return generate


def gemini_measure_query(embedder: GeminiEmbedder) -> Callable[[str, bool], dict[str, Any]]:
    """Times one uncached query embedding (with its HyDE draft when asked) and counts its tokens.

    Runs on a throwaway in-memory cache, so the measured call is a real API call
    and the ranking still uses the persistent cached vectors.
    """
    def measure(question, hyde):
        fresh = GeminiEmbedder(DiskCache(":memory:"), dim=embedder.dim, model=embedder.model,
                               client=embedder.client)
        start = time.perf_counter()
        draft = ""
        if hyde:
            fresh.hyde_query(question)
            draft = fresh.draft_hypothetical(question)
        else:
            fresh.embed_query(question)
        embed_ms = (time.perf_counter() - start) * 1000

        def count(model, text):
            if not text:
                return 0
            return call_with_retry(lambda: embedder.client.models.count_tokens(
                model=model, contents=text)).total_tokens or 0

        return {
            "embed_ms": embed_ms,
            "embed_tokens": count(EMBEDDING_MODEL, question) + count(EMBEDDING_MODEL, draft),
            "hyde_input_tokens": count(UTILITY_MODEL, HYDE_PROMPT.format(question=question)) if hyde else 0,
            "hyde_output_tokens": count(UTILITY_MODEL, draft) if hyde else 0,
        }
    return measure


def _load_bytes(meta: dict[str, Any]) -> tuple[bytes, str] | None:
    return read_figure(meta) if meta.get("image") else None


def real_services(cfg: RunConfig, with_judge: bool, fresh: bool = False) -> Services:
    """`fresh` recomputes answers and query embeddings against an in-memory cache,
    so a traced run measures real calls. Judge verdicts still use the disk cache."""
    judge_cache = DiskCache()
    cache = DiskCache(":memory:") if fresh else judge_cache
    embedder = GeminiEmbedder(cache, dim=DEFAULT_DIM)
    generate = None

    def lazy_generate(model, contents, gen_config):
        # the Gemini client is only created when an answer is not cached
        nonlocal generate
        generate = generate or gemini_generate(embedder.client)
        return generate(model, contents, gen_config)

    return Services(
        store=open_store(exact=True),  # deterministic reruns, see ChromaStore
        embedder=embedder,
        reranker=CrossEncoderReranker() if cfg.rerank else None,
        judge=FoundryJudge(judge_cache) if with_judge else None,
        cache=cache,
        generate=lazy_generate,
        measure_query=gemini_measure_query(embedder),
        load_bytes=_load_bytes,
    )


# --- cached steps ---------------------------------------------------------------

def _cached_json(svc: Services, key: str, kind: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    raw = svc.cache.get_text(key)
    if raw is not None:
        return json.loads(raw)
    value = compute()
    svc.paid_calls[kind] += 1
    svc.cache.put_text(key, json.dumps(value))
    return value


def _query_key(cfg: RunConfig, question: str) -> str:
    return cache_key("eval-query", EMBEDDING_MODEL, DEFAULT_DIM, int(cfg.hyde), question)


def _answer_key(cfg: RunConfig, question: str, selected: list[dict[str, Any]]) -> str:
    return cache_key("eval-answer", cfg.generator, question, *[row["id"] for row in selected])


def _judge_key(svc: Services, row: dict[str, Any], answer: str, contexts: list[str]) -> str:
    return cache_key("eval-judge", svc.judge.deployment, JUDGE_VERSION, int(row["answerable"]),
                     row["question"], answer, *contexts)


def _contexts(selected: list[dict[str, Any]]) -> list[str]:
    # Figures reach the judge as their caption text; the judge never sees the image.
    return [row.get("text") or row.get("name", "") for row in selected]


def judge_row(svc: Services, row: dict[str, Any], answer: str, contexts: list[str]) -> dict[str, Any]:
    def compute():
        svc.judge.take_usage()
        verdict: dict[str, Any] = {"refused": is_refusal(svc.judge, row["question"], answer)}
        if row["answerable"] and not verdict["refused"]:
            verdict.update(score_generation(svc.judge, row["question"], answer, contexts))
        usage = svc.judge.take_usage()
        verdict["judge_input_tokens"] = usage["input_tokens"]
        verdict["judge_output_tokens"] = usage["output_tokens"]
        return verdict
    return _cached_json(svc, _judge_key(svc, row, answer, contexts), "judge_row", compute)


# --- one row ------------------------------------------------------------------------

def run_row(row: dict[str, Any], cfg: RunConfig, svc: Services, prices: dict[str, Any]) -> dict[str, Any]:
    pcfg = cfg.pipeline_config()
    question = row["question"]
    timings: dict[str, float] = {}
    hits: list[Hit] = retrieve(question, pcfg, svc.store, svc.embedder, svc.reranker, timings=timings)
    ranked = [h.chunk_id for h in hits]

    latency = {stage: timings[stage] for stage in LOCAL_STAGES if stage in timings}
    query = {"embed_ms": 0.0, "embed_tokens": 0, "hyde_input_tokens": 0, "hyde_output_tokens": 0}
    if pcfg.mode != "bm25":
        query = _cached_json(svc, _query_key(cfg, question), "query", lambda: svc.measure_query(question, cfg.hyde))
        latency["embed"] = query["embed_ms"]
    latency["retrieve"] = sum(latency.values())
    latency["rerank"] = timings.get("rerank", 0.0)

    contents, gen_config, selected = answer_request(question, hits, pcfg, svc.load_bytes)
    generated = _cached_json(svc, _answer_key(cfg, question, selected), "answer",
                             lambda: svc.generate(cfg.generator, contents, gen_config))
    answer = generated["text"]
    latency["generate"] = generated["latency_ms"]
    latency["total"] = latency["retrieve"] + latency["rerank"] + latency["generate"]

    out: dict[str, Any] = {
        "id": row["id"],
        "type": row["type"],
        "answerable": row["answerable"],
        "question": question,
        "gold_chunk_ids": row["gold_chunk_ids"],
        "ranked": ranked,
        "context_ids": [s["id"] for s in selected],
        "answer": answer,
        "cited_ids": cited_ids(answer, selected),
    }
    if row["answerable"]:
        gold = row["gold_chunk_ids"]
        out.update({
            "hit@1": hit_at_k(ranked, gold, 1),
            "hit@3": hit_at_k(ranked, gold, 3),
            "hit@5": hit_at_k(ranked, gold, 5),
            "mrr@10": mrr_at_k(ranked, gold, 10),
            "recall@5": recall_at_k(ranked, gold, 5),
            "citation_correctness": citation_correctness(answer, selected, gold),
        })

    if svc.judge is not None:
        with tracing.stage("judge"):
            out.update(judge_row(svc, row, answer, _contexts(selected)))

    tokens = {
        "generate_input": generated["input_tokens"],
        "generate_output": generated["output_tokens"],
        "embed": query["embed_tokens"],
        "hyde_input": query["hyde_input_tokens"],
        "hyde_output": query["hyde_output_tokens"],
    }
    out["latency_ms"] = latency
    out["tokens"] = tokens
    out["cost_usd"] = (token_cost(prices, cfg.generator, tokens["generate_input"], tokens["generate_output"])
                       + token_cost(prices, EMBEDDING_MODEL, tokens["embed"])
                       + token_cost(prices, UTILITY_MODEL, tokens["hyde_input"], tokens["hyde_output"]))
    if "judge_input_tokens" in out:
        out["judge_cost_usd"] = token_cost(prices, JUDGE_MODEL, out["judge_input_tokens"], out["judge_output_tokens"])
    return out


# --- totals ---------------------------------------------------------------------------

def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    multihop = [r for r in answerable if r["type"] == "multihop"]
    out: dict[str, Any] = {"n": len(rows), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}

    for key in ("hit@1", "hit@3", "hit@5", "mrr@10", "recall@5"):
        out[key] = mean(_values(answerable, key))
    out["hit@5_ci95"] = bootstrap_ci(_values(answerable, "hit@5"))
    out["multihop_recall@5"] = mean(_values(multihop, "recall@5"))

    citing = _values(answerable, "citation_correctness")
    out["citation_correctness"] = mean(citing)
    out["no_citation_rate"] = (1 - len(citing) / len(answerable)) if answerable else None

    judged = [r for r in rows if "refused" in r]
    if judged:
        refused_unans = [r["refused"] for r in unanswerable if "refused" in r]
        refused_ans = [r["refused"] for r in answerable if "refused" in r]
        out["refusal_accuracy"] = mean(float(x) for x in refused_unans)
        out["false_refusal_rate"] = mean(float(x) for x in refused_ans)
        faith = _values(answerable, "faithfulness")
        out["faithfulness"] = mean(faith)
        out["faithfulness_ci95"] = bootstrap_ci(faith)
        out["answer_relevancy"] = mean(_values(answerable, "answer_relevancy"))
        out["n_generation_judged"] = len(faith)

    latency: dict[str, Any] = {}
    for stage in ("embed", "retrieve", "rerank", "generate", "total"):
        vals = [r["latency_ms"][stage] for r in rows if stage in r["latency_ms"]]
        latency[stage] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95)}
    out["latency_ms"] = latency
    out["tokens_per_query"] = {k: mean(r["tokens"][k] for r in rows) for k in (rows[0]["tokens"] if rows else {})}
    out["cost_usd_per_query"] = mean(r["cost_usd"] for r in rows)
    out["cost_usd_per_100_queries"] = out["cost_usd_per_query"] * 100 if rows else None
    out["judge_cost_usd_total"] = sum(r.get("judge_cost_usd", 0.0) for r in rows)
    return out


def summarize_by_type(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {t: summarize([r for r in rows if r["type"] == t]) for t in TYPES if any(r["type"] == t for r in rows)}


# --- driver -------------------------------------------------------------------------------

def estimate(rows: list[dict[str, Any]], cfg: RunConfig, svc: Services, prices: dict[str, Any]) -> dict[str, Any]:
    """Count uncached paid calls before making any. Retrieval runs first (cheap)."""
    pcfg = cfg.pipeline_config()
    need = {"query": 0, "answer": 0, "judge_row": 0}
    for row in rows:
        if pcfg.mode != "bm25" and svc.cache.get_text(_query_key(cfg, row["question"])) is None:
            need["query"] += 1
        hits = retrieve(row["question"], pcfg, svc.store, svc.embedder, svc.reranker)
        selected = [{"id": h.chunk_id} for h in hits[:pcfg.context_k]]
        raw = svc.cache.get_text(_answer_key(cfg, row["question"], selected))
        if raw is None:
            need["answer"] += 1
            if svc.judge is not None:
                need["judge_row"] += 1
        elif svc.judge is not None:
            selected_full = answer_request(row["question"], hits, pcfg)[2]
            key = _judge_key(svc, row, json.loads(raw)["text"], _contexts(selected_full))
            if svc.cache.get_text(key) is None:
                need["judge_row"] += 1
    usd = (need["answer"] * token_cost(prices, cfg.generator, EST_ANSWER_TOKENS["input"], EST_ANSWER_TOKENS["output"])
           + need["judge_row"] * token_cost(prices, JUDGE_MODEL, EST_JUDGE_TOKENS["input"], EST_JUDGE_TOKENS["output"]))
    return {"calls": need, "usd": usd}


def git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


def run(rows: list[dict[str, Any]], cfg: RunConfig, svc: Services, prices: dict[str, Any],
        progress: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    out_rows = []
    for i, row in enumerate(rows, start=1):
        out_rows.append(run_row(row, cfg, svc, prices))
        progress(f"[{i}/{len(rows)}] {row['id']}")
    return assemble(out_rows, cfg, svc, prices)


SCORED = ("hit@5", "mrr@10", "citation_correctness", "faithfulness", "answer_relevancy")


def row_scores(out: dict[str, Any]) -> list[Evaluation]:
    """The metrics run_row already computed, reported to Langfuse as scores (no new judge calls)."""
    scores = [Evaluation(name=key, value=float(out[key])) for key in SCORED if out.get(key) is not None]
    if "refused" in out:
        scores.append(Evaluation(name="refused", value=float(out["refused"])))
    return scores


def run_traced(rows: list[dict[str, Any]], cfg: RunConfig, svc: Services, prices: dict[str, Any],
               dataset, run_name: str, progress: Callable[[str], None] = lambda _: None) -> tuple[dict[str, Any], Any]:
    """Run the rows as a Langfuse experiment on the synced golden dataset.

    One trace per question: a `query` span holding retrieve, rerank, generate and
    judge, tagged with the config name and question id. The rows run one at a
    time, in golden order, through the same run_row as an untraced run.
    """
    by_item = {item_id(r["id"]): r for r in rows}
    missing = sorted(set(by_item) - {item.id for item in dataset.items})
    if missing:
        raise SystemExit(f"{len(missing)} golden rows aren't in the Langfuse dataset; run python -m eval.langfuse_sync")
    order = {key: i for i, key in enumerate(by_item)}
    dataset.items = sorted((item for item in dataset.items if item.id in by_item), key=lambda item: order[item.id])
    results: dict[str, dict[str, Any]] = {}

    def task(*, item, **kwargs):
        row = by_item[item.id]
        with (tracing.attributes(tags=[cfg.name, row["id"]], metadata={"config": cfg.name, "question_id": row["id"]}),
              tracing.stage("query", question_id=row["id"], type=row["type"]) as span):
            out = run_row(row, cfg, svc, prices)
            if span is not None:
                span.update(input=row["question"], output=out["answer"])
        results[row["id"]] = out
        progress(f"[{len(results)}/{len(rows)}] {row['id']}")
        return out["answer"]

    def scores(*, metadata=None, **kwargs):
        return row_scores(results[metadata["golden_id"]])

    experiment = dataset.run_experiment(
        name=cfg.name, run_name=run_name, task=task, evaluators=[scores], max_concurrency=1,
        description=f"eval.run --config {cfg.name} --trace",
        metadata={"config": cfg.name, "generator": cfg.generator, "judge": f"{JUDGE_MODEL} {JUDGE_VERSION}"},
    )
    tracing.flush()
    out_rows = [results[r["id"]] for r in rows if r["id"] in results]
    if len(out_rows) != len(rows):
        raise SystemExit(f"only {len(out_rows)} of {len(rows)} rows finished; see the Langfuse experiment")
    return assemble(out_rows, cfg, svc, prices), experiment


def assemble(out_rows: list[dict[str, Any]], cfg: RunConfig, svc: Services, prices: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": {
            "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
            "retrieval_depth": RETRIEVAL_DEPTH,
            "embedding_model": EMBEDDING_MODEL,
            "judge": {"model": JUDGE_MODEL, "version": JUDGE_VERSION,
                      "deployment": svc.judge.deployment, "temperature": 0} if svc.judge else None,
            "golden_sha256_16": golden_hash(),
            "prices_checked": prices.get("checked"),
            "git": git_state(),
            "paid_calls_this_run": dict(svc.paid_calls),
        },
        "totals": summarize(out_rows),
        "by_type": summarize_by_type(out_rows),
        "rows": out_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run")
    parser.add_argument("--config", required=True, help="name of a file in eval/configs/")
    parser.add_argument("--limit", type=int, help="only the first N golden rows")
    parser.add_argument("--ids", help="comma-separated golden ids, e.g. g001,g005")
    parser.add_argument("--no-judge", action="store_true", help="skip faithfulness, relevancy and refusal")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument("--out", help="output path (default results/<config>.json)")
    parser.add_argument("--trace", action="store_true",
                        help="trace to Langfuse as an experiment on the synced golden dataset; "
                             "recomputes answers and writes results/traced/<config>.json")
    args = parser.parse_args(argv)

    cfg = load_run_config(args.config)
    rows = load_golden()
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("no golden rows selected")

    prices = load_prices()
    if args.trace:
        # Traced runs make real calls so the spans show real latency, and write to
        # results/traced/ so the ablation results and the README tables stay as they are.
        tracing.require_keys()
        os.environ[tracing.FLAG] = "1"
    svc = real_services(cfg, with_judge=not args.no_judge, fresh=args.trace)
    est = estimate(rows, cfg, svc, prices)
    print(f"{cfg.name}: {len(rows)} rows; uncached calls {est['calls']}, about ${est['usd']:.2f}")
    if any(est["calls"].values()) and not args.yes and input("Run? [y/N] ").strip().lower() != "y":
        return 1

    partial = bool(args.limit or args.ids)
    name = f"{cfg.name}{'.partial' if partial else ''}.json"
    if args.trace:
        run_name = f"{cfg.name} {time.strftime('%Y-%m-%d %H:%M')}"
        dataset = tracing.client().get_dataset(DATASET)
        result, experiment = run_traced(rows, cfg, svc, prices, dataset, run_name,
                                        progress=lambda msg: print(msg, flush=True))
        result["meta"]["langfuse"] = {"dataset": DATASET, "run_name": run_name,
                                      "experiment_url": getattr(experiment, "experiment_url", None)}
        out = Path(args.out) if args.out else TRACED_DIR / name
    else:
        result = run(rows, cfg, svc, prices, progress=lambda msg: print(msg, flush=True))
        out = Path(args.out) if args.out else RESULTS_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    t = result["totals"]
    print(f"wrote {out}")
    print(f"hit@5 {t['hit@5']:.3f} {t['hit@5_ci95']}  mrr@10 {t['mrr@10']:.3f}  "
          f"faithfulness {t.get('faithfulness')}  p95 total {t['latency_ms']['total']['p95']:.0f} ms  "
          f"paid calls {result['meta']['paid_calls_this_run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
