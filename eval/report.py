"""Build the README results tables from results/*.json.

    uv run python -m eval.report            # rewrite the tables in README.md
    uv run python -m eval.report --check    # exit 1 if README.md is out of date

Everything between <!-- RESULTS:START --> and <!-- RESULTS:END --> is generated;
nothing outside the markers is touched. Every number in the tables comes from a
results file, and the same files always produce byte-identical output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.golden import TYPES
from eval.metrics import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
BASELINE = "hybrid_rrf_rerank"
ORDER = ["dense", "bm25", "hybrid_rrf", "hybrid_rrf_rerank", "hybrid_rrf_rerank_hyde", "hybrid_rrf_rerank_ctx1"]
LABELS = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid_rrf": "Hybrid (RRF)",
    "hybrid_rrf_rerank": "Hybrid + rerank",
    "hybrid_rrf_rerank_hyde": "Hybrid + rerank + HyDE",
    "hybrid_rrf_rerank_ctx1": "Hybrid + rerank, 1 context chunk",
}


class ReportError(ValueError):
    pass


def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        results[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        raise ReportError(f"no results in {results_dir}; run python -m eval.run --config <name> first")
    _check_comparable(results)
    return dict(sorted(results.items(), key=lambda kv: (kv[0] not in ORDER, ORDER.index(kv[0])
                                                        if kv[0] in ORDER else 0, kv[0])))


def _check_comparable(results: dict[str, dict[str, Any]]) -> None:
    def signature(r):
        judge = r["meta"].get("judge") or {}
        return (r["meta"]["golden_sha256_16"], judge.get("model"), judge.get("version"),
                tuple(row["id"] for row in r["rows"]))
    signatures = {name: signature(r) for name, r in results.items()}
    if len(set(signatures.values())) > 1:
        detail = "; ".join(f"{n}: golden {s[0]}, judge {s[1]} {s[2]}, {len(s[3])} rows"
                           for n, s in signatures.items())
        raise ReportError(f"results are not comparable (golden set, judge or rows differ): {detail}")


def _f(x: float | None, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _ci(value: float | None, ci: list[float] | None) -> str:
    if value is None:
        return "–"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{value:.3f}"


def _delta(d: tuple[float, float, float] | None) -> str:
    if d is None:
        return "–"
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def _ms(x: float | None) -> str:
    return "–" if x is None else f"{x:,.0f}"


def _paired(base: dict[str, Any], other: dict[str, Any], key: str) -> tuple[float, float, float] | None:
    """Paired bootstrap of `key` over the questions where both configs have a value."""
    b = {r["id"]: r.get(key) for r in base["rows"]}
    pairs = [(b[r["id"]], r.get(key)) for r in other["rows"]
             if b.get(r["id"]) is not None and r.get(key) is not None]
    if not pairs:
        return None
    return paired_bootstrap_ci([p[0] for p in pairs], [p[1] for p in pairs])


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def render(results: dict[str, dict[str, Any]]) -> str:
    base = results.get(BASELINE)
    any_result = next(iter(results.values()))
    meta, totals = any_result["meta"], any_result["totals"]

    retrieval, answers, by_type = [], [], []
    for name, r in results.items():
        t = r["totals"]
        label = LABELS.get(name, name)
        is_base = name == BASELINE
        retrieval.append([
            label, _f(t["hit@1"]), _ci(t["hit@5"], t["hit@5_ci95"]),
            "baseline" if is_base else (_delta(_paired(base, r, "hit@5")) if base else "–"),
            _f(t["mrr@10"]), _f(t["multihop_recall@5"]),
        ])
        lat = t["latency_ms"]
        answers.append([
            label, _ci(t.get("faithfulness"), t.get("faithfulness_ci95")),
            "baseline" if is_base else (_delta(_paired(base, r, "faithfulness")) if base else "–"),
            _f(t.get("answer_relevancy")), _f(t["citation_correctness"]),
            _f(t.get("refusal_accuracy")), _f(t.get("false_refusal_rate")),
            _ms(lat["retrieve"]["p95"]),
            _ms(lat["rerank"]["p95"]) if r["meta"]["config"].get("rerank", True) else "–",
            _ms(lat["total"]["p95"]),
            f"${t['cost_usd_per_100_queries']:.2f}",
        ])
        cells = [label]
        for kind in TYPES:
            if kind == "unanswerable":
                continue
            s = r["by_type"].get(kind)
            cells.append("–" if not s else f"{_f(s['hit@5'])} / {_f(s['mrr@10'])}")
        by_type.append(cells)

    judge = meta.get("judge") or {}
    type_counts = ", ".join(f"{s['n']} {kind}" for kind, s in any_result["by_type"].items())
    lines = [
        START,
        "<!-- Generated by `python -m eval.report` from results/*.json. Do not edit by hand. -->",
        "",
        (f"{totals['n']} golden questions ({type_counts}). Retrieval metrics use the "
         f"{totals['n_answerable']} answerable ones. Generator `{meta['config']['generator']}`, "
         f"judge `{judge.get('model')}` version `{judge.get('version')}` at temperature {judge.get('temperature')}. "
         f"Δ columns are paired bootstrap differences against Hybrid + rerank, with 95% intervals: "
         f"an interval that excludes zero is a difference these questions can tell apart."),
        "",
        "**Retrieval**",
        "",
        *_table(["Config", "hit@1", "hit@5 [95% CI]", "Δ hit@5", "MRR@10", "Multihop recall@5"], retrieval),
        "",
        "**Answers, latency and cost**",
        "",
        *_table(["Config", "Faithfulness [95% CI]", "Δ faithfulness", "Answer relevancy", "Citation correctness",
                 "Refusal accuracy", "False refusals", "p95 retrieve ms", "p95 rerank ms", "p95 total ms",
                 "Cost / 100 queries"], answers),
        "",
        "**hit@5 / MRR@10 by question type**",
        "",
        *_table(["Config", *[k for k in TYPES if k != "unanswerable"]], by_type),
        "",
        (f"Cost uses the prices in `eval/prices.json` (checked {meta.get('prices_checked')}) and excludes the judge. "
         f"Latency is per query, measured on the original call and replayed from the cache on reruns."),
        END,
    ]
    return "\n".join(lines)


def splice(readme: str, block: str) -> str:
    start, end = readme.find(START), readme.find(END)
    if start < 0 or end < start:
        raise ReportError(f"README needs a {START} ... {END} block to write the tables into")
    return readme[:start] + block + readme[end + len(END):]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.report")
    parser.add_argument("--results", default=str(RESULTS_DIR))
    parser.add_argument("--readme", default=str(README))
    parser.add_argument("--check", action="store_true", help="fail if the README tables are out of date")
    args = parser.parse_args(argv)

    readme_path = Path(args.readme)
    current = readme_path.read_text(encoding="utf-8")
    updated = splice(current, render(load_results(Path(args.results))))
    if args.check:
        if updated != current:
            print("README results tables are out of date; run python -m eval.report", file=sys.stderr)
            return 1
        return 0
    if updated != current:
        readme_path.write_text(updated, encoding="utf-8")
        print(f"updated {readme_path}")
    else:
        print(f"{readme_path} already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
