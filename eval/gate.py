"""The CI eval gate: compare a smoke run against the committed baseline.

    uv run python -m eval.gate --make-subset            # write eval/smoke_ids.json (done once)
    uv run python -m eval.gate --baseline results/baseline.json --current results/gate/current.json \\
        [--summary $GITHUB_STEP_SUMMARY]

A pull request fails when, on the smoke subset, hit@5 falls by more than
HIT5_DROP, faithfulness falls by more than FAITH_DROP, or the false-refusal
rate rises by more than FALSE_REFUSAL_RISE. The third check is there because
narrowing the answer context (context_k 3 to 1) keeps hit@5 and even raises
faithfulness, yet makes the model refuse answerable questions far more often.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from eval.golden import EVAL_DIR, TARGET_MIX, load_golden

SMOKE_PATH = EVAL_DIR / "smoke_ids.json"
SMOKE_SEED = 20
SMOKE_QUOTA = {"factual": 8, "keyword": 4, "paraphrase": 4, "multihop": 2, "unanswerable": 2}

HIT5_DROP = 0.05
FAITH_DROP = 0.05
FALSE_REFUSAL_RISE = 0.10


class GateError(ValueError):
    pass


def make_subset(rows: list[dict[str, Any]], quota: dict[str, int] = SMOKE_QUOTA,
                seed: int = SMOKE_SEED) -> list[str]:
    """A fixed, type-stratified sample of golden ids; same rows and seed, same ids."""
    rng = random.Random(seed)
    ids: list[str] = []
    for kind in TARGET_MIX:  # fixed type order, so the RNG draws are reproducible
        pool = sorted(r["id"] for r in rows if r["type"] == kind)
        k = quota.get(kind, 0)
        if k > len(pool):
            raise GateError(f"need {k} {kind} rows for the smoke subset, have {len(pool)}")
        ids += sorted(rng.sample(pool, k))
    return ids


def load_smoke_ids(path: Path = SMOKE_PATH) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))["ids"]


def _signature(result: dict[str, Any]) -> tuple:
    judge = result["meta"].get("judge") or {}
    return (result["meta"].get("golden_sha256_16"), judge.get("model"), judge.get("version"),
            tuple(r["id"] for r in result["rows"]))


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """One check per gated metric: values, change, limit and whether it passed."""
    if _signature(baseline) != _signature(current):
        raise GateError("baseline and current runs differ in golden set, judge or question ids; "
                        "they can't be compared")
    b, c = baseline["totals"], current["totals"]
    checks = []
    for metric, limit, worse_when in (
        ("hit@5", HIT5_DROP, "drop"),
        ("faithfulness", FAITH_DROP, "drop"),
        ("false_refusal_rate", FALSE_REFUSAL_RISE, "rise"),
    ):
        before, after = b.get(metric), c.get(metric)
        if before is None or after is None:
            raise GateError(f"{metric} is missing from a run; was the judge skipped?")
        change = after - before
        harm = -change if worse_when == "drop" else change
        # round so a change exactly at the limit isn't failed by float noise
        checks.append({"metric": metric, "baseline": before, "current": after, "change": change,
                       "limit": f"{worse_when} ≤ {limit:.2f}", "passed": round(harm, 9) <= limit})
    return checks


def changed_rows(baseline: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Questions whose hit@5, refusal or faithfulness differ between the runs."""
    base = {r["id"]: r for r in baseline["rows"]}
    out = []
    for row in current["rows"]:
        old = base[row["id"]]
        diffs = {k: (old.get(k), row.get(k)) for k in ("hit@5", "refused", "faithfulness")
                 if old.get(k) != row.get(k)}
        if diffs:
            out.append({"id": row["id"], "type": row["type"], **diffs})
    return out


def _fmt(x: Any) -> str:
    if isinstance(x, bool):
        return "yes" if x else "no"
    return "–" if x is None else f"{x:.3f}"


def render_summary(checks: list[dict[str, Any]], rows: list[dict[str, Any]], current: dict[str, Any]) -> str:
    passed = all(c["passed"] for c in checks)
    cfg = current["meta"]["config"]
    lines = [
        f"## Eval gate: {'passed' if passed else 'FAILED'}",
        "",
        f"Config `{cfg['name']}` (mode {cfg['mode']}, rerank {cfg['rerank']}, hyde {cfg['hyde']}, "
        f"context_k {cfg['context_k']}) on {len(current['rows'])} smoke questions, against `results/baseline.json`.",
        "",
        "| Metric | Baseline | Current | Change | Allowed | Result |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c['metric']} | {_fmt(c['baseline'])} | {_fmt(c['current'])} | {c['change']:+.3f} | "
                     f"{c['limit']} | {'pass' if c['passed'] else '**fail**'} |")
    if rows:
        lines += ["", "Questions that changed (baseline → current):", "",
                  "| Question | Type | hit@5 | Refused | Faithfulness |", "|---|---|---|---|---|"]
        for r in rows:
            cells = [" → ".join(_fmt(v) for v in r[k]) if k in r else "" for k in ("hit@5", "refused", "faithfulness")]
            lines.append(f"| {r['id']} | {r['type']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


RUN_METRICS = ("hit@1", "hit@5", "mrr@10", "citation_correctness", "faithfulness", "answer_relevancy",
               "refusal_accuracy", "false_refusal_rate")


def render_run(result: dict[str, Any]) -> str:
    """Totals of one run as a markdown table, for the manual full-run job (no pass/fail)."""
    t, cfg = result["totals"], result["meta"]["config"]
    lines = [f"## Eval run: `{cfg['name']}` on {t['n']} questions", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {m} | {_fmt(t.get(m))} |" for m in RUN_METRICS]
    lines.append(f"| p95 total latency ms | {t['latency_ms']['total']['p95']:,.0f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.gate")
    parser.add_argument("--make-subset", action="store_true", help=f"write {SMOKE_PATH.name} from the golden set")
    parser.add_argument("--baseline", default="results/baseline.json")
    parser.add_argument("--current", default="results/gate/current.json")
    parser.add_argument("--summary", help="append the markdown summary here (CI: $GITHUB_STEP_SUMMARY)")
    parser.add_argument("--run", help="summarize one results file instead of gating (manual full runs)")
    args = parser.parse_args(argv)

    if args.run:
        summary = render_run(json.loads(Path(args.run).read_text(encoding="utf-8")))
        print(summary)
        if args.summary:
            with open(args.summary, "a", encoding="utf-8") as fh:
                fh.write(summary)
        return 0

    if args.make_subset:
        ids = make_subset(load_golden())
        SMOKE_PATH.write_text(json.dumps({"seed": SMOKE_SEED, "quota": SMOKE_QUOTA, "ids": ids}, indent=1) + "\n",
                              encoding="utf-8")
        print(f"wrote {len(ids)} ids to {SMOKE_PATH}")
        return 0

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    checks = compare(baseline, current)
    summary = render_summary(checks, changed_rows(baseline, current), current)
    print(summary)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(summary)
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
