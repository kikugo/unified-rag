"""Scoring functions for one eval row, plus the summary statistics.

Pure functions with no I/O, so every number in `results/*.json` can be traced
to a hand-checkable rule. Retrieval metrics only make sense for answerable rows;
the runner never calls them on unanswerable ones (they have no gold ids).
"""
from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable, Sequence

# "[S1]", and the grouped form the model also writes: "[S1, S2]"
CITATION_RE = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")


def _dedupe(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for chunk_id in ranked:
        if chunk_id not in seen:
            seen.add(chunk_id)
            out.append(chunk_id)
    return out


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold chunk is in the top k, else 0.0."""
    gold = set(gold)
    if not gold:
        raise ValueError("hit@k needs at least one gold chunk")
    return 1.0 if gold & set(_dedupe(ranked)[:k]) else 0.0


def mrr_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = 10) -> float:
    """Reciprocal rank of the first gold chunk within the top k, else 0.0."""
    gold = set(gold)
    if not gold:
        raise ValueError("MRR needs at least one gold chunk")
    for rank, chunk_id in enumerate(_dedupe(ranked)[:k], start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = 5) -> float:
    """Share of gold chunks found in the top k (the multihop metric)."""
    gold = set(gold)
    if not gold:
        raise ValueError("recall@k needs at least one gold chunk")
    return len(gold & set(_dedupe(ranked)[:k])) / len(gold)


def cited_ids(answer: str, selected: Sequence[dict]) -> list[str]:
    """Chunk ids behind the [S<n>] markers in an answer, in first-cited order.

    A marker with no matching source (the model invented [S7] with three
    sources) maps to None, so it counts against citation correctness.
    """
    by_citation = {row["citation_id"]: row["id"] for row in selected}
    out: list[str | None] = []
    markers = [cid.strip() for group in CITATION_RE.findall(answer or "") for cid in group.split(",")]
    for cid in dict.fromkeys(markers):
        out.append(by_citation.get(cid))
    return out


def citation_correctness(answer: str, selected: Sequence[dict], gold: Iterable[str]) -> float | None:
    """Share of cited sources that are gold chunks; None when nothing is cited."""
    cited = cited_ids(answer, selected)
    if not cited:
        return None
    gold = set(gold)
    return sum(1 for c in cited if c is not None and c in gold) / len(cited)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def percentile(values: Iterable[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default), p in [0, 100]."""
    values = sorted(values)
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    pos = (len(values) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_ci(values: Sequence[float], n: int = 10_000, seed: int = 0, level: float = 0.95
                 ) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for the mean."""
    values = list(values)
    if not values:
        return None
    rng = random.Random(seed)
    size = len(values)
    means = sorted(sum(rng.choices(values, k=size)) / size for _ in range(n))
    tail = (1 - level) / 2 * 100
    return percentile(means, tail), percentile(means, 100 - tail)


def paired_bootstrap_ci(baseline: Sequence[float], other: Sequence[float], n: int = 10_000, seed: int = 0,
                        level: float = 0.95) -> tuple[float, float, float] | None:
    """Mean of `other - baseline` over the same questions, with a percentile bootstrap CI.

    Pairing removes the between-question variance that makes two unpaired CIs
    overlap even when one config wins on almost every question.
    """
    if len(baseline) != len(other):
        raise ValueError("paired values need the same length")
    diffs = [b - a for a, b in zip(baseline, other)]
    ci = bootstrap_ci(diffs, n=n, seed=seed, level=level)
    if ci is None:
        return None
    return sum(diffs) / len(diffs), ci[0], ci[1]
