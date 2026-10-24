"""Draft golden-set candidates from sampled corpus chunks.

    uv run python -m eval.draft_questions --dry-run   # sample + cost estimate, no API
    uv run python -m eval.draft_questions             # draft into eval/candidates.csv

An LLM reads one chunk (two for multihop) and writes a question, an answer and
verbatim evidence quotes. Gold chunk ids are then derived from where those
quotes actually sit in the committed corpus, not from which chunk was sampled,
so an answer that falls in the 200-character overlap between two windows gets
both windows as gold. Every candidate still goes through `eval.review` by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from google.genai import types

from eval.golden import EVAL_DIR
from rag.cache import DiskCache, cache_key
from rag.corpus import CORPUS_DIR
from rag.gemini_client import make_client
from rag.index import load_chunks, read_figure
from rag.models import ANSWER_MODEL
from rag_core import bm25_rank, call_with_retry, tokenize_for_search

CANDIDATES_PATH = EVAL_DIR / "candidates.csv"
DRAFTER_MODEL = ANSWER_MODEL
# Multihop is drawn at 3x its target: about two thirds of pairs turn out to share no
# natural link, and the drafter is allowed to decline those.
DRAFT_COUNTS = {"factual": 60, "keyword": 30, "paraphrase": 30, "multihop": 32, "unanswerable": 16}
FIGURE_FACTUAL = 6
AI_ACT_ID = "eu-ai-act"
# Share of each type's draws taken from the AI Act; the rest rotate across the
# Wikipedia articles. Identifier questions lean on the regulation's numbering.
AI_ACT_SHARE = {"keyword": 0.4}
DEFAULT_AI_ACT_SHARE = 0.25
MIN_CHUNK_CHARS = 300
PARAPHRASE_OVERLAP_LIMIT = 0.3
# A chunk elsewhere in the corpus sharing at least this share of an evidence
# quote's 5-word shingles is offered to the reviewer as a possible extra gold
# chunk. Legal text restates itself (a recital and its article, one amendment
# applied to several regulations), so an answer often has more than one home.
NEAR_DUPLICATE_SHARE = 0.5
# Paid-tier prices for the drafter, per 1M tokens (ai.google.dev/gemini-api/docs/pricing, 2026-10-24).
PRICE_IN, PRICE_OUT = 0.75, 3.75
EST_OUTPUT_TOKENS = 700  # answer JSON plus thinking, a deliberate overestimate

IDENTIFIER_RE = re.compile(
    r"\b(Article\s+\d+[a-z]?(\(\d+\))?|Annex\s+[IVX]+|Regulation\s+\((EU|EC)\)\s+(No\s+)?\d+/\d+"
    r"|Directive\s+(\(EU\)\s+)?\d+/\d+|Executive Order\s+\d+|(19|20)\d{2}"
    r"|[A-Z][A-Za-z]*-?\d+[A-Za-z0-9.]*|[A-Z]{2,}\d*)\b"
)
STOPWORDS = set("""
a an the and or of to in on for by with from as at is are was were be been being this that these those
it its which who whom what when where why how does do did can could should would may might must shall will
not no than then there their they them he she his her we our you your i into about over under between
such any all each other more most some also only very much many one two
""".split())

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer", "description": "1 for SOURCE 1, 2 for SOURCE 2"},
                    "quote": {"type": "string", "description": "verbatim sentence(s) copied from that source"},
                },
                "required": ["source", "quote"],
            },
        },
    },
    "required": ["question", "answer", "evidence"],
}

COMMON_RULES = (
    "Write ONE question a curious reader could ask a search assistant over a library about AI and its regulation. "
    "The question must make sense on its own, without saying 'this text', 'the passage' or 'the source'. "
    "Copy every evidence quote character for character from the source: one to three consecutive sentences."
)
TYPE_RULES = {
    "factual": "Ask for a specific fact the source states. The answer is a short phrase or one sentence.",
    "keyword": (
        "The question must hinge on an exact identifier that appears in the source (an article or annex number, "
        "a regulation number, a year, a model or product name, a code) and must contain that identifier verbatim."
    ),
    "paraphrase": (
        "Ask for a fact the source states, but phrase the question so it reuses NONE of the source's content words: "
        "use synonyms and describe things indirectly. Names and numbers may not appear either."
    ),
    "multihop": (
        "Ask a question that can only be answered by combining a fact from SOURCE 1 with a fact from SOURCE 2, "
        "where combining them has a real reason: a comparison of two comparable things, a cause in one and its "
        "effect in the other, an entity named in one and described in the other, or a rule in one applied to a "
        "case in the other. Never join two unrelated facts with 'and'. Give one evidence quote from each source. "
        "If the sources share no natural link, return an empty question and empty evidence."
    ),
    "unanswerable": (
        "Ask a natural, specific question on this source's topic whose answer is NOT in the source and is "
        "unlikely to be in an encyclopedia article or the EU AI Act, for example an internal detail, a future "
        "figure or a private opinion. Set answer to an empty string and evidence to an empty list."
    ),
}
FIGURE_RULE = "The source is a figure (image plus caption). Ask about what the figure shows; quote the caption as evidence."


def content_words(text: str) -> set[str]:
    return {t for t in tokenize_for_search(text) if t not in STOPWORDS and len(t) > 2}


def overlap_score(question: str, source_text: str) -> float:
    """Share of the question's content words that also appear in the source."""
    q = content_words(question)
    return len(q & content_words(source_text)) / len(q) if q else 0.0


def identifiers(text: str) -> set[str]:
    return {m.group(0) for m in IDENTIFIER_RE.finditer(text)}


def _keyword_ids(text: str) -> set[str]:
    # Bare years and all-caps acronyms are too common to anchor a keyword
    # question on their own; prefer chunks with structured identifiers.
    return {i for i in identifiers(text) if not re.fullmatch(r"(19|20)\d{2}|[A-Z]{2,}", i)}


# --- sampling ------------------------------------------------------------------

def _eligible(chunk: dict[str, Any], kind: str) -> bool:
    if chunk["kind"] != "text" or len(chunk["text"]) < MIN_CHUNK_CHARS:
        return False
    return kind != "keyword" or bool(_keyword_ids(chunk["text"]))


def _draw(rng: random.Random, by_doc: dict[str, list[dict]], kind: str, n: int,
          used: set[str]) -> list[dict[str, Any]]:
    """n chunks for one type: a fixed AI Act share, the rest round-robin over articles."""
    ai_n = round(n * AI_ACT_SHARE.get(kind, DEFAULT_AI_ACT_SHARE))
    wiki_docs = sorted(d for d in by_doc if d != AI_ACT_ID)
    rng.shuffle(wiki_docs)
    picks: list[dict[str, Any]] = []

    def take(doc_id: str) -> bool:
        pool = [c for c in by_doc[doc_id] if c["chunk_id"] not in used and _eligible(c, kind)]
        if not pool:
            return False
        chunk = rng.choice(pool)
        used.add(chunk["chunk_id"])
        picks.append(chunk)
        return True

    for _ in range(ai_n):
        take(AI_ACT_ID)
    i = 0
    stalled = 0
    while len(picks) < n and stalled < len(wiki_docs):
        stalled = 0 if take(wiki_docs[i % len(wiki_docs)]) else stalled + 1
        i += 1
    return picks


def _partner(seed: dict[str, Any], text_chunks: list[dict[str, Any]], used: set[str]) -> dict[str, Any] | None:
    """Best BM25 match for the seed's topic from a different document."""
    query = " ".join(seed["text"].split()[:80])
    docs = [{"id": c["chunk_id"], "searchable_text": c["searchable_text"]}
            for c in text_chunks if c["doc_id"] != seed["doc_id"] and c["chunk_id"] not in used
            and len(c["text"]) >= MIN_CHUNK_CHARS]
    ranked = bm25_rank(query, docs)
    if not ranked:
        return None
    by_id = {c["chunk_id"]: c for c in text_chunks}
    return by_id[ranked[0][0]]


def sample_candidates(chunks: list[dict[str, Any]], seed: int = 7,
                      counts: dict[str, int] = DRAFT_COUNTS) -> list[dict[str, Any]]:
    """Deterministic list of candidate specs: {candidate_id, type, sources}."""
    rng = random.Random(seed)
    text_chunks = [c for c in chunks if c["kind"] == "text"]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in text_chunks:
        by_doc[c["doc_id"]].append(c)
    used: set[str] = set()
    specs: list[dict[str, Any]] = []

    figures = [c for c in chunks if c["kind"] == "figure"]
    for fig in rng.sample(figures, min(FIGURE_FACTUAL, len(figures), counts.get("factual", 0))):
        used.add(fig["chunk_id"])
        specs.append({"type": "factual", "sources": [fig]})

    for kind in ("factual", "keyword", "paraphrase", "unanswerable"):
        n = counts.get(kind, 0) - (sum(1 for s in specs if s["type"] == kind))
        specs.extend({"type": kind, "sources": [c]} for c in _draw(rng, by_doc, kind, n, used))

    for seed_chunk in _draw(rng, by_doc, "multihop", counts.get("multihop", 0), used):
        partner = _partner(seed_chunk, text_chunks, used)
        if partner:
            used.add(partner["chunk_id"])
            specs.append({"type": "multihop", "sources": [seed_chunk, partner]})

    for n, spec in enumerate(specs, start=1):
        spec["candidate_id"] = f"c{n:03d}"
    return specs


# --- prompting -----------------------------------------------------------------

def build_prompt(spec: dict[str, Any]) -> str:
    lines = [COMMON_RULES, TYPE_RULES[spec["type"]]]
    if spec["sources"][0]["kind"] == "figure":
        lines.append(FIGURE_RULE)
    lines.append("Reply as JSON with question, answer and evidence.")
    for n, src in enumerate(spec["sources"], start=1):
        lines += ["", f"SOURCE {n} ({src['title']} — {src['heading']}):", src["text"]]
    return "\n".join(lines)


def parse_draft(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("draft is not a JSON object")
    question = str(data.get("question", "")).strip()
    if not question:
        raise ValueError("draft has no question")
    evidence = data.get("evidence") or []
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list")
    clean = []
    for item in evidence:
        if not isinstance(item, dict) or "quote" not in item:
            raise ValueError("evidence items need a quote")
        clean.append({"source": int(item.get("source", 1)), "quote": str(item["quote"]).strip()})
    return {"question": question, "answer": str(data.get("answer", "")).strip(), "evidence": clean}


class GeminiDrafter:
    def __init__(self, cache: DiskCache, model: str = DRAFTER_MODEL):
        self.cache = cache
        self.model = model
        self._client = None
        self.calls = 0

    def __call__(self, prompt: str, images: list[tuple[bytes, str]]) -> str:
        key = cache_key(self.model, "DRAFT", prompt, *(b for b, _ in images))
        cached = self.cache.get_text(key)
        if cached is not None:
            return cached
        if self._client is None:
            self._client = make_client()
        parts = [types.Part.from_bytes(data=b, mime_type=m) for b, m in images] + [types.Part(text=prompt)]
        resp = call_with_retry(lambda: self._client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=SCHEMA,
            ),
        ), attempts=6, base_delay=5.0)
        self.calls += 1
        text = resp.text or ""
        self.cache.put_text(key, text)
        return text


# --- gold ids from evidence ----------------------------------------------------

def load_page_texts(corpus_dir: Path = CORPUS_DIR) -> dict[tuple[str, int], str]:
    pages = {}
    for path in sorted((corpus_dir / "docs").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            pages[(row["doc_id"], row["page"])] = row["text"]
    return pages


def _find_span(quote: str, text: str, lo: int = 0, hi: int | None = None) -> tuple[int, int] | None:
    """Where `quote` sits in text[lo:hi], as absolute offsets into `text`."""
    region = text[lo:hi]
    start = region.find(quote)
    if start >= 0:
        return lo + start, lo + start + len(quote)
    words = quote.split()
    if not words:
        return None
    # Tolerate whitespace differences (a newline between sections, double spaces).
    match = re.search(r"\s+".join(re.escape(w) for w in words), region)
    return (lo + match.start(), lo + match.end()) if match else None


def gold_ids_for_quote(quote: str, source: dict[str, Any], page_text: str,
                       page_chunks: list[dict[str, Any]]) -> list[str]:
    """Chunks on the source's page that hold the quote.

    Windows that contain the whole quote win; if the quote straddles a window
    boundary, every window it touches is gold. Empty when the quote is not in
    the page text at all.
    """
    if source["kind"] == "figure":
        return [source["chunk_id"]]
    # Search the window the drafter actually read first: legal text repeats
    # sentences, and the first match on the page can sit in a different window.
    lo = source.get("char_offset", 0)
    span = _find_span(quote, page_text, lo, lo + len(source.get("text", ""))) or _find_span(quote, page_text)
    if span is None:
        return []
    s, e = span
    windows = [(c["char_offset"], c["char_offset"] + len(c["text"]), c["chunk_id"]) for c in page_chunks]
    full = [cid for lo, hi, cid in windows if lo <= s and e <= hi]
    if full:
        return full
    return [cid for lo, hi, cid in windows if lo < e and hi > s]


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class DuplicateIndex:
    """Finds other chunks holding an evidence quote, verbatim or nearly."""

    def __init__(self, text_chunks: list[dict[str, Any]]):
        self.chunks = text_chunks
        self.norm = {c["chunk_id"]: _norm_ws(c["text"]) for c in text_chunks}
        self.shingles = {c["chunk_id"]: shingles(c["text"]) for c in text_chunks}

    def find(self, quote: str, exclude: set[str]) -> tuple[list[str], list[str]]:
        """(exact copies, near copies) of `quote` outside `exclude`."""
        q = _norm_ws(quote)
        qs = shingles(quote)
        exact, near = [], []
        for cid, text in self.norm.items():
            if cid in exclude:
                continue
            if q and q in text:
                exact.append(cid)
            elif qs and len(qs & self.shingles[cid]) / len(qs) >= NEAR_DUPLICATE_SHARE:
                near.append(cid)
        return exact, near


def finalize(spec: dict[str, Any], draft: dict[str, Any], pages: dict[tuple[str, int], str],
             chunks_by_page: dict[tuple[str, int], list[dict[str, Any]]],
             duplicates: DuplicateIndex | None = None) -> dict[str, Any]:
    flags: list[str] = []
    gold: list[str] = []
    alt: list[str] = []
    if spec["type"] != "unanswerable":
        sources_hit = set()
        for ev in draft["evidence"]:
            idx = ev["source"] - 1
            if not 0 <= idx < len(spec["sources"]):
                flags.append(f"bad_source_index={ev['source']}")
                continue
            src = spec["sources"][idx]
            key = (src["doc_id"], src["page"])
            ids = gold_ids_for_quote(ev["quote"], src, pages.get(key, ""), chunks_by_page.get(key, []))
            if not ids:
                flags.append(f"quote_not_found(source {ev['source']})")
            for cid in ids:
                if cid not in gold:
                    gold.append(cid)
            if ids:
                sources_hit.add(idx)
        if duplicates and gold:
            for ev in draft["evidence"]:
                exact, near = duplicates.find(ev["quote"], set(gold) | set(alt))
                for cid in exact:
                    gold.append(cid)
                if exact and "gold_has_exact_copy" not in flags:
                    flags.append("gold_has_exact_copy")
                alt.extend(cid for cid in near if cid not in gold and cid not in alt)
        if not draft["evidence"]:
            flags.append("no_evidence")
        if spec["type"] == "multihop" and len(sources_hit) < 2:
            flags.append("multihop_missing_a_source")
    source_text = " ".join(s["text"] for s in spec["sources"])
    if spec["type"] == "paraphrase":
        score = overlap_score(draft["question"], source_text)
        if score > PARAPHRASE_OVERLAP_LIMIT:
            flags.append(f"paraphrase_overlap={score:.2f}")
    if spec["type"] == "keyword":
        ids_in_q = identifiers(draft["question"]) & identifiers(source_text)
        if not ids_in_q:
            flags.append("keyword_identifier_missing")
    if spec["type"] == "unanswerable" and draft["answer"]:
        flags.append("unanswerable_has_answer")
    return {
        "candidate_id": spec["candidate_id"],
        "type": spec["type"],
        "question": draft["question"],
        "answer": "" if spec["type"] == "unanswerable" else draft["answer"],
        "gold_chunk_ids": gold,
        "alt_gold_candidates": alt if spec["type"] != "unanswerable" else [],
        "source_chunk_ids": [s["chunk_id"] for s in spec["sources"]],
        "evidence": draft["evidence"],
        "flags": flags,
        "drafter_model": DRAFTER_MODEL,
    }


# --- csv -----------------------------------------------------------------------

CSV_FIELDS = ["candidate_id", "type", "question", "answer", "gold_chunk_ids", "alt_gold_candidates", "source_chunk_ids",
              "evidence", "flags", "drafter_model"]
JSON_FIELDS = {"gold_chunk_ids", "alt_gold_candidates", "source_chunk_ids", "evidence", "flags"}


def write_candidates(rows: list[dict[str, Any]], path: Path = CANDIDATES_PATH) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if k in JSON_FIELDS else row[k]
                             for k in CSV_FIELDS})


def read_candidates(path: Path = CANDIDATES_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{k: json.loads(v) if k in JSON_FIELDS else v for k, v in row.items()}
                for row in csv.DictReader(handle)]


# --- command -------------------------------------------------------------------

def _images(spec: dict[str, Any]) -> list[tuple[bytes, str]]:
    return [read_figure(s) for s in spec["sources"] if s["kind"] == "figure"]


def estimate_cost(specs: list[dict[str, Any]]) -> tuple[int, float]:
    tokens_in = sum(len(build_prompt(s)) // 4 + 258 * sum(src["kind"] == "figure" for src in s["sources"])
                    for s in specs)
    tokens_out = EST_OUTPUT_TOKENS * len(specs)
    return tokens_in, tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT


def draft_all(specs: list[dict[str, Any]], drafter: Callable[[str, list], str],
              chunks: list[dict[str, Any]], image_loader=_images) -> list[dict[str, Any]]:
    pages = load_page_texts()
    chunks_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for c in chunks:
        if c["kind"] == "text":
            chunks_by_page[(c["doc_id"], c["page"])].append(c)
    duplicates = DuplicateIndex([c for c in chunks if c["kind"] == "text"])
    rows = []
    for spec in specs:
        raw = drafter(build_prompt(spec), image_loader(spec))
        try:
            draft = parse_draft(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"{spec['candidate_id']}: no usable draft ({exc}), skipped")
            continue
        rows.append(finalize(spec, draft, pages, chunks_by_page, duplicates))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.draft_questions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="show the sample and cost, call nothing")
    parser.add_argument("--out", type=Path, default=CANDIDATES_PATH)
    args = parser.parse_args(argv)

    chunks = load_chunks()
    specs = sample_candidates(chunks, seed=args.seed)
    per_type = defaultdict(int)
    per_doc = defaultdict(int)
    for s in specs:
        per_type[s["type"]] += 1
        for src in s["sources"]:
            per_doc[src["doc_id"]] += 1
    tokens_in, cost = estimate_cost(specs)
    print(f"{len(specs)} candidates: {dict(per_type)}")
    print(f"{len(per_doc)} documents covered; AI Act sources: {per_doc[AI_ACT_ID]}")
    print(f"~{tokens_in:,} input tokens, ~${cost:.2f} at paid-tier prices for {DRAFTER_MODEL}")
    if args.dry_run:
        return 0

    drafter = GeminiDrafter(DiskCache())
    rows = draft_all(specs, drafter, chunks)
    write_candidates(rows, args.out)
    flagged = sum(1 for r in rows if r["flags"])
    print(f"wrote {len(rows)} candidates to {args.out} ({flagged} flagged, {drafter.calls} API calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
