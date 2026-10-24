"""Terminal review of drafted candidates into the golden set.

    uv run python -m eval.review

Shows one candidate at a time with its source text, evidence highlighted.
k keep · e edit then keep · d drop · s skip for now · q quit.

Every decision is appended to eval/review_log.jsonl, so quitting loses nothing
and the next run picks up where you stopped. Kept rows are appended to
eval/golden.jsonl and never rewritten. A type stops being offered once it hits
its target in TARGET_MIX.

Only unanswerable candidates show what the system retrieves (top 5, hybrid +
rerank), so you can check nothing in the corpus answers them. Answerable rows
never show system output, so it can't anchor your judgment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from eval.draft_questions import CANDIDATES_PATH, read_candidates
from eval.golden import EVAL_DIR, GOLDEN_PATH, TARGET_MIX, TYPES, GoldenSetError, load_golden, mix, validate_row
from rag.cache import DiskCache
from rag.gemini_client import GeminiEmbedder
from rag.index import load_chunks, open_store
from rag.pipeline import Config, retrieve
from rag.rerank import CrossEncoderReranker

LOG_PATH = EVAL_DIR / "review_log.jsonl"
BOLD, YELLOW, RED, DIM, RESET = "\033[1m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def highlight(text: str, quotes: list[str]) -> str:
    for quote in quotes:
        if quote and quote in text:
            text = text.replace(quote, f"{YELLOW}{BOLD}{quote}{RESET}", 1)
    return text


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def next_golden_id(rows: list[dict[str, Any]]) -> str:
    nums = [int(r["id"][1:]) for r in rows if r["id"][1:].isdigit()]
    return f"g{(max(nums) if nums else 0) + 1:03d}"


class Reviewer:
    def __init__(
        self,
        candidates: list[dict[str, Any]],
        chunks_by_id: dict[str, dict[str, Any]],
        golden_path: Path = GOLDEN_PATH,
        log_path: Path = LOG_PATH,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        preview_fn: Callable[[str], list[dict[str, Any]]] | None = None,
        targets: dict[str, int] = TARGET_MIX,
    ):
        self.candidates = candidates
        self.chunks = chunks_by_id
        self.golden_path = golden_path
        self.log_path = log_path
        self.ask = input_fn
        self.say = output_fn
        self.preview_fn = preview_fn
        self.targets = targets

    def pending(self) -> list[dict[str, Any]]:
        decided = {e["candidate_id"] for e in read_log(self.log_path)}
        return [c for c in self.candidates if c["candidate_id"] not in decided]

    def _header(self, golden: list[dict[str, Any]], remaining: int) -> str:
        counts = mix(golden)
        parts = [f"{t} {counts[t]}/{self.targets[t]}" for t in TYPES]
        return f"{BOLD}golden {len(golden)}/{sum(self.targets.values())}{RESET} · " + " · ".join(parts) \
            + f" · {remaining} candidates left"

    def _show(self, cand: dict[str, Any]) -> None:
        self.say("")
        self.say(f"{BOLD}[{cand['candidate_id']}] {cand['type'].upper()}{RESET}")
        self.say(f"{BOLD}Q:{RESET} {cand['question']}")
        if cand["type"] != "unanswerable":
            self.say(f"{BOLD}A:{RESET} {cand['answer']}")
        if cand["flags"]:
            self.say(f"{RED}flags: {', '.join(cand['flags'])}{RESET}")
        self.say(f"{DIM}gold chunks: {', '.join(cand['gold_chunk_ids']) or '(none)'}{RESET}")
        for n, cid in enumerate(cand["source_chunk_ids"], start=1):
            chunk = self.chunks.get(cid, {})
            quotes = [e["quote"] for e in cand["evidence"] if e.get("source") == n]
            self.say(f"\n{BOLD}SOURCE {n}{RESET} {DIM}{cid} · {chunk.get('title')} — {chunk.get('heading')}{RESET}")
            if chunk.get("image"):
                self.say(f"{DIM}image: corpus/{chunk['image']}{RESET}")
            self.say(highlight(chunk.get("text", "(chunk not found)"), quotes))
        if cand["type"] == "unanswerable" and self.preview_fn:
            self.say(f"\n{BOLD}What the system retrieves (check none of these answers it):{RESET}")
            for hit in self.preview_fn(cand["question"]):
                self.say(f"  {DIM}{hit['chunk_id']} · {hit['doc_id']} / {hit['page']}{RESET}  {hit['text'][:300]}")

    def _edit(self, cand: dict[str, Any]) -> dict[str, Any]:
        edited = dict(cand)
        question = self.ask(f"question [{cand['question']}]: ").strip()
        edited["question"] = question or cand["question"]
        if cand["type"] != "unanswerable":
            answer = self.ask(f"answer [{cand['answer']}]: ").strip()
            edited["answer"] = answer or cand["answer"]
        new_type = self.ask(f"type {'/'.join(TYPES)} [{cand['type']}]: ").strip()
        edited["type"] = new_type or cand["type"]
        if edited["type"] == "unanswerable":
            edited["answer"], edited["gold_chunk_ids"] = "", []
        return edited

    def _golden_row(self, cand: dict[str, Any], golden: list[dict[str, Any]]) -> dict[str, Any]:
        row = {
            "id": next_golden_id(golden),
            "question": cand["question"],
            "gold_chunk_ids": list(cand["gold_chunk_ids"]),
            "gold_answer": cand["answer"],
            "type": cand["type"],
            "answerable": cand["type"] != "unanswerable",
            "source_candidate": cand["candidate_id"],
        }
        validate_row(row)
        return row

    def run(self) -> int:
        kept = 0
        for cand in self.pending():
            golden = load_golden(self.golden_path)
            if mix(golden)[cand["type"]] >= self.targets[cand["type"]]:
                continue
            if len(golden) >= sum(self.targets.values()):
                break
            self.say("\n" + "─" * 80)
            self.say(self._header(golden, len(self.pending())))
            self._show(cand)
            while True:
                choice = self.ask("\n[k]eep [e]dit [d]rop [s]kip [q]uit > ").strip().lower()
                if choice == "q":
                    self.say(f"saved; {kept} kept this session")
                    return kept
                if choice == "s":
                    break
                if choice == "d":
                    append_jsonl(self.log_path, {"candidate_id": cand["candidate_id"], "decision": "drop"})
                    break
                if choice in ("k", "e"):
                    final = self._edit(cand) if choice == "e" else cand
                    if final["type"] not in TYPES:
                        self.say(f"{RED}unknown type {final['type']!r}{RESET}")
                        continue
                    if mix(golden)[final["type"]] >= self.targets[final["type"]]:
                        self.say(f"{RED}{final['type']} is already at its target; drop or pick another type{RESET}")
                        continue
                    try:
                        row = self._golden_row(final, golden)
                    except GoldenSetError as exc:
                        self.say(f"{RED}can't keep this as is: {exc}{RESET}")
                        continue
                    append_jsonl(self.golden_path, row)
                    entry = {"candidate_id": cand["candidate_id"], "decision": "edit" if choice == "e" else "keep",
                             "golden_id": row["id"]}
                    if choice == "e":
                        entry["edited"] = {k: final[k] for k in ("question", "answer", "type") if final[k] != cand[k]}
                    append_jsonl(self.log_path, entry)
                    kept += 1
                    break
                self.say("type k, e, d, s or q")
        golden = load_golden(self.golden_path)
        self.say("\n" + self._header(golden, len(self.pending())))
        self.say(f"{kept} kept this session")
        return kept


def _system_preview() -> Callable[[str], list[dict[str, Any]]]:
    """Top 5 hybrid + rerank hits from the bench index, built on first use."""
    state: dict[str, Any] = {}

    def preview(question: str) -> list[dict[str, Any]]:
        if not state:
            state.update(store=open_store(), embedder=GeminiEmbedder(DiskCache()), reranker=CrossEncoderReranker())
        hits = retrieve(question, Config(mode="hybrid_rrf", rerank=True, top_k=5, context_k=5),
                        state["store"], state["embedder"], state["reranker"])
        return [{"chunk_id": h.chunk_id, "doc_id": h.metadata.get("doc_id"), "page": h.metadata.get("page"),
                 "text": h.metadata.get("text", "")} for h in hits]

    return preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.review")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    parser.add_argument("--no-preview", action="store_true", help="don't query the index for unanswerable rows")
    args = parser.parse_args(argv)

    chunks = {c["chunk_id"]: c for c in load_chunks()}
    reviewer = Reviewer(read_candidates(args.candidates), chunks, args.golden, args.log,
                        preview_fn=None if args.no_preview else _system_preview())
    reviewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
