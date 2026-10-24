"""Download and extract the fixed benchmark corpus.

    uv run python -m rag.corpus fetch           # download raw, extract, write lock
    uv run python -m rag.corpus fetch --verify  # re-download and diff against the lock

Sources are pinned in corpus/manifest.json: the AI Act as the Official Journal
PDF and Wikipedia articles by revision id. Raw downloads go to corpus/raw/
(gitignored). The extracted text in corpus/docs/ and corpus/figures.jsonl is
committed, because an old Wikipedia revision is re-rendered with today's
templates and its text can drift; committing it freezes the character offsets
that chunk ids are built from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import fitz
import requests
from bs4 import BeautifulSoup, Tag

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
USER_AGENT = "unified-rag-eval/0.1 (https://github.com/kikugo/unified-rag; karthikchangal39@gmail.com)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
REQUEST_PAUSE_SEC = 0.3

# Sections that hold citations or navigation rather than article content.
SKIP_HEADINGS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "sources", "citations", "works cited", "footnotes",
    "notes and references", "general and cited references", "explanatory notes",
}
# Elements removed before any text is read.
DROP_SELECTORS = [
    "style", "script", "table", "sup.reference", ".mw-ref", ".mw-references-wrap",
    ".reflist", ".navbox", ".hatnote", ".metadata", ".ambox", ".shortdescription",
    ".mw-editsection", ".noprint", ".gallery", "link", "meta",
]
# OJ page furniture repeated on every page of the AI Act PDF.
OJ_FURNITURE = re.compile(
    r"^(EN|OJ L, 12\.7\.2024|ELI: http://data\.europa\.eu/eli/reg/2024/1689/oj|\d{1,3}/144)$"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_manifest(corpus_dir: Path = CORPUS_DIR) -> dict[str, Any]:
    return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))


# --- extraction (pure, tested offline) ---------------------------------------

def extract_oj_pdf(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """One row per PDF page, page numbers 1-based, OJ headers and footers dropped."""
    rows = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc, start=1):
            raw = page.get_text("text", flags=fitz.TEXT_DEHYPHENATE)
            lines = [ln.strip() for ln in raw.splitlines()]
            kept = [ln for ln in lines if ln and not OJ_FURNITURE.match(ln)]
            text = collapse(" ".join(kept))
            if text:
                rows.append({"page": index, "heading": f"page {index}", "text": text})
    return rows


def _math_to_text(soup: BeautifulSoup) -> None:
    # Parsoid renders formulas as MathML plus a fallback image; get_text() on that
    # spews every MathML token. The `alttext` is the readable TeX source.
    for span in soup.select(".mwe-math-element"):
        math = span.find("math")
        alt = math.get("alttext", "") if isinstance(math, Tag) else ""
        span.replace_with(f" {alt} " if alt else " ")


def _heading_of(section: Tag) -> str:
    for child in section.children:
        if isinstance(child, Tag) and re.fullmatch(r"h[2-6]", child.name or ""):
            return collapse(child.get_text())
    return ""


def _figure_record(figure: Tag) -> dict[str, Any] | None:
    img = figure.find("img")
    caption_tag = figure.find("figcaption")
    if not isinstance(img, Tag) or not isinstance(caption_tag, Tag):
        return None
    src = img.get("src", "")
    # Only Commons-hosted files: enwiki-local images are often non-free.
    if "/wikipedia/commons/" not in src:
        return None
    caption = collapse(caption_tag.get_text())
    resource = img.get("resource", "")
    if len(caption) < 10 or not resource.startswith("./File:"):
        return None
    return {"file": "File:" + resource[len("./File:"):], "caption": caption}


def extract_wiki_html(html: str, max_figures: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split Parsoid HTML into section rows and figure records.

    `page` is the section's position in document order (0 is the lead). Nested
    sections are read deepest first and then detached, so a parent's text never
    repeats its subsections. A figure's `char_offset` is where it sits in its
    section's text, which keeps its chunk id tied to its position.
    """
    soup = BeautifulSoup(html, "html.parser")
    for selector in DROP_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    _math_to_text(soup)

    sections = soup.find_all("section")
    order = {id(sec): index for index, sec in enumerate(sections)}
    skipped: set[int] = set()
    for sec in sections:
        if _heading_of(sec).lower() in SKIP_HEADINGS:
            skipped.add(id(sec))
            skipped.update(id(s) for s in sec.find_all("section"))

    rows: dict[int, dict[str, Any]] = {}
    figures: list[dict[str, Any]] = []
    for sec in reversed(sections):
        page = order[id(sec)]
        heading = _heading_of(sec)
        parts: list[str] = []
        length = 0
        section_figures = []
        if id(sec) not in skipped:
            for child in list(sec.children):
                if not isinstance(child, Tag) or child.name == "section":
                    continue
                if child.name == "figure":
                    record = _figure_record(child)
                    if record:
                        section_figures.append({**record, "page": page, "char_offset": length})
                    continue
                for nested in child.find_all("figure"):
                    nested.decompose()
                if re.fullmatch(r"h[2-6]", child.name or ""):
                    continue
                text = collapse(child.get_text())
                if text:
                    parts.append(text)
                    length += len(text) + 1
        text = "\n".join(parts)
        if text:
            rows[page] = {"page": page, "heading": heading or "Introduction", "text": text}
        figures = section_figures + figures
        sec.decompose()

    ordered_rows = [rows[p] for p in sorted(rows)]
    figures.sort(key=lambda f: (f["page"], f["char_offset"]))
    return ordered_rows, figures[:max_figures]


# --- network -------------------------------------------------------------------

def _get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    for attempt in range(4):
        resp = session.get(url, timeout=60, **kwargs)
        if resp.status_code in (429, 503) and attempt < 3:
            time.sleep(2 ** attempt * 2)
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_PAUSE_SEC)
        return resp
    raise RuntimeError(f"gave up on {url}")


def fetch_raw(session: requests.Session, doc: dict[str, Any]) -> bytes:
    if doc["source"] == "eurlex":
        # eur-lex.europa.eu answers scripts with a 202 bot challenge; the
        # Publications Office Cellar serves the same OJ PDF by CELEX number.
        resp = _get(session, doc["url"], headers={"Accept": "application/pdf", "Accept-Language": "eng"})
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"{doc['doc_id']}: Cellar did not return a PDF")
        return resp.content
    return _get(session, doc["url"]).content


def resolve_figures(session: requests.Session, figures: list[dict[str, Any]], width: int) -> None:
    """Fill in the thumbnail url and Commons sha1 for each figure, in place."""
    titles = sorted({f["file"] for f in figures})
    info: dict[str, dict[str, Any]] = {}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        data = _get(session, WIKI_API, params={
            "action": "query", "format": "json", "formatversion": 2,
            "titles": "|".join(batch), "prop": "imageinfo",
            "iiprop": "url|sha1|mime", "iiurlwidth": width,
        }).json()
        normalized = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
        for page in data["query"]["pages"]:
            ii = (page.get("imageinfo") or [None])[0]
            if ii:
                info[normalized.get(page["title"], page["title"])] = ii
    for fig in figures:
        ii = info.get(fig["file"])
        fig["thumb_url"] = ii.get("thumburl") if ii else None
        fig["commons_sha1"] = ii.get("sha1") if ii else None


def thumb_ext(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix.lower()
    return ".jpg" if suffix in (".jpeg", ".jpg") else (suffix or ".png")


# --- commands ------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> bytes:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return data


def fetch(corpus_dir: Path = CORPUS_DIR, verify: bool = False) -> int:
    manifest = load_manifest(corpus_dir)
    raw_dir = corpus_dir / "raw"
    fig_dir = raw_dir / "figures"
    docs_dir = corpus_dir / "docs"
    for d in (raw_dir, fig_dir, docs_dir):
        d.mkdir(parents=True, exist_ok=True)
    lock_path = corpus_dir / "manifest.lock.json"
    old_lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    lock: dict[str, Any] = {"documents": {}, "figures": {}}
    all_figures: list[dict[str, Any]] = []
    extracted: dict[str, bytes] = {}

    for doc in manifest["documents"]:
        doc_id = doc["doc_id"]
        raw = fetch_raw(session, doc)
        ext = ".pdf" if doc["source"] == "eurlex" else ".html"
        (raw_dir / f"{doc_id}{ext}").write_bytes(raw)
        if doc["source"] == "eurlex":
            rows, figures = extract_oj_pdf(raw), []
        else:
            rows, figures = extract_wiki_html(raw.decode("utf-8"), manifest["max_figures_per_doc"])
        rows = [{"doc_id": doc_id, **r} for r in rows]
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
        extracted[doc_id] = data
        for n, fig in enumerate(figures):
            all_figures.append({"doc_id": doc_id, "figure_index": n, **fig})
        lock["documents"][doc_id] = {
            "raw_sha256": sha256_bytes(raw),
            "text_sha256": sha256_bytes(data),
            "rows": len(rows),
            "chars": sum(len(r["text"]) for r in rows),
        }
        print(f"{doc_id}: {len(rows)} rows, {len(figures)} figures", flush=True)

    resolve_figures(session, all_figures, manifest["figure_thumb_width"])
    kept_figures = []
    for fig in all_figures:
        if not fig.get("thumb_url"):
            print(f"  skipped figure without a thumbnail: {fig['file']}")
            continue
        img = _get(session, fig["thumb_url"]).content
        name = f"{fig['doc_id']}-{fig['figure_index']}{thumb_ext(fig['thumb_url'])}"
        (fig_dir / name).write_bytes(img)
        fig["image"] = f"raw/figures/{name}"
        lock["figures"][name] = {"thumb_sha256": sha256_bytes(img), "commons_sha1": fig["commons_sha1"]}
        kept_figures.append(fig)

    if verify:
        return _diff_lock(old_lock, lock)

    for doc_id, data in extracted.items():
        (docs_dir / f"{doc_id}.jsonl").write_bytes(data)
    write_jsonl(corpus_dir / "figures.jsonl", kept_figures)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(extracted)} documents, {len(kept_figures)} figures written")
    return 0


def _diff_lock(old: dict[str, Any], new: dict[str, Any]) -> int:
    if not old:
        print("no manifest.lock.json to verify against; run fetch first")
        return 1
    problems = []
    for section in ("documents", "figures"):
        for key in sorted(set(old.get(section, {})) | set(new.get(section, {}))):
            if old.get(section, {}).get(key) != new.get(section, {}).get(key):
                problems.append(f"{section}/{key}")
    for p in problems:
        print(f"changed: {p}")
    print("corpus matches the lock" if not problems else f"{len(problems)} entries differ from the lock")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_p = sub.add_parser("fetch", help="download the pinned sources and extract text")
    fetch_p.add_argument("--verify", action="store_true",
                         help="re-download and compare with manifest.lock.json without writing")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return fetch(verify=args.verify)
    return 2


if __name__ == "__main__":
    sys.exit(main())
