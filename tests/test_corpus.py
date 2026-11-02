import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import requests

from rag.corpus import CORPUS_DIR, _get, extract_wiki_html, fetch_figures, load_manifest, sha256_bytes

# Trimmed to the Parsoid shapes the extractor depends on: nested <section>s, a
# Commons figure, an enwiki-local figure, inline refs, an infobox table, a math
# span and a References section.
PARSOID_HTML = """
<html><body>
<section data-mw-section-id="0">
  <table class="infobox"><tr><td>Infobox noise</td></tr></table>
  <p>The <b>Example Act</b> is a law.<sup class="mw-ref reference"><a>[1]</a></sup> It applies widely.</p>
</section>
<section data-mw-section-id="1">
  <h2>Provisions</h2>
  <p>Article 5 bans some practices.</p>
  <figure typeof="mw:File/Thumb"><a><img resource="./File:Chart_one.png"
    src="//thumb.wikimedia.org/wikipedia/commons/thumb/a/ab/Chart_one.png/250px-Chart_one.png"/></a>
    <figcaption>Share of systems by risk tier</figcaption></figure>
  <figure typeof="mw:File/Thumb"><a><img resource="./File:Local_logo.png"
    src="//upload.wikimedia.org/wikipedia/en/a/ab/Local_logo.png"/></a>
    <figcaption>A non-free logo hosted on enwiki</figcaption></figure>
  <p>The loss is <span class="mwe-math-element"><math alttext="{\\displaystyle L=x^{2}}"><mi>L</mi></math></span> here.</p>
  <section data-mw-section-id="2">
    <h3>Penalties</h3>
    <p>Fines reach 7% of turnover.</p>
  </section>
</section>
<section data-mw-section-id="3">
  <h2>References</h2>
  <ol class="mw-references references"><li>Some citation</li></ol>
</section>
</body></html>
"""


class ExtractWikiHtmlTest(unittest.TestCase):
    def setUp(self):
        self.rows, self.figures = extract_wiki_html(PARSOID_HTML, max_figures=5)
        self.by_page = {r["page"]: r for r in self.rows}

    def test_sections_in_document_order_without_references(self):
        self.assertEqual([r["page"] for r in self.rows], [0, 1, 2])
        self.assertEqual(self.by_page[1]["heading"], "Provisions")
        self.assertEqual(self.by_page[2]["heading"], "Penalties")

    def test_parent_section_does_not_repeat_subsection_text(self):
        self.assertNotIn("Fines reach", self.by_page[1]["text"])
        self.assertIn("Fines reach 7% of turnover.", self.by_page[2]["text"])

    def test_refs_infobox_and_math_markup_are_cleaned(self):
        lead = self.by_page[0]["text"]
        self.assertEqual(lead, "The Example Act is a law. It applies widely.")
        self.assertIn("L=x^{2}", self.by_page[1]["text"])
        self.assertNotIn("Infobox", lead)

    def test_only_commons_figures_with_position(self):
        self.assertEqual(len(self.figures), 1)
        fig = self.figures[0]
        self.assertEqual(fig["file"], "File:Chart_one.png")
        self.assertEqual(fig["caption"], "Share of systems by risk tier")
        self.assertEqual(fig["page"], 1)
        self.assertEqual(fig["char_offset"], len("Article 5 bans some practices.") + 1)
        self.assertNotIn("risk tier", self.by_page[1]["text"])

    def test_max_figures_caps_the_list(self):
        _, figures = extract_wiki_html(PARSOID_HTML, max_figures=0)
        self.assertEqual(figures, [])


class ManifestTest(unittest.TestCase):
    def test_manifest_pins_forty_unique_documents(self):
        docs = load_manifest()["documents"]
        self.assertEqual(len(docs), 40)
        self.assertEqual(len({d["doc_id"] for d in docs}), 40)
        for doc in docs:
            if doc["source"] == "wikipedia":
                self.assertIn(str(doc["revision"]), doc["url"])

    def test_committed_text_matches_the_lock(self):
        lock_path = CORPUS_DIR / "manifest.lock.json"
        if not lock_path.exists():
            self.skipTest("corpus not fetched yet")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for doc_id, entry in lock["documents"].items():
            data = (CORPUS_DIR / "docs" / f"{doc_id}.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["text_sha256"], doc_id)

class FakeSession:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def get(self, url, timeout=None, **kwargs):
        self.calls += 1
        resp = requests.Response()
        resp.status_code = self.statuses.pop(0)
        resp._content = b"ok"
        resp.url = url
        return resp


class RetryTest(unittest.TestCase):
    def test_transient_server_errors_are_retried(self):
        for status in (500, 502, 503, 504, 429):
            session = FakeSession([status, 200])
            self.assertEqual(_get(session, "https://example.invalid", sleep=lambda _: None).content, b"ok")
            self.assertEqual(session.calls, 2)

    def test_client_errors_fail_at_once_and_retries_run_out(self):
        session = FakeSession([404])
        with self.assertRaises(requests.HTTPError):
            _get(session, "https://example.invalid", sleep=lambda _: None)
        self.assertEqual(session.calls, 1)
        session = FakeSession([500, 500, 500, 500])
        with self.assertRaises(requests.HTTPError):
            _get(session, "https://example.invalid", sleep=lambda _: None)
        self.assertEqual(session.calls, 4)

class FigureFetchTest(unittest.TestCase):
    def _corpus(self, tmp, images):
        root = Path(tmp)
        figs = [{"image": f"raw/figures/{name}", "thumb_url": f"https://example.invalid/{name}"} for name in images]
        (root / "figures.jsonl").write_text("".join(json.dumps(f) + "\n" for f in figs))
        lock = {"figures": {name: {"thumb_sha256": sha256_bytes(b"ok")} for name in images}}
        (root / "manifest.lock.json").write_text(json.dumps(lock))
        return root

    def test_downloads_missing_figures_and_skips_present_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._corpus(tmp, ["a-0.png", "b-0.png"])
            (root / "raw/figures").mkdir(parents=True)
            (root / "raw/figures/a-0.png").write_bytes(b"ok")
            session = FakeSession([200])
            self.assertEqual(fetch_figures(root, session), 0)
            self.assertEqual(session.calls, 1, "only the missing figure is downloaded")
            self.assertEqual((root / "raw/figures/b-0.png").read_bytes(), b"ok")


if __name__ == "__main__":
    unittest.main()
