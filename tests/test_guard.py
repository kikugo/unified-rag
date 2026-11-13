import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fitz
from PIL import Image

from app import gemini
from app.guard import (
    UTILITY_MAX_OUTPUT_TOKENS,
    BudgetedClient,
    DailyBudget,
    DemoBudgetExceeded,
    UploadLimits,
    check_upload,
    clamp_question,
    friendly_error,
    js_string_literal,
)
from rag_core import call_with_retry, is_transient_error

LIMITS = UploadLimits(max_mb=1, max_pdf_pages=2, max_audio_seconds=1,
                      max_video_seconds=1, max_image_pixels=10_000)


def _pdf(pages: int) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("1", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def _wav(seconds: float, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class DailyBudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "budget.json"
        self.day = "2026-11-13"

    def tearDown(self):
        self.tmp.cleanup()

    def _budget(self, limit=2):
        return DailyBudget(limit, path=self.path, today=lambda: self.day)

    def test_spends_until_limit_then_raises(self):
        budget = self._budget()
        budget.spend()
        budget.spend()
        self.assertEqual(budget.remaining(), 0)
        with self.assertRaises(DemoBudgetExceeded):
            budget.spend()

    def test_count_survives_a_new_instance(self):
        # A reload or container restart must not hand out a fresh budget.
        self._budget().spend()
        self.assertEqual(self._budget().remaining(), 1)

    def test_resets_on_a_new_day(self):
        budget = self._budget()
        budget.spend()
        budget.spend()
        self.day = "2026-11-14"
        self.assertEqual(budget.remaining(), 2)

    def test_spent_budget_is_not_retried(self):
        self.assertFalse(is_transient_error(DemoBudgetExceeded()))
        calls = []

        def fn():
            calls.append(1)
            raise DemoBudgetExceeded()

        with self.assertRaises(DemoBudgetExceeded):
            call_with_retry(fn, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)


class BudgetedClientTest(unittest.TestCase):
    def test_charges_paid_calls_and_blocks_once_spent(self):
        inner = mock.MagicMock()
        budget = DailyBudget(1)
        client = BudgetedClient(inner, budget)

        client.models.generate_content(model="m", contents="hi")
        inner.models.generate_content.assert_called_once()
        with self.assertRaises(DemoBudgetExceeded):
            client.models.embed_content(model="m", contents="hi")
        inner.models.embed_content.assert_not_called()
        with self.assertRaises(DemoBudgetExceeded):
            client.file_search_stores.upload_to_file_search_store(file="x")

    def test_free_calls_pass_through(self):
        inner = mock.MagicMock()
        client = BudgetedClient(inner, DailyBudget(0))
        client.file_search_stores.delete(name="s")
        inner.file_search_stores.delete.assert_called_once_with(name="s")


class JsStringLiteralTest(unittest.TestCase):
    def test_cannot_close_the_script_tag(self):
        text = 'bye </script><script>alert(1)</script> <!-- x & y   '
        literal = js_string_literal(text)
        self.assertNotIn("</", literal)
        self.assertNotIn("<!--", literal)
        self.assertNotIn(" ", literal)
        self.assertNotIn(" ", literal)
        self.assertEqual(json.loads(literal), text)


class FriendlyErrorTest(unittest.TestCase):
    def test_budget(self):
        self.assertIn("today's budget", friendly_error(DemoBudgetExceeded()))

    def test_rate_limit_hides_raw_error(self):
        msg = friendly_error(Exception("429 RESOURCE_EXHAUSTED secret-detail"))
        self.assertIn("rate limited", msg)
        self.assertNotIn("secret-detail", msg)

    def test_bad_key(self):
        msg = friendly_error(Exception("400 INVALID_ARGUMENT API key not valid"), using_demo_key=False)
        self.assertIn("rejected the API key", msg)
        self.assertIn("rejected", friendly_error(Exception("403 PERMISSION_DENIED")))

    def test_unknown_error_is_generic(self):
        msg = friendly_error(ValueError("internal path /srv/app"))
        self.assertNotIn("/srv/app", msg)


class CheckUploadTest(unittest.TestCase):
    def test_size_cap(self):
        self.assertIn("over 1 MB", check_upload(b"x" * (1024 * 1024 + 1), "image/png", LIMITS))

    def test_pdf_pages(self):
        self.assertIsNone(check_upload(_pdf(2), "application/pdf", LIMITS))
        self.assertIn("3 pages", check_upload(_pdf(3), "application/pdf", LIMITS))
        self.assertIn("not a PDF", check_upload(b"not a pdf", "application/pdf", LIMITS))

    def test_image_pixels_from_header(self):
        self.assertIsNone(check_upload(_png(100, 100), "image/png", LIMITS))
        self.assertIn("megapixel", check_upload(_png(200, 100), "image/png", LIMITS))

    def test_audio_duration(self):
        self.assertIsNone(check_upload(_wav(0.5), "audio/wav", LIMITS))
        self.assertIn("longer than", check_upload(_wav(3), "audio/wav", LIMITS))

    def test_unknown_type(self):
        self.assertIsNotNone(check_upload(b"x", "text/html", LIMITS))

    def test_clamp_question(self):
        self.assertEqual(len(clamp_question("a" * 5000)), 2000)
        self.assertEqual(clamp_question("short"), "short")


class SessionClientTest(unittest.TestCase):
    def test_each_session_uses_its_own_client(self):
        # The client used to be a module global, so the last session to set it
        # served everyone. Now it lives in each session's state.
        session_a = {}
        session_b = {}
        client_a, client_b = mock.MagicMock(), mock.MagicMock()
        for client in (client_a, client_b):
            client.models.generate_content.return_value = SimpleNamespace(text="local", usage_metadata=None)

        with mock.patch.object(gemini, "st", SimpleNamespace(session_state=session_a)):
            gemini.set_client(client_a)
        with mock.patch.object(gemini, "st", SimpleNamespace(session_state=session_b)):
            gemini.set_client(client_b)
        with mock.patch.object(gemini, "st", SimpleNamespace(session_state=session_a)):
            self.assertEqual(gemini.route_query("q"), "local")

        client_a.models.generate_content.assert_called_once()
        client_b.models.generate_content.assert_not_called()
        config = client_a.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.max_output_tokens, UTILITY_MAX_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main()
