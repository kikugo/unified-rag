"""The demo's first screen, run headless with a fake key: no Gemini call happens before an upload."""

import os
import unittest
from unittest import mock

from streamlit.testing.v1 import AppTest

from app.ui import SAMPLE_QUESTION


class DemoFirstLoadTest(unittest.TestCase):
    def setUp(self):
        env = {"DEMO_MODE": "true", "GOOGLE_API_KEY": "fake-demo-key", "DEMO_DAILY_CALL_LIMIT": "1000"}
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_empty_library_offers_upload_samples_quota_and_an_example(self):
        at = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(at.exception, at.exception)

        # Upload and samples live in the main column, not only in the (closed on a phone) sidebar.
        main_buttons = {b.key for b in at.main.button}
        self.assertIn("welcome_load_samples", main_buttons)
        self.assertIn("sample_ask_live", main_buttons)
        self.assertTrue(any(u.proto.id.endswith("-main_uploader") for u in at.main.get("file_uploader")))

        main_text = " ".join(m.value for m in at.main.markdown)
        self.assertIn(SAMPLE_QUESTION, main_text)
        self.assertNotIn("from the sidebar", main_text)

        self.assertTrue(any("calls left today" in p.proto.text for p in at.main.get("progress")))

        # The remote Chroma fields are not offered on the demo.
        self.assertFalse([t for t in at.sidebar.text_input if t.key and t.key.startswith("chroma_")])


if __name__ == "__main__":
    unittest.main()
