import unittest

from app.gemini import rewrite_search_query


class RewriteSearchQueryTest(unittest.TestCase):
    def test_no_history_returns_query_unchanged(self):
        # No prior turns -> no LLM call, just the original query back.
        self.assertEqual(rewrite_search_query("at which timestamp", []), "at which timestamp")
        self.assertEqual(rewrite_search_query("hello", None), "hello")


if __name__ == "__main__":
    unittest.main()
