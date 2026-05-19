import unittest

from rag_core import bm25_rank, expand_query_variants, reciprocal_rank_fusion


class RetrievalRegressionTest(unittest.TestCase):
    def test_exact_match_bm25_finds_keyword_doc(self):
        docs = [
            {"id": "a", "searchable_text": "unrelated quarterly memo"},
            {"id": "b", "searchable_text": "Tesla delivery count exceeded expectations"},
        ]
        ranked = bm25_rank("Tesla delivery", docs)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], "b")

    def test_rrf_prefers_docs_present_in_both_rankings(self):
        fused = reciprocal_rank_fusion([["x", "y", "z"], ["z", "x"]])
        self.assertEqual(fused[0], "x")
        self.assertIn("z", fused[:2])

    def test_expand_query_variants_for_short_query(self):
        variants = expand_query_variants("margin")
        self.assertTrue(any("margin" in v for v in variants))
        self.assertGreaterEqual(len(variants), 1)

    def test_timestamp_label_present_in_sparse_doc(self):
        doc = {
            "id": "clip-1",
            "searchable_text": "earnings.mp4 frame timestamp 1:05 earnings beat",
        }
        ranked = bm25_rank("earnings 1:05", [doc])
        self.assertEqual(ranked[0][0], "clip-1")


if __name__ == "__main__":
    unittest.main()
