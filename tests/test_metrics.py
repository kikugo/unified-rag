"""Metric checks against examples worked out by hand."""
import unittest

from eval.metrics import (
    bootstrap_ci,
    citation_correctness,
    cited_ids,
    hit_at_k,
    mean,
    mrr_at_k,
    percentile,
    recall_at_k,
)

RANKED = ["a", "g1", "b", "c", "g2", "d"]


class RetrievalMetricTest(unittest.TestCase):
    def test_hit_at_k(self):
        # first gold chunk sits at rank 2
        self.assertEqual(hit_at_k(RANKED, ["g1"], 1), 0.0)
        self.assertEqual(hit_at_k(RANKED, ["g1"], 3), 1.0)
        self.assertEqual(hit_at_k(RANKED, ["zz"], 5), 0.0)
        self.assertEqual(hit_at_k(RANKED, ["g1"], 50), 1.0)  # k past the end of the list

    def test_mrr(self):
        self.assertEqual(mrr_at_k(RANKED, ["g1"]), 0.5)
        self.assertEqual(mrr_at_k(RANKED, ["g2", "g1"]), 0.5)  # first gold wins
        self.assertAlmostEqual(mrr_at_k(RANKED, ["g2"]), 0.2)
        self.assertEqual(mrr_at_k(RANKED, ["g2"], k=4), 0.0)
        self.assertEqual(mrr_at_k(RANKED, ["zz"]), 0.0)

    def test_recall(self):
        self.assertEqual(recall_at_k(RANKED, ["g1", "g2"], 5), 1.0)
        self.assertEqual(recall_at_k(RANKED, ["g1", "g2"], 3), 0.5)
        self.assertEqual(recall_at_k(RANKED, ["g1", "g2", "zz"], 5), 2 / 3)

    def test_duplicate_ids_in_ranking_count_once(self):
        ranked = ["a", "a", "a", "g1"]
        self.assertEqual(hit_at_k(ranked, ["g1"], 2), 1.0)
        self.assertEqual(mrr_at_k(ranked, ["g1"]), 0.5)

    def test_empty_gold_is_an_error(self):
        for fn in (lambda: hit_at_k(RANKED, [], 5), lambda: mrr_at_k(RANKED, []),
                   lambda: recall_at_k(RANKED, [])):
            with self.assertRaises(ValueError):
                fn()


SELECTED = [{"citation_id": "S1", "id": "g1"}, {"citation_id": "S2", "id": "a"},
            {"citation_id": "S3", "id": "g2"}]


class CitationTest(unittest.TestCase):
    def test_cited_ids_in_order_without_repeats(self):
        self.assertEqual(cited_ids("x [S3] y [S1] z [S3]", SELECTED), ["g2", "g1"])

    def test_grouped_citations(self):
        self.assertEqual(cited_ids("both [S1, S3] and [S2,S1]", SELECTED), ["g1", "g2", "a"])
        self.assertEqual(cited_ids("not a citation [S1 and S2]", SELECTED), [])

    def test_correctness(self):
        self.assertEqual(citation_correctness("Fines [S1].", SELECTED, ["g1"]), 1.0)
        self.assertEqual(citation_correctness("[S1] and [S2].", SELECTED, ["g1"]), 0.5)
        self.assertEqual(citation_correctness("[S2]", SELECTED, ["g1"]), 0.0)

    def test_invented_source_counts_as_wrong(self):
        self.assertEqual(citation_correctness("[S1] [S7]", SELECTED, ["g1"]), 0.5)

    def test_no_citation_is_none_not_zero(self):
        self.assertIsNone(citation_correctness("No sources cited.", SELECTED, ["g1"]))


class StatsTest(unittest.TestCase):
    def test_mean_and_percentile(self):
        self.assertIsNone(mean([]))
        self.assertEqual(mean([1, 2, 3]), 2)
        values = [10, 20, 30, 40, 50]
        self.assertEqual(percentile(values, 50), 30)
        self.assertEqual(percentile(values, 95), 48)  # 40 + 0.8 * 10
        self.assertEqual(percentile([7], 95), 7)
        self.assertIsNone(percentile([], 50))

    def test_bootstrap_is_seeded_and_brackets_the_mean(self):
        values = [1.0] * 30 + [0.0] * 70
        lo, hi = bootstrap_ci(values, n=2000, seed=3)
        self.assertEqual((lo, hi), bootstrap_ci(values, n=2000, seed=3))
        self.assertLess(lo, 0.3)
        self.assertGreater(hi, 0.3)
        self.assertLess(hi - lo, 0.25)

    def test_bootstrap_of_constant_is_a_point(self):
        self.assertEqual(bootstrap_ci([1.0] * 10, n=500), (1.0, 1.0))
        self.assertIsNone(bootstrap_ci([]))


if __name__ == "__main__":
    unittest.main()
