import unittest

from app.ui import (
    SAMPLE_ANSWER,
    SAMPLE_SOURCES,
    cited_ids,
    ingest_fraction,
    md_escape,
    quota_fraction,
    quota_text,
)


class SampleAnswerTest(unittest.TestCase):
    def test_every_citation_has_a_listed_source(self):
        cited = cited_ids(SAMPLE_ANSWER)
        self.assertTrue(cited)
        self.assertLessEqual(max(cited), len(SAMPLE_SOURCES))

    def test_cited_ids_reads_grouped_markers(self):
        self.assertEqual(cited_ids("a [S1, S3] b [S2] c [S4](http://x)"), {1, 2, 3})


class QuotaTest(unittest.TestCase):
    def test_label_counts_down(self):
        self.assertEqual(quota_text(987, 1000), "Shared demo key: 987 of 1,000 calls left today")

    def test_spent_budget_says_when_it_resets(self):
        self.assertIn("00:00 UTC", quota_text(0, 1000))
        self.assertIn("00:00 UTC", quota_text(-3, 1000))

    def test_fraction_is_clamped(self):
        self.assertEqual(quota_fraction(500, 1000), 0.5)
        self.assertEqual(quota_fraction(2000, 1000), 1.0)
        self.assertEqual(quota_fraction(5, 0), 0.0)


class IngestProgressTest(unittest.TestCase):
    def test_counts_steps_of_the_current_file(self):
        self.assertEqual(ingest_fraction(1, 4, step=1, steps=2), 0.375)
        self.assertEqual(ingest_fraction(4, 4), 1.0)
        self.assertEqual(ingest_fraction(0, 0), 1.0)


class MarkdownEscapeTest(unittest.TestCase):
    def test_file_names_stay_literal(self):
        self.assertEqual(md_escape("q4_report[final].pdf"), r"q4\_report\[final\].pdf")


if __name__ == "__main__":
    unittest.main()
