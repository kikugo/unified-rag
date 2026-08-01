"""Pins the retrieval ablation reported in the README.

Runs offline against committed embeddings, so it needs no API key and gives the
same answer every time. The point of these assertions is not the exact decimals,
it is the shape of the result: BM25 owns exact-identifier lookup, dense owns
paraphrase, and fusing them lands between the two rather than above both.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from eval_harness import (
    evaluate_queries,
    make_bm25_search_fn,
    make_dense_search_fn,
    make_hybrid_search_fn,
)

DATA = Path(__file__).parent / "data"


def _load(name: str):
    with (DATA / name).open(encoding="utf-8") as handle:
        return json.load(handle)


class AblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = _load("eval_corpus_v4.json")
        cls.gold = _load("eval_gold_set_v2.json")
        cls.embeddings = _load("eval_embeddings_v4.json")
        cls.arms = {
            "bm25": make_bm25_search_fn(cls.corpus),
            "dense": make_dense_search_fn(cls.corpus, cls.embeddings),
            "hybrid": make_hybrid_search_fn(cls.corpus, cls.embeddings),
        }

    def _score(self, arm: str, prefix: str | None = None, k: int = 1) -> float:
        cases = self.gold
        if prefix:
            cases = [c for c in cases if all(i.startswith(prefix) for i in c["expected_ids"])]
        report = evaluate_queries(cases, self.arms[arm], k_values=(k,))
        return report["summary"][f"recall@{k}"]

    def test_corpus_is_realistic_passage_length(self):
        # The whole finding depends on documents being long enough to dilute an
        # identifier. If someone shortens them, the ablation stops meaning anything.
        lengths = [len(d["searchable_text"].split()) for d in self.corpus]
        self.assertGreater(sum(lengths) / len(lengths), 90)

    def test_no_arm_is_saturated_overall(self):
        # A set where an arm cannot lose cannot compare arms. An earlier version of
        # this benchmark was saturated and had to be rebuilt.
        for arm in self.arms:
            self.assertLess(self._score(arm, k=1), 1.0, f"{arm} is saturated at recall@1")

    def test_bm25_beats_dense_on_exact_identifiers(self):
        self.assertGreater(self._score("bm25", "exact"), self._score("dense", "exact"))

    def test_dense_beats_bm25_on_paraphrase(self):
        self.assertGreater(self._score("dense", "para"), self._score("bm25", "para"))

    def test_hybrid_recovers_dense_losses_on_exact_identifiers(self):
        # This is the case hybrid retrieval exists for.
        self.assertGreater(self._score("hybrid", "exact"), self._score("dense", "exact"))
        self.assertEqual(self._score("hybrid", "exact", k=3), 1.0)

    def test_hybrid_costs_accuracy_on_paraphrase(self):
        # Reported honestly in the README rather than left out. Unweighted RRF gives
        # a useless arm the same vote as a good one.
        self.assertLess(self._score("hybrid", "para"), self._score("dense", "para"))


if __name__ == "__main__":
    unittest.main()
