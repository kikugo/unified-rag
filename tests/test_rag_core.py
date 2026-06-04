import unittest

from rag_core import (
    bm25_rank,
    build_local_answer_prompt,
    build_rehydrated_sources,
    call_with_retry,
    is_rate_limit_error,
    reciprocal_rank_fusion,
    tokenize_for_search,
    pop_source_and_embedding,
    select_context_results,
)


class RagCoreTest(unittest.TestCase):
    def test_pop_managed_source_does_not_remove_local_embedding(self):
        sources = [
            {"id": "local-1", "name": "chart.png", "backend": "local"},
            {"name": "cloud.pdf", "backend": "managed"},
            {"id": "local-2", "name": "slide.png", "backend": "local"},
        ]
        embeddings = [[1.0], [2.0]]

        removed, chroma_ids = pop_source_and_embedding(sources, embeddings, 1)

        self.assertEqual(removed["name"], "cloud.pdf")
        self.assertEqual(chroma_ids, [])
        self.assertEqual([s["name"] for s in sources], ["chart.png", "slide.png"])
        self.assertEqual(embeddings, [[1.0], [2.0]])

    def test_pop_local_source_after_managed_removes_matching_embedding(self):
        sources = [
            {"id": "local-1", "name": "chart.png", "backend": "local"},
            {"name": "cloud.pdf", "backend": "managed"},
            {"id": "local-2", "name": "slide.png", "backend": "local"},
        ]
        embeddings = [[1.0], [2.0]]

        removed, chroma_ids = pop_source_and_embedding(sources, embeddings, 2)

        self.assertEqual(removed["name"], "slide.png")
        self.assertEqual(chroma_ids, ["local-2"])
        self.assertEqual([s["name"] for s in sources], ["chart.png", "cloud.pdf"])
        self.assertEqual(embeddings, [[1.0]])

    def test_select_context_results_caps_results_and_assigns_citation_ids(self):
        results = [
            {"id": f"doc-{i}", "name": f"doc-{i}.pdf", "score": 1 - i / 10}
            for i in range(6)
        ]

        selected = select_context_results(results, max_results=3)

        self.assertEqual([r["citation_id"] for r in selected], ["S1", "S2", "S3"])
        self.assertEqual([r["name"] for r in selected], ["doc-0.pdf", "doc-1.pdf", "doc-2.pdf"])

    def test_build_local_answer_prompt_instructs_citations_for_all_sources(self):
        selected = [
            {"citation_id": "S1", "name": "contract.pdf pages 1-2", "type": "pdf", "score": 0.91},
            {"citation_id": "S2", "name": "contract.pdf pages 3-4", "type": "pdf", "score": 0.84},
        ]

        prompt = build_local_answer_prompt("What is the termination clause?", selected)

        self.assertIn("Use the retrieved sources below", prompt)
        self.assertIn("[S1] contract.pdf pages 1-2", prompt)
        self.assertIn("[S2] contract.pdf pages 3-4", prompt)
        self.assertIn("cite the source IDs", prompt)
        self.assertIn("What is the termination clause?", prompt)

    def test_tokenize_for_search_keeps_exact_match_terms(self):
        tokens = tokenize_for_search("Gemini-2.5 Flash handles SKU ABC-123 and error_code 0xDEAD.")

        self.assertIn("gemini", tokens)
        self.assertIn("2", tokens)
        self.assertIn("5", tokens)
        self.assertIn("abc", tokens)
        self.assertIn("123", tokens)
        self.assertIn("error_code", tokens)
        self.assertIn("0xdead", tokens)

    def test_bm25_rank_prefers_exact_keyword_matches(self):
        docs = [
            {"id": "a", "searchable_text": "general pricing and support terms"},
            {"id": "b", "searchable_text": "release notes mention ERR-9427 timeout failure"},
            {"id": "c", "searchable_text": "installation and onboarding guide"},
        ]

        ranking = bm25_rank("ERR-9427", docs)

        self.assertEqual(ranking[0][0], "b")
        self.assertGreater(ranking[0][1], 0)

    def test_reciprocal_rank_fusion_combines_dense_and_sparse_rankings(self):
        fused = reciprocal_rank_fusion([
            ["dense-top", "shared", "dense-third"],
            ["shared", "sparse-second", "dense-top"],
        ])

        self.assertEqual(fused[0], "shared")
        self.assertEqual(set(fused), {"dense-top", "shared", "dense-third", "sparse-second"})

    def test_build_rehydrated_sources_restores_local_rows_and_maps_doc_id(self):
        rows = [
            {"source_id": "s1", "doc_id": "d1", "name": "chart.png", "type": "image",
             "backend": "local", "chroma_ids": ["d1"]},
            {"source_id": "s2", "name": "cloud.pdf", "type": "pdf", "backend": "managed"},
            {"source_id": "s3", "name": "clip.mp4", "type": "video_frame", "backend": "local",
             "chroma_ids": ["d3"]},
        ]

        restored = build_rehydrated_sources(rows)

        self.assertEqual([r["source_id"] for r in restored], ["s1", "s3"])
        self.assertEqual(restored[0]["id"], "d1")
        # falls back to first chroma id when doc_id is absent
        self.assertEqual(restored[1]["id"], "d3")
        self.assertTrue(all(r["backend"] == "local" for r in restored))

    def test_is_rate_limit_error_matches_quota_and_429(self):
        self.assertTrue(is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED")))
        self.assertTrue(is_rate_limit_error(Exception("Quota exceeded for requests")))
        self.assertFalse(is_rate_limit_error(Exception("400 invalid argument")))

    def test_call_with_retry_returns_first_success_without_sleeping(self):
        sleeps = []
        result = call_with_retry(lambda: "ok", sleep=sleeps.append)
        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [])

    def test_call_with_retry_retries_rate_limit_then_succeeds(self):
        sleeps = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("429 rate limit")
            return "done"

        result = call_with_retry(flaky, attempts=3, base_delay=0.1, sleep=sleeps.append)
        self.assertEqual(result, "done")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(sleeps, [0.1, 0.2])

    def test_call_with_retry_reraises_non_rate_limit_immediately(self):
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise ValueError("400 bad request")

        with self.assertRaises(ValueError):
            call_with_retry(boom, attempts=3, sleep=lambda _: None)
        self.assertEqual(calls["n"], 1)

    def test_call_with_retry_raises_after_exhausting_attempts(self):
        with self.assertRaises(Exception):
            call_with_retry(
                lambda: (_ for _ in ()).throw(Exception("429")),
                attempts=2,
                sleep=lambda _: None,
            )


if __name__ == "__main__":
    unittest.main()
