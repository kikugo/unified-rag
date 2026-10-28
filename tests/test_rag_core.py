import unittest

from rag_core import (
    bm25_rank,
    build_local_answer_prompt,
    build_rehydrated_sources,
    call_with_retry,
    format_prompt_cost,
    format_token_usage,
    is_rate_limit_error,
    is_transient_error,
    reciprocal_rank_fusion,
    tokenize_for_search,
    pop_source_by_id,
    source_artifacts,
    select_context_results,
    within_demo_upload_cap,
)


class RagCoreTest(unittest.TestCase):
    def test_pop_source_by_id_removes_row_and_returns_all_chunk_ids(self):
        sources = [
            {"source_id": "f1", "id": "chunk-a", "name": "doc.pdf", "backend": "local",
             "chroma_ids": ["chunk-a", "chunk-b"]},
            {"source_id": "f2", "id": "img", "name": "chart.png", "backend": "local",
             "chroma_ids": ["img"]},
        ]

        removed, chroma_ids = pop_source_by_id(sources, "f1")

        self.assertEqual(removed["name"], "doc.pdf")
        self.assertEqual(chroma_ids, ["chunk-a", "chunk-b"])
        self.assertEqual([s["source_id"] for s in sources], ["f2"])

    def test_pop_source_by_id_missing_is_noop(self):
        sources = [{"source_id": "f1", "id": "x", "backend": "local"}]
        removed, chroma_ids = pop_source_by_id(sources, "nope")
        self.assertEqual(removed, {})
        self.assertEqual(chroma_ids, [])
        self.assertEqual(len(sources), 1)

    def test_source_artifacts_falls_back_to_row_paths(self):
        with_artifacts = {"artifacts": [{"chroma_id": "a", "file_path": "p1"}]}
        self.assertEqual(source_artifacts(with_artifacts), [{"chroma_id": "a", "file_path": "p1"}])
        legacy = {"file_path": "p2", "preview_path": "pv"}
        self.assertEqual(source_artifacts(legacy), [{"file_path": "p2", "preview_path": "pv"}])
        self.assertEqual(source_artifacts({}), [])

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
        # guards against the model miscounting, mislabeling frames, or inventing timestamps
        self.assertIn("not the full library", prompt)
        self.assertIn("video frame", prompt)
        self.assertIn("do not invent", prompt)

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

    def test_format_prompt_cost(self):
        self.assertEqual(format_prompt_cost(0), "")
        self.assertEqual(format_prompt_cost(500), "🔢 500 tokens this prompt")
        out = format_prompt_cost(1500, {"total": 1234})
        self.assertIn("1,500 tokens", out)
        self.assertIn("answer 1,234", out)
        self.assertIn("266", out)

    def test_format_token_usage(self):
        self.assertEqual(format_token_usage(None), "")
        out = format_token_usage({"prompt": 980, "output": 254, "total": 1234})
        self.assertEqual(out, "🔢 1,234 tokens (in 980 · out 254)")
        with_think = format_token_usage({"prompt": 8, "output": 2, "total": 30, "thinking": 20})
        self.assertIn("thinking 20", with_think)

    def test_is_transient_error_includes_503_and_rate_limits(self):
        self.assertTrue(is_transient_error(Exception("503 UNAVAILABLE: model overloaded")))
        self.assertTrue(is_transient_error(Exception("429 rate limit")))
        self.assertFalse(is_transient_error(Exception("400 invalid argument")))

    def test_is_transient_error_includes_dropped_connections(self):
        self.assertTrue(is_transient_error(Exception("Server disconnected without sending a response.")))
        self.assertTrue(is_transient_error(ConnectionResetError("Connection reset by peer")))

    def test_call_with_retry_retries_503_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise Exception("503 UNAVAILABLE")
            return "ok"

        self.assertEqual(call_with_retry(flaky, attempts=3, sleep=lambda _: None), "ok")
        self.assertEqual(calls["n"], 2)

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


    def test_demo_upload_cap_rejects_beyond_limit(self):
        self.assertTrue(within_demo_upload_cap(current=2, limit=3))
        self.assertFalse(within_demo_upload_cap(current=3, limit=3))


if __name__ == "__main__":
    unittest.main()
