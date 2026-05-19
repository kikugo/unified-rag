import unittest

from rag_core import (
    bm25_rank,
    build_local_answer_prompt,
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


if __name__ == "__main__":
    unittest.main()
