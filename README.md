# Unified RAG

I built this multimodal RAG app to see how far I could push Google Gemini alongside ChromaDB. It handles text, images, audio, and video directly without relying on heavy OCR pipelines. 

[Live demo](https://unified-rag.streamlit.app/) — no signup, no API key. Hit **Load 6 sample charts** in the sidebar and ask it something. Uploads are capped there so one visitor cannot exhaust the container; clone the repo to run it without limits.

The demo runs on a key the deployment supplies (`GOOGLE_API_KEY` in Streamlit secrets, see `.streamlit/secrets.toml.example`). Paste your own key in the sidebar and it takes priority. With no key configured anywhere, the app falls back to asking each visitor for one, which is how it behaves on a fresh clone.

## What's inside

- **Gemini Embedding 2**: Maps everything (text, charts, audio clips, video frames) into a single vector space.
- **Gemini 3.6 Flash**: Reads the retrieved chunks and generates the final answer, dropping to `gemini-3.5-flash-lite` when `DEMO_MODE` is on. The whole app was on the 2.5 line until 2026-08-01, when that generation started returning 404 to newly issued API keys — model ids now live in one place (`app/gemini.py`) so the next retirement is a one-line fix.
- **ChromaDB**: Local vector storage. It's fast enough for what I need and doesn't require setting up cloud infrastructure just to test an idea.
- **Google File Search (Managed RAG)**: I added this to bypass local limits when dealing with massive document dumps. It offloads the hassle of ingestion squarely to Google Cloud.

## How it works

1. **Upload**: You drop files in. The app either embeds them locally into Chroma (using `gemini-embedding-2-preview`) or pushes them straight to an ephemeral Google File Search store. Local uploads go through hashed file storage under `.unified_rag_store/` instead of stuffing raw bytes into Chroma metadata when possible.
2. **Search**: You ask a question. The backend can route to Google File Search, or run hybrid local retrieval: dense Chroma search fused with BM25 keyword ranking via Reciprocal Rank Fusion (RRF).
3. **Answer**: the answer model gets the top retrieved chunks (not just the first hit) with source IDs like `[S1]` and `[S2]` so citations actually mean something.

## Does the hybrid retrieval actually help?

I kept claiming it did, so I built a benchmark and measured it. The answer turned out to be "yes, but only for some queries", which is not what I expected.

The set is 151 passages and 50 queries, in `tests/data/`. Queries split three ways: exact-identifier lookups (SKU codes, CVE numbers, version strings), paraphrases that share no vocabulary with the target passage, and neutral queries that should be easy for anything. Every passage runs to about 128 words, because that is roughly what a chunk looks like once a PDF is split, and passage length turned out to matter more than anything else here.

Recall over all 50 queries:

| Retrieval | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| BM25 only | 0.660 | 0.700 | 0.700 |
| Dense only | 0.720 | 0.960 | 1.000 |
| Hybrid (RRF) | 0.540 | 0.760 | 0.860 |

On that headline number, dense alone wins and fusing BM25 into it makes things worse. Splitting by query type shows why:

| Query type | BM25 r@1 | Dense r@1 | Hybrid r@1 |
|---|---|---|---|
| Exact identifier | 1.000 | 0.471 | 0.647 |
| Paraphrase | 0.059 | 0.824 | 0.118 |
| Neutral | 0.938 | 0.875 | 0.875 |

Dense retrieval finds fewer than half of the exact-identifier passages at rank one. BM25 finds all of them. On paraphrases it is the other way round, and not close.

The interesting part is what passage length does to that. Same 151 documents, same queries, same embedding model, with only the amount of surrounding text changed:

| Mean passage length | Dense, exact-identifier r@1 |
|---|---|
| 11 words | 0.941 |
| 128 words | 0.471 |

One embedding vector has to stand in for the whole passage, so a serial number surrounded by three hundred words of context gets averaged away. BM25 scored 1.000 in both cases, because a literal token match does not care how much text is around it. This is the case hybrid retrieval exists for, and it only appears once the documents are realistic. On one-line documents dense looks flawless and the whole argument for BM25 disappears.

So the honest summary: hybrid recovers most of what dense loses on identifier lookups (0.471 to 0.647 at rank one, and 0.882 to 1.000 at rank three), and it costs a lot on paraphrases, where RRF gives an arm that scored 0.059 the same weight as one that scored 0.824. Plain RRF has no idea that one of its inputs is useless for a given query. Whether the trade is worth it depends on the query mix, which for a document search tool leans toward identifier lookups more than this overall average suggests.

The obvious next step is weighting the fusion by query type rather than treating both rankings as equal. This app already routes queries between backends, so routing between retrievers is the same idea one level down.

Reproduce it with `tests/test_ablation.py`. Embeddings are cached in the repo, so it runs offline and without an API key.

## Features that matter

- **It takes basically anything**: PNGs, JPGs, MP3s, WAVs, MP4s, MOVs, and regular PDFs.
- **Hybrid Routing**: You don't have to manually pick the backend. Gemini Flash runs a sub-second check to figure out if your query belongs in the local vector store or the managed Google backend.
- **Hybrid local retrieval**: Dense vectors plus BM25/RRF so exact keywords and serial numbers are less likely to disappear.
- **Multimodal embeddings**: Per-image captions (including an optional auto-caption path), image+text interleaved embeddings, PDF overlap chunking with extracted text for search, timestamped audio chunks decoded from MP3, and scene-based video frame sampling that skips near-duplicate frames.
- **Speech-to-text**: Audio files and the audio track inside videos get transcribed with Gemini (there's a sidebar toggle), and each video frame inherits the transcript of the window it falls in. Spoken words become searchable.
- **One source per file**: Every uploaded file is one sidebar entry and one registry row that tracks all its chunks, so deleting it cleans up every chunk's blobs. Re-uploading the same bytes is skipped via a content hash.
- **Source registry**: A JSON registry tracks file paths, page ranges, timestamps, captions, and Chroma IDs, written atomically and rebuilt into the sidebar on restart.
- **Citation click-through**: Answers cite `[S1]`, `[S2]`, etc.; opening one shows the exact cited chunk's pages or frame and seeks audio/video to the matched timestamp.
- **Image queries**: You can upload an image instead of typing a question. The app hits ChromaDB's HNSW index to find visually or semantically similar content.
- **Browser TTS**: I threw in a quick Web Speech API integration so the app can read answers out loud.
- **Adjustable MRL**: You can slide the embedding dimensions between 3072, 1536, and 768.

Most of the non-UI logic lives in helper modules now (`rag_core.py`, `media_ingest.py`, `storage.py`, `source_registry.py`, `eval_harness.py`, and `app/gemini.py`) so I can unit test retrieval and ingest without launching Streamlit.

## Running it locally

You'll need Python 3.10 or higher.

```bash
pip install -r requirements.txt
streamlit run app.py
```

You just plug your Gemini API key into the sidebar when the app loads. I set it up specifically to only hold your key in session memory. There are no accounts, and it doesn't save your key to disk.

For heavy usage, you can connect it to a hosted Chroma Cloud instance by dropping in your tenant details. Otherwise, it defaults to creating a `./chroma_db` folder locally.

There are tests under `tests/` if you want to sanity-check helpers (`python3 -m unittest discover -s tests`). PDF tests skip automatically when PyMuPDF isn't installed.

## Retrieval from the command line

Retrieval and answering live in `rag/pipeline.py` now, behind one `Config` (mode `dense`, `bm25` or `hybrid_rrf`, plus `rerank`, `hyde`, `top_k`, `context_k`). The Streamlit app and the CLI both call it, so anything I measure from the terminal is the code the app runs.

The benchmark corpus is fixed: the EU AI Act as published in the Official Journal, plus a set of Wikipedia articles pinned by revision id, all listed in `corpus/manifest.json`. The extracted text is committed under `corpus/docs/`, so chunk boundaries can't drift when Wikipedia re-renders an old revision. Figures are indexed as image + caption chunks; the images themselves are downloaded, not committed.

```bash
uv pip install -r requirements.txt
uv run python -m rag.corpus fetch            # download raw sources and figures (text is already committed)
uv run python -m rag.index build             # embed into .eval_index/, asks before spending API calls
uv run python -m rag.cli "Which AI practices does Article 5 prohibit?" --mode hybrid_rrf --rerank
```

The CLI prints each hit's chunk id with its dense, BM25, RRF and rerank scores. Add `--answer` for a cited answer, or `--hyde` to embed a drafted answer alongside the question. Reranking is a local cross-encoder (`ms-marco-MiniLM-L-6-v2` through fastembed's ONNX runtime), so it needs no API key.

Chunk ids are a hash of document id, page (or section), character offset and kind, so rebuilding the index from the same text gives the same ids. Embeddings are cached in `.cache/`, which makes a rebuild free. `uv run python -m rag.index verify-stable` checks both: it deletes the index, rebuilds it and confirms every ranking comes back identical.

## Notes to self: what's left

My professor reviewed the codebase a while back and pointed out a pile of real embarrassments. Most of that list is gone now. Retrieval is hybrid BM25+RRF with top-K context. Audio and video get transcribed. MP3 decodes properly, re-uploads dedupe by content hash, each file is one source row, citations seek to the right spot, and the corpus survives a restart. There's even a recall@k test on a seeded corpus so I can tell when I break something.

What I still want to get to:

- **Grow the eval set**: The harness runs a 14-query gold set as a regression guard, but I need more real multimodal questions before I trust bigger retrieval changes. Everything below should be justified by those numbers, not vibes.
- **Transcription cost**: Speech-to-text is one Gemini call per chunk. Fine for short clips, rough for a long video. I want a batch path or a per-file cap before heavy use.
- **Semantic PDF chunking**: Overlap page windows are in place (4 pages with 1-page overlap). I might try `semchunk` if the eval shows boundary errors.
- **Cross-encoder reranking**: Most of my content is visual, so a text-to-text cross encoder would probably choke. If I ever want reranking, a Gemini-based reranker fits the data better and adds no dependency. Waiting on eval numbers either way.
- **The cutting-edge stuff**: ColPali (treats pages as image patches, skips OCR), ColBERT-style late interaction, RAPTOR tree summarization, full agentic RAG. Interesting, each a real dependency and a real architecture change. Only worth it if this becomes a product or a research demo, not before.
