# Unified RAG

I built this multimodal RAG app to see how far I could push Google Gemini alongside ChromaDB. It handles text, images, audio, and video directly without relying on heavy OCR pipelines. 

## What's inside

- **Gemini Embedding 2**: Maps everything (text, charts, audio clips, video frames) into a single vector space.
- **Gemini 2.5 Flash / Pro**: Reads the retrieved chunks and generates the final answer.
- **ChromaDB**: Local vector storage. It's fast enough for what I need and doesn't require setting up cloud infrastructure just to test an idea.
- **Google File Search (Managed RAG)**: I added this to bypass local limits when dealing with massive document dumps. It offloads the hassle of ingestion squarely to Google Cloud.

## How it works

1. **Upload**: You drop files in. The app either embeds them locally into Chroma (using `gemini-embedding-2-preview`) or pushes them straight to an ephemeral Google File Search store. Local uploads go through hashed file storage under `.unified_rag_store/` instead of stuffing raw bytes into Chroma metadata when possible.
2. **Search**: You ask a question. The backend can route to Google File Search, or run hybrid local retrieval: dense Chroma search fused with BM25 keyword ranking via Reciprocal Rank Fusion (RRF).
3. **Answer**: `gemini-2.5-flash` gets the top retrieved chunks (not just the first hit) with source IDs like `[S1]` and `[S2]` so citations actually mean something.

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

## Notes to self: what's left

My professor reviewed the codebase a while back and pointed out a pile of real embarrassments. Most of that list is gone now. Retrieval is hybrid BM25+RRF with top-K context. Audio and video get transcribed. MP3 decodes properly, re-uploads dedupe by content hash, each file is one source row, citations seek to the right spot, and the corpus survives a restart. There's even a recall@k test on a seeded corpus so I can tell when I break something.

What I still want to get to:

- **Grow the eval set**: The harness runs a 14-query gold set as a regression guard, but I need more real multimodal questions before I trust bigger retrieval changes. Everything below should be justified by those numbers, not vibes.
- **Transcription cost**: Speech-to-text is one Gemini call per chunk. Fine for short clips, rough for a long video. I want a batch path or a per-file cap before heavy use.
- **Semantic PDF chunking**: Overlap page windows are in place (4 pages with 1-page overlap). I might try `semchunk` if the eval shows boundary errors.
- **Cross-encoder reranking**: Most of my content is visual, so a text-to-text cross encoder would probably choke. If I ever want reranking, a Gemini-based reranker fits the data better and adds no dependency. Waiting on eval numbers either way.
- **The cutting-edge stuff**: ColPali (treats pages as image patches, skips OCR), ColBERT-style late interaction, RAPTOR tree summarization, full agentic RAG. Interesting, each a real dependency and a real architecture change. Only worth it if this becomes a product or a research demo, not before.
