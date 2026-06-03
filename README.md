# Unified RAG

I built this multimodal RAG app to see how far I could push Google Gemini alongside ChromaDB. It handles text, images, audio, and video directly without relying on heavy OCR pipelines. 

## What's inside

- **Gemini Embedding 2**: Maps everything (text, charts, audio clips, video frames) into a single vector space.
- **Gemini 2.5 Flash / Pro**: Reads the retrieved chunks and generates the final answer.
- **ChromaDB**: Local vector storage. It's fast enough for what I need and doesn't require setting up cloud infrastructure just to test an idea.
- **Google File Search (Managed RAG)**: I added this to bypass local limits when dealing with massive document dumps. It offloads the hassle of ingestion squarely to Google Cloud.

## How it works

1. **Upload**: You drop files in. The app either embeds them locally into Chroma (using `gemini-embedding-2-preview`) or pushes them straight to an ephemeral Google File Search store. Local uploads go through hashed file storage under `.unified_rag_store/` instead of stuffing raw bytes into Chroma metadata when possible.
2. **Search**: You ask a question. The backend can route to Google File Search, or run hybrid local retrieval: dense Chroma search fused with BM25 keyword ranking via Reciprocal Rank Fusion (RRF), plus lightweight query expansion on short questions.
3. **Answer**: `gemini-2.5-flash` gets the top retrieved chunks (not just the first hit) with source IDs like `[S1]` and `[S2]` so citations actually mean something.

## Features that matter

- **It takes basically anything**: PNGs, JPGs, MP3s, WAVs, MP4s, MOVs, and regular PDFs.
- **Hybrid Routing**: You don't have to manually pick the backend. Gemini Flash runs a sub-second check to figure out if your query belongs in the local vector store or the managed Google backend.
- **Hybrid local retrieval**: Dense vectors plus BM25/RRF so exact keywords and serial numbers are less likely to disappear.
- **Multimodal embeddings**: Per-image captions (including an optional auto-caption path), image+text interleaved embeddings, PDF overlap chunking with extracted text for search, timestamped audio chunks, and adaptive video frame sampling for longer files.
- **Source registry**: A JSON registry tracks file paths, page ranges, timestamps, captions, and Chroma IDs so deletion and citation panels have something consistent to read.
- **Citation click-through**: Answers cite `[S1]`, `[S2]`, etc., and you can open a source detail panel with previews, audio, or the original video.
- **Image queries**: You can upload an image instead of typing a question. The app hits ChromaDB's HNSW index to find visually or semantically similar content.
- **Browser TTS**: I threw in a quick Web Speech API integration so the app can read answers out loud.
- **Adjustable MRL**: You can slide the embedding dimensions between 3072, 1536, and 768.

Most of the non-UI logic lives in helper modules now (`rag_core.py`, `storage.py`, `source_registry.py`, `eval_harness.py`) so I can unit test retrieval and ingest without launching Streamlit.

## Running it locally

You'll need Python 3.10 or higher.

```bash
pip install -r requirements.txt
streamlit run app.py
```

You just plug your Gemini API key into the sidebar when the app loads. I set it up specifically to only hold your key in session memory. There are no accounts, and it doesn't save your key to disk.

For heavy usage, you can connect it to a hosted Chroma Cloud instance by dropping in your tenant details. Otherwise, it defaults to creating a `./chroma_db` folder locally.

There are tests under `tests/` if you want to sanity-check helpers (`python3 -m unittest discover -s tests`). PDF tests skip automatically when PyMuPDF isn't installed.

## Notes to self: Upcoming fixes

My professor recently reviewed the codebase and pointed out some real embarrassments. Retrieval is in better shape now (Top-K context, BM25+RRF, basic query expansion), but there's still plenty that feels prototype-grade.

Here is the hit-list of upgrades I still need to work on:

- **Citation UX**: Click-through exists, but I still want seek-to-timestamp for audio/video and tighter PDF page focus when you open a cited chunk.
- **Measure retrieval properly**: I added a small `eval_harness.py` and starter gold set, but I need more real questions and recall@K runs against a seeded corpus before I trust changes.
- **Transcripts**: Audio and video ingest store transcript fields, but I'm not actually generating speech-to-text yet. That's probably the biggest win left for video search.
- **MP3 chunking**: WAV chunks decode cleanly; MP3 still uses a byte-proportional fallback instead of PyAV.
- **Semantic PDF chunking**: Overlap page windows are in place (4 pages with 1-page overlap). I might still try `semchunk` if overlap isn't enough.
- **HyDE / multi-embed query expansion**: Short queries get simple variant phrases today. I still want to test drafting a fake answer or averaging multiple query embeddings like my professor suggested.
- **Rehydration**: Chroma and the registry persist on disk, but the Streamlit session doesn't rebuild the sidebar source list from them after a restart.
- **Caching & Async**: The upload loop currently blocks the main thread. I really ought to wrap it in a `ThreadPoolExecutor` so people don't have to wait. I also need to add LRU caching on the router, and drop an SHA-256 content hash check in so I stop re-embedding files if the Streamlit server reloads.
- **Cross-Encoders**: I'm debating whether a lightweight `SentenceTransformer` re-ranker makes sense here. Most of my documents are visual, so a standard text-to-text cross encoder might just choke on them. I need to test if relying purely on Gemini Flash is enough.
- **The cutting edge stuff**: I've got a lot of SOTA papers to read. I'm wondering if I should rip out my page-level PDF parsing and replace it with **ColPali**, which treats pages as pure image patches and completely bypasses OCR errors (though the HuggingFace bloat scares me). My professor also mentioned looking into Late Interaction methods like ColBERT, hierarchical tree summarization like RAPTOR, and even expanding the simple LLM router into full-blown Agentic RAG. I don't know if I have the energy to build all of that yet, but I need to read up on them.
