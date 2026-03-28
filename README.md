# Unified RAG

I built this multimodal RAG app to see how far I could push Google Gemini alongside ChromaDB. It handles text, images, audio, and video directly without relying on heavy OCR pipelines. 

## What's inside

- **Gemini Embedding 2**: Maps everything (text, charts, audio clips, video frames) into a single vector space.
- **Gemini 2.5 Flash / Pro**: Reads the retrieved chunks and generates the final answer.
- **ChromaDB**: Local vector storage. It's fast enough for what I need and doesn't require setting up cloud infrastructure just to test an idea.
- **Google File Search (Managed RAG)**: I added this to bypass local limits when dealing with massive document dumps. It offloads the hassle of ingestion squarely to Google Cloud.

## How it works

1. **Upload**: You drop files in. The app either embeds them locally into Chroma (using `gemini-embedding-2-preview`) or pushes them straight to an ephemeral Google File Search store.
2. **Search**: You ask a question. The backend runs a quick classification to route it to either Chroma or Google File Search, retrieving the most relevant chunks.
3. **Answer**: `gemini-2.5-flash` uses those chunks to write a response with actual citations.

## Features that matter

- **It takes basically anything**: PNGs, JPGs, MP3s, WAVs, MP4s, MOVs, and regular PDFs.
- **Hybrid Routing**: You don't have to manually pick the backend. Gemini Flash runs a sub-second check to figure out if your query belongs in the local vector store or the managed Google backend.
- **Multimodal embeddings**: You can add text captions to uploaded images, and the embedding model sees both together. I also wired it up so PDFs, video frames, and audio up to 80 seconds process cleanly.
- **Image queries**: You can upload an image instead of typing a question. The app hits ChromaDB's HNSW index to find visually or semantically similar content.
- **Browser TTS**: I threw in a quick Web Speech API integration so the app can read answers out loud.
- **Adjustable MRL**: You can slide the embedding dimensions between 3072, 1536, and 768.

## Running it locally

You'll need Python 3.10 or higher.

```bash
pip install -r requirements.txt
streamlit run app.py
```

You just plug your Gemini API key into the sidebar when the app loads. I set it up specifically to only hold your key in session memory. There are no accounts, and it doesn't save your key to disk.

For heavy usage, you can connect it to a hosted Chroma Cloud instance by dropping in your tenant details. Otherwise, it defaults to creating a `./chroma_db` folder locally.

## Notes to self: Upcoming fixes

My professor recently reviewed the codebase and pointed out some real embarrassments. For one, my retrieval pipeline is entirely dense (pure ANN), which means it completely fails if I try to look up an exact serial number or a rare keyword.

Here is the hit-list of upgrades I need to work on:

- **Stop throwing away Top-K context**: Right now, the app pulls the Top-3 results but I only pass the #1 hit to the LLM context. Gemini has a massive token window, so I'm basically wasting it. I need to feed it everything I retrieve.
- **BM25 Sparse Retrieval**: I need to fuse my Chroma dense vectors with a standard BM25 keyword index using Reciprocal Rank Fusion (RRF). This should stop the exact-match queries from failing.
- **Fixing PDF boundaries**: I'm currently chunking PDFs at a hard 6-page limit, which eventually cuts a sentence or chart in half. I'm debating between using proper semantic chunking (with a library like `semchunk`), or just taking the easy route and using a sliding window with a 1-to-2 page overlap. Both feel better than what I have now.
- **Temporal constraints on video frames**: Extracting video frames natively works fine, but the isolated image vectors have no concept of time. I wonder if I should construct a caption like "Video segment from 2:15-2:30" and embed that alongside the frame. In theory, questions about sequences might actually start working.
- **Query Expansion (HyDE / Multi-query)**: I'm thinking about running a fast Flash call before search to draft a fake answer, and embedding that text instead of the user's short question. My professor also suggested splitting the query into three variants and averaging the vectors. I need to test if this tricks the vector space into matching much broader context.
- **Caching & Async**: The upload loop currently blocks the main thread. I really ought to wrap it in a `ThreadPoolExecutor` so people don't have to wait. I also need to add LRU caching on the router, and drop an SHA-256 content hash check in so I stop re-embedding files if the Streamlit server reloads.
- **Cross-Encoders**: I'm debating whether a lightweight `SentenceTransformer` re-ranker makes sense here. Most of my documents are visual, so a standard text-to-text cross encoder might just choke on them. I need to test if relying purely on Gemini Flash is enough.
- **The cutting edge stuff**: I've got a lot of SOTA papers to read. I'm wondering if I should rip out my page-level PDF parsing and replace it with **ColPali**, which treats pages as pure image patches and completely bypasses OCR errors (though the HuggingFace bloat scares me). My professor also mentioned looking into Late Interaction methods like ColBERT, hierarchical tree summarization like RAPTOR, and even expanding the simple LLM router into full-blown Agentic RAG. I don't know if I have the energy to build all of that yet, but I need to read up on them.
