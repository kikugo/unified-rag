# Unified RAG

A multimodal Retrieval-Augmented Generation app powered entirely by Google Gemini and ChromaDB.

- **Gemini Embedding 2** — embeds images, text, audio, and video into a unified vector space.
- **Gemini 2.5 Flash / Pro** — reads retrieved content and generates highly grounded answers.
- **ChromaDB** — Scalable, persistent local vector storage for fast semantic retrieval.
- **Managed RAG** — Direct integration with Google's backend File Search API for massive documents.

## Features

- **Multi-Modal Uploads**: Support for images (PNG, JPG), audio (MP3, WAV), video (MP4, MOV), and PDFs.
- **Hybrid Multi-Backend Retrieval**:
  - **Auto (Hybrid)**: Dynamically routes questions to the appropriate backend using a fast Gemini Flash classifier.
  - **Local Vector Store (ChromaDB)**: Embed files locally and query them using a persistent, embedded database (`./chroma_db`). Optionally connect to a hosted Chroma Cloud instance.
  - **Managed RAG (Google File Search)**: Offload ingestion and semantics entirely to Google Cloud. Upload files directly to an ephemeral backend store, bypassing local limitations and receiving built-in citation metadata.
- **Interleaved/Multimodal Embedding**: Add optional captions to uploaded images to embed the text and image together for richer context. Videos, massive PDFs, and audio up to 80 seconds are parsed cleanly natively automatically.
- **Image-as-Query**: Upload an image to visually search your library for similar content, seamlessly executing across `ChromaDB` HNSW semantic vectors.
- **Browser TTS**: Includes a native 'Read Aloud' Text-to-Speech integration leveraging the headless Web Speech API for assistant responses.
- **One-Click Demo**: Load sample financial charts instantly to test the app without uploading your own files.
- **Citation Parsing**: Generated answers in Managed mode are cross-referenced with your documents and explicitly cited.
- **Smart Adjustments**: Adjustable embedding dimension (3072 / 1536 / 768) via MRL, per-document removal, and clear-all controls.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Get a Gemini API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

## API Keys & Privacy

- **Bring Your Own Key Strategy:** Enter your Google Gemini API key in the sidebar each time you use the app. No mandatory accounts or logins required.
- **Session-Only Security:** Your Google API key is held in session memory only — it is never stored locally or sent anywhere other than Google's API.
- **Optional Chroma Cloud:** For users running intensive workloads, you can optionally provide a Chroma Tenant and API Key to offload vector storage to the cloud. If left blank, it relies seamlessly on a local `./chroma_db` database folder!

## How it works

1. **Upload** — files are either embedded locally into Chroma using `gemini-embedding-2-preview`, or uploaded directly into an ephemeral Google File Search store.
2. **Search** — your question is passed to the backend, either surfacing the highest scoring semantic vectors from ChromaDB, or utilizing the Google FileSearch tool.
3. **Answer** — `gemini-2.5-flash` or `gemini-2.5-pro` generate a comprehensive, grounded answer with traceable citations based on the retrieved context.

## Requirements

- Python 3.10+
- Google API key with Gemini access
- `chromadb` for local vector persistence
- `PyMuPDF` for PDF chunking handling
- `av` (PyAV) for robust video container parsing and hardware abstraction
