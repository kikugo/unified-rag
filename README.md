# Unified RAG

A multimodal Retrieval-Augmented Generation app powered entirely by Google Gemini.

- **Gemini Embedding 2** — embeds images and PDFs into a unified vector space (no OCR, no preprocessing)
- **Gemini 2.5 Flash** — reads the retrieved content and generates a grounded answer

## Features

- **Multi-Modal Uploads**: Support for images (PNG, JPG), audio (MP3, WAV), video (MP4, MOV), and PDFs of any length (auto-chunked).
- **Interleaved Embedding**: Add optional captions to uploaded images to embed the text and image together for richer context.
- **Image-as-Query**: Upload an image to visually search your library for similar content (uses `RETRIEVAL_QUERY` task type).
- **One-Click Demo**: Load sample financial charts instantly to test the app without uploading your own files.
- **Visual Library**: View all your loaded documents in an interactive gallery expander.
- **Answer Generation**: Gemini 2.5 Flash answers questions based on the top retrieved result.
- **Smart Adjustments**: Adjustable embedding dimension (3072 / 1536 / 768) via MRL, per-document removal, and clear-all controls.
- Single Google API key — no other accounts needed.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Get a Gemini API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

## API Key

- Enter your Google Gemini API key in the sidebar each time you use the app.
- Your key is held in session memory only — it is never stored, logged, or sent anywhere other than directly to Google's API.
- Refreshing the page clears the key. This is intentional and keeps the app stateless.
- For a live demo, you can share your own key with people you trust, or direct them to [Google AI Studio](https://aistudio.google.com/app/apikey) to get their own (free).

## How it works

1. **Upload** — images and PDFs are embedded on upload using `gemini-embedding-2-preview`
2. **Search** — your question is embedded and scored against all stored embeddings via cosine similarity
3. **Answer** — the top result is passed to `gemini-2.5-flash` with your question for a full generated answer

## Requirements

- Python 3.10+
- Google API key with Gemini access
