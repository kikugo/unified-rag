# Unified RAG

A multimodal Retrieval-Augmented Generation app powered entirely by Google Gemini.

- **Gemini Embedding 2** — embeds images and PDFs into a unified vector space (no OCR, no preprocessing)
- **Gemini 2.5 Flash** — reads the retrieved content and generates a grounded answer

## Features

- Upload images (PNG, JPG) and PDFs of any length
- PDFs over 6 pages are automatically chunked (API limit)
- Cosine similarity search returns the most relevant content for your question
- Gemini Flash answers the question based on the top retrieved result
- Adjustable embedding dimension (3072 / 1536 / 768) via MRL
- Per-document remove and clear-all session controls
- Single Google API key — no other accounts needed

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Get a Gemini API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

## How it works

1. **Upload** — images and PDFs are embedded on upload using `gemini-embedding-2-preview`
2. **Search** — your question is embedded and scored against all stored embeddings via cosine similarity
3. **Answer** — the top result is passed to `gemini-2.5-flash` with your question for a full generated answer

## Requirements

- Python 3.10+
- Google API key with Gemini access
