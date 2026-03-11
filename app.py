import streamlit as st
import numpy as np
from google import genai
from google.genai import types

# page config
st.set_page_config(page_title="Unified RAG", layout="wide")
st.title("Unified RAG 🔍")
st.caption("Multimodal search powered by Gemini Embedding 2")

# sidebar: api key
with st.sidebar:
    st.header("🔑 API Key")
    google_api_key = st.text_input("Google API Key", type="password", key="google_key")
    "[Get a Google API key](https://aistudio.google.com/app/apikey)"

# init client
client = None
if google_api_key:
    client = genai.Client(api_key=google_api_key)
    st.sidebar.success("Client ready!")
else:
    st.info("Enter your Google API key in the sidebar to get started.")
    st.stop()

# embedding helper
def embed_text(text: str) -> np.ndarray | None:
    """Embed a text string using Gemini Embedding 2."""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def embed_image(image_bytes: bytes, mime_type: str = "image/png") -> np.ndarray | None:
    """Embed an image using Gemini Embedding 2."""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def embed_pdf(pdf_bytes: bytes) -> list[np.ndarray]:
    """Embed a PDF using Gemini Embedding 2 (up to 6 pages per call)."""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            ],
        )
        return [np.array(e.values) for e in result.embeddings]
    except Exception as e:
        st.error(f"PDF embedding error: {e}")
        return []

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# session state
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = []  # list of np.ndarray
if "doc_sources" not in st.session_state:
    st.session_state.doc_sources = []  # list of dicts: {name, type, bytes, mime}

# main ui
st.markdown("---")
st.subheader("📤 Upload Files")

uploaded_files = st.file_uploader(
    "Upload images or PDFs",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    new_files = [
        f for f in uploaded_files
        if f.name not in [s["name"] for s in st.session_state.doc_sources]
    ]

    if new_files:
        progress = st.progress(0, text="Embedding files...")
        for i, file in enumerate(new_files):
            file_bytes = file.read()
            mime = file.type  # e.g. "image/png" or "application/pdf"

            if mime == "application/pdf":
                embeddings = embed_pdf(file_bytes)
                for idx, emb in enumerate(embeddings):
                    st.session_state.doc_embeddings.append(emb)
                    st.session_state.doc_sources.append({
                        "name": f"{file.name} · page {idx + 1}",
                        "type": "pdf",
                        "bytes": file_bytes,
                        "mime": mime,
                    })
            else:
                emb = embed_image(file_bytes, mime_type=mime)
                if emb is not None:
                    st.session_state.doc_embeddings.append(emb)
                    st.session_state.doc_sources.append({
                        "name": file.name,
                        "type": "image",
                        "bytes": file_bytes,
                        "mime": mime,
                    })

            progress.progress((i + 1) / len(new_files), text=f"Embedded {file.name}")

        progress.empty()
        st.success(f"Added {len(new_files)} file(s). Total docs: {len(st.session_state.doc_sources)}")
    else:
        st.info("All uploaded files are already loaded.")

# show loaded docs count in sidebar
with st.sidebar:
    st.markdown("---")
    st.caption(f"📚 {len(st.session_state.doc_sources)} document(s) loaded")

st.markdown("---")
if not st.session_state.doc_sources:
    st.warning("Upload at least one file to start searching.")
else:
    st.info(f"{len(st.session_state.doc_sources)} document chunk(s) ready. Search coming next.")
