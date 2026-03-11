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

def search(query: str, top_k: int = 3) -> list[dict]:
    """Embed query and return top-K most similar docs with scores."""
    if not st.session_state.doc_embeddings:
        return []
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_emb = np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Query embedding error: {e}")
        return []

    scores = [
        cosine_similarity(query_emb, doc_emb)
        for doc_emb in st.session_state.doc_embeddings
    ]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**st.session_state.doc_sources[i], "score": scores[i]}
        for i in top_indices
    ]

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
st.subheader("🔍 Search")

if not st.session_state.doc_sources:
    st.warning("Upload at least one file to start searching.")
else:
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g. What is the revenue trend?",
        key="search_query",
    )
    top_k = st.slider("Number of results", min_value=1, max_value=5, value=3, key="top_k")

    if st.button("Search", key="search_btn", disabled=not query):
        with st.spinner("Searching..."):
            results = search(query, top_k=top_k)

        if results:
            st.markdown(f"**Top {len(results)} result(s) for:** *{query}*")
            cols = st.columns(len(results))
            for col, res in zip(cols, results):
                with col:
                    if res["type"] == "image":
                        st.image(res["bytes"], use_container_width=True)
                    elif res["type"] == "pdf":
                        st.image(res["bytes"], use_container_width=True)
                    st.caption(f"**{res['name']}**")
                    st.caption(f"Score: `{res['score']:.4f}`")
        else:
            st.warning("No results found.")
