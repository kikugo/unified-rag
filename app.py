import io
import streamlit as st
import numpy as np
import pypdf
from google import genai
from google.genai import types

# page config
st.set_page_config(page_title="Unified RAG", layout="wide")
st.title("Unified RAG 🔍")
st.caption("Multimodal search powered by Gemini Embedding 2")

# session state — must be before sidebar renders
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = []
if "doc_sources" not in st.session_state:
    st.session_state.doc_sources = []
if "active_dim" not in st.session_state:
    st.session_state.active_dim = 3072

# sidebar
with st.sidebar:
    st.header("🔑 API Key")
    google_api_key = st.text_input("Google API Key", type="password", key="google_key")
    "[Get a Google API key](https://aistudio.google.com/app/apikey)"

    st.markdown("---")
    st.subheader("⚙️ Settings")
    st.selectbox("Embedding dimension", [3072, 1536, 768], key="embedding_dim")

    # auto-clear if dimension changed with docs loaded
    if st.session_state.embedding_dim != st.session_state.active_dim:
        if st.session_state.doc_sources:
            st.session_state.doc_embeddings = []
            st.session_state.doc_sources = []
            st.toast("Dimension changed — documents cleared. Re-upload to re-embed.", icon="⚠️")
        st.session_state.active_dim = st.session_state.embedding_dim

    st.markdown("---")
    doc_count = len(st.session_state.doc_sources)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"📚 {doc_count} document(s) loaded")
    with c2:
        if st.button("🗑️", key="clear_btn", disabled=doc_count == 0, help="Clear all"):
            st.session_state.doc_embeddings = []
            st.session_state.doc_sources = []
            st.rerun()

    for i, src in enumerate(st.session_state.doc_sources):
        c1, c2 = st.columns([5, 1])
        with c1:
            icon = "📄" if src["type"] == "pdf" else "🖼️"
            st.caption(f"{icon} {src['name']}")
        with c2:
            if st.button("✕", key=f"rm_{i}", help=f"Remove {src['name']}"):
                st.session_state.doc_embeddings.pop(i)
                st.session_state.doc_sources.pop(i)
                st.rerun()

# init client
client = None
if google_api_key:
    client = genai.Client(api_key=google_api_key)
    st.sidebar.success("Client ready!")
else:
    st.info("Enter your Google API key in the sidebar to get started.")
    st.stop()

# embedding helpers
def embed_text(text: str) -> np.ndarray | None:
    """Embed a text string using Gemini Embedding 2."""
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=dim,
            ),
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def embed_image(image_bytes: bytes, mime_type: str = "image/png") -> np.ndarray | None:
    """Embed an image using Gemini Embedding 2."""
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
            config=types.EmbedContentConfig(output_dimensionality=dim),
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def embed_pdf(pdf_bytes: bytes) -> list[np.ndarray]:
    """Embed a single PDF chunk (max 6 pages) using Gemini Embedding 2."""
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            ],
            config=types.EmbedContentConfig(output_dimensionality=dim),
        )
        return [np.array(e.values) for e in result.embeddings]
    except Exception as e:
        st.error(f"PDF embedding error: {e}")
        return []

def chunk_pdf(pdf_bytes: bytes, chunk_size: int = 6) -> list[tuple[bytes, str]]:
    """Split a PDF into chunks of up to chunk_size pages.
    Returns a list of (chunk_bytes, page_range_label) tuples.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    chunks = []

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        writer = pypdf.PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        buf = io.BytesIO()
        writer.write(buf)
        chunk_bytes = buf.getvalue()

        label = f"pages {start + 1}–{end}" if total_pages > chunk_size else f"page {'1' if total_pages == 1 else f'1–{total_pages}'}"
        chunks.append((chunk_bytes, label))

    return chunks

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search(query: str, top_k: int = 3) -> list[dict]:
    """Embed query and return top-K most similar docs with scores."""
    if not st.session_state.doc_embeddings:
        return []
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=dim,
            ),
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

def answer(question: str, image_bytes: bytes, mime_type: str) -> str:
    """Pass the top retrieved image + question to Gemini 2.5 Flash for an answer."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                question,
            ],
        )
        return response.text
    except Exception as e:
        return f"Generation error: {e}"



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
    # for PDFs, stored names have " · pages X-Y" suffix — match by base filename
    loaded_base_names = {s["name"].split(" · ")[0] for s in st.session_state.doc_sources}
    new_files = [f for f in uploaded_files if f.name not in loaded_base_names]

    if new_files:
        progress = st.progress(0, text="Embedding files...")
        for i, file in enumerate(new_files):
            file_bytes = file.read()
            mime = file.type  # e.g. "image/png" or "application/pdf"

            if mime == "application/pdf":
                chunks = chunk_pdf(file_bytes)
                for chunk_bytes, page_label in chunks:
                    embeddings = embed_pdf(chunk_bytes)
                    for emb in embeddings:
                        st.session_state.doc_embeddings.append(emb)
                        st.session_state.doc_sources.append({
                            "name": f"{file.name} · {page_label}",
                            "type": "pdf",
                            "bytes": chunk_bytes,
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

    if st.button("Search & Answer", key="search_btn", disabled=not query):
        with st.spinner("Searching..."):
            results = search(query, top_k=top_k)

        if results:
            st.markdown(f"**Top {len(results)} result(s) for:** *{query}*")
            cols = st.columns(len(results))
            for col, res in zip(cols, results):
                with col:
                    st.image(res["bytes"], use_container_width=True)
                    st.caption(f"**{res['name']}**")
                    st.caption(f"Score: `{res['score']:.4f}`")

            # generate answer from top result
            st.markdown("---")
            st.subheader("💬 Answer")
            top = results[0]
            with st.spinner("Generating answer from top result..."):
                generated = answer(query, top["bytes"], top["mime"])
            st.markdown(generated)
            st.caption(f"Answer based on: **{top['name']}** (score: `{top['score']:.4f}`)")        
        else:
            st.warning("No results found.")
