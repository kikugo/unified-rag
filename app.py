import io
import requests
import streamlit as st
import numpy as np
import pypdf
from PIL import Image
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
    st.caption("🔒 Your key is held in memory for this session only and is never stored or sent anywhere.")

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

def embed_image_with_caption(image_bytes: bytes, caption: str, mime_type: str = "image/png") -> np.ndarray | None:
    """Embed an image interleaved with a text caption as one unified vector.
    Produces richer embeddings than image-only when caption provides useful context.
    """
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=types.Content(
                parts=[
                    types.Part(text=caption),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ]
            ),
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

def embed_audio(audio_bytes: bytes, mime_type: str) -> np.ndarray | None:
    """Embed an audio file using Gemini Embedding 2 (max 80 seconds)."""
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            ],
            config=types.EmbedContentConfig(output_dimensionality=dim),
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Audio embedding error: {e}")
        return None

def embed_video(video_bytes: bytes, mime_type: str) -> np.ndarray | None:
    """Embed a video using Gemini Embedding 2 (max 128 seconds)."""
    dim = st.session_state.get("embedding_dim", 3072)
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
            ],
            config=types.EmbedContentConfig(output_dimensionality=dim),
        )
        return np.array(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Video embedding error: {e}")
        return None



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

SAMPLE_IMAGES = {
    "Tesla Q4 2024": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbef936e6-3efa-43b3-88d7-7ec620cdb33b_2744x1539.png",
    "Netflix Q4 2024": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23bd84c9-5b62-4526-b467-3088e27e4193_2744x1539.png",
    "Nike Q4 2024": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
    "Google Q4 2024": "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F395dd3b9-b38e-4d1f-91bc-d37b642ee920_2741x1541.png",
    "Accenture Q4 2024": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08b2227c-7dc8-49f7-b3c5-13cab5443ba6_2741x1541.png",
    "Tencent Q4 2024": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ec8448c-c4d1-4aab-a8e9-2ddebe0c95fd_2741x1541.png",
}

def load_sample_images():
    """Download sample financial chart images and embed them into session state."""
    loaded_base_names = {s["name"].split(" · ")[0] for s in st.session_state.doc_sources}
    to_load = {name: url for name, url in SAMPLE_IMAGES.items() if name not in loaded_base_names}

    if not to_load:
        st.info("Sample images already loaded.")
        return

    progress = st.progress(0, text="Loading sample images...")
    for i, (name, url) in enumerate(to_load.items()):
        progress.progress(i / len(to_load), text=f"Downloading {name}...")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            # Convert to PNG via Pillow — ensures a supported format regardless of what
            # the CDN returns (WebP, JPEG, etc.). Pillow is already in requirements.txt.
            raw = Image.open(io.BytesIO(resp.content)).convert("RGB")
            buf = io.BytesIO()
            raw.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            mime = "image/png"

            progress.progress(i / len(to_load), text=f"Embedding {name}...")
            emb = embed_image(img_bytes, mime_type=mime)
            if emb is not None:
                st.session_state.doc_embeddings.append(emb)
                st.session_state.doc_sources.append({
                    "name": name,
                    "type": "image",
                    "bytes": img_bytes,
                    "mime": mime,
                })
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

    progress.progress(1.0, text="Done!")
    progress.empty()
    st.success(f"Loaded {len(to_load)} sample image(s).")

# about expander
with st.expander("ℹ️ About this app"):
    st.markdown("""
    **Unified RAG** uses **Gemini Embedding 2** (`gemini-embedding-2-preview`) to embed your content
    into a single unified vector space, and **Gemini 2.5 Flash** to generate answers from retrieved content.

    **Supported content:**
    - 🖼️ Images (PNG, JPG)
    - 📄 PDFs (any length — auto-chunked into 6-page segments)
    - 🎧 Audio (MP3, WAV) — *coming soon*
    - 🎥 Video (MP4, MOV) — *coming soon*

    **How it works:** Upload content → Gemini Embedding 2 vectorizes it → your question is
    embedded and scored against all vectors via cosine similarity → top result is passed to
    Gemini Flash for a grounded answer.
    """)

st.markdown("---")
st.subheader("📊 Load Sample Images")
st.caption("Try the app instantly with pre-loaded financial charts from Tesla, Netflix, Nike, Google, Accenture, and Tencent.")
if st.button("Load Sample Images", key="load_samples_btn"):
    load_sample_images()

# main ui
st.markdown("---")
st.subheader("📤 Upload Files")

image_caption = st.text_input(
    "Optional caption for uploaded images",
    placeholder="e.g. Tesla Q4 2024 earnings slide — embeds text + image together for better retrieval",
    key="image_caption",
)

uploaded_files = st.file_uploader(
    "Upload images, PDFs, audio, or video files",
    type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "mp4", "mov"],
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
            elif mime in ("audio/mpeg", "audio/wav"):
                emb = embed_audio(file_bytes, mime_type=mime)
                if emb is not None:
                    st.session_state.doc_embeddings.append(emb)
                    st.session_state.doc_sources.append({
                        "name": file.name,
                        "type": "audio",
                        "bytes": file_bytes,
                        "mime": mime,
                    })
            elif mime in ("video/mp4", "video/quicktime"):
                emb = embed_video(file_bytes, mime_type=mime)
                if emb is not None:
                    st.session_state.doc_embeddings.append(emb)
                    st.session_state.doc_sources.append({
                        "name": file.name,
                        "type": "video",
                        "bytes": file_bytes,
                        "mime": mime,
                    })
            else:
                if image_caption:
                    emb = embed_image_with_caption(file_bytes, caption=image_caption, mime_type=mime)
                else:
                    emb = embed_image(file_bytes, mime_type=mime)
                if emb is not None:
                    label = f"{file.name} · {image_caption}" if image_caption else file.name
                    st.session_state.doc_embeddings.append(emb)
                    st.session_state.doc_sources.append({
                        "name": label,
                        "type": "image",
                        "bytes": file_bytes,
                        "mime": mime,
                    })

            progress.progress((i + 1) / len(new_files), text=f"Embedded {file.name}")

        progress.empty()
        st.success(f"Added {len(new_files)} file(s). Total docs: {len(st.session_state.doc_sources)}")
    else:
        st.info("All uploaded files are already loaded.")

# loaded content gallery
if st.session_state.doc_sources:
    with st.expander(f"View Loaded Content ({len(st.session_state.doc_sources)} items)", expanded=False):
        cols = st.columns(5)
        for i, src in enumerate(st.session_state.doc_sources):
            with cols[i % 5]:
                if src["type"] == "image":
                    st.image(src["bytes"], width='stretch')
                elif src["type"] == "audio":
                    st.audio(src["bytes"], format=src["mime"])
                elif src["type"] == "video":
                    st.markdown("🎥")
                else:
                    st.markdown("📄")
                st.caption(src["name"])



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
                    if res["type"] == "image":
                        st.image(res["bytes"], width='stretch')
                    elif res["type"] == "audio":
                        st.audio(res["bytes"], format=res["mime"])
                    elif res["type"] == "video":
                        st.video(res["bytes"])
                    else:
                        st.markdown("📄 PDF")
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

# image-as-query search
st.markdown("---")
st.subheader("🖼️ Search by Image")
st.caption("Upload an image as your query — Gemini Embedding 2 will find the most visually and semantically similar content in your library.")

if not st.session_state.doc_sources:
    st.warning("Upload at least one file to your library first.")
else:
    query_image = st.file_uploader(
        "Upload a query image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="image_query_uploader",
        label_visibility="collapsed",
    )
    top_k_img = st.slider("Number of results", min_value=1, max_value=5, value=3, key="top_k_img")

    if query_image and st.button("Find Similar", key="img_search_btn"):
        query_bytes = query_image.read()
        query_mime = query_image.type
        dim = st.session_state.get("embedding_dim", 3072)

        with st.spinner("Embedding query image..."):
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-2-preview",
                    contents=[types.Part.from_bytes(data=query_bytes, mime_type=query_mime)],
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=dim,
                    ),
                )
                query_emb = np.array(result.embeddings[0].values)
            except Exception as e:
                st.error(f"Query embedding error: {e}")
                query_emb = None

        if query_emb is not None:
            scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in st.session_state.doc_embeddings]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_img]
            img_results = [{**st.session_state.doc_sources[i], "score": scores[i]} for i in top_indices]

            st.markdown(f"**Top {len(img_results)} result(s) for your query image:**")
            cols = st.columns(len(img_results))
            for col, res in zip(cols, img_results):
                with col:
                    if res["type"] == "image":
                        st.image(res["bytes"], width='stretch')
                    elif res["type"] == "audio":
                        st.audio(res["bytes"], format=res["mime"])
                    elif res["type"] == "video":
                        st.video(res["bytes"])
                    else:
                        st.markdown("📄 PDF")
                    st.caption(f"**{res['name']}**")
                    st.caption(f"Score: `{res['score']:.4f}`")

# footer
st.markdown("---")
st.caption("Unified RAG · Powered by Gemini Embedding 2 and Gemini 2.5 Flash")
