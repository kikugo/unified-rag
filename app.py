import io
import struct
import wave
import requests
import streamlit as st
import numpy as np
import pypdf
import base64
import uuid
from PIL import Image
from google import genai
from google.genai import types
import chromadb

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
    
    with st.expander("Advanced Vector Storage (ChromaDB)", expanded=False):
        st.caption("Leave blank to use a local, embedded database.")
        st.text_input("Chroma Tenant", value="", key="chroma_tenant")
        st.text_input("Chroma Database", value="", key="chroma_database")
        st.text_input("Chroma API Key", type="password", key="chroma_api_key")

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

# init chroma client
@st.cache_resource
def get_chroma_client(tenant: str, database: str, api_key: str):
    if tenant and database and api_key:
        return chromadb.HttpClient(
            tenant=tenant,
            database=database,
            headers={"x-chroma-token": api_key}
        )
    return chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_tenant = st.session_state.get("chroma_tenant", "")
    chroma_database = st.session_state.get("chroma_database", "")
    chroma_api_key = st.session_state.get("chroma_api_key", "")
    
    chroma_client = get_chroma_client(chroma_tenant, chroma_database, chroma_api_key)
    chroma_collection = chroma_client.get_or_create_collection(
        name="unified_rag",
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {e}")
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
    image_bytes = resize_image_if_needed(image_bytes, mime_type)
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

# ── API limits ────────────────────────────────────────────────────────────────

AUDIO_MAX_SECONDS = 80          # Gemini Embedding 2 hard limit for audio
VIDEO_MAX_SECONDS = 80          # 80s for video-with-audio; 120s without — use conservative limit
IMAGE_MAX_DIM = 4096            # Resize images above this dimension

def get_video_duration_seconds(video_bytes: bytes) -> float | None:
    """Parse MP4/MOV container (pure Python) to extract duration from mvhd box.
    Returns duration in seconds, or None if parsing fails.
    """
    data = video_bytes
    i = 0
    while i + 8 <= len(data):
        try:
            box_size = struct.unpack('>I', data[i:i+4])[0]
            box_type = data[i+4:i+8]
        except struct.error:
            break
        if box_size < 8:
            break
        if box_type == b'moov':
            j = i + 8
            while j + 8 <= i + box_size:
                try:
                    inner_size = struct.unpack('>I', data[j:j+4])[0]
                    inner_type = data[j+4:j+8]
                except struct.error:
                    break
                if inner_type == b'mvhd' and inner_size >= 32:
                    version = data[j+8]
                    if version == 0:
                        timescale = struct.unpack('>I', data[j+20:j+24])[0]
                        duration  = struct.unpack('>I', data[j+24:j+28])[0]
                    else:  # version 1 (64-bit timestamps)
                        timescale = struct.unpack('>I', data[j+28:j+32])[0]
                        duration  = struct.unpack('>Q', data[j+32:j+40])[0]
                    return duration / timescale if timescale > 0 else None
                if inner_size == 0:
                    break
                j += inner_size
        if box_size == 0:
            break
        i += box_size
    return None

def resize_image_if_needed(image_bytes: bytes, mime_type: str) -> bytes:
    """Resize an image that exceeds IMAGE_MAX_DIM using Pillow.
    Preserves aspect ratio. Returns original bytes if already within limit.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if max(w, h) <= IMAGE_MAX_DIM:
            return image_bytes
        ratio = IMAGE_MAX_DIM / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "JPEG" if mime_type in ("image/jpeg", "image/jpg") else "PNG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    except Exception:
        return image_bytes

# ── Audio limit helper ────────────────────────────────────────────────────────
def trim_audio_to_limit(audio_bytes: bytes, mime_type: str, max_seconds: int = AUDIO_MAX_SECONDS) -> bytes:
    """Trim audio to max_seconds. WAV: uses built-in wave module.
    MP3: byte-based estimate (constant bitrate assumption).
    Returns original bytes if already within limit.
    """
    if mime_type == "audio/wav":
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                framerate = wf.getframerate()
                total_frames = wf.getnframes()
                duration = total_frames / framerate
                if duration <= max_seconds:
                    return audio_bytes
                max_frames = int(framerate * max_seconds)
                wf.rewind()
                params = wf.getparams()
                frames = wf.readframes(max_frames)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setparams(params)
                out.writeframes(frames)
            return buf.getvalue()
        except Exception:
            return audio_bytes
    elif mime_type == "audio/mp3":
        # Rough trim by byte proportion (assumes constant bitrate)
        # 128kbps CBR: ~16KB/s. Estimate duration from file size.
        estimated_duration = len(audio_bytes) / 16_000
        if estimated_duration <= max_seconds:
            return audio_bytes
        trim_ratio = max_seconds / estimated_duration
        return audio_bytes[:int(len(audio_bytes) * trim_ratio)]
    return audio_bytes

def embed_audio(audio_bytes: bytes, mime_type: str) -> np.ndarray | None:
    """Embed an audio file using Gemini Embedding 2 (max 80 seconds).
    Automatically trims to 80s if the file is longer.
    """
    audio_bytes = trim_audio_to_limit(audio_bytes, mime_type)
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



def add_document(emb: np.ndarray, name: str, doc_type: str, mime: str, file_bytes: bytes):
    """Helper to add an embedding to ChromaDB and session state."""
    doc_id = str(uuid.uuid4())
    try:
        b64_data = base64.b64encode(file_bytes).decode('utf-8')
        chroma_collection.upsert(
            ids=[doc_id],
            embeddings=[emb.tolist()],
            metadatas=[{"name": name, "type": doc_type, "mime": mime, "data_b64": b64_data}]
        )
        # Keep in session state for UI rendering and temporary search fallback (until Commit 4)
        st.session_state.doc_embeddings.append(emb)
        st.session_state.doc_sources.append({
            "id": doc_id,
            "name": name,
            "type": doc_type,
            "bytes": file_bytes,
            "mime": mime,
        })
    except Exception as e:
        st.error(f"Error saving to database: {e}")

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

def search(query: str, top_k: int = 3) -> list[dict]:
    """Embed query and query ChromaDB for top-K results."""
    try:
        if chroma_collection.count() == 0:
            return []
    except Exception:
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
        query_emb = result.embeddings[0].values
    except Exception as e:
        st.error(f"Query embedding error: {e}")
        return []

    # Query Chroma
    try:
        results = chroma_collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, chroma_collection.count())
        )
    except Exception as e:
        st.error(f"Database query error: {e}")
        return []
    
    # Format the results
    out = []
    if results and results.get("ids") and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
            
            # Reconstruct the file bytes from base64 stored in metadata
            b64_data = meta.get("data_b64", "")
            file_bytes = base64.b64decode(b64_data) if b64_data else b""
            
            out.append({
                "id": doc_id,
                "name": meta.get("name", "Unknown Document"),
                "type": meta.get("type", "unknown"),
                "mime": meta.get("mime", "application/octet-stream"),
                "bytes": file_bytes,
                "score": 1.0 - dist  # Chroma returns cosine distance; metric is 1 - similarity
            })
    return out

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
                add_document(emb, name, "image", mime, img_bytes)
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

    **Supported content and API limits:**

    | Modality | Formats | Limit | Handling |
    |---|---|---|---|
    | Images | PNG, JPG | 4096px (auto-resized) | Auto |
    | PDFs | PDF | 6 pages per call | Auto-chunked |
    | Audio | MP3, WAV | 80 seconds | Auto-trimmed |
    | Video | MP4, MOV | 80 seconds | Rejected with warning |

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
                        add_document(emb, f"{file.name} · {page_label}", "pdf", mime, chunk_bytes)
            elif mime in ("audio/mpeg", "audio/mp3", "audio/wav"):
                # Gemini Embedding API expects "audio/mp3", not "audio/mpeg"
                audio_mime = "audio/mp3" if mime == "audio/mpeg" else mime
                emb = embed_audio(file_bytes, mime_type=audio_mime)
                if emb is not None:
                    add_document(emb, file.name, "audio", mime, file_bytes)
            elif mime in ("video/mp4", "video/quicktime"):
                duration = get_video_duration_seconds(file_bytes)
                if duration is not None and duration > VIDEO_MAX_SECONDS:
                    st.warning(
                        f"⏱️ **{file.name}** is {duration:.0f}s — the Gemini Embedding API "
                        f"only supports videos up to {VIDEO_MAX_SECONDS}s. "
                        f"Please trim your video to under {VIDEO_MAX_SECONDS} seconds and re-upload."
                    )
                else:
                    emb = embed_video(file_bytes, mime_type=mime)
                    if emb is not None:
                        add_document(emb, file.name, "video", mime, file_bytes)
            else:
                if image_caption:
                    emb = embed_image_with_caption(file_bytes, caption=image_caption, mime_type=mime)
                else:
                    emb = embed_image(file_bytes, mime_type=mime)
                if emb is not None:
                    label = f"{file.name} · {image_caption}" if image_caption else file.name
                    add_document(emb, label, "image", mime, file_bytes)

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
