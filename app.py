import io
import json
import struct
import wave
import requests
import re
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import fitz  # PyMuPDF
import av
import base64
import uuid
import tempfile
import os
from PIL import Image
from google import genai
from google.genai import types
import chromadb
from rag_core import (
    MANAGED_BACKEND,
    bm25_rank,
    build_local_answer_prompt,
    pop_source_and_embedding,
    reciprocal_rank_fusion,
    select_context_results,
)
from storage import delete_blob, persist_blob, read_blob
from source_registry import delete_source, list_sources, upsert_source

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
if "google_store" not in st.session_state:
    st.session_state.google_store = None
if "messages" not in st.session_state:
    # each msg: {"role": "user"|"assistant", "content": str, "results": [...], "citations": [...]}
    st.session_state.messages = []
if "tts_pending" not in st.session_state:
    st.session_state.tts_pending = None
if "corpus_id" not in st.session_state:
    st.session_state.corpus_id = str(uuid.uuid4())
if "pending_clear_all" not in st.session_state:
    st.session_state.pending_clear_all = False
if "pending_delete_chroma_ids" not in st.session_state:
    st.session_state.pending_delete_chroma_ids = []
if "pending_cleanup_paths" not in st.session_state:
    st.session_state.pending_cleanup_paths = []
if "load_samples_requested" not in st.session_state:
    st.session_state.load_samples_requested = False
if "selected_citation_source_id" not in st.session_state:
    st.session_state.selected_citation_source_id = None

client = None


def speak_text(text: str):
    """Fire a Web Speech API utterance in the browser. Zero-height iframe, no dependencies."""
    # Sanitise the text so it's safe to embed in a JS string literal via json.dumps
    js_text = json.dumps(text)
    components.html(
        f"""
        <script>
            (function() {{
                if (!window.speechSynthesis) {{ return; }}
                var u = new SpeechSynthesisUtterance({js_text});
                u.rate = 1.0;
                u.pitch = 1.05;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
            }})();
        </script>
        """,
        height=0,
    )


# Fire TTS if a message was queued last run
if st.session_state.tts_pending:
    speak_text(st.session_state.tts_pending)
    st.session_state.tts_pending = None

# sidebar
with st.sidebar:
    st.header("🔑 API Key")
    google_api_key = st.text_input("Google API Key", type="password", key="google_key")
    "[Get a Google API key](https://aistudio.google.com/app/apikey)"
    st.caption("🔒 Your key is held in memory for this session only and is never stored or sent anywhere.")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    retrieval_strategy = st.radio(
        "Retrieval Strategy",
        ["Auto (Hybrid)", "Local Vector Store (Chroma)", "Managed RAG (Google File Search)"],
        help="Auto: Gemini classifies each query and picks the best backend. Local: always searches your embedded files. Managed: always queries Google File Search.",
        key="retrieval_strategy"
    )
    
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
            st.session_state.corpus_id = str(uuid.uuid4())
            st.toast("Dimension changed — documents cleared. Re-upload to re-embed.", icon="⚠️")
        st.session_state.active_dim = st.session_state.embedding_dim

    st.markdown("---")
    doc_count = len(st.session_state.doc_sources)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"📚 {doc_count} document(s) loaded")
    with c2:
        if st.button("🗑️", key="clear_btn", disabled=doc_count == 0, help="Clear all"):
            st.session_state.pending_clear_all = True

    # Document Manager
    TYPE_ICONS = {"pdf": "📄", "audio": "🎵", "video": "🎥", "image": "🖼️", "video_frame": "🎞️"}
    sources = st.session_state.doc_sources

    if not sources:
        st.caption("📭 No documents loaded yet.")
    else:
        import pandas as pd

        type_options = ["All"] + sorted({s["type"] for s in sources})
        type_filter = st.selectbox("Filter by type", type_options, key="doc_type_filter", label_visibility="collapsed")

        # Build dataframe; "#" tracks original index for safe deletion
        df = pd.DataFrame([
            {"#": i, "Icon": TYPE_ICONS.get(s["type"], "📎"), "Name": s["name"], "Type": s["type"]}
            for i, s in enumerate(sources)
            if type_filter == "All" or s["type"] == type_filter
        ])

        event = st.dataframe(
            df[["Icon", "Name", "Type"]],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="doc_table",
        )

        selected_orig_idx = (
            int(df.iloc[event.selection.rows[0]]["#"]) if event.selection.rows else None
        )

        # Delete selected row
        col_del, col_count = st.columns([1, 2])
        with col_del:
            if st.button("🗑 Delete", key="delete_row_btn", disabled=selected_orig_idx is None):
                removed, chroma_ids = pop_source_and_embedding(
                    st.session_state.doc_sources,
                    st.session_state.doc_embeddings,
                    selected_orig_idx,
                )
                for path_key in ("file_path", "preview_path", "video_path"):
                    if removed.get(path_key):
                        st.session_state.pending_cleanup_paths.append(removed[path_key])
                st.session_state.pending_delete_chroma_ids.extend(chroma_ids)
                st.rerun()
        with col_count:
            shown = len(df)
            total = len(sources)
            label = f"{shown} of {total}" if shown != total else f"{total}"
            st.caption(f"📚 {label} document(s)")

    st.markdown("---")
    st.subheader("🔍 Image Query")
    st.caption("Search by uploading an image instead of typing.")
    query_image_sidebar = st.file_uploader(
        "Query image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="image_query_uploader",
        label_visibility="collapsed",
    )
    top_k_img = st.slider("Results", min_value=1, max_value=5, value=3, key="top_k_img")
    run_img_query = st.button("🔍 Find Similar", key="img_search_btn",
                              disabled=not (query_image_sidebar and st.session_state.doc_sources))

    st.markdown("---")
    st.subheader("📤 Upload")
    image_caption = st.text_input(
        "Image caption (optional)",
        placeholder="e.g. Tesla Q4 2024 earnings slide",
        key="image_caption",
    )
    image_captions_map_raw = st.text_area(
        "Per-image captions (one per line: filename|caption)",
        value="",
        key="image_captions_map",
        help="Example: chart1.png|Revenue by quarter",
    )
    auto_caption_images = st.checkbox("Auto-caption images when caption missing", value=True, key="auto_caption_images")
    uploaded_files = st.file_uploader(
        "Files",
        type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "mp4", "mov"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("⚡ Load Sample Images", key="load_samples_btn", help="Pre-load 6 financial charts"):
        st.session_state.load_samples_requested = True

# init client
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
        name=f"unified_rag_{st.session_state.active_dim}",
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {e}")
    st.stop()

def delete_chroma_ids(ids: list[str]):
    if not ids:
        return
    try:
        chroma_collection.delete(ids=ids)
    except Exception as e:
        st.warning(f"Could not delete some local index entries: {e}")

def cleanup_local_artifacts(source: dict):
    for path_key in ("file_path", "preview_path", "video_path"):
        delete_blob(source.get(path_key))
    source_id = source.get("source_id")
    if source_id:
        delete_source(source_id)

def delete_current_corpus_from_chroma():
    try:
        existing = chroma_collection.get(where={"corpus_id": st.session_state.corpus_id})
        for meta in existing.get("metadatas", []) or []:
            cleanup_local_artifacts(meta or {})
        delete_chroma_ids(existing.get("ids", []))
    except Exception as e:
        st.warning(f"Could not clear local index entries: {e}")

if st.session_state.pending_delete_chroma_ids:
    delete_chroma_ids(st.session_state.pending_delete_chroma_ids)
    st.session_state.pending_delete_chroma_ids = []

if st.session_state.pending_cleanup_paths:
    for path in st.session_state.pending_cleanup_paths:
        delete_blob(path)
    st.session_state.pending_cleanup_paths = []

if st.session_state.pending_clear_all:
    delete_current_corpus_from_chroma()
    st.session_state.doc_embeddings = []
    st.session_state.doc_sources = []
    st.session_state.messages = []
    st.session_state.corpus_id = str(uuid.uuid4())
    if st.session_state.google_store is not None and client is not None:
        try:
            client.file_search_stores.delete(name=st.session_state.google_store.name, config={'force': True})
        except Exception as e:
            st.warning(f"Could not delete Google File Search store: {e}")
        st.session_state.google_store = None
    st.session_state.pending_clear_all = False
    st.rerun()

# Google File Search helper
def get_or_create_google_store():
    """Create an ephemeral Google File Search store on-demand to avoid empty cloud resources."""
    if st.session_state.google_store is None:
        try:
            st.session_state.google_store = client.file_search_stores.create(
                config={'display_name': 'unified-rag-ephemeral-store'}
            )
        except Exception as e:
            st.error(f"Failed to create Google File Search Store: {e}")
    return st.session_state.google_store

def add_file_to_google_store(file_name: str, file_bytes: bytes, mime: str):
    """Save bytes to a temp file, upload directly to the managed Google File Search store."""
    store = get_or_create_google_store()
    if not store:
        return
        
    ext = os.path.splitext(file_name)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        # Directly upload to the file search store
        client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store.name,
            config={'display_name': file_name}
        )
        # Add to UI state so it shows in the gallery
        doc_type = "pdf" if mime == "application/pdf" else ("audio" if "audio" in mime else ("video" if "video" in mime else "image"))
        st.session_state.doc_sources.append({
            "source_id": str(uuid.uuid4()),
            "name": file_name,
            "type": doc_type,
            "mime": mime,
            "backend": MANAGED_BACKEND,
        })
    except Exception as e:
        st.error(f"Google Upload Error for {file_name}: {e}")
    finally:
        os.remove(tmp_path)


def route_query(question: str) -> str:
    """Classify whether a question is best answered by the local vector store or
    the Google Managed RAG backend, using a fast Gemini Flash call.

    Returns 'local' or 'managed'.
    - 'local'   → best for specific visual content, charts, images, audio, video clips
    - 'managed' → best for broad text summarisation, multi-document synthesis,
                  or anything needing full-text search across large corpora
    """
    prompt = (
        "You are a query router for a multimodal RAG application. "
        "Given a user question, decide which backend is best:\n"
        "- 'local': precise lookup in locally embedded images, PDF pages, audio, or video frame timestamps\n"
        "- 'managed': broad text search / summarisation across large text documents\n\n"
        f"User question: {question}\n\n"
        "Reply with exactly one word: local OR managed"
    )
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        answer = resp.text.strip().lower()
        return "managed" if "managed" in answer else "local"
    except Exception:
        return "local"  # safe fallback


def parse_caption_map(raw: str) -> dict[str, str]:
    mapping = {}
    for line in (raw or "").splitlines():
        if "|" not in line:
            continue
        name, caption = line.split("|", 1)
        name = name.strip()
        caption = caption.strip()
        if name and caption:
            mapping[name] = caption
    return mapping


def auto_caption_image(image_bytes: bytes, mime_type: str) -> str:
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text="Write a concise search caption for this image in one sentence."),
            ],
        )
        return (resp.text or "").strip()
    except Exception:
        return ""


def expand_query_variants(query: str) -> list[str]:
    q = query.strip()
    if not q:
        return []
    variants = {q}
    if len(q.split()) <= 5:
        variants.add(f"{q} details")
        variants.add(f"explain {q}")
    return [v for v in variants if v != q]


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

def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return ""

# ── API limits ────────────────────────────────────────────────────────────────

AUDIO_MAX_SECONDS = 80          # Gemini Embedding 2 hard limit for a single audio chunk
VIDEO_MAX_SECONDS = 80          # Legacy limit kept for compatibility; long videos now chunked via frame sampling
FRAME_INTERVAL_SEC = 5          # Extract a frame every N seconds for Deep Video Search
IMAGE_MAX_DIM = 4096            # Resize images above this dimension

def extract_video_frames(video_bytes: bytes, interval_sec: int = FRAME_INTERVAL_SEC) -> list[tuple[bytes, float]]:
    """Extract frames from an MP4/MOV at a given interval using PyAV.
    Yields (frame_png_bytes, timestamp_sec) tuples.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name
    try:
        container = av.open(tmp_path)
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if not video_stream:
            return []
        fps = float(video_stream.average_rate or 25)
        # Avoid zero/negative interval sizes
        interval_sec = max(1, interval_sec)
        last_yielded_sec = -interval_sec
        frames = []
        for i, frame in enumerate(container.decode(video_stream)):
            ts = float(frame.pts * video_stream.time_base) if frame.pts else i / fps
            if ts - last_yielded_sec >= interval_sec:
                img = frame.to_image()                  # PIL Image
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                frames.append((buf.getvalue(), ts))
                last_yielded_sec = ts
        container.close()
        return frames
    except Exception as e:
        print(f"Error extracting video frames: {e}")
        return []
    finally:
        os.remove(tmp_path)

def format_timestamp(sec: float) -> str:
    """Convert seconds into MM:SS format."""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"

def get_video_duration_seconds(video_bytes: bytes) -> float | None:
    """Use PyAV to extract duration in seconds."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        with av.open(tmp_path) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            return None
    except Exception:
        return None
    finally:
        os.remove(tmp_path)


def choose_frame_interval(duration_sec: float | None) -> int:
    if not duration_sec:
        return FRAME_INTERVAL_SEC
    if duration_sec <= 120:
        return 5
    if duration_sec <= 600:
        return 10
    return 20

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
def chunk_audio_bytes(audio_bytes: bytes, mime_type: str, chunk_seconds: int = AUDIO_MAX_SECONDS) -> list[tuple[bytes, float, float]]:
    """Chunk audio into windows with timestamps; MP3 uses proportional chunking fallback."""
    if mime_type == "audio/wav":
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                framerate = wf.getframerate()
                total_frames = wf.getnframes()
                params = wf.getparams()
                duration = total_frames / framerate
                if duration <= chunk_seconds:
                    return [(audio_bytes, 0.0, duration)]
                windows = []
                start_sec = 0.0
                while start_sec < duration:
                    start_frame = int(start_sec * framerate)
                    end_frame = int(min((start_sec + chunk_seconds), duration) * framerate)
                    wf.setpos(start_frame)
                    frames = wf.readframes(end_frame - start_frame)
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as out:
                        out.setparams(params)
                        out.writeframes(frames)
                    windows.append((buf.getvalue(), start_sec, min(start_sec + chunk_seconds, duration)))
                    start_sec += chunk_seconds
                return windows
        except Exception:
            return [(audio_bytes, 0.0, float(chunk_seconds))]
    estimated_duration = len(audio_bytes) / 16_000
    if estimated_duration <= chunk_seconds:
        return [(audio_bytes, 0.0, estimated_duration)]
    windows = []
    start_sec = 0.0
    while start_sec < estimated_duration:
        end_sec = min(start_sec + chunk_seconds, estimated_duration)
        start_byte = int((start_sec / estimated_duration) * len(audio_bytes))
        end_byte = int((end_sec / estimated_duration) * len(audio_bytes))
        windows.append((audio_bytes[start_byte:end_byte], start_sec, end_sec))
        start_sec += chunk_seconds
    return windows

def embed_audio(audio_bytes: bytes, mime_type: str) -> np.ndarray | None:
    """Embed an audio file using Gemini Embedding 2 (max 80 seconds).
    Automatically trims to 80s if the file is longer.
    """
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



def add_document(emb: np.ndarray, name: str, doc_type: str, mime: str, file_bytes: bytes,
                 preview_bytes: bytes | None = None, video_path: str | None = None,
                 searchable_text: str = "", source_meta: dict | None = None):
    """Helper to add an embedding to ChromaDB and session state."""
    doc_id = str(uuid.uuid4())
    try:
        file_path = persist_blob(file_bytes, mime, name)
        metadata: dict = {
            "name": name,
            "type": doc_type,
            "mime": mime,
            "file_path": file_path,
            "corpus_id": st.session_state.corpus_id,
            "searchable_text": searchable_text or name,
        }
        if preview_bytes:
            metadata["preview_path"] = persist_blob(preview_bytes, "image/png", f"{name}_preview.png")
        if video_path:
            metadata["video_path"] = video_path
        if source_meta:
            metadata.update(source_meta)
        chroma_collection.upsert(
            ids=[doc_id],
            embeddings=[emb.tolist()],
            metadatas=[metadata]
        )
        source_id = metadata.get("source_id") or str(uuid.uuid4())
        metadata["source_id"] = source_id
        st.session_state.doc_embeddings.append(emb)
        source_row = {
            "source_id": source_id,
            "id": doc_id,
            "name": name,
            "type": doc_type,
            "mime": mime,
            "file_path": metadata["file_path"],
            "preview_path": metadata.get("preview_path"),
            "video_path": metadata.get("video_path"),
            "backend": "local",
            "searchable_text": searchable_text or name,
        }
        if source_meta:
            source_row.update(source_meta)
        st.session_state.doc_sources.append(source_row)
        upsert_source({
            "source_id": source_id,
            "doc_id": doc_id,
            "name": name,
            "type": doc_type,
            "backend": "local",
            "mime": mime,
            "file_path": metadata["file_path"],
            "preview_path": metadata.get("preview_path"),
            "video_path": metadata.get("video_path"),
            "searchable_text": searchable_text or name,
            "corpus_id": st.session_state.corpus_id,
            "timestamp_start_sec": (source_meta or {}).get("timestamp_start_sec"),
            "timestamp_end_sec": (source_meta or {}).get("timestamp_end_sec"),
            "page_range": (source_meta or {}).get("page_range"),
            "caption": (source_meta or {}).get("caption"),
            "transcript": (source_meta or {}).get("transcript", ""),
            "chroma_ids": [doc_id],
        })
    except Exception as e:
        st.error(f"Error saving to database: {e}")

def chunk_pdf(pdf_bytes: bytes, chunk_size: int = 4, overlap_pages: int = 1) -> list[tuple[bytes, str, bytes]]:
    """Split a PDF into overlapping page windows using PyMuPDF.

    Returns a list of (chunk_pdf_bytes, page_range_label, preview_png_bytes).
    preview_png_bytes is a rendered image of the first page of that chunk at 72 DPI.
    """
    src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = src_doc.page_count
    chunks = []

    step = max(1, chunk_size - overlap_pages)
    for start in range(0, total_pages, step):
        end = min(start + chunk_size, total_pages)

        # Build a sub-document for this page range
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(src_doc, from_page=start, to_page=end - 1)
        chunk_bytes = chunk_doc.tobytes()

        # Render first page of the chunk as a PNG thumbnail
        page = chunk_doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))  # 72 DPI
        preview_png = pix.tobytes(output="png")
        chunk_doc.close()

        if total_pages > chunk_size:
            label = f"pages {start + 1}\u2013{end}"
        else:
            label = "page 1" if total_pages == 1 else f"pages 1\u2013{total_pages}"

        chunks.append((chunk_bytes, label, preview_png))
        if end == total_pages:
            break

    src_doc.close()
    return chunks

def search(query: str = None, top_k: int = 3, query_emb: np.ndarray = None) -> list[dict]:
    """Embed query and query ChromaDB for top-K results."""
    try:
        corpus_docs = chroma_collection.get(where={"corpus_id": st.session_state.corpus_id})
        if not corpus_docs.get("ids"):
            return []
    except Exception:
        return []

    effective_query = query or ""
    variants = expand_query_variants(effective_query)
    expanded_query = " ".join([effective_query, *variants]).strip()

    if query_emb is None and query:
        dim = st.session_state.get("embedding_dim", 3072)
        try:
            result = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=expanded_query or query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=dim,
                ),
            )
            query_emb = np.array(result.embeddings[0].values)
        except Exception as e:
            st.error(f"Query embedding error: {e}")
            return []
    elif query_emb is None:
        return []

    candidate_count = min(max(top_k * 4, top_k), len(corpus_docs["ids"]))

    # Query Chroma
    try:
        results = chroma_collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=candidate_count,
            where={"corpus_id": st.session_state.corpus_id},
        )
    except Exception as e:
        st.error(f"Database query error: {e}")
        return []
    
    corpus_metas = corpus_docs.get("metadatas") or []
    sparse_docs = [
        {"id": doc_id, "searchable_text": meta.get("searchable_text", meta.get("name", ""))}
        for doc_id, meta in zip(corpus_docs["ids"], corpus_metas)
        if meta
    ]
    dense_ids = results["ids"][0] if results and results.get("ids") and results["ids"] else []
    sparse_ids = [doc_id for doc_id, _ in bm25_rank(expanded_query or query or "", sparse_docs)[:candidate_count]]
    fused_ids = reciprocal_rank_fusion([dense_ids, sparse_ids])[:top_k]
    fused_position = {doc_id: rank for rank, doc_id in enumerate(fused_ids)}

    result_rows = []
    if dense_ids:
        for i, doc_id in enumerate(dense_ids):
            result_rows.append((
                doc_id,
                results["metadatas"][0][i],
                results["distances"][0][i] if "distances" in results and results["distances"] else 0.0,
            ))

    missing_ids = [doc_id for doc_id in fused_ids if doc_id not in dense_ids]
    if missing_ids:
        try:
            sparse_only = chroma_collection.get(ids=missing_ids)
            for doc_id, meta in zip(sparse_only.get("ids", []), sparse_only.get("metadatas", [])):
                result_rows.append((doc_id, meta, 1.0))
        except Exception as e:
            st.warning(f"Could not load keyword-only matches: {e}")

    result_rows.sort(key=lambda row: fused_position.get(row[0], len(fused_position)))

    # Format the results
    out = []
    for doc_id, meta, dist in result_rows[:top_k]:
        # Preferred path-based storage with fallback to legacy base64 metadata.
        file_bytes = read_blob(meta.get("file_path"))
        if not file_bytes:
            b64_data = meta.get("data_b64", "")
            file_bytes = base64.b64decode(b64_data) if b64_data else b""

        preview_bytes = read_blob(meta.get("preview_path"))
        if not preview_bytes:
            preview_b64 = meta.get("preview_b64", "")
            preview_bytes = base64.b64decode(preview_b64) if preview_b64 else None

        video_bytes = read_blob(meta.get("video_path"))

        out.append({
            "source_id": meta.get("source_id"),
            "id": doc_id,
            "name": meta.get("name", "Unknown Document"),
            "type": meta.get("type", "unknown"),
            "mime": meta.get("mime", "application/octet-stream"),
            "bytes": file_bytes,
            "preview_bytes": preview_bytes,
            "video_bytes": video_bytes if video_bytes else None,
            "score": 1.0 - dist,  # Chroma returns cosine distance; metric is 1 - similarity
            "searchable_text": meta.get("searchable_text", ""),
        })
    # Light rerank pass combining dense score and lexical overlap with query terms.
    query_terms = set(re.findall(r"[a-z0-9_]+", (expanded_query or "").lower()))
    for row in out:
        text_terms = set(re.findall(r"[a-z0-9_]+", row.get("searchable_text", "").lower()))
        overlap = len(query_terms & text_terms) / max(1, len(query_terms))
        row["rerank_score"] = row.get("score", 0.0) * 0.8 + overlap * 0.2
    out.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
    return out

def answer(question: str, selected_results: list[dict],
           history: list[dict] | None = None):
    """Stream a grounded answer from Gemini 2.5 Flash, with optional conversation history.

    history: list of {role, content} dicts from st.session_state.messages (most recent last).
    """
    # Build multi-turn contents: [prior turns...] + [retrieved doc] + [current question]
    MAX_HISTORY = 6  # keep last 3 exchanges to stay within context limits
    contents: list = []
    if history:
        for msg in history[-(MAX_HISTORY):]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    # Append current turn: retrieved document + the question
    prompt = build_local_answer_prompt(question, selected_results)
    parts = []
    for result in selected_results:
        parts.append(types.Part(text=f"Source [{result['citation_id']}] content follows."))
        if result.get("bytes"):
            parts.append(types.Part.from_bytes(
                data=result["bytes"],
                mime_type=result.get("mime", "application/octet-stream"),
            ))
    parts.append(types.Part(text=prompt))
    contents.append(types.Content(
        role="user",
        parts=parts,
    ))
    try:
        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Generation error: {e}"

def answer_managed(question: str,
                   history: list[dict] | None = None) -> tuple[object, list[str]]:
    """Stream a grounded answer from Gemini 2.5 Pro using the Google File Search store,
    with optional conversation history for multi-turn context."""
    store = get_or_create_google_store()
    if not store:
        def _error():
            yield "No Managed File Search store available."
        return _error(), []

    # Build multi-turn contents list
    MAX_HISTORY = 6
    contents: list = []
    if history:
        for msg in history[-(MAX_HISTORY):]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    citations = []
    try:
        stream = client.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store.name]
                        )
                    )
                ]
            )
        )
        full_chunks = list(stream)
        if full_chunks and full_chunks[-1].candidates:
            meta = full_chunks[-1].candidates[0].grounding_metadata
            if meta and hasattr(meta, 'grounding_chunks'):
                for chunk in meta.grounding_chunks:
                    if hasattr(chunk, 'retrieved_context') and chunk.retrieved_context:
                        title = getattr(chunk.retrieved_context, 'title', None)
                        if title and title not in citations:
                            citations.append(title)

        def _gen():
            for c in full_chunks:
                if c.text:
                    yield c.text

        return _gen(), citations
    except Exception as e:
        def _error():
            yield f"Managed generation error: {e}"
        return _error(), []

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
                add_document(emb, name, "image", mime, img_bytes,
                             searchable_text=name)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

    progress.progress(1.0, text="Done!")
    progress.empty()
    st.success(f"Loaded {len(to_load)} sample image(s).")

if st.session_state.load_samples_requested:
    st.session_state.load_samples_requested = False
    load_sample_images()

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
    embedded and scored against all vectors via cosine similarity → top retrieved results are passed to
    Gemini Flash for a grounded answer.
    """)

# process uploads from sidebar
if uploaded_files:
    per_file_captions = parse_caption_map(image_captions_map_raw)
    loaded_base_names = {s["name"].split(" · ")[0] for s in st.session_state.doc_sources}
    new_files = [f for f in uploaded_files if f.name not in loaded_base_names]

    if new_files:
        progress = st.sidebar.progress(0, text="Embedding…")
        for i, file in enumerate(new_files):
            file_bytes = file.read()
            mime = file.type

            if retrieval_strategy == "Managed RAG (Google File Search)":
                add_file_to_google_store(file.name, file_bytes, mime)
            elif mime == "application/pdf":
                for chunk_bytes, page_label, preview_png in chunk_pdf(file_bytes):
                    chunk_text = extract_pdf_text(chunk_bytes)
                    for emb in embed_pdf(chunk_bytes):
                        add_document(emb, f"{file.name} · {page_label}", "pdf", mime, chunk_bytes,
                                     preview_bytes=preview_png,
                                     searchable_text=f"{file.name} {page_label}\n{chunk_text}",
                                     source_meta={"page_range": page_label})
            elif mime in ("audio/mpeg", "audio/mp3", "audio/wav"):
                audio_mime = "audio/mp3" if mime == "audio/mpeg" else mime
                audio_chunks = chunk_audio_bytes(file_bytes, audio_mime)
                for chunk_idx, (chunk_bytes, start_sec, end_sec) in enumerate(audio_chunks, start=1):
                    emb = embed_audio(chunk_bytes, mime_type=audio_mime)
                    if emb is not None:
                        label = f"{file.name} · {format_timestamp(start_sec)}-{format_timestamp(end_sec)}"
                        add_document(
                            emb,
                            label,
                            "audio",
                            mime,
                            chunk_bytes,
                            searchable_text=f"{file.name} audio {format_timestamp(start_sec)} {format_timestamp(end_sec)}",
                            source_meta={
                                "timestamp_start_sec": start_sec,
                                "timestamp_end_sec": end_sec,
                                "transcript": "",
                            },
                        )
            elif mime in ("video/mp4", "video/quicktime"):
                duration = get_video_duration_seconds(file_bytes)
                if duration is not None and duration > VIDEO_MAX_SECONDS:
                    st.sidebar.info(f"⏱️ {file.name} is long; using adaptive frame sampling.")
                stored_video_path = persist_blob(file_bytes, mime, file.name)
                interval = choose_frame_interval(duration)
                frames = extract_video_frames(file_bytes, interval_sec=interval)
                for frame_png, ts in frames:
                    emb = embed_image(frame_png, mime_type="image/png")
                    if emb is not None:
                        add_document(
                            emb,
                            f"{file.name} · {format_timestamp(ts)}",
                            "video_frame",
                            "image/png",
                            file_bytes=frame_png,
                            preview_bytes=frame_png,
                            video_path=stored_video_path,
                            searchable_text=f"{file.name} frame timestamp {format_timestamp(ts)}",
                            source_meta={
                                "timestamp_start_sec": ts,
                                "timestamp_end_sec": ts + interval,
                                "transcript": "",
                            },
                        )

            else:
                cap = per_file_captions.get(file.name, image_caption)
                if not cap and auto_caption_images:
                    cap = auto_caption_image(file_bytes, mime)
                if cap:
                    emb = embed_image_with_caption(file_bytes, caption=cap, mime_type=mime)
                else:
                    emb = embed_image(file_bytes, mime_type=mime)
                if emb is not None:
                    label = f"{file.name} · {cap}" if cap else file.name
                    add_document(emb, label, "image", mime, file_bytes,
                                 searchable_text=f"{file.name} {cap or ''}",
                                 source_meta={"caption": cap or ""})

            progress.progress((i + 1) / len(new_files), text=f"Processed {file.name}")

        progress.empty()
        st.sidebar.success(f"Added {len(new_files)} file(s).")
    else:
        st.sidebar.info("All files already loaded.")

def render_results_gallery(results, show_score=False):
    if not results:
        return
    cols = st.columns(min(len(results), 5))
    for col, res in zip(cols, results):
        with col:
            res_type = res.get("type", "")
            if res_type == "image":
                st.image(res["bytes"], width="stretch")
            elif res_type == "pdf":
                if res.get("preview_bytes"):
                    st.image(res["preview_bytes"], width="stretch")
                else:
                    st.markdown("📄 PDF")
            elif res_type == "audio":
                st.audio(res["bytes"], format=res.get("mime"))
            elif res_type == "video":
                st.video(res["bytes"])
            elif res_type == "video_frame":
                if res.get("preview_bytes"):
                    st.image(res["preview_bytes"], width="stretch")
                else:
                    st.markdown("🎞️ Frame")
                if res.get("video_bytes"):
                    st.video(res["video_bytes"])
            else:
                st.markdown("📄")
            
            citation_prefix = f"[{res['citation_id']}] " if res.get("citation_id") else ""
            if show_score:
                st.caption(f"**{citation_prefix}{res.get('name', 'Unknown')}**")
                st.caption(f"Score: `{res.get('score', 0):.4f}`")
            else:
                st.caption(f"{citation_prefix}{res.get('name', 'Unknown')}")


def render_source_detail_panel():
    source_id = st.session_state.get("selected_citation_source_id")
    if not source_id:
        return
    all_sources = list_sources(st.session_state.corpus_id)
    source = next((s for s in all_sources if s.get("source_id") == source_id), None)
    if not source:
        return
    with st.expander("Selected Source Detail", expanded=True):
        st.caption(f"Source ID: {source.get('source_id')}")
        st.caption(f"Type: {source.get('type')}")
        if source.get("page_range"):
            st.caption(f"Page range: {source.get('page_range')}")
        if source.get("timestamp_start_sec") is not None:
            start = format_timestamp(float(source.get("timestamp_start_sec", 0)))
            end = format_timestamp(float(source.get("timestamp_end_sec", source.get("timestamp_start_sec", 0))))
            st.caption(f"Timestamp: {start}-{end}")
        if source.get("caption"):
            st.caption(f"Caption: {source.get('caption')}")
        preview = read_blob(source.get("preview_path"))
        payload = read_blob(source.get("file_path"))
        video = read_blob(source.get("video_path"))
        if source.get("type") == "pdf" and preview:
            st.image(preview, width="stretch")
        elif source.get("type") in ("image", "video_frame") and preview:
            st.image(preview, width="stretch")
        elif source.get("type") == "audio" and payload:
            st.audio(payload, format=source.get("mime"))
        if video:
            st.video(video)

st.markdown("---")
st.subheader("💬 Chat with your Documents")

if not st.session_state.doc_sources:
    st.info("Upload at least one file to start chatting.")
else:
    top_k = st.slider("Results to retrieve", min_value=1, max_value=5, value=3, key="top_k")

    # Render the full chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("results"):
                render_results_gallery(msg["results"])
            if msg.get("citations"):
                st.caption(f"**Sources:** {', '.join(msg['citations'])}")
            # Read Aloud button — only for assistant messages
            if msg["role"] == "assistant" and msg.get("content"):
                msg_idx = st.session_state.messages.index(msg)
                if st.button("🔊 Read", key=f"tts_{msg_idx}", help="Read this answer aloud"):
                    st.session_state.tts_pending = msg["content"]
                    st.rerun()

    # Chat input
    query = st.chat_input("Ask a question about your documents…", key="chat_input")

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Resolve strategy — Auto runs the router first
        if retrieval_strategy == "Auto (Hybrid)":
            with st.spinner("🤖 Routing query to best backend…"):
                route = route_query(query)
            st.toast(f"Routed to **{'Google Managed RAG' if route == 'managed' else 'Local Vector Store'}**", icon="🔀")
            use_managed = (route == "managed")
        else:
            use_managed = (retrieval_strategy == "Managed RAG (Google File Search)")

        if use_managed:
            with st.chat_message("assistant"):
                with st.spinner("Searching Google backend…"):
                    gen, citations = answer_managed(query,
                                                   history=st.session_state.messages)
                full_text = st.write_stream(gen)
                if citations:
                    st.caption(f"**Sources:** {', '.join(citations)}")
                if st.button("🔊 Read", key="tts_live_managed", help="Read this answer aloud"):
                    st.session_state.tts_pending = full_text
                    st.rerun()
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "results": [],
                "citations": citations,
            })
        else:
            with st.spinner("Searching Local Vector Store…"):
                results = search(query, top_k=top_k)

            with st.chat_message("assistant"):
                if results:
                    selected_results = select_context_results(results, max_results=top_k)
                    full_text = st.write_stream(
                        answer(query, selected_results,
                               history=st.session_state.messages)
                    )
                    source_labels = [
                        f"[{res['citation_id']}] {res['name']}" for res in selected_results
                    ]
                    st.caption(f"Based on: {', '.join(source_labels)}")
                    for src in selected_results:
                        source_id = src.get("source_id")
                        if source_id and st.button(f"Open {src['citation_id']}", key=f"open_src_{source_id}"):
                            st.session_state.selected_citation_source_id = source_id
                            st.rerun()
                    if st.button("🔊 Read", key="tts_live_local", help="Read this answer aloud"):
                        st.session_state.tts_pending = full_text
                        st.rerun()
                    render_results_gallery(selected_results)
                    render_source_detail_panel()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "results": selected_results,
                        "citations": source_labels,
                    })
                else:
                    msg = "No results found — try rephrasing your question."
                    st.warning(msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg,
                        "results": [],
                        "citations": [],
                    })

# Image query processing (fired by the sidebar Find Similar button)
if run_img_query and query_image_sidebar:
    query_bytes = query_image_sidebar.read()
    query_mime = query_image_sidebar.type
    dim = st.session_state.get("embedding_dim", 3072)

    # Post the query image as a user message
    with st.chat_message("user"):
        st.image(query_bytes, width=200, caption="Image query")
    st.session_state.messages.append({
        "role": "user",
        "content": "🔍 Image similarity search",
    })

    with st.spinner("Embedding query image…"):
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
        img_results = search(query=None, top_k=top_k_img, query_emb=query_emb)

        with st.chat_message("assistant"):
            st.markdown(f"Found **{len(img_results)}** similar item(s):")
            render_results_gallery(img_results, show_score=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Found {len(img_results)} similar item(s) for your image query.",
            "results": img_results,
            "citations": [],
        })

# footer
st.markdown("---")
st.caption("Unified RAG · Powered by Gemini Embedding 2 and Gemini 2.5 Flash")
