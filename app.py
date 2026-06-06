import io
import json
import requests
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import base64
import uuid
import tempfile
import time
import os
from PIL import Image
from google import genai
from google.genai import types
import chromadb
from rag_core import (
    MANAGED_BACKEND,
    bm25_rank,
    build_local_answer_prompt,
    build_rehydrated_sources,
    call_with_retry,
    is_transient_error,
    pop_source_by_id,
    reciprocal_rank_fusion,
    select_context_results,
    source_artifacts,
)
from media_ingest import (
    AUDIO_MAX_SECONDS,
    FRAME_INTERVAL_SEC,
    VIDEO_MAX_SECONDS,
    attach_transcript_metadata,
    build_audio_searchable_text,
    build_image_searchable_text,
    build_video_frame_searchable_text,
    choose_frame_interval,
    chunk_audio_bytes,
    chunk_pdf,
    extract_pdf_text,
    extract_video_frames,
    format_timestamp,
    get_video_duration_seconds,
    normalize_image,
    parse_caption_map,
    select_scene_frames,
    transcript_for_timestamp,
)
from storage import delete_blob, hash_bytes, load_or_create_corpus_id, persist_blob, read_blob, set_corpus_id
from source_registry import delete_source, list_sources, upsert_source
from app import gemini
from app.gemini import (
    auto_caption_image,
    embed_audio,
    embed_image,
    embed_image_with_caption,
    embed_pdf,
    hyde_query_embedding,
    route_query,
    transcribe_audio_bytes,
    transcribe_video_audio,
)

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
    st.session_state.corpus_id = load_or_create_corpus_id()
if "pending_clear_all" not in st.session_state:
    st.session_state.pending_clear_all = False
if "pending_delete_chroma_ids" not in st.session_state:
    st.session_state.pending_delete_chroma_ids = []
if "pending_cleanup_paths" not in st.session_state:
    st.session_state.pending_cleanup_paths = []
if "load_samples_requested" not in st.session_state:
    st.session_state.load_samples_requested = False
if "selected_citation" not in st.session_state:
    # the exact cited chunk: {source_id, chroma_id, label, page_range, timestamp_start_sec}
    st.session_state.selected_citation = None

# Rebuild the sidebar source list from the persisted registry once per session,
# so a restart or browser refresh doesn't drop everything that's still indexed.
if "rehydrated" not in st.session_state:
    st.session_state.rehydrated = True
    if not st.session_state.doc_sources:
        restored = build_rehydrated_sources(list_sources(st.session_state.corpus_id))
        if restored:
            st.session_state.doc_sources = restored

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

    st.checkbox(
        "Expand query (HyDE)",
        value=False,
        key="use_hyde",
        help="Draft a hypothetical answer, embed it, and average with your question's embedding. Better recall on vague questions; adds a Gemini call per search.",
    )

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
            set_corpus_id(st.session_state.corpus_id)
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
        type_options = ["All"] + sorted({s["type"] for s in sources})
        type_filter = st.selectbox("Filter by type", type_options, key="doc_type_filter", label_visibility="collapsed")

        # Build dataframe; "source_id" tracks the row for safe deletion
        df = pd.DataFrame([
            {"source_id": s["source_id"], "Icon": TYPE_ICONS.get(s["type"], "📎"), "Name": s["name"], "Type": s["type"]}
            for s in sources
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

        selected_source_id = (
            df.iloc[event.selection.rows[0]]["source_id"] if event.selection.rows else None
        )

        # Delete selected row
        col_del, col_count = st.columns([1, 2])
        with col_del:
            if st.button("🗑 Delete", key="delete_row_btn", disabled=selected_source_id is None):
                removed, chroma_ids = pop_source_by_id(
                    st.session_state.doc_sources,
                    selected_source_id,
                )
                for artifact in source_artifacts(removed):
                    for path_key in ("file_path", "preview_path", "video_path"):
                        if artifact.get(path_key):
                            st.session_state.pending_cleanup_paths.append(artifact[path_key])
                if removed.get("source_id"):
                    delete_source(removed["source_id"])
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
    transcribe_audio = st.checkbox(
        "Transcribe audio (speech-to-text)",
        value=True,
        key="transcribe_audio",
        help="Use Gemini to transcribe each audio chunk so spoken words become searchable. Adds one API call per chunk.",
    )
    scene_sampling = st.checkbox(
        "Scene-based video frames",
        value=True,
        key="scene_sampling",
        help="Skip near-duplicate frames so a held slide indexes once instead of many times. Fewer embeddings, less redundancy.",
    )
    uploaded_files = st.file_uploader(
        "Files",
        type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "mp4", "mov"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("⚡ Load Sample Images", key="load_samples_btn", help="Pre-load 6 financial charts"):
        st.session_state.load_samples_requested = True

# init client
@st.cache_resource
def get_gemini_client(api_key: str):
    """Build the Gemini client once per key instead of on every Streamlit rerun."""
    return genai.Client(api_key=api_key)

if google_api_key:
    client = get_gemini_client(google_api_key)
    gemini.set_client(client)
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
    set_corpus_id(st.session_state.corpus_id)
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

def add_file_to_google_store(file_name: str, file_bytes: bytes, mime: str, content_hash: str = ""):
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
            "content_hash": content_hash,
        })
    except Exception as e:
        st.error(f"Google Upload Error for {file_name}: {e}")
    finally:
        os.remove(tmp_path)


def add_document(emb: np.ndarray, name: str, doc_type: str, mime: str, file_bytes: bytes,
                 preview_bytes: bytes | None = None, video_path: str | None = None,
                 searchable_text: str = "", source_meta: dict | None = None,
                 content_hash: str = "", source_id: str | None = None):
    """Add one chunk to ChromaDB and fold it into its file's source row.

    All chunks of an uploaded file share `source_id`, so the sidebar and registry
    show one entry per file whose `chroma_ids`/`artifacts` cover every chunk.
    """
    doc_id = str(uuid.uuid4())
    try:
        file_path = persist_blob(file_bytes, mime, name)
        preview_path = persist_blob(preview_bytes, "image/png", f"{name}_preview.png") if preview_bytes else None
        sid = source_id or (source_meta or {}).get("source_id") or str(uuid.uuid4())

        metadata: dict = {
            "name": name,
            "type": doc_type,
            "mime": mime,
            "file_path": file_path,
            "corpus_id": st.session_state.corpus_id,
            "searchable_text": searchable_text or name,
            "content_hash": content_hash,
        }
        if preview_path:
            metadata["preview_path"] = preview_path
        if video_path:
            metadata["video_path"] = video_path
        if source_meta:
            metadata.update(source_meta)
        metadata["source_id"] = sid
        chroma_collection.upsert(ids=[doc_id], embeddings=[emb.tolist()], metadatas=[metadata])

        artifact = {"chroma_id": doc_id, "file_path": file_path,
                    "preview_path": preview_path, "video_path": video_path}
        display_name = name.split(" · ")[0]

        existing = next((s for s in st.session_state.doc_sources if s.get("source_id") == sid), None)
        if existing:
            existing.setdefault("chroma_ids", [existing.get("id")]).append(doc_id)
            existing.setdefault("artifacts", []).append(artifact)
        else:
            source_row = {
                "source_id": sid,
                "id": doc_id,
                "name": display_name,
                "type": doc_type,
                "mime": mime,
                "file_path": file_path,
                "preview_path": preview_path,
                "video_path": video_path,
                "backend": "local",
                "searchable_text": searchable_text or name,
                "content_hash": content_hash,
                "chroma_ids": [doc_id],
                "artifacts": [artifact],
            }
            if source_meta:
                source_row.update({k: v for k, v in source_meta.items() if k != "source_id"})
            st.session_state.doc_sources.append(source_row)

        upsert_source({
            "source_id": sid,
            "doc_id": doc_id,
            "name": display_name,
            "type": doc_type,
            "backend": "local",
            "mime": mime,
            "file_path": file_path,
            "preview_path": preview_path,
            "video_path": video_path,
            "searchable_text": searchable_text or name,
            "corpus_id": st.session_state.corpus_id,
            "timestamp_start_sec": (source_meta or {}).get("timestamp_start_sec"),
            "timestamp_end_sec": (source_meta or {}).get("timestamp_end_sec"),
            "page_range": (source_meta or {}).get("page_range"),
            "caption": (source_meta or {}).get("caption"),
            "transcript": (source_meta or {}).get("transcript", ""),
            "content_hash": content_hash,
            "chroma_ids": [doc_id],
            "artifacts": [artifact],
        })
    except Exception as e:
        st.error(f"Error saving to database: {e}")

def search(query: str = None, top_k: int = 3, query_emb: np.ndarray = None) -> list[dict]:
    """Embed query and query ChromaDB for top-K results."""
    try:
        corpus_docs = chroma_collection.get(where={"corpus_id": st.session_state.corpus_id})
        if not corpus_docs.get("ids"):
            return []
    except Exception:
        return []

    effective_query = query or ""

    if query_emb is None and query:
        dim = st.session_state.get("embedding_dim", 3072)
        if st.session_state.get("use_hyde"):
            query_emb = hyde_query_embedding(query, dim)
        if query_emb is None:
            try:
                result = call_with_retry(lambda: client.models.embed_content(
                    model="gemini-embedding-2-preview",
                    contents=effective_query or query,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=dim,
                    ),
                ))
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
    corpus_meta_by_id = {doc_id: meta for doc_id, meta in zip(corpus_docs["ids"], corpus_metas) if meta}
    sparse_docs = [
        {"id": doc_id, "searchable_text": meta.get("searchable_text", meta.get("name", ""))}
        for doc_id, meta in corpus_meta_by_id.items()
    ]
    dense_ids = results["ids"][0] if results and results.get("ids") and results["ids"] else []
    sparse_ids = [doc_id for doc_id, _ in bm25_rank(effective_query or query or "", sparse_docs)[:candidate_count]]
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

    # Reuse the corpus metadata already fetched above instead of a second Chroma call.
    missing_ids = [doc_id for doc_id in fused_ids if doc_id not in dense_ids]
    for doc_id in missing_ids:
        meta = corpus_meta_by_id.get(doc_id)
        if meta:
            result_rows.append((doc_id, meta, 1.0))

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
            "timestamp_start_sec": meta.get("timestamp_start_sec"),
            "page_range": meta.get("page_range"),
        })
    # Already ordered by RRF fusion above; a second sort here only penalized
    # keyword-only hits (whose dense score is 0), so we trust the fused order.
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
    for attempt in range(3):
        produced = False
        try:
            stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=contents,
            )
            for chunk in stream:
                if chunk.text:
                    produced = True
                    yield chunk.text
            return
        except Exception as e:
            # Retry transient failures, but only if nothing has streamed yet —
            # once text is on screen we can't cleanly restart.
            if not produced and is_transient_error(e) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if produced:
                yield "\n\n⚠️ The answer was cut off — the model hiccupped. Please ask again."
            else:
                yield "⚠️ The model is busy right now (try again in a moment)."
            return

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
        full_chunks = call_with_retry(lambda: list(client.models.generate_content_stream(
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
        )))
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
    except Exception:
        def _error():
            yield "⚠️ The managed backend is unavailable right now (try again in a moment)."
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
    | Images | PNG, JPG | 4096px (auto-resized) | Per-file or auto caption |
    | PDFs | PDF | 4 pages per embed call | Overlap chunking + text for BM25 |
    | Audio | MP3, WAV | 80s per chunk | Timestamped chunks |
    | Video | MP4, MOV | Long files sampled | Adaptive frame intervals + timestamps |

    **How it works:** Upload → embed into Chroma (or Google File Search) → hybrid local search
    (dense + BM25 + RRF) → top-K chunks and source IDs → Gemini Flash answer.
    """)

# process uploads from sidebar
if uploaded_files:
    per_file_captions = parse_caption_map(image_captions_map_raw)
    # Dedup by content hash so re-uploads (and server reloads) don't re-embed the
    # same bytes, even if the filename changed.
    existing_hashes = {s.get("content_hash") for s in st.session_state.doc_sources if s.get("content_hash")}

    progress = st.sidebar.progress(0, text="Embedding…")
    added = 0
    skipped = 0
    total = len(uploaded_files)
    for i, file in enumerate(uploaded_files):
        file_bytes = file.read()
        mime = file.type
        content_hash = hash_bytes(file_bytes)
        if content_hash in existing_hashes:
            skipped += 1
            progress.progress((i + 1) / total, text=f"Skipped {file.name} (already indexed)")
            continue
        existing_hashes.add(content_hash)
        file_source_id = str(uuid.uuid4())  # one source row per uploaded file

        if retrieval_strategy == "Managed RAG (Google File Search)":
            add_file_to_google_store(file.name, file_bytes, mime, content_hash=content_hash)
        elif mime == "application/pdf":
            for chunk_bytes, page_label, preview_png in chunk_pdf(file_bytes):
                chunk_text = extract_pdf_text(chunk_bytes)
                for emb in embed_pdf(chunk_bytes):
                    add_document(emb, f"{file.name} · {page_label}", "pdf", mime, chunk_bytes,
                                 preview_bytes=preview_png,
                                 searchable_text=f"{file.name} {page_label}\n{chunk_text}",
                                 source_meta={"page_range": page_label},
                                 content_hash=content_hash, source_id=file_source_id)
        elif mime in ("audio/mpeg", "audio/mp3", "audio/wav"):
            audio_mime = "audio/mp3" if mime == "audio/mpeg" else mime
            audio_chunks = chunk_audio_bytes(file_bytes, audio_mime)
            for chunk_idx, (chunk_bytes, start_sec, end_sec, chunk_mime) in enumerate(audio_chunks, start=1):
                emb = embed_audio(chunk_bytes, mime_type=chunk_mime)
                if emb is not None:
                    label = f"{file.name} · {format_timestamp(start_sec)}-{format_timestamp(end_sec)}"
                    transcript = transcribe_audio_bytes(chunk_bytes, chunk_mime) if transcribe_audio else ""
                    audio_meta = attach_transcript_metadata(
                        {
                            "timestamp_start_sec": start_sec,
                            "timestamp_end_sec": end_sec,
                        },
                        transcript=transcript,
                    )
                    add_document(
                        emb,
                        label,
                        "audio",
                        chunk_mime,
                        chunk_bytes,
                        searchable_text=build_audio_searchable_text(
                            file.name, start_sec, end_sec, audio_meta.get("transcript", "")
                        ),
                        source_meta=audio_meta,
                        content_hash=content_hash,
                        source_id=file_source_id,
                    )
        elif mime in ("video/mp4", "video/quicktime"):
            duration = get_video_duration_seconds(file_bytes)
            if duration is not None and duration > VIDEO_MAX_SECONDS:
                st.sidebar.info(f"⏱️ {file.name} is long; using adaptive frame sampling.")
            stored_video_path = persist_blob(file_bytes, mime, file.name)
            interval = choose_frame_interval(duration)
            frames = extract_video_frames(file_bytes, interval_sec=interval)
            if scene_sampling:
                frames = select_scene_frames(frames)
            transcript_windows = transcribe_video_audio(file_bytes) if transcribe_audio else []
            for frame_png, ts in frames:
                emb = embed_image(frame_png, mime_type="image/png")
                if emb is not None:
                    ts_label = format_timestamp(ts)
                    frame_transcript = transcript_for_timestamp(ts, transcript_windows)
                    video_meta = attach_transcript_metadata(
                        {
                            "timestamp_start_sec": ts,
                            "timestamp_end_sec": ts + interval,
                        },
                        transcript=frame_transcript,
                    )
                    add_document(
                        emb,
                        f"{file.name} · {ts_label}",
                        "video_frame",
                        "image/png",
                        file_bytes=frame_png,
                        preview_bytes=frame_png,
                        video_path=stored_video_path,
                        searchable_text=build_video_frame_searchable_text(
                            file.name, ts_label, video_meta.get("transcript", "")
                        ),
                        source_meta=video_meta,
                        content_hash=content_hash,
                        source_id=file_source_id,
                    )
        else:
            norm_bytes, norm_mime = normalize_image(file_bytes, mime)
            cap = per_file_captions.get(file.name, image_caption)
            if not cap and auto_caption_images:
                cap = auto_caption_image(norm_bytes, norm_mime)
            if cap:
                emb = embed_image_with_caption(norm_bytes, caption=cap, mime_type=norm_mime)
            else:
                emb = embed_image(norm_bytes, mime_type=norm_mime)
            if emb is not None:
                label = f"{file.name} · {cap}" if cap else file.name
                add_document(emb, label, "image", norm_mime, norm_bytes,
                             searchable_text=build_image_searchable_text(file.name, cap or ""),
                             source_meta={"caption": cap or ""},
                             content_hash=content_hash, source_id=file_source_id)

        added += 1
        progress.progress((i + 1) / total, text=f"Processed {file.name}")

    progress.empty()
    if added:
        st.sidebar.success(f"Added {added} file(s).")
    if skipped:
        st.sidebar.info(f"Skipped {skipped} already-indexed file(s).")
    if not added and not skipped:
        st.sidebar.info("No new files to add.")

def _seek_seconds(value) -> int:
    """Coerce a stored timestamp into an int start_time for st.audio/st.video."""
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _escaped_stream(chunks):
    """Escape `$` in streamed answer text so dollar amounts don't trigger
    Streamlit's LaTeX math mode (which eats spaces and italicizes whole runs)."""
    for chunk in chunks:
        yield chunk.replace("$", "\\$") if chunk else chunk


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
                st.audio(res["bytes"], format=res.get("mime"), start_time=_seek_seconds(res.get("timestamp_start_sec")))
            elif res_type == "video":
                st.video(res["bytes"], start_time=_seek_seconds(res.get("timestamp_start_sec")))
            elif res_type == "video_frame":
                # Show just the frame image. Rendering the full source video per
                # result is enormous (one big file copied per hit); the source
                # video stays one click away under "Open Sn".
                if res.get("preview_bytes"):
                    st.image(res["preview_bytes"], width="stretch")
                    st.caption("🎥 video frame")
                else:
                    st.markdown("🎞️ Frame")
            else:
                st.markdown("📄")
            
            citation_prefix = f"[{res['citation_id']}] " if res.get("citation_id") else ""
            if show_score:
                st.caption(f"**{citation_prefix}{res.get('name', 'Unknown')}**")
                st.caption(f"Score: `{res.get('score', 0):.4f}`")
            else:
                st.caption(f"{citation_prefix}{res.get('name', 'Unknown')}")


def render_source_detail_panel():
    citation = st.session_state.get("selected_citation")
    if not citation:
        return
    all_sources = list_sources(st.session_state.corpus_id)
    source = next((s for s in all_sources if s.get("source_id") == citation.get("source_id")), None)
    if not source:
        return
    # Focus on the exact cited chunk (its page render / frame), not the file's first chunk.
    chunk = next(
        (a for a in source_artifacts(source) if a.get("chroma_id") == citation.get("chroma_id")),
        None,
    ) or {}
    seek = _seek_seconds(citation.get("timestamp_start_sec"))
    with st.expander("Selected Source Detail", expanded=True):
        st.caption(f"Source: {citation.get('label') or source.get('name')}")
        st.caption(f"Type: {source.get('type')}")
        if citation.get("page_range"):
            st.caption(f"Showing {citation.get('page_range')}")
        if citation.get("timestamp_start_sec") is not None:
            st.caption(f"Timestamp: {format_timestamp(float(citation.get('timestamp_start_sec') or 0))}")
        if source.get("caption"):
            st.caption(f"Caption: {source.get('caption')}")
        preview = read_blob(chunk.get("preview_path") or source.get("preview_path"))
        payload = read_blob(chunk.get("file_path") or source.get("file_path"))
        video = read_blob(chunk.get("video_path") or source.get("video_path"))
        if source.get("type") in ("pdf", "image", "video_frame") and preview:
            st.image(preview, width="stretch")
        elif source.get("type") == "audio" and payload:
            st.audio(payload, format=source.get("mime"), start_time=seek)
        if video:
            st.video(video, start_time=seek)

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
            # Only use managed if there are actually managed docs; otherwise the
            # store is empty and the router would send the query nowhere useful.
            has_managed_docs = any(s.get("backend") == MANAGED_BACKEND for s in st.session_state.doc_sources)
            use_managed = (route == "managed" and has_managed_docs)
            st.toast(f"Routed to **{'Google Managed RAG' if use_managed else 'Local Vector Store'}**", icon="🔀")
        else:
            use_managed = (retrieval_strategy == "Managed RAG (Google File Search)")

        if use_managed:
            with st.chat_message("assistant"):
                with st.spinner("Searching Google backend…"):
                    gen, citations = answer_managed(query,
                                                   history=st.session_state.messages)
                full_text = st.write_stream(_escaped_stream(gen))
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
                    full_text = st.write_stream(_escaped_stream(
                        answer(query, selected_results,
                               history=st.session_state.messages)
                    ))
                    source_labels = [
                        f"[{res['citation_id']}] {res['name']}" for res in selected_results
                    ]
                    st.caption(f"Based on: {', '.join(source_labels)}")
                    for src in selected_results:
                        source_id = src.get("source_id")
                        if source_id and st.button(f"Open {src['citation_id']}", key=f"open_{src['citation_id']}"):
                            st.session_state.selected_citation = {
                                "source_id": source_id,
                                "chroma_id": src.get("id"),
                                "label": src.get("name"),
                                "page_range": src.get("page_range"),
                                "timestamp_start_sec": src.get("timestamp_start_sec"),
                            }
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
            result = call_with_retry(lambda: client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=[types.Part.from_bytes(data=query_bytes, mime_type=query_mime)],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=dim,
                ),
            ))
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
