import base64
import io
import logging
import os
import re
import tempfile
import time
import uuid

import chromadb
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from PIL import Image

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
from app.guard import (
    ANSWER_MAX_OUTPUT_TOKENS,
    DEMO_LIMITS,
    LOCAL_LIMITS,
    MAX_QUESTION_CHARS,
    BudgetedClient,
    DailyBudget,
    DemoBudgetExceeded,
    check_upload,
    clamp_question,
    friendly_error,
    image_too_large,
    js_string_literal,
)
from media_ingest import (
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
from rag.models import ANSWER_MODEL, DEMO_ANSWER_MODEL, EMBEDDING_MODEL
from rag.pipeline import ChromaStore, Config, build_answer_contents, retrieve
from rag_core import (
    MANAGED_BACKEND,
    SOURCE_TYPE_LABELS,
    build_rehydrated_sources,
    call_with_retry,
    format_prompt_cost,
    is_transient_error,
    mark_citations,
    pop_source_by_id,
    select_context_results,
    source_artifacts,
    source_locator,
    strip_citations,
    within_demo_upload_cap,
)
from source_registry import delete_corpus_sources, delete_source, list_sources, upsert_source
from storage import (
    STORAGE_ROOT,
    delete_blob,
    forget_corpus,
    hash_bytes,
    load_or_create_corpus_id,
    new_corpus_id,
    persist_blob,
    read_blob,
    set_corpus_id,
    stale_corpora,
    touch_corpus,
)

log = logging.getLogger("unified_rag")

# Demo deployment settings. Unset (the default) runs with no limits and the
# full Pro-backed Managed RAG path, same as a local clone.
DEMO_MODE = os.getenv("DEMO_MODE", "false").strip().lower() in {"1", "true", "yes"}
MAX_UPLOADS_PER_SESSION = int(os.getenv("MAX_UPLOADS_PER_SESSION", "5"))
# Gemini requests per UTC day on the shared demo key, across every visitor and
# session in this process. A visitor's own key is not counted. A reload used to
# reset the only limit there was (the per-session upload counter).
DEMO_DAILY_CALL_LIMIT = int(os.getenv("DEMO_DAILY_CALL_LIMIT", "1000"))
# Demo libraries are per session; ones idle this long are deleted.
DEMO_CORPUS_TTL_HOURS = float(os.getenv("DEMO_CORPUS_TTL_HOURS", "6"))
UPLOAD_LIMITS = DEMO_LIMITS if DEMO_MODE else LOCAL_LIMITS
# A demo in front of strangers stays on the cheaper model so one visitor cannot
# exhaust the daily quota. Both were on the 2.5 line until 2026-08-01, when that
# generation started returning 404 to newly issued API keys; 2.5 Pro was also
# only 1K requests a day against 3.6 Flash's 10K.
MANAGED_ANSWER_MODEL = DEMO_ANSWER_MODEL if DEMO_MODE else ANSWER_MODEL

# page config. Colours and fonts live in .streamlit/config.toml.
st.set_page_config(page_title="Unified RAG", page_icon=":material/manage_search:", layout="centered")
st.title("Unified RAG")
st.markdown(
    "Ask one question across PDFs, images, audio and video. Every answer cites the "
    "page, frame or clip it came from, so you can open the source and check it."
)

AVATARS = {"user": ":material/person:", "assistant": ":material/find_in_page:"}

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
    # The demo gives every session its own library, so one visitor's uploads
    # and "Clear all" never reach another's. A local run keeps one persisted
    # library that survives restarts.
    st.session_state.corpus_id = new_corpus_id() if DEMO_MODE else load_or_create_corpus_id()
if "attempted_uploads" not in st.session_state:
    # file_ids already processed, whether they worked or not, so a failed
    # upload still sitting in the uploader is not retried on every rerun.
    st.session_state.attempted_uploads = set()
if "pending_clear_all" not in st.session_state:
    st.session_state.pending_clear_all = False
if "confirm_clear_all" not in st.session_state:
    st.session_state.confirm_clear_all = False
if "pending_delete_chroma_ids" not in st.session_state:
    st.session_state.pending_delete_chroma_ids = []
if "pending_cleanup_paths" not in st.session_state:
    st.session_state.pending_cleanup_paths = []
if "load_samples_requested" not in st.session_state:
    st.session_state.load_samples_requested = False
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0
if "prompt_tokens" not in st.session_state:
    st.session_state.prompt_tokens = 0
if "demo_uploads_this_session" not in st.session_state:
    st.session_state.demo_uploads_this_session = 0

# Rebuild the sidebar source list from the persisted registry once per session,
# so a restart or browser refresh doesn't drop everything that's still indexed.
if "rehydrated" not in st.session_state:
    st.session_state.rehydrated = True
    if not st.session_state.doc_sources and not DEMO_MODE:
        restored = build_rehydrated_sources(list_sources(st.session_state.corpus_id))
        if restored:
            st.session_state.doc_sources = restored

client = None


def speak_text(text: str):
    """Fire a Web Speech API utterance in the browser. Zero-height iframe, no dependencies."""
    # json.dumps alone let "</script>" in an answer close the tag; this escapes < > &.
    js_text = js_string_literal(text)
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

def resolve_hosted_key() -> str:
    """Key the deployment supplies so visitors are not met with a key prompt.

    Accepts either name: the secrets template shipped GEMINI_API_KEY while the
    local .env uses GOOGLE_API_KEY, and neither was ever actually read before,
    so honouring both avoids a silent empty-key deploy.

    st.secrets raises rather than returning a default when no secrets file
    exists, which is the normal case for a local checkout, so this has to be
    guarded rather than written as a .get() with a fallback.
    """
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            key = str(st.secrets.get(name, "")).strip()
            if key:
                return key
        except Exception:
            pass
        key = os.getenv(name, "").strip()
        if key:
            return key
    return ""


HOSTED_KEY = resolve_hosted_key()


def start_new_corpus():
    """Point this session at a fresh, empty library.

    In the demo the old corpus is left to the idle sweep, which also removes
    its rows from the other dimension's collection.
    """
    st.session_state.corpus_id = new_corpus_id()
    st.session_state.attempted_uploads = set()
    if not DEMO_MODE:
        set_corpus_id(st.session_state.corpus_id)

# sidebar: the library (add, browse, delete), image search, settings, key
SAMPLE_CHARTS_HELP = ("Six Q4 2024 earnings charts (Tesla, Netflix, Nike, Google, Accenture, "
                      "Tencent), enough to try a question in a few seconds.")


def api_key_input() -> str:
    """Key field plus a line on where the key goes. Returns what was typed."""
    typed = st.text_input(
        "Google API key",
        type="password",
        key="google_key",
        placeholder="Using the shared demo key" if HOSTED_KEY else "Paste your key",
        help="Optional while the demo key is available. Your own key takes priority.",
    )
    if not typed and HOSTED_KEY:
        st.caption("A shared demo key is active, so you can try the app without signing up. "
                   "Paste your own key to use it instead.")
    else:
        st.caption("[Get a Google API key](https://aistudio.google.com/app/apikey). "
                   "It is held in memory for this session only and never stored.")
    return typed


with st.sidebar:
    # Without a demo key the key is the first thing anyone needs, so it leads.
    # With one, it is a setting and sits at the bottom.
    if not HOSTED_KEY:
        _typed_key = api_key_input()
        st.divider()

    # ---------- Add sources ----------
    st.subheader("Add sources")
    uploaded_files = st.file_uploader(
        "Files",
        type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "mp4", "mov"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        max_upload_size=UPLOAD_LIMITS.max_mb,
    )
    if DEMO_MODE:
        st.caption(f"Demo limits: {UPLOAD_LIMITS.max_mb} MB per file, "
                   f"{UPLOAD_LIMITS.max_pdf_pages} PDF pages, "
                   f"{UPLOAD_LIMITS.max_audio_seconds // 60} min audio, "
                   f"{UPLOAD_LIMITS.max_video_seconds // 60} min video.")

    # Show ingest options only for the file types actually selected, so a PDF
    # upload isn't buried under image-caption and audio fields.
    _sel_mimes = {(f.type or "") for f in (uploaded_files or [])}
    _has_images = any(m.startswith("image") for m in _sel_mimes)
    _has_audio = any(m in ("audio/mpeg", "audio/mp3", "audio/wav") for m in _sel_mimes)
    _has_video = any(m in ("video/mp4", "video/quicktime") for m in _sel_mimes)

    image_caption = ""
    image_captions_map_raw = ""
    auto_caption_images = True
    transcribe_audio = True
    scene_sampling = True

    if _has_images:
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
        auto_caption_images = st.checkbox(
            "Auto-caption images when caption missing", value=True, key="auto_caption_images"
        )
    if _has_audio or _has_video:
        transcribe_audio = st.checkbox(
            "Transcribe audio (speech-to-text)",
            value=True,
            key="transcribe_audio",
            help="Use Gemini to transcribe each audio chunk so spoken words become searchable. "
                 "Adds one API call per chunk.",
        )
    if _has_video:
        scene_sampling = st.checkbox(
            "Scene-based video frames",
            value=True,
            key="scene_sampling",
            help="Skip near-duplicate frames so a held slide indexes once instead of many times.",
        )

    if st.button("Load 6 sample charts", key="load_samples_btn", width="stretch",
                 help=SAMPLE_CHARTS_HELP):
        st.session_state.load_samples_requested = True

    st.divider()
    # ---------- Library ----------
    st.subheader("Library")
    sources = st.session_state.doc_sources

    if not sources:
        st.caption("Nothing indexed yet. Add files above or load the sample charts.")
    else:
        if st.session_state.get("confirm_clear_all"):
            st.warning("Remove every source from your library? This also deletes the index entries.")
            cc1, cc2 = st.columns(2)
            if cc1.button("Remove all", key="confirm_clear_yes", type="primary", width="stretch"):
                st.session_state.pending_clear_all = True
                st.session_state.confirm_clear_all = False
                st.rerun()
            if cc2.button("Cancel", key="confirm_clear_no", width="stretch"):
                st.session_state.confirm_clear_all = False
                st.rerun()
        else:
            # chunk count = vectors actually in the local index for this corpus
            chunk_count = sum(len(s.get("chroma_ids") or []) for s in sources)
            top1, top2 = st.columns([3, 1], vertical_alignment="center")
            top1.caption(
                f"{len(sources)} {'source' if len(sources) == 1 else 'sources'}, "
                f"{chunk_count} {'chunk' if chunk_count == 1 else 'chunks'} indexed"
            )
            if top2.button("Clear all", key="clear_all_btn", type="tertiary", width="stretch"):
                st.session_state.confirm_clear_all = True
                st.rerun()

        type_options = ["All"] + sorted({s["type"] for s in sources})
        type_filter = st.selectbox(
            "Filter by type", type_options, key="doc_type_filter", label_visibility="collapsed",
            format_func=lambda t: "All types" if t == "All" else SOURCE_TYPE_LABELS.get(t, t),
        )

        df = pd.DataFrame([
            {"source_id": s["source_id"], "Name": s["name"],
             "Type": SOURCE_TYPE_LABELS.get(s["type"], s["type"])}
            for s in sources
            if type_filter == "All" or s["type"] == type_filter
        ])
        event = st.dataframe(
            df[["Name", "Type"]],
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="doc_table",
        )
        selected_ids = [df.iloc[r]["source_id"] for r in event.selection.rows]
        if st.button(
            f"Delete {len(selected_ids)} selected" if selected_ids else "Select rows to delete",
            key="delete_sel_btn",
            disabled=not selected_ids,
            width="stretch",
        ):
            for sid in selected_ids:
                removed, chroma_ids = pop_source_by_id(st.session_state.doc_sources, sid)
                for artifact in source_artifacts(removed):
                    for path_key in ("file_path", "preview_path", "video_path"):
                        if artifact.get(path_key):
                            st.session_state.pending_cleanup_paths.append(artifact[path_key])
                if removed.get("source_id"):
                    delete_source(removed["source_id"])
                st.session_state.pending_delete_chroma_ids.extend(chroma_ids)
            st.rerun()

    st.divider()
    # ---------- Search by image ----------
    with st.expander("Search by image", icon=":material/image_search:", expanded=False):
        st.caption("Upload an image to find the closest charts, pages and video frames in your library.")
        query_image_sidebar = st.file_uploader(
            "Query image",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="image_query_uploader",
            label_visibility="collapsed",
            max_upload_size=UPLOAD_LIMITS.max_mb,
        )
        top_k_img = st.slider("Results", min_value=1, max_value=5, value=3, key="top_k_img")
        run_img_query = st.button(
            "Find similar",
            key="img_search_btn",
            width="stretch",
            disabled=not (query_image_sidebar and st.session_state.doc_sources),
        )

    # ---------- Settings ----------
    with st.expander("Search settings", icon=":material/tune:", expanded=False):
        retrieval_strategy = st.selectbox(
            "Retrieval strategy",
            ["Auto (Hybrid)", "Local Vector Store (Chroma)", "Managed RAG (Google File Search)"],
            key="retrieval_strategy",
            help="Auto: Gemini picks the backend per query. Local: your embedded files. Managed: Google File Search.",
        )
        st.slider("Results to retrieve", min_value=1, max_value=5, value=3, key="top_k")
        st.selectbox("Embedding dimension", [3072, 1536, 768], key="embedding_dim")
        st.checkbox(
            "Expand query (HyDE)",
            value=False,
            key="use_hyde",
            help="Draft a hypothetical answer, embed it, and average with your question's embedding. "
                 "Adds a Gemini call per search.",
        )
        if not DEMO_MODE:
            # Not offered on the public demo: it would point the server at
            # whatever host a visitor typed.
            st.caption("Remote ChromaDB (optional). Leave blank to use the local database.")
            st.text_input("Chroma Tenant", value="", key="chroma_tenant")
            st.text_input("Chroma Database", value="", key="chroma_database")
            st.text_input("Chroma API Key", type="password", key="chroma_api_key")

    if HOSTED_KEY:
        with st.expander("API key", icon=":material/key:", expanded=False):
            _typed_key = api_key_input()
    google_api_key = (_typed_key or "").strip() or HOSTED_KEY

    # auto-clear if dimension changed with docs loaded
    if st.session_state.embedding_dim != st.session_state.active_dim:
        if st.session_state.doc_sources:
            st.session_state.doc_embeddings = []
            st.session_state.doc_sources = []
            start_new_corpus()
            st.toast("Embedding dimension changed, so the library was cleared. Re-upload to re-embed.",
                     icon=":material/warning:")
            st.session_state.active_dim = st.session_state.embedding_dim
            st.rerun()
        st.session_state.active_dim = st.session_state.embedding_dim

# init client
@st.cache_resource
def get_demo_budget() -> DailyBudget:
    """One budget per process, shared by every session on the demo key."""
    return DailyBudget(DEMO_DAILY_CALL_LIMIT, path=STORAGE_ROOT / "demo_budget.json")


def session_client(api_key: str, budgeted: bool):
    """This session's Gemini client, rebuilt only when its key changes.

    Held in session state rather than a process-wide cache or module global, so
    a visitor's own key never serves another session's calls.
    """
    fingerprint = (hash_bytes(api_key.encode()), budgeted)
    if st.session_state.get("_client_fingerprint") != fingerprint:
        raw = genai.Client(api_key=api_key)
        st.session_state.gemini_session_client = BudgetedClient(raw, get_demo_budget()) if budgeted else raw
        st.session_state._client_fingerprint = fingerprint
    return st.session_state.gemini_session_client


# The budget applies to the deployment's key in demo mode only.
USING_DEMO_KEY = bool(HOSTED_KEY) and google_api_key == HOSTED_KEY
BUDGETED = DEMO_MODE and USING_DEMO_KEY
st.session_state.using_demo_key = USING_DEMO_KEY


def budget_spent() -> bool:
    return BUDGETED and get_demo_budget().remaining() <= 0


def show_error(e: Exception, warn: bool = False):
    """Show a message a visitor can act on; the raw error goes to the log."""
    (st.warning if warn else st.error)(friendly_error(e, using_demo_key=USING_DEMO_KEY))


BUDGET_SPENT_MSG = ("The shared demo key has used today's budget. Paste your own Google API key "
                    "in the sidebar to keep going, or come back tomorrow.")

if google_api_key:
    client = session_client(google_api_key, BUDGETED)
    gemini.set_client(client)
else:
    st.info(
        "Add a Google API key in the sidebar to start. "
        "[Getting one](https://aistudio.google.com/app/apikey) is free and takes about a minute.",
        icon=":material/key:",
    )
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
except Exception:
    log.exception("chroma init failed")
    st.error("The search index could not be opened. Reload the page, and check the Chroma "
             "settings if you entered any.")
    st.stop()


@st.cache_resource
def _sweep_clock() -> dict:
    return {"last": 0.0}


def sweep_idle_corpora():
    """Delete demo libraries nobody has used for DEMO_CORPUS_TTL_HOURS.

    Runs at most every ten minutes per process. Removes the index rows from
    every dimension's collection, the registry rows and the corpus's blobs.
    """
    clock = _sweep_clock()
    now = time.time()
    if now - clock["last"] < 600:
        return
    clock["last"] = now
    stale = stale_corpora(DEMO_CORPUS_TTL_HOURS * 3600, now=now)
    stale = [cid for cid in stale if cid != st.session_state.corpus_id]
    if not stale:
        return
    try:
        collections = [c for c in chroma_client.list_collections()
                       if getattr(c, "name", str(c)).startswith("unified_rag_")]
    except Exception:
        log.exception("could not list collections for the sweep")
        return
    for cid in stale:
        try:
            for coll in collections:
                if not hasattr(coll, "delete"):
                    coll = chroma_client.get_collection(str(coll))
                coll.delete(where={"corpus_id": cid})
            delete_corpus_sources(cid)
            forget_corpus(cid)
        except Exception:
            log.exception("could not sweep corpus %s", cid)


if DEMO_MODE:
    touch_corpus(st.session_state.corpus_id)
    sweep_idle_corpora()


def delete_chroma_ids(ids: list[str]):
    if not ids:
        return
    try:
        chroma_collection.delete(ids=ids)
    except Exception:
        log.exception("chroma delete failed")
        st.warning("Some index entries could not be deleted. Try again.")

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
    except Exception:
        log.exception("clearing the corpus failed")
        st.warning("The library could not be fully cleared. Try again.")

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
    start_new_corpus()
    if st.session_state.google_store is not None and client is not None:
        try:
            client.file_search_stores.delete(name=st.session_state.google_store.name, config={'force': True})
        except Exception as e:
            show_error(e, warn=True)
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
            show_error(e)
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
        doc_type = ("pdf" if mime == "application/pdf"
                    else "audio" if "audio" in mime else "video" if "video" in mime else "image")
        st.session_state.doc_sources.append({
            "source_id": str(uuid.uuid4()),
            "name": file_name,
            "type": doc_type,
            "mime": mime,
            "backend": MANAGED_BACKEND,
            "content_hash": content_hash,
        })
    except Exception as e:
        st.error(f"{file_name} could not be added to Google File Search. "
                 + friendly_error(e, using_demo_key=USING_DEMO_KEY))
    finally:
        os.remove(tmp_path)


def blob_namespace() -> str | None:
    """Demo blobs live in a per-session folder; a local run keeps the shared store."""
    return st.session_state.corpus_id if DEMO_MODE else None


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
        file_path = persist_blob(file_bytes, mime, name, corpus_id=blob_namespace())
        preview_path = (persist_blob(preview_bytes, "image/png", f"{name}_preview.png", corpus_id=blob_namespace())
                        if preview_bytes else None)
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
    except Exception:
        log.exception("saving a chunk failed")
        st.error(f"Part of {name.split(' · ')[0]} could not be saved to the index.")

class _QueryEmbeddingError(Exception):
    """Raised by the app's query embedder after it has already shown the error."""


class _AppQueryEmbedder:
    """Adapts the app's Gemini calls to the pipeline's embedder interface."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed_query(self, text: str) -> np.ndarray:
        try:
            result = call_with_retry(lambda: client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self.dim,
                ),
            ))
        except Exception as e:
            show_error(e)
            raise _QueryEmbeddingError() from e
        gemini.record_usage(getattr(result, "usage_metadata", None))
        return np.array(result.embeddings[0].values)

    def hyde_query(self, text: str) -> np.ndarray:
        vector = hyde_query_embedding(text, self.dim)
        return vector if vector is not None else self.embed_query(text)


def search(query: str = None, top_k: int = 3, query_emb: np.ndarray = None) -> list[dict]:
    """Hybrid dense + BM25 search over this corpus through the shared pipeline.

    Same config the app has always run (RRF over a 4x candidate pool, HyDE when
    toggled, no reranker), now in `rag.pipeline` so the CLI and evals share it.
    """
    if query_emb is None and not query:
        return []
    store = ChromaStore(chroma_collection, where={"corpus_id": st.session_state.corpus_id})
    try:
        if store.count() == 0:
            return []
    except Exception:
        return []

    cfg = Config(
        mode="hybrid_rrf",
        hyde=bool(st.session_state.get("use_hyde")),
        top_k=top_k,
        context_k=top_k,
        candidate_k=top_k * 4,
    )
    embedder = _AppQueryEmbedder(st.session_state.get("embedding_dim", 3072))
    try:
        hits = retrieve(query, cfg, store, embedder, query_emb=query_emb)
    except _QueryEmbeddingError:
        return []
    except Exception:
        log.exception("search failed")
        st.error("The search failed. Try again.")
        return []

    # Format the results
    out = []
    for hit in hits:
        meta = hit.metadata
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
            "id": hit.chunk_id,
            "name": meta.get("name", "Unknown Document"),
            "type": meta.get("type", "unknown"),
            "mime": meta.get("mime", "application/octet-stream"),
            "bytes": file_bytes,
            "preview_bytes": preview_bytes,
            "video_bytes": video_bytes if video_bytes else None,
            # Cosine similarity; keyword-only hits have no dense score and show 0.
            "score": hit.dense_score if hit.dense_score is not None else 0.0,
            "searchable_text": meta.get("searchable_text", ""),
            "timestamp_start_sec": meta.get("timestamp_start_sec"),
            "page_range": meta.get("page_range"),
        })
    # Already ordered by RRF fusion in the pipeline; a second sort here only
    # penalized keyword-only hits (whose dense score is 0), so we trust that order.
    return out

def _usage_dict(usage) -> dict | None:
    """Flatten a Gemini usage_metadata object into a plain token dict."""
    if not usage:
        return None
    return {
        "prompt": getattr(usage, "prompt_token_count", 0) or 0,
        "output": getattr(usage, "candidates_token_count", 0) or 0,
        "total": getattr(usage, "total_token_count", 0) or 0,
        "thinking": getattr(usage, "thoughts_token_count", 0) or 0,
    }


def answer(question: str, selected_results: list[dict],
           history: list[dict] | None = None):
    """Stream a grounded answer from Gemini 2.5 Flash, with optional conversation history.

    history: list of {role, content} dicts from st.session_state.messages (most recent last).
    """
    # Prior turns, then each retrieved source, then the grounded prompt.
    contents = build_answer_contents(question, selected_results, history)
    for attempt in range(3):
        produced = False
        try:
            stream = client.models.generate_content_stream(
                model=MANAGED_ANSWER_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS),
            )
            raw_usage = None
            for chunk in stream:
                u = getattr(chunk, "usage_metadata", None)
                if u:
                    raw_usage = u
                if chunk.text:
                    produced = True
                    yield chunk.text
            if raw_usage:
                st.session_state.last_usage = _usage_dict(raw_usage)
                gemini.record_usage(raw_usage)
            return
        except Exception as e:
            # Retry transient failures, but only if nothing has streamed yet —
            # once text is on screen we can't cleanly restart.
            if not produced and is_transient_error(e) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if produced:
                yield "\n\n*The answer was cut off by a model error. Ask again to get the rest.*"
            else:
                yield friendly_error(e, using_demo_key=USING_DEMO_KEY)
            return

def answer_managed(question: str,
                   history: list[dict] | None = None) -> tuple[object, list[str]]:
    """Stream a grounded answer from MANAGED_ANSWER_MODEL using the Google File Search store,
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
            model=MANAGED_ANSWER_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS,
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store.name]
                        )
                    )
                ]
            )
        )))
        if full_chunks:
            _mu = getattr(full_chunks[-1], "usage_metadata", None)
            st.session_state.last_usage = _usage_dict(_mu)
            gemini.record_usage(_mu)
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
        message = friendly_error(e, using_demo_key=USING_DEMO_KEY)

        def _error():
            yield message
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
        st.session_state.flash = "The sample charts are already in your library."
        return
    if budget_spent():
        st.session_state.flash = BUDGET_SPENT_MSG
        st.rerun()

    failed = []
    progress = st.progress(0, text="Loading sample charts…")
    for i, (name, url) in enumerate(to_load.items()):
        progress.progress(i / len(to_load), text=f"Downloading {name}…")
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

            progress.progress(i / len(to_load), text=f"Embedding {name}…")
            emb = embed_image(img_bytes, mime_type=mime)
            if emb is not None:
                add_document(emb, name, "image", mime, img_bytes,
                             searchable_text=name)
        except Exception:
            log.exception("sample %s failed", name)
            failed.append(name)

    progress.empty()
    loaded = len(to_load) - len(failed)
    note = f"Added {loaded} sample {'chart' if loaded == 1 else 'charts'} to your library."
    if failed:
        note += " Could not load " + ", ".join(failed) + ". Try again in a moment."
    # Rerun so the sidebar library, drawn earlier in this run, lists them too.
    st.session_state.flash = note
    st.rerun()

if st.session_state.load_samples_requested:
    st.session_state.load_samples_requested = False
    load_sample_images()

if _flash := st.session_state.pop("flash", None):
    st.toast(_flash, icon=":material/library_add_check:")

# process uploads from sidebar. Each uploaded file is attempted once: the
# uploader keeps its files across reruns, so without the attempted set a file
# that failed (or was deleted from the library) was embedded again on every
# click anywhere in the app.
new_uploads = [f for f in (uploaded_files or []) if f.file_id not in st.session_state.attempted_uploads]
if new_uploads:
    per_file_captions = parse_caption_map(image_captions_map_raw)
    # Dedup by content hash so re-uploads (and server reloads) don't re-embed the
    # same bytes, even if the filename changed.
    existing_hashes = {s.get("content_hash") for s in st.session_state.doc_sources if s.get("content_hash")}

    progress = st.sidebar.progress(0, text="Embedding…")
    added = 0
    skipped = 0
    total = len(new_uploads)
    for i, file in enumerate(new_uploads):
        st.session_state.attempted_uploads.add(file.file_id)
        file_bytes = file.getvalue()
        mime = file.type or ""
        content_hash = hash_bytes(file_bytes)
        if content_hash in existing_hashes:
            skipped += 1
            progress.progress((i + 1) / total, text=f"Skipped {file.name} (already indexed)")
            continue
        if DEMO_MODE and not within_demo_upload_cap(
            current=st.session_state.demo_uploads_this_session, limit=MAX_UPLOADS_PER_SESSION
        ):
            skipped += 1
            st.sidebar.warning(
                f"Demo limit: {MAX_UPLOADS_PER_SESSION} files per session. "
                "Clone the repo to run it without limits."
            )
            progress.progress((i + 1) / total, text=f"Skipped {file.name} (demo limit reached)")
            continue
        if budget_spent():
            skipped += 1
            st.sidebar.warning(BUDGET_SPENT_MSG)
            progress.progress((i + 1) / total, text=f"Skipped {file.name} (demo budget spent)")
            continue
        # Size, pages, duration and pixels are checked from headers, before any decoding.
        refusal = check_upload(file_bytes, mime, UPLOAD_LIMITS)
        if refusal:
            skipped += 1
            st.sidebar.warning(f"Skipped {file.name}: {refusal}.")
            progress.progress((i + 1) / total, text=f"Skipped {file.name}")
            continue
        existing_hashes.add(content_hash)
        st.session_state.demo_uploads_this_session += 1
        file_source_id = str(uuid.uuid4())  # one source row per uploaded file

        try:
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
                for chunk_bytes, start_sec, end_sec, chunk_mime in audio_chunks:
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
                    st.sidebar.info(f"{file.name} is long, so frames are sampled at a wider interval.")
                stored_video_path = persist_blob(file_bytes, mime, file.name, corpus_id=blob_namespace())
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

        except DemoBudgetExceeded:
            st.sidebar.warning(BUDGET_SPENT_MSG)
            break
        except Exception:
            log.exception("ingesting %s failed", file.name)
            st.sidebar.error(f"{file.name} could not be processed. Try another file.")
            continue

        added += 1
        progress.progress((i + 1) / total, text=f"Processed {file.name}")

    progress.empty()
    if added:
        st.sidebar.success(f"Added {added} file(s).")
    if skipped:
        st.sidebar.info(f"Skipped {skipped} file(s).")

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


def stream_answer(chunks) -> str:
    """Stream answer text into the page with [S1] markers drawn as highlights.

    Returns the raw text (markers intact) for the chat history.
    """
    slot = st.empty()
    text = ""
    for chunk in _escaped_stream(chunks):
        if chunk:
            text += chunk
            slot.markdown(mark_citations(text))
    slot.markdown(mark_citations(text))
    return text


_MD_SPECIAL = re.compile(r"([\\`*_\[\]#|~<>])")


def _md_escape(text: str) -> str:
    """Escape a file name so underscores or brackets in it don't turn into markdown."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def _display_name_and_locator(res: dict) -> tuple[str, str]:
    """Split a chunk name like "report.pdf · pages 1–4" into file name and locator."""
    base, _, rest = (res.get("name") or "Unknown source").partition(" · ")
    if res.get("type") == "image" and rest:
        return base, rest  # an image's caption says more than "image"
    return base, source_locator(res)


def render_thumbnail(res: dict):
    """The cited page, chart or frame itself, or a player seeked to the clip."""
    res_type = res.get("type", "")
    if res_type == "image" and res.get("bytes"):
        st.image(res["bytes"], width="stretch")
    elif res_type in ("pdf", "video_frame") and res.get("preview_bytes"):
        st.image(res["preview_bytes"], width="stretch")
    elif res_type == "audio" and res.get("bytes"):
        st.audio(res["bytes"], format=res.get("mime"), start_time=_seek_seconds(res.get("timestamp_start_sec")))
    elif res_type == "video" and res.get("bytes"):
        st.video(res["bytes"], start_time=_seek_seconds(res.get("timestamp_start_sec")))
    # Full source videos are not rendered per card: one big file copied per hit.
    # The player sits one click away in the "Open Sn" expander.


def render_evidence(results, show_score=False, with_video=True):
    """Source cards for an answer (or image-search matches), then one expander
    per cited source that opens the exact page, frame or clip.

    with_video=False skips the full source-video player. Expander contents render
    even when collapsed, so older answers in the history would otherwise re-send
    every cited video on each rerun.
    """
    if not results:
        return
    ncols = min(len(results), 3)
    cols = st.columns(ncols)
    for i, res in enumerate(results):
        name, locator = _display_name_and_locator(res)
        with cols[i % ncols], st.container(border=True):
            render_thumbnail(res)
            chip = f":yellow-background[**{res['citation_id']}**] " if res.get("citation_id") else ""
            st.markdown(f"{chip}{_md_escape(name)}")
            detail = locator
            if show_score:
                detail += f", similarity {res.get('score', 0):.3f}"
            st.caption(detail)
    for res in results:
        if not res.get("citation_id"):
            continue
        name, locator = _display_name_and_locator(res)
        with st.expander(f"Open {res['citation_id']}: {name} ({locator})", expanded=False):
            render_source_inline(res, with_video=with_video)


def render_source_inline(res: dict, with_video: bool = True):
    """Render one cited source's detail inside its expander: the exact page/frame,
    or a player seeked to the matched timestamp."""
    rtype = res.get("type")
    seek = _seek_seconds(res.get("timestamp_start_sec"))
    if rtype in ("pdf", "video_frame") and res.get("preview_bytes"):
        st.image(res["preview_bytes"], width="stretch")
    elif rtype == "image" and (res.get("preview_bytes") or res.get("bytes")):
        st.image(res.get("preview_bytes") or res.get("bytes"), width="stretch")
    elif rtype == "audio" and res.get("bytes"):
        st.audio(res["bytes"], format=res.get("mime"), start_time=seek)
    if with_video and rtype == "video_frame" and res.get("video_bytes"):
        st.video(res["video_bytes"], start_time=seek)
    if res.get("searchable_text") and rtype in ("pdf", "audio", "video_frame"):
        with st.container(height=180, border=False):
            st.caption(_md_escape(res["searchable_text"]))


def render_answer_footer(text: str, citations: list[str], prompt_tokens: int, usage, key: str):
    """Managed-backend source names, read-aloud and token cost under an answer."""
    if citations:
        st.caption("Sources: " + "; ".join(citations))
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    if c1.button("Read aloud", key=key, icon=":material/volume_up:", type="tertiary",
                 help="Read this answer aloud"):
        st.session_state.tts_pending = strip_citations(text)
        st.rerun()
    if prompt_tokens:
        c2.caption(format_prompt_cost(prompt_tokens, usage))


def render_about():
    with st.expander("How it works", icon=":material/info:"):
        st.markdown(f"""
Every file is embedded with **Gemini Embedding 2** (`{EMBEDDING_MODEL}`) into one shared
vector space, so a text question can land on a chart, a PDF page or a video frame.
Answers come from `{MANAGED_ANSWER_MODEL}`, which only sees the retrieved chunks and cites them as [S1], [S2] and so on.

| Type | Formats | Limit | How it is indexed |
|---|---|---|---|
| Images | PNG, JPG | 4096px (auto-resized) | Per-file or auto caption |
| PDFs | PDF | 4 pages per embed call | Overlapping page chunks, text kept for keyword search |
| Audio | MP3, WAV | 80s per chunk | Timestamped chunks, optional transcript |
| Video | MP4, MOV | Long files sampled | Frames at adaptive intervals, with timestamps |

Search runs dense vectors and BM25 keyword ranking side by side and fuses them with
reciprocal rank fusion. The top chunks go to the answer model with their source IDs.
Managed mode hands the files to Google File Search instead.
""")


if not st.session_state.doc_sources:
    with st.container(border=True):
        st.subheader("Your library is empty")
        st.markdown(
            "Add PDFs, images, audio or video from the sidebar. Each file is split into pages, "
            "frames or clips and embedded into one index, so a single question searches all of them."
        )
        st.caption("PDF, PNG, JPG, MP3, WAV, MP4 and MOV")
        wc1, wc2 = st.columns([2, 3], vertical_alignment="center")
        if wc1.button("Load 6 sample charts", key="welcome_load_samples", type="primary", width="stretch"):
            st.session_state.load_samples_requested = True
            st.rerun()
        wc2.caption(SAMPLE_CHARTS_HELP)
else:
    top_k = st.session_state.get("top_k", 3)  # set in the sidebar Settings expander

    # Starter prompts before the first question, so a new user sees it work fast.
    if not st.session_state.messages:
        st.markdown("**Try a question**")
        starters = ["What are these files about?", "Summarize the key points", "What figures stand out?"]
        for _col, _prompt in zip(st.columns(len(starters)), starters):
            if _col.button(_prompt, key=f"starter_{_prompt}", width="stretch"):
                st.session_state.pending_query = _prompt
                st.rerun()

    # Render the full chat history
    last_idx = len(st.session_state.messages) - 1
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            st.markdown(mark_citations(msg["content"]) if msg["role"] == "assistant" else msg["content"])
            if msg.get("results"):
                render_evidence(msg["results"], show_score=msg.get("image_search", False),
                                with_video=msg_idx == last_idx)
            if msg["role"] == "assistant" and msg.get("content"):
                render_answer_footer(
                    msg["content"],
                    [] if msg.get("results") else msg.get("citations", []),
                    msg.get("prompt_tokens", 0),
                    msg.get("usage"),
                    key=f"tts_{msg_idx}",
                )

    # Chat input — a starter-prompt click feeds in via pending_query
    query = (st.chat_input("Ask about your library…", key="chat_input", max_chars=MAX_QUESTION_CHARS)
             or st.session_state.pop("pending_query", None))
    query = clamp_question(query) if query else query

    if query and budget_spent():
        for role, content in (("user", query), ("assistant", BUDGET_SPENT_MSG)):
            with st.chat_message(role, avatar=AVATARS[role]):
                st.markdown(content)
            st.session_state.messages.append({"role": role, "content": content, "results": [], "citations": []})
        query = None

    if query:
        st.session_state.last_usage = None  # reset; populated by the answer call
        st.session_state.prompt_tokens = 0  # accumulates this prompt's full token cost
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Resolve strategy — Auto runs the router first
        if retrieval_strategy == "Auto (Hybrid)":
            with st.spinner("Choosing a search backend…"):
                route = route_query(query)
            # Only use managed if there are actually managed docs; otherwise the
            # store is empty and the router would send the query nowhere useful.
            has_managed_docs = any(s.get("backend") == MANAGED_BACKEND for s in st.session_state.doc_sources)
            use_managed = (route == "managed" and has_managed_docs)
            st.toast(f"Searching {'Google File Search' if use_managed else 'the local index'}",
                     icon=":material/alt_route:")
        else:
            use_managed = (retrieval_strategy == "Managed RAG (Google File Search)")

        if use_managed:
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                with st.spinner("Searching Google File Search…"):
                    gen, citations = answer_managed(query,
                                                   history=st.session_state.messages)
                full_text = stream_answer(gen)
                usage = st.session_state.get("last_usage")
                prompt_tokens = st.session_state.get("prompt_tokens", 0)
                render_answer_footer(full_text, citations, prompt_tokens, usage, key="tts_live_managed")
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "results": [],
                "citations": citations,
                "usage": usage,
                "prompt_tokens": prompt_tokens,
            })
        else:
            # Rewrite contextual follow-ups ("at which timestamp?") into a
            # standalone query so retrieval stays on the prior subject. The answer
            # still uses the original wording.
            search_query = gemini.rewrite_search_query(query, st.session_state.messages[:-1])
            with st.spinner("Searching the local index…"):
                results = search(search_query, top_k=top_k)

            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                if results:
                    selected_results = select_context_results(results, max_results=top_k)
                    full_text = stream_answer(
                        answer(query, selected_results,
                               history=st.session_state.messages)
                    )
                    source_labels = [
                        f"[{res['citation_id']}] {res['name']}" for res in selected_results
                    ]
                    usage = st.session_state.get("last_usage")
                    prompt_tokens = st.session_state.get("prompt_tokens", 0)
                    render_evidence(selected_results)
                    render_answer_footer(full_text, [], prompt_tokens, usage, key="tts_live_local")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "results": selected_results,
                        "citations": source_labels,
                        "usage": usage,
                        "prompt_tokens": prompt_tokens,
                    })
                else:
                    msg = ("Nothing in your library matched that question. "
                           "Try other wording, or check the file is listed in the library.")
                    st.markdown(msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg,
                        "results": [],
                        "citations": [],
                    })

# Image query processing (fired by the sidebar Find similar button)
_img_refusal = None
if run_img_query and query_image_sidebar:
    _img_refusal = (BUDGET_SPENT_MSG if budget_spent()
                    else image_too_large(query_image_sidebar.getvalue(), UPLOAD_LIMITS.max_image_pixels))
    if _img_refusal:
        st.sidebar.warning(_img_refusal if _img_refusal == BUDGET_SPENT_MSG
                           else f"That image can't be searched: {_img_refusal}.")

if run_img_query and query_image_sidebar and not _img_refusal:
    query_bytes = query_image_sidebar.getvalue()
    query_mime = query_image_sidebar.type
    dim = st.session_state.get("embedding_dim", 3072)

    # Post the query image as a user message
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.image(query_bytes, width=200, caption="Image query")
    st.session_state.messages.append({
        "role": "user",
        "content": f"Image search: {query_image_sidebar.name}",
    })

    with st.spinner("Embedding the query image…"):
        try:
            result = call_with_retry(lambda: client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[types.Part.from_bytes(data=query_bytes, mime_type=query_mime)],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=dim,
                ),
            ))
            gemini.record_usage(getattr(result, "usage_metadata", None))
            query_emb = np.array(result.embeddings[0].values)
        except Exception as e:
            show_error(e)
            query_emb = None

    if query_emb is not None:
        img_results = search(query=None, top_k=top_k_img, query_emb=query_emb)
        found = (f"The {len(img_results)} closest {'match' if len(img_results) == 1 else 'matches'} "
                 "in your library.")

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.markdown(found)
            render_evidence(img_results, show_score=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": found,
            "results": img_results,
            "citations": [],
            "image_search": True,
        })

# Running token total for the session — rendered last so it reflects this run's calls.
st.sidebar.caption(f"Session usage: {st.session_state.get('session_tokens', 0):,} tokens")

render_about()
st.caption(f"Unified RAG, built on Gemini Embedding 2 and {MANAGED_ANSWER_MODEL}.")
