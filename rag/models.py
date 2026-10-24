"""Model ids in one place.

gemini-2.5-flash started returning 404 "no longer available to new users" on
recently issued API keys while it was hardcoded in five call sites, so a key
rotation would have broken all of them at once. This module has no Streamlit
import, so the app, the CLI and the eval code can all read it.
"""

UTILITY_MODEL = "gemini-3.5-flash-lite"   # routing, rewriting, captions, transcription, HyDE drafts
EMBEDDING_MODEL = "gemini-embedding-2-preview"
ANSWER_MODEL = "gemini-3.6-flash"
DEMO_ANSWER_MODEL = "gemini-3.5-flash-lite"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
