"""
Central config for the Streaming, Real-Time RAG service.

Time model (read this first; everything else depends on it). Every chunk has
four timestamps, all epoch seconds:

  published_at   EVENT time  - when the source says the thing happened
  ingested_at    SYSTEM time - when this service learned about it
  expires_at     end of the relevance window (published_at + source TTL)
  superseded_at  when a newer version of the same item replaced this chunk

A chunk is visible at time T iff
      published_at <= T  AND  ingested_at <= T  AND  expires_at > T  AND  superseded_at > T

"Never expires" / "never superseded" are stored as the NEVER sentinel (year 2100)
rather than NULL, so every filter is one uniform numeric range condition in both
Qdrant and SQLite. Because visibility is a pure function of T, the system can be
asked what it knew at any past moment, which is what audit mode uses.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# --- Storage -------------------------------------------------------------------
REGISTRY_DB = os.environ.get("REGISTRY_DB", str(ROOT / "registry.db"))
QDRANT_URL = os.environ.get("QDRANT_URL", "")                    # set for a real Qdrant server
QDRANT_PATH = os.environ.get("QDRANT_PATH", str(ROOT / "qdrant_data"))  # local mode; ":memory:" for tests
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "live_chunks")

# --- Embeddings ----------------------------------------------------------------
EMBEDDER = os.environ.get("EMBEDDER", "sentence-transformers")   # or "hash" (offline, no model download)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# --- Chunking ------------------------------------------------------------------
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "120"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "20"))
MAX_ITEM_CHARS = 20_000
MAX_PUSH_ITEMS = 500

# --- Freshness -----------------------------------------------------------------
NEVER = 4_102_444_800.0                                           # 2100-01-01 UTC
DEFAULT_HALF_LIFE_S = int(os.environ.get("DEFAULT_HALF_LIFE_S", str(6 * 3600)))
DECAY_FLOOR = float(os.environ.get("DECAY_FLOOR", "0.1"))         # decay never zeroes a relevant old chunk
DEFAULT_STALE_AFTER_S = int(os.environ.get("DEFAULT_STALE_AFTER_S", str(24 * 3600)))
OVERSAMPLE = int(os.environ.get("OVERSAMPLE", "4"))               # candidates fetched = top_k * OVERSAMPLE
AUDIT_RETENTION_S = int(os.environ.get("AUDIT_RETENTION_S", str(90 * 24 * 3600)))

# --- Auth / security -------------------------------------------------------------
API_KEY_PEPPER = os.environ.get("API_KEY_PEPPER", "dev-only-pepper-change-me")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")                       # if set, required to onboard tenants
ALLOW_PRIVATE_FETCH = _flag("ALLOW_PRIVATE_FETCH")                # dev only: lets connectors hit localhost/LAN
EXPORTS_DIR = Path(os.environ.get("EXPORTS_DIR", ROOT / "exports"))

# --- Runtime -------------------------------------------------------------------
ENABLE_SCHEDULER = _flag("ENABLE_SCHEDULER", "1")
INGEST_WORKER = _flag("INGEST_WORKER", "1")                       # 0 = run ingestion inline (tests)

# --- Grok (answer generation) ------------------------------------------------------
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4")
