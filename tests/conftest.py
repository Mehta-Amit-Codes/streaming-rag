import os
import tempfile

# Must be set before any `app.*` import: in-memory Qdrant, offline hash embedder, inline ingestion,
# no background scheduler, tiny chunks so short test texts split into several chunks.
os.environ.update({
    "QDRANT_PATH": ":memory:", "EMBEDDER": "hash", "INGEST_WORKER": "0", "ENABLE_SCHEDULER": "0",
    "REGISTRY_DB": os.path.join(tempfile.mkdtemp(), "registry.db"),
    "CHUNK_WORDS": "20", "CHUNK_OVERLAP": "5", "EXPORTS_DIR": tempfile.mkdtemp(),
})
os.environ.pop("GROK_API_KEY", None)  # tests must never call the real LLM
os.environ.pop("ADMIN_KEY", None)
os.environ.pop("ALLOW_PRIVATE_FETCH", None)

import pytest

from app import registry

registry.init()


@pytest.fixture()
def tenant():
    """A fresh tenant per test. The vector store is shared, so isolation is exercised everywhere."""
    import uuid
    return registry.create_tenant(f"t-{uuid.uuid4().hex[:6]}", uuid.uuid4().hex)


@pytest.fixture()
def make_source(tenant):
    def _make(tenant_id=None, name="feed", type_="webhook", ttl=None, half_life=3600, config=None):
        return registry.create_source(tenant_id or tenant, name, type_, config or {}, 0, ttl, half_life, None)
    return _make
