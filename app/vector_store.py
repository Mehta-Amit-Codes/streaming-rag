"""
Qdrant wrapper. One collection for all tenants (Qdrant's recommended multitenancy
pattern) with `tenant_id` in the payload.

Isolation model, stated honestly: unlike the Postgres version (Row-Level Security),
Qdrant has no database-enforced equivalent. Isolation here depends on `search()` being
the ONLY read path and it having no way to omit the tenant filter: `tenant_id` is a
required positional argument and is always the first `must` condition. Tests assert
cross-tenant leakage is impossible through every public entry point. If you need
isolation that survives a code bug, use collection-per-tenant or the Postgres design.

Visibility at time T is a pure range-filter predicate over four payload fields (see
config.py), so asking "what was visible at T" for any past T needs no special storage.
"""
import os
import threading
import uuid
import warnings

from qdrant_client import QdrantClient, models

from app.config import QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL

_lock = threading.RLock()  # qdrant local mode is not thread-safe; also serialises ensure_collection
_client: QdrantClient | None = None
_ready = False


def chunk_point_id(tenant_id: str, item_key: str, chunk_hash: str) -> str:
    """Deterministic id: re-ingesting identical content upserts the same point, never a duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}|{item_key}|{chunk_hash}"))


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY") or None)
        elif QDRANT_PATH == ":memory:":
            _client = QdrantClient(":memory:")
        else:
            _client = QdrantClient(path=QDRANT_PATH)
    return _client


def ensure_collection(dim: int) -> None:
    global _ready
    with _lock:
        if _ready:
            return
        c = _get_client()
        if not c.collection_exists(QDRANT_COLLECTION):
            c.create_collection(QDRANT_COLLECTION, vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE))
            with warnings.catch_warnings():
                if not QDRANT_URL:  # local mode just says "indexes have no effect"; true and irrelevant there
                    warnings.simplefilter("ignore", UserWarning)
                _create_indexes(c)
        else:
            existing = c.get_collection(QDRANT_COLLECTION).config.params.vectors.size
            if existing != dim:
                raise RuntimeError(
                    f"Collection '{QDRANT_COLLECTION}' has dimension {existing} but the embedder produces {dim}. "
                    "Changing embedding models requires re-indexing into a new collection.")
        _ready = True


def _create_indexes(c: QdrantClient) -> None:
    """Payload indexes matter on a real server (they make the filters fast); local mode ignores them."""
    try:
        c.create_payload_index(QDRANT_COLLECTION, "tenant_id", models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True))
    except Exception:
        try:
            c.create_payload_index(QDRANT_COLLECTION, "tenant_id", models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
    for field, schema in [("source_id", models.PayloadSchemaType.KEYWORD),
                          ("published_at", models.PayloadSchemaType.FLOAT),
                          ("ingested_at", models.PayloadSchemaType.FLOAT),
                          ("expires_at", models.PayloadSchemaType.FLOAT),
                          ("superseded_at", models.PayloadSchemaType.FLOAT)]:
        try:
            c.create_payload_index(QDRANT_COLLECTION, field, schema)
        except Exception:
            pass


def upsert(dim: int, points: list[dict]) -> None:
    """points: [{"id", "vector", "payload"}]"""
    if not points:
        return
    ensure_collection(dim)
    with _lock:
        _get_client().upsert(QDRANT_COLLECTION, points=[
            models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points])


def set_superseded(dim: int, ids: list[str], ts: float) -> None:
    if not ids:
        return
    ensure_collection(dim)
    with _lock:
        _get_client().set_payload(QDRANT_COLLECTION, payload={"superseded_at": ts}, points=ids)


def delete(dim: int, ids: list[str]) -> None:
    if not ids:
        return
    ensure_collection(dim)
    with _lock:
        _get_client().delete(QDRANT_COLLECTION, points_selector=models.PointIdsList(points=ids))


def search(dim: int, tenant_id: str, vector: list[float], limit: int, at: float,
           source_ids: list[str] | None = None) -> list[dict]:
    """
    Everything visible to `tenant_id` at time `at`, nearest first. `tenant_id` has no default
    on purpose: there is no way to call this without scoping to a tenant.
    """
    if not tenant_id:
        raise ValueError("tenant_id is required")
    ensure_collection(dim)
    must = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
        models.FieldCondition(key="published_at", range=models.Range(lte=at)),
        models.FieldCondition(key="ingested_at", range=models.Range(lte=at)),
        models.FieldCondition(key="expires_at", range=models.Range(gt=at)),
        models.FieldCondition(key="superseded_at", range=models.Range(gt=at)),
    ]
    if source_ids:
        must.append(models.FieldCondition(key="source_id", match=models.MatchAny(any=source_ids)))
    with _lock:
        res = _get_client().query_points(
            QDRANT_COLLECTION, query=vector, query_filter=models.Filter(must=must),
            limit=limit, with_payload=True)
    return [{"id": str(p.id), "score": float(p.score), "payload": p.payload} for p in res.points]
