from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app import ingestion, query as query_svc, registry, scheduler
from app.auth import get_tenant, new_secret, require_admin, verify_secret
from app.config import ENABLE_SCHEDULER, MAX_ITEM_CHARS, MAX_PUSH_ITEMS
from app.connectors import POLLABLE_TYPES, SOURCE_TYPES, Item, validate_source_config
from app.freshness import MODES
from app.timeutil import iso, parse_time
from app.worker import ingest_queue

registry.init()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest_queue.start()
    if ENABLE_SCHEDULER:
        scheduler.start()
    yield
    scheduler.shutdown()
    ingest_queue.stop()


app = FastAPI(title="Streaming, Real-Time RAG", version="0.1.0", lifespan=lifespan,
              description="Freshness-aware, multi-tenant RAG over continuously updating sources.")


# ------------------------------------------------------------------ tenants ----
class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1)


@app.post("/tenants", status_code=201, dependencies=[Depends(require_admin)])
def create_tenant(req: TenantCreate):
    key, key_hash = new_secret("srag_live_")
    tid = registry.create_tenant(req.name, key_hash)
    return {"tenant_id": tid, "api_key": key}  # the plaintext key is shown exactly once


# ------------------------------------------------------------------ sources ----
class SourceCreate(BaseModel):
    name: str = Field(..., min_length=1)
    type: str
    config: dict = {}
    poll_interval_seconds: int = Field(300, ge=0)       # 0 = manual polling only
    ttl_seconds: int | None = Field(None, ge=1)         # relevance window; None = never expires
    half_life_seconds: int = Field(6 * 3600, ge=0)      # 0 = evergreen (no decay)


class SourcePatch(BaseModel):
    enabled: bool | None = None
    poll_interval_seconds: int | None = Field(None, ge=0)
    ttl_seconds: int | None = Field(None, ge=1)
    half_life_seconds: int | None = Field(None, ge=0)


def _public_source(s: dict) -> dict:
    return {"id": s["id"], "name": s["name"], "type": s["type"], "enabled": s["enabled"],
            "poll_interval_seconds": s["poll_interval_seconds"], "ttl_seconds": s["ttl_seconds"],
            "half_life_seconds": s["half_life_seconds"], "last_polled_at": s["last_polled_at"],
            "last_error": s["last_error"], "consecutive_failures": s["consecutive_failures"],
            "next_run_at": scheduler.next_run(s["id"]),
            "config": s["config"] if s["type"] != "slack_export" or "path" in s["config"] else {"channel": s["config"].get("channel")}}


@app.post("/sources", status_code=201)
def create_source(req: SourceCreate, tenant: dict = Depends(get_tenant)):
    try:
        validate_source_config(req.type, req.config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    token, token_hash = (new_secret("whk_") if req.type == "webhook" else (None, None))
    src = registry.create_source(tenant["id"], req.name, req.type, req.config, req.poll_interval_seconds,
                                 req.ttl_seconds, req.half_life_seconds, token_hash)
    scheduler.sync_jobs()
    out = _public_source(src)
    if token:
        out["webhook_token"] = token  # shown once; webhook callers send it as X-Webhook-Token
    return out


@app.get("/sources")
def list_sources(tenant: dict = Depends(get_tenant)):
    return [_public_source(s) for s in registry.list_sources(tenant["id"])]


@app.patch("/sources/{source_id}")
def patch_source(source_id: str, req: SourcePatch, tenant: dict = Depends(get_tenant)):
    if registry.get_source(tenant["id"], source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")
    src = registry.update_source(tenant["id"], source_id, **req.model_dump(exclude_unset=True))
    scheduler.sync_jobs()
    return _public_source(src)


@app.delete("/sources/{source_id}")
def delete_source(source_id: str, tenant: dict = Depends(get_tenant)):
    if registry.get_source(tenant["id"], source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")
    hidden = ingestion.retire_source(tenant["id"], source_id)
    registry.delete_source(tenant["id"], source_id)
    scheduler.sync_jobs()
    return {"deleted": True, "chunks_hidden": hidden}


@app.post("/sources/{source_id}/poll")
def poll_source(source_id: str, wait: bool = True, tenant: dict = Depends(get_tenant)):
    src = registry.get_source(tenant["id"], source_id)
    if src is None:
        raise HTTPException(status_code=404, detail="Source not found")
    if src["type"] not in POLLABLE_TYPES:
        raise HTTPException(status_code=422, detail="Push-only source; send items instead of polling")
    task = ingest_queue.submit(ingestion.poll_source, source_id, "manual")
    if not wait:
        return {"queued": True, "pending": ingest_queue.pending()}
    result = task.wait(60)
    if task.error:
        raise HTTPException(status_code=500, detail=task.error)
    return result or {"queued": True, "pending": ingest_queue.pending()}


# ------------------------------------------------------------------- pushes ----
class ItemIn(BaseModel):
    id: str = Field(..., min_length=1)
    title: str = ""
    text: str = Field(..., min_length=1, max_length=MAX_ITEM_CHARS)
    url: str = ""
    published_at: str | float | None = None   # ISO-8601 or epoch; omitted = now (flagged as assumed)


class ItemsIn(BaseModel):
    items: list[ItemIn] = Field(..., min_length=1, max_length=MAX_PUSH_ITEMS)


def _to_items(batch: ItemsIn) -> list[Item]:
    import time
    now, out = time.time(), []
    for i in batch.items:
        try:
            ts = parse_time(i.published_at)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"bad published_at for item '{i.id}'")
        out.append(Item(i.id, i.title, i.text, i.url, ts if ts is not None else now, ts is None))
    return out


def _push(source: dict, batch: ItemsIn, trigger: str, wait: bool):
    if source["type"] != "webhook":
        raise HTTPException(status_code=422, detail="Only 'webhook' sources accept pushed items")
    task = ingest_queue.submit(ingestion.push_items, source["id"], _to_items(batch), trigger)
    if not wait:
        return {"queued": True, "pending": ingest_queue.pending()}
    result = task.wait(60)
    if task.error:
        raise HTTPException(status_code=500, detail=task.error)
    return result or {"queued": True, "pending": ingest_queue.pending()}


@app.post("/sources/{source_id}/items")
def push_items(source_id: str, batch: ItemsIn, wait: bool = True, tenant: dict = Depends(get_tenant)):
    src = registry.get_source(tenant["id"], source_id)
    if src is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return _push(src, batch, "push", wait)


@app.post("/webhooks/{source_id}")
def webhook(source_id: str, batch: ItemsIn, x_webhook_token: str = Header(default=""), wait: bool = False):
    """For external systems (no API key): authenticated by the per-source webhook token."""
    src = registry.get_source_by_id(source_id)
    if src is None or not verify_secret(x_webhook_token, src["webhook_token_hash"]):
        raise HTTPException(status_code=401, detail="Invalid webhook token")  # same error for both: no source probing
    return _push(src, batch, "webhook", wait)


# -------------------------------------------------------------------- query ----
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    mode: str = "freshness"
    as_of: str | float | None = None
    half_life_seconds: float | None = Field(None, gt=0)
    source_ids: list[str] | None = None
    min_similarity: float = Field(0.25, ge=0, le=1)


def _as_of(value) -> float | None:
    try:
        return parse_time(value)
    except ValueError:
        raise HTTPException(status_code=422, detail="as_of must be ISO-8601 or epoch seconds")


def _check_mode(mode: str):
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")


@app.post("/query")
def query(req: QueryRequest, tenant: dict = Depends(get_tenant)):
    _check_mode(req.mode)
    return query_svc.run_query(tenant["id"], req.question, top_k=req.top_k, mode=req.mode, as_of=_as_of(req.as_of),
                               half_life_override=req.half_life_seconds, source_ids=req.source_ids,
                               min_similarity=req.min_similarity)


@app.post("/query/retrieve")
def retrieve(req: QueryRequest, tenant: dict = Depends(get_tenant)):
    """Retrieval only, same shape as the multi-tenant RAG service, so the cost-control and
    eval-monitoring projects can point at this service unchanged."""
    _check_mode(req.mode)
    return query_svc.retrieve_only(tenant["id"], req.question, top_k=req.top_k, mode=req.mode,
                                   as_of=_as_of(req.as_of), half_life_override=req.half_life_seconds,
                                   source_ids=req.source_ids, min_similarity=req.min_similarity)


# -------------------------------------------------------------------- audit ----
@app.get("/audit/queries")
def audit_queries(limit: int = 30, tenant: dict = Depends(get_tenant)):
    return [{"query_id": q["id"], "ts": q["ts"], "ts_iso": iso(q["ts"]), "question": q["question"],
             "mode": q["params"]["mode"], "as_of": q["params"].get("as_of"), "n_chunks": len(q["chunk_ids"])}
            for q in registry.recent_queries(tenant["id"], limit)]


@app.get("/audit/{query_id}")
def audit_replay(query_id: str, regenerate: bool = False, tenant: dict = Depends(get_tenant)):
    result = query_svc.replay(tenant["id"], query_id, regenerate)
    if result is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return result


# ------------------------------------------------------------ stats / maint ----
@app.get("/stats")
def stats(tenant: dict = Depends(get_tenant)):
    import time
    runs = registry.recent_runs(tenant["id"], 200)
    embedded, reused = sum(r["chunks_embedded"] or 0 for r in runs), sum(r["chunks_reused"] or 0 for r in runs)
    seen = sum((r["fetched"] or 0) for r in runs)
    skipped = sum((r["duplicate_items"] or 0) + (r["duplicate_content"] or 0) for r in runs)
    return {"chunks": registry.chunk_stats(tenant["id"], time.time()),
            "ingestion": {"items_seen": seen, "items_skipped_as_duplicates": skipped,
                          "dedup_ratio": round(skipped / seen, 3) if seen else 0.0,
                          "chunks_embedded": embedded, "chunks_reused": reused},
            "queue_pending": ingest_queue.pending()}


@app.get("/runs")
def runs(limit: int = 30, tenant: dict = Depends(get_tenant)):
    return registry.recent_runs(tenant["id"], limit)


class PurgeRequest(BaseModel):
    retention_seconds: int | None = Field(None, ge=0)


@app.post("/maintenance/purge")
def purge(req: PurgeRequest, tenant: dict = Depends(get_tenant)):
    from app.config import AUDIT_RETENTION_S
    return ingestion.purge_expired(retention_s=req.retention_seconds if req.retention_seconds is not None
                                   else AUDIT_RETENTION_S, tenant_id=tenant["id"])


@app.get("/health")
def health():
    return {"status": "ok"}
