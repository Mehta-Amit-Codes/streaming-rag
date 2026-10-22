"""
Streaming ingestion: pull (or receive) items, embed ONLY what is new.

Per-item decision tree
  1. expired on arrival?            published_at + TTL already in the past -> skip (a backfill must
                                    not resurrect stale facts)
  2. same item, same content hash?  duplicate -> skip (just touch last_seen)
  3. other live item, same hash?    syndicated copy of the same story -> skip
  4. otherwise it is new or changed. Chunk it, then at CHUNK granularity:
        unchanged chunk (hash seen in the previous version)  -> reuse, no re-embed
        new chunk                                            -> embed + upsert
        old chunk no longer present                          -> superseded (kept for audit, hidden from search)

Write order matters for crash safety: embed -> upsert vectors -> registry. Vector ids are
deterministic, so a crash between steps is healed by simply re-running (the item is not yet
registered, so it is re-processed and the upsert is idempotent). The reverse order could register an
item whose vectors never landed, and dedup would then block it from ever being fixed.
"""
import time
from dataclasses import asdict, dataclass

from app import registry, vector_store
from app.chunking import chunk_words, normalize_text, sha256
from app.config import AUDIT_RETENTION_S, CHUNK_OVERLAP, CHUNK_WORDS, MAX_ITEM_CHARS, NEVER
from app.connectors import Item, NotPollable, build_connector
from app.embedding import get_embedder


@dataclass
class IngestStats:
    fetched: int = 0
    new_items: int = 0
    updated_items: int = 0
    duplicate_items: int = 0
    duplicate_content: int = 0
    expired_on_arrival: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_superseded: int = 0


def _make_chunks(item: Item) -> list[str]:
    body = f"{item.title}\n\n{item.text}".strip()[:MAX_ITEM_CHARS]
    pieces = chunk_words(body, CHUNK_WORDS, CHUNK_OVERLAP)
    # Later chunks lose the headline, so re-attach it: "price is $10" alone is meaningless.
    return [p if i == 0 or not item.title else f"{item.title} - {p}" for i, p in enumerate(pieces)]


def ingest_items(tenant_id: str, source: dict, items: list[Item], now: float | None = None) -> IngestStats:
    now = now or time.time()
    stats = IngestStats()
    embedder = get_embedder()

    for item in items:
        stats.fetched += 1
        key = f"{source['id']}:{item.external_id}"
        # Clamp future-dated items (clock skew) so they don't stay invisible until their date arrives.
        published = min(item.published_at, now)
        ttl = source["ttl_seconds"]
        expires = published + ttl if ttl else NEVER
        if expires <= now:
            stats.expired_on_arrival += 1
            continue

        content_hash = sha256(normalize_text(f"{item.title}\n{item.text}"))
        existing = registry.get_item(tenant_id, key)
        if existing and existing["content_hash"] == content_hash:
            registry.touch_item(tenant_id, key, now)
            stats.duplicate_items += 1
            continue
        if registry.find_live_item_by_hash(tenant_id, content_hash, key, now):
            stats.duplicate_content += 1
            continue

        pieces = _make_chunks(item)
        if not pieces:
            continue
        hashes = [sha256(p) for p in pieces]

        version = existing["version"] + 1 if existing else 1
        old_live = registry.live_chunks_for_item(tenant_id, key, now) if existing else []
        old_hashes = {c["chunk_hash"] for c in old_live}
        new_hash_set = set(hashes)
        to_supersede = [c["id"] for c in old_live if c["chunk_hash"] not in new_hash_set]

        fresh_idx, seen = [], set()
        for idx, h in enumerate(hashes):
            if h in old_hashes or h in seen:
                stats.chunks_reused += 1
                continue
            seen.add(h)
            fresh_idx.append(idx)

        points, rows = [], []
        if fresh_idx:
            vectors = embedder.embed([pieces[i] for i in fresh_idx])  # 1. embed (pure; nothing written yet)
            for i, vec in zip(fresh_idx, vectors):
                cid = vector_store.chunk_point_id(tenant_id, key, hashes[i])
                payload = {
                    "tenant_id": tenant_id, "source_id": source["id"], "source_name": source["name"],
                    "item_key": key, "chunk_index": i, "chunk_hash": hashes[i], "title": item.title,
                    "url": item.url, "text": pieces[i], "version": version,
                    "published_at": published, "ingested_at": now, "expires_at": expires,
                    "superseded_at": NEVER, "published_assumed": item.published_assumed,
                }
                points.append({"id": cid, "vector": vec, "payload": payload})
                rows.append({"id": cid, "tenant_id": tenant_id, "item_key": key, "source_id": source["id"],
                             "version": version, "chunk_index": i, "chunk_hash": hashes[i], "ingested_at": now,
                             "published_at": published, "expires_at": expires, "superseded_at": NEVER})
            vector_store.upsert(embedder.dim, points)  # 2. vectors (idempotent ids)
        registry.insert_chunks(rows)                   # 3. bookkeeping
        if to_supersede:
            vector_store.set_superseded(embedder.dim, to_supersede, now)
            registry.supersede_chunks(to_supersede, now)
        registry.upsert_item(tenant_id, key, source["id"], item.external_id, content_hash, version,
                             published, expires, item.title, item.url, now)

        stats.chunks_embedded += len(points)
        stats.chunks_superseded += len(to_supersede)
        if existing:
            stats.updated_items += 1
        else:
            stats.new_items += 1
    return stats


def _record_run(source: dict, trigger: str, started: float, stats: IngestStats | None, error: str | None) -> dict:
    s = asdict(stats) if stats else asdict(IngestStats())
    run_id = registry.add_run(
        tenant_id=source["tenant_id"], source_id=source["id"], source_name=source["name"], trigger=trigger,
        started_at=started, finished_at=time.time(), status="error" if error else "ok", error=error, **s)
    return {"run_id": run_id, "status": "error" if error else "ok", "error": error, **s}


def poll_source(source_id: str, trigger: str = "schedule", now: float | None = None) -> dict:
    """Fetch from a pollable source and ingest. Errors are recorded, never raised: one bad feed must
    not kill the worker thread or the scheduler."""
    source = registry.get_source_by_id(source_id)
    if source is None:
        return {"status": "error", "error": "source not found"}
    started = time.time()
    now = now or started
    try:
        items, new_state = build_connector(source).fetch(source["state"], now)
        stats = ingest_items(source["tenant_id"], source, items, now)
        registry.record_poll_result(source_id, new_state, None, now)
        return _record_run(source, trigger, started, stats, None)
    except NotPollable as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        registry.record_poll_result(source_id, None, str(e), now)
        return _record_run(source, trigger, started, None, f"{type(e).__name__}: {e}")


def push_items(source_id: str, items: list[Item], trigger: str = "push", now: float | None = None) -> dict:
    source = registry.get_source_by_id(source_id)
    if source is None:
        return {"status": "error", "error": "source not found"}
    started = time.time()
    try:
        stats = ingest_items(source["tenant_id"], source, items, now or started)
        return _record_run(source, trigger, started, stats, None)
    except Exception as e:
        return _record_run(source, trigger, started, None, f"{type(e).__name__}: {e}")


def retire_source(tenant_id: str, source_id: str, now: float | None = None) -> int:
    """Deleting a source hides its data from search immediately (supersede) but keeps it replayable
    for audit until the retention purge removes it."""
    now = now or time.time()
    ids = [c["id"] for c in registry.live_chunks_for_source(tenant_id, source_id, now)]
    vector_store.set_superseded(get_embedder().dim, ids, now)
    registry.supersede_chunks(ids, now)
    return len(ids)


def purge_expired(now: float | None = None, retention_s: int = AUDIT_RETENTION_S,
                  tenant_id: str | None = None) -> dict:
    """Physically delete chunks that expired/were superseded more than `retention_s` ago. Until then they
    are only HIDDEN, which is what lets audit replay reconstruct past answers. Rows stay in the registry
    flagged purged=1, so audit can say 'this chunk existed but was purged' instead of silently differing."""
    now = now or time.time()
    ids = registry.purge_candidates(now - retention_s, tenant_id)
    vector_store.delete(get_embedder().dim, ids)  # vectors first: a crash leaves rows to retry next sweep
    registry.mark_purged(ids)
    return {"purged": len(ids), "retention_seconds": retention_s}
