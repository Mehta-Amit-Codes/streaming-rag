"""
SQLite registry: the system of record for everything that is NOT a vector.

  tenants    API-key auth (same hashed-key scheme as the multi-tenant RAG service)
  sources    per-tenant live feeds + their TTL / half-life / poll interval
  items      one row per external item (feed entry, Slack message...), with a content hash
  chunks     lifecycle of every chunk ever embedded (ingested / superseded / purged).
             Rows are KEPT after purge (purged=1) so audit can say "this chunk existed".
  runs       ingestion run log (new / updated / duplicate / expired counts)
  query_log  every answered query, with the exact params and chunk ids (audit replay)

Vectors + searchable payload live in Qdrant; this DB is the bookkeeping that makes
dedup, supersession, expiry and audit possible.
"""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from app.config import NEVER, REGISTRY_DB


@contextmanager
def _conn():
    conn = sqlite3.connect(REGISTRY_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # API threads and the ingest worker share this file
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY, name TEXT, api_key_hash TEXT UNIQUE, created_at REAL);
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, type TEXT,
            config TEXT, state TEXT, poll_interval_seconds INTEGER,
            ttl_seconds INTEGER, half_life_seconds INTEGER, webhook_token_hash TEXT,
            enabled INTEGER DEFAULT 1, created_at REAL, last_polled_at REAL,
            last_error TEXT, consecutive_failures INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_sources_tenant ON sources(tenant_id);
        CREATE TABLE IF NOT EXISTS items (
            tenant_id TEXT, item_key TEXT, source_id TEXT, external_id TEXT,
            content_hash TEXT, version INTEGER, published_at REAL, expires_at REAL,
            title TEXT, url TEXT, first_seen_at REAL, last_seen_at REAL,
            PRIMARY KEY (tenant_id, item_key));
        CREATE INDEX IF NOT EXISTS idx_items_hash ON items(tenant_id, content_hash);
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, tenant_id TEXT, item_key TEXT, source_id TEXT,
            version INTEGER, chunk_index INTEGER, chunk_hash TEXT,
            ingested_at REAL, published_at REAL, expires_at REAL,
            superseded_at REAL, purged INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_chunks_item ON chunks(tenant_id, item_key);
        CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, source_id TEXT, source_name TEXT,
            trigger TEXT, started_at REAL, finished_at REAL, status TEXT, error TEXT,
            fetched INTEGER, new_items INTEGER, updated_items INTEGER, duplicate_items INTEGER,
            duplicate_content INTEGER, expired_on_arrival INTEGER, chunks_embedded INTEGER,
            chunks_reused INTEGER, chunks_superseded INTEGER);
        CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id, started_at);
        CREATE TABLE IF NOT EXISTS query_log (
            id TEXT PRIMARY KEY, tenant_id TEXT, ts REAL, question TEXT, params TEXT,
            chunk_ids TEXT, scores TEXT, answer TEXT, disclosure TEXT);
        CREATE INDEX IF NOT EXISTS idx_qlog_tenant ON query_log(tenant_id, ts);
        """)


# ------------------------------------------------------------------ tenants ----
def create_tenant(name: str, api_key_hash: str) -> str:
    tid = str(uuid.uuid4())
    with _conn() as c:
        c.execute("INSERT INTO tenants (id, name, api_key_hash, created_at) VALUES (?,?,?,?)",
                  (tid, name, api_key_hash, time.time()))
    return tid


def get_tenant_by_key_hash(key_hash: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM tenants WHERE api_key_hash = ?", (key_hash,)).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------------ sources ----
def _src(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    d["config"] = json.loads(d["config"] or "{}")
    d["state"] = json.loads(d["state"] or "{}")
    d["enabled"] = bool(d["enabled"])
    return d


def create_source(tenant_id: str, name: str, type_: str, config: dict, poll_interval_seconds: int,
                  ttl_seconds: int | None, half_life_seconds: int, webhook_token_hash: str | None) -> dict:
    sid = str(uuid.uuid4())
    with _conn() as c:
        c.execute("""INSERT INTO sources (id, tenant_id, name, type, config, state, poll_interval_seconds,
                     ttl_seconds, half_life_seconds, webhook_token_hash, enabled, created_at)
                     VALUES (?,?,?,?,?,?,?,?,?,?,1,?)""",
                  (sid, tenant_id, name, type_, json.dumps(config), "{}", poll_interval_seconds,
                   ttl_seconds, half_life_seconds, webhook_token_hash, time.time()))
    return get_source_by_id(sid)


def get_source(tenant_id: str, source_id: str) -> dict | None:
    with _conn() as c:
        return _src(c.execute("SELECT * FROM sources WHERE id = ? AND tenant_id = ?",
                              (source_id, tenant_id)).fetchone())


def get_source_by_id(source_id: str) -> dict | None:
    """Tenant-unscoped: only for the scheduler, worker and webhook auth, which already hold a source id."""
    with _conn() as c:
        return _src(c.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone())


def list_sources(tenant_id: str) -> list[dict]:
    with _conn() as c:
        return [_src(r) for r in c.execute(
            "SELECT * FROM sources WHERE tenant_id = ? ORDER BY created_at", (tenant_id,))]


def list_all_sources() -> list[dict]:
    with _conn() as c:
        return [_src(r) for r in c.execute("SELECT * FROM sources")]


def update_source(tenant_id: str, source_id: str, **fields) -> dict | None:
    allowed = {"enabled", "poll_interval_seconds", "ttl_seconds", "half_life_seconds"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if sets:
        with _conn() as c:
            c.execute(f"UPDATE sources SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ? AND tenant_id = ?",
                      [*sets.values(), source_id, tenant_id])
    return get_source(tenant_id, source_id)


def record_poll_result(source_id: str, state: dict | None, error: str | None, now: float) -> None:
    with _conn() as c:
        if error is None:
            c.execute("""UPDATE sources SET state = COALESCE(?, state), last_polled_at = ?, last_error = NULL,
                         consecutive_failures = 0 WHERE id = ?""",
                      (json.dumps(state) if state is not None else None, now, source_id))
        else:
            c.execute("""UPDATE sources SET last_polled_at = ?, last_error = ?,
                         consecutive_failures = consecutive_failures + 1 WHERE id = ?""",
                      (now, error[:500], source_id))


def delete_source(tenant_id: str, source_id: str) -> bool:
    with _conn() as c:
        return c.execute("DELETE FROM sources WHERE id = ? AND tenant_id = ?",
                         (source_id, tenant_id)).rowcount > 0


def source_half_lives(tenant_id: str) -> dict[str, int]:
    with _conn() as c:
        return {r["id"]: r["half_life_seconds"] for r in c.execute(
            "SELECT id, half_life_seconds FROM sources WHERE tenant_id = ?", (tenant_id,))}


# -------------------------------------------------------------------- items ----
def get_item(tenant_id: str, item_key: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM items WHERE tenant_id = ? AND item_key = ?",
                        (tenant_id, item_key)).fetchone()
    return dict(row) if row else None


def touch_item(tenant_id: str, item_key: str, now: float) -> None:
    with _conn() as c:
        c.execute("UPDATE items SET last_seen_at = ? WHERE tenant_id = ? AND item_key = ?",
                  (now, tenant_id, item_key))


def find_live_item_by_hash(tenant_id: str, content_hash: str, exclude_key: str, now: float) -> dict | None:
    with _conn() as c:
        row = c.execute("""SELECT * FROM items WHERE tenant_id = ? AND content_hash = ?
                           AND item_key != ? AND expires_at > ? LIMIT 1""",
                        (tenant_id, content_hash, exclude_key, now)).fetchone()
    return dict(row) if row else None


def upsert_item(tenant_id: str, item_key: str, source_id: str, external_id: str, content_hash: str,
                version: int, published_at: float, expires_at: float, title: str, url: str, now: float) -> None:
    with _conn() as c:
        c.execute("""INSERT INTO items (tenant_id, item_key, source_id, external_id, content_hash, version,
                     published_at, expires_at, title, url, first_seen_at, last_seen_at)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(tenant_id, item_key) DO UPDATE SET content_hash = excluded.content_hash,
                     version = excluded.version, published_at = excluded.published_at,
                     expires_at = excluded.expires_at, title = excluded.title, url = excluded.url,
                     last_seen_at = excluded.last_seen_at""",
                  (tenant_id, item_key, source_id, external_id, content_hash, version,
                   published_at, expires_at, title, url, now, now))


# ------------------------------------------------------------------- chunks ----
def insert_chunks(rows: list[dict]) -> None:
    with _conn() as c:
        c.executemany("""INSERT OR REPLACE INTO chunks (id, tenant_id, item_key, source_id, version, chunk_index,
                         chunk_hash, ingested_at, published_at, expires_at, superseded_at, purged)
                         VALUES (:id,:tenant_id,:item_key,:source_id,:version,:chunk_index,:chunk_hash,
                         :ingested_at,:published_at,:expires_at,:superseded_at,0)""", rows)


def live_chunks_for_item(tenant_id: str, item_key: str, now: float) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            """SELECT * FROM chunks WHERE tenant_id = ? AND item_key = ? AND purged = 0
               AND superseded_at > ? AND expires_at > ?""", (tenant_id, item_key, now, now))]


def live_chunks_for_source(tenant_id: str, source_id: str, now: float) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            """SELECT * FROM chunks WHERE tenant_id = ? AND source_id = ? AND purged = 0
               AND superseded_at > ? AND expires_at > ?""", (tenant_id, source_id, now, now))]


def supersede_chunks(ids: list[str], ts: float) -> None:
    if ids:
        with _conn() as c:
            c.executemany("UPDATE chunks SET superseded_at = ? WHERE id = ?", [(ts, i) for i in ids])


def purge_candidates(cutoff: float, tenant_id: str | None = None) -> list[str]:
    """Chunks that expired / were superseded before `cutoff` (now - retention). NEVER sentinels never match."""
    sql = "SELECT id FROM chunks WHERE purged = 0 AND (expires_at <= ? OR superseded_at <= ?)"
    params: list = [cutoff, cutoff]
    if tenant_id:
        sql += " AND tenant_id = ?"
        params.append(tenant_id)
    with _conn() as c:
        return [r["id"] for r in c.execute(sql, params)]


def mark_purged(ids: list[str]) -> None:
    if ids:
        with _conn() as c:
            c.executemany("UPDATE chunks SET purged = 1 WHERE id = ?", [(i,) for i in ids])


def chunks_by_ids(tenant_id: str, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    with _conn() as c:
        q = ",".join("?" * len(ids))
        return [dict(r) for r in c.execute(
            f"SELECT * FROM chunks WHERE tenant_id = ? AND id IN ({q})", [tenant_id, *ids])]


AGE_BUCKETS = [("< 1h", 3600), ("1-6h", 6 * 3600), ("6-24h", 86400), ("1-7d", 7 * 86400), ("> 7d", float("inf"))]


def chunk_stats(tenant_id: str, now: float) -> dict:
    with _conn() as c:
        row = c.execute("""SELECT
            COALESCE(SUM(CASE WHEN purged=0 AND expires_at>? AND superseded_at>? THEN 1 ELSE 0 END),0) AS live,
            COALESCE(SUM(CASE WHEN purged=0 AND expires_at<=? AND superseded_at>? THEN 1 ELSE 0 END),0) AS expired,
            COALESCE(SUM(CASE WHEN purged=0 AND superseded_at<=? THEN 1 ELSE 0 END),0) AS superseded,
            COALESCE(SUM(CASE WHEN purged=1 THEN 1 ELSE 0 END),0) AS purged,
            COUNT(*) AS total FROM chunks WHERE tenant_id = ?""",
                        (now, now, now, now, now, tenant_id)).fetchone()
        live_rows = c.execute(
            """SELECT COALESCE(s.name, '(deleted source)') AS source, c.published_at FROM chunks c
               LEFT JOIN sources s ON s.id = c.source_id
               WHERE c.tenant_id = ? AND c.purged = 0 AND c.expires_at > ? AND c.superseded_at > ?""",
            (tenant_id, now, now)).fetchall()
    ages: dict[str, dict[str, int]] = {}
    for r in live_rows:
        age = max(0.0, now - r["published_at"])
        label = next(lbl for lbl, limit in AGE_BUCKETS if age < limit)
        ages.setdefault(r["source"], {lbl: 0 for lbl, _ in AGE_BUCKETS})[label] += 1
    return {**dict(row), "age_buckets": ages}


# --------------------------------------------------------------------- runs ----
def add_run(**f) -> int:
    cols = ["tenant_id", "source_id", "source_name", "trigger", "started_at", "finished_at", "status", "error",
            "fetched", "new_items", "updated_items", "duplicate_items", "duplicate_content",
            "expired_on_arrival", "chunks_embedded", "chunks_reused", "chunks_superseded"]
    with _conn() as c:
        cur = c.execute(f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [f.get(k) for k in cols])
        return cur.lastrowid


def recent_runs(tenant_id: str, limit: int = 30) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM runs WHERE tenant_id = ? ORDER BY started_at DESC, id DESC LIMIT ?",
            (tenant_id, limit))]


# ---------------------------------------------------------------- query log ----
def log_query(tenant_id: str, ts: float, question: str, params: dict, chunk_ids: list[str],
              scores: list[float], answer: str, disclosure: str) -> str:
    qid = str(uuid.uuid4())
    with _conn() as c:
        c.execute("INSERT INTO query_log VALUES (?,?,?,?,?,?,?,?,?)",
                  (qid, tenant_id, ts, question, json.dumps(params), json.dumps(chunk_ids),
                   json.dumps(scores), answer, disclosure))
    return qid


def _q(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("params", "chunk_ids", "scores"):
        d[k] = json.loads(d[k])
    return d


def get_query(tenant_id: str, query_id: str) -> dict | None:
    with _conn() as c:
        return _q(c.execute("SELECT * FROM query_log WHERE id = ? AND tenant_id = ?",
                            (query_id, tenant_id)).fetchone())


def recent_queries(tenant_id: str, limit: int = 30) -> list[dict]:
    with _conn() as c:
        return [_q(r) for r in c.execute(
            "SELECT * FROM query_log WHERE tenant_id = ? ORDER BY ts DESC LIMIT ?", (tenant_id, limit))]
