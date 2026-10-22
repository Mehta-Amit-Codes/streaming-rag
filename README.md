# Streaming, Real-Time RAG over Live Data

Freshness-aware, **multi-tenant** RAG over continuously updating sources. Most RAG
demos index a static folder of PDFs. This one ingests from live feeds, embeds only what
is new, lets stale facts fade or expire, tells the user *how old* its sources are, and can
reconstruct exactly what it would have answered at any past moment.

It keeps the multi-tenant design of the RAG-as-a-Service project (per-tenant API keys,
`tenant_id` on every chunk) and uses the same `tenant_id` as the `namespace` in the cost-control
and eval-monitoring projects, so all of them line up per tenant.

## Pipeline

```
 RSS ─┐                         ┌─ exact duplicate? ─────────► skip
 Slack/Discord export ─┤  poll  │  same story, other feed? ──► skip
 JSON API ─┼─(APScheduler)─►  ingest queue ─► already expired? ────► skip
 simulator ─┤  push           │  changed item? ──► re-embed ONLY changed chunks,
 webhook ───┘ (token auth)    │                    supersede the rest (kept for audit)
                              └─ new ──► embed ──► Qdrant (vectors + 4 timestamps) + SQLite registry

 question ─► embed ─► Qdrant search (tenant + visibility-at-T filters) ─► freshness re-rank
          ─► answer (Grok) + provenance disclosure built by code ─► logged for audit
```

## The time model (everything hinges on this)

Each chunk carries four epoch timestamps:

| field | meaning |
|---|---|
| `published_at` | **event time**: when the source says it happened |
| `ingested_at` | **system time**: when this service learned of it |
| `expires_at` | end of the relevance window (`published_at` + source TTL) |
| `superseded_at` | when a newer version of the same item replaced it |

A chunk is visible at time **T** iff `published_at <= T < expires_at`, `ingested_at <= T` and
`superseded_at > T`. "Never" is stored as a year-2100 sentinel rather than NULL, so every
condition is one uniform range filter. Since visibility is a pure function of T:

- **Expiry** is just a filter; nothing is deleted when a chunk expires.
- **Supersession** (a price changes) hides the old chunk without destroying it.
- **Audit** is "run the same search at the old T". Events published long ago but only
  *ingested* later are correctly invisible to a replay of the earlier moment.

## How it maps to the blueprint

| Blueprint step | Where |
|---|---|
| Live source (RSS / Slack-Discord / polling API) | `connectors.py` (+ a deterministic simulator for demos) |
| Streaming ingestion, hash dedup, embed only what's new | `ingestion.py`, `worker.py`, `scheduler.py` |
| Freshness metadata, as-of retrieval | `vector_store.py`, `query.py` |
| Expiry (time-decay and hard expiry) | `freshness.py` (half-life decay) and `expires_at` filter |
| Answer with freshness disclosure | `freshness.build_disclosure`, `llm.py` |
| Stretch: compliance / audit mode | `query.replay`, `/audit/*`, Streamlit *Audit* tab |

## Design decisions worth knowing

**Dedup at two levels.** Item level (stable external id + content hash; cosmetic edits like
case/whitespace don't count) and chunk level (an edited article re-embeds only the chunks whose
hash changed). Syndicated copies of one story across two feeds are skipped by content hash.

**Freshness ranking is multiplicative:** `score = similarity × max(floor, 0.5^(age/half_life))`.
A fresh but irrelevant chunk stays low; an additive blend would let new noise outrank good
answers. Each source has its own half-life (minutes for prices, hours for news, none for evergreen
docs). Three modes: `relevance` (static-RAG baseline), `freshness` (default), `latest`.

**Disclosure is built by code, not by the LLM.** The model is asked to say "As of …" and to prefer
the newest source on conflict, but the provenance line (source dates, newest age, staleness
warning, "date assumed" flags) is computed from chunk timestamps and always returned.

**Crash-safe write order:** embed → upsert vectors (deterministic ids, idempotent) → registry.
The reverse could register an item whose vectors never landed, and dedup would then block it from
ever being repaired.

**Audit replay is deterministic by construction.** Live queries, replays and what-if queries all
go through one retrieval function. Each query logs its time, half-lives, mode, `top_k`, embedder
and result chunk ids, so a replay uses the *logged* parameters (editing a source's half-life later
doesn't change history). Purged chunks are flagged ("existed, then purged") instead of causing a
silent mismatch.

## Security

| Risk | Mitigation |
|---|---|
| **SSRF**: tenants choose URLs the server fetches | scheme allow-list, block private/loopback/link-local/reserved addresses, redirects followed manually with per-hop re-validation |
| **Arbitrary file read** via chat-export paths | paths resolved inside `EXPORTS_DIR` only; traversal rejected |
| API tokens for JSON sources | read from a named env var at fetch time, never stored in the DB |
| Webhook abuse | per-source token, stored hashed; same 401 for "no such source" and "bad token" |
| Tenant creation | optional `ADMIN_KEY` |
| **Prompt injection via feed content** | sources are delimited and the model is told they are data. **Not a full defence**; wire in the injection scanner from the agentic-guardrails project for stronger protection |

**Isolation, stated honestly:** the Postgres project enforces tenant isolation in the database
(Row-Level Security). Qdrant has no equivalent, so isolation here depends on `search()` being the
only read path and taking a *required* `tenant_id`. Tests assert no cross-tenant leakage through
search, retrieve, sources, polling, deletion, stats and audit. If you need isolation that survives
an application bug, use collection-per-tenant or the Postgres design.

## Setup

```bash
cp .env.example .env            # set API_KEY_PEPPER, optionally GROK_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --port 8005 --reload
streamlit run streamlit_app.py  # in another terminal
```

No Grok key? Answers fall back to an extractive summary of the top source plus the provenance
line, and everything else works. No internet or model download? `EMBEDDER=hash` runs fully offline
(lexical similarity only; fine for demos, not for real retrieval quality).

Real Qdrant server and API together: `docker compose up -d` (set `API_KEY_PEPPER` in `.env`).

```bash
pip install -r requirements-dev.txt && pytest      # 61 tests, no network, no API key
```

## Demo script (≈5 minutes in the UI)

1. **Live Sources** → create a `simulated` source (poll every 30 s, TTL 3600, half-life 600). Click *Poll now* a few times.
   Watch the ingestion log: prices that didn't move are *duplicates* (no work), prices that moved are
   *updated* with the old chunk *superseded*, news headlines are *new*.
2. **Ask** → "What is the price of Widget X?". Note the provenance banner and the age/freshness of each source.
3. Poll again until the price changes, ask again: the answer follows the new price, and the old chunk is hidden.
4. Tick **Compare with pure relevance ranking** to see where freshness weighting changes the order.
5. **Freshness** → live / expired / superseded counts, age distribution per source, the decay curve.
6. **Audit** → *Replay & verify* the first answer: it reproduces exactly even though the index has moved on.
   Then use *Time travel* to ask what the system knew at an earlier ingestion time.

## What was verified, and what wasn't

| Part | Status |
|---|---|
| Dedup (item, chunk-level, cross-source), supersession, expiry, clamping, retirement | 61 automated tests |
| Time-travel correctness, bitemporal late arrivals, ranking modes, disclosure | automated tests |
| Tenant isolation (search, retrieve, API, polling, deletion, audit) | automated tests |
| Connectors (RSS incl. double-escaped HTML, Slack/Discord, JSON API), SSRF and path-traversal guards | automated tests |
| Real server process: threaded worker, APScheduler firing unattended, on-disk Qdrant, restart persistence, audit replay still matching after restart + further background ingestion | run manually end to end |
| Streamlit UI against a live server (sign-in gate, all tabs, Ask, Replay, time travel) | driven headlessly |
| **Qdrant server mode** (`QDRANT_URL`, payload indexes incl. `is_tenant`) | **not tested** (no Docker here); only embedded local mode was exercised |
| **sentence-transformers embedder** | **not exercised** (no model download here); tests use the hash embedder |
| **Live Grok answers** | **not exercised**; the offline fallback is tested. Same OpenAI-compatible call pattern as the other projects |

## Known limitations

- **Age = time since published/changed, not since last confirmed.** A price that is re-polled and
  unchanged is skipped as a duplicate, so its chunk keeps ageing even though it is still true. Using
  `last_seen_at` as a "confirmed at" signal would fix this; the column is already tracked.
- **Syndicated duplicates share fate with the copy that was kept.** If the original expires first (different TTLs), the other feed's copy is gone too.
- **Single node.** The ingest worker, APScheduler and SQLite registry are single-process. Several API replicas would each run the scheduler (dedup keeps that *correct* but wasteful); use one scheduler and a Postgres registry for HA.
- **No rate limiting** here (the RAG-as-a-Service project has a Redis token bucket you can reuse).
- Audit can only reconstruct answers **inside the retention window**; after the purge, replays report the purged chunks instead of matching.
- The decay candidate pool is `top_k × OVERSAMPLE` re-ranked in Python (Qdrant's server-side decay formulas aren't available in local mode); a very old but highly relevant chunk outside the pool can be missed on large indexes.

## Integration with the other projects

- `POST /query/retrieve` returns `sources[{document_id, chunk_index, text, …}]`, the same shape as the
  RAG-as-a-Service retrieval endpoint, so the **eval-monitoring** drift check and the **cost-control**
  layer can point `RAG_API_URL` here unchanged.
- Use the tenant id as the `namespace` in both to correlate cost, quality and freshness per tenant.
- Drift checks are especially meaningful on a live index: the corpus *is* changing, so overlap between
  snapshots measures how fast answers move.

## Layout

```
app/
  config.py        env config + the time model
  timeutil.py      parse/format/humanise timestamps
  chunking.py      HTML stripping, normalisation, hashing, chunking
  embedding.py     sentence-transformers or offline hash embedder
  registry.py      SQLite: tenants, sources, items, chunk lifecycle, runs, query log
  vector_store.py  Qdrant wrapper; mandatory tenant filter + visibility-at-T filter
  freshness.py     decay, ranking modes, provenance disclosure
  connectors.py    RSS, chat export, JSON API, simulator; SSRF / path guards
  ingestion.py     dedup, supersession, expiry, poll/push, retire, purge
  worker.py        ingest queue (the "cron + queue")
  scheduler.py     APScheduler jobs reconciled with the sources table
  llm.py           Grok answer generation + extractive fallback
  query.py         live query, retrieval-only, audit replay
  auth.py, main.py FastAPI app
streamlit_app.py   UI: sources, ask, freshness, audit
tests/             61 tests
```
