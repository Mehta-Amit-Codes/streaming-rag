"""
Streamlit UI for the Streaming, Real-Time RAG service.

Tabs
  1. Live Sources   connect feeds, poll/push, watch ingestion dedupe and supersede
  2. Ask            freshness-aware answers with a provenance banner; compare ranking modes
  3. Freshness      what is live / expired / superseded, age distribution, decay curve
  4. Audit          replay a past answer and verify it; time-travel to "what did we know at T"

Run:  streamlit run streamlit_app.py      (API must be running; see README)
"""
import os
import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("STREAMING_RAG_URL", "http://localhost:8005")
LABEL_ICON = {"fresh": "🟢 fresh", "aging": "🟡 aging", "stale": "🔴 stale", "evergreen": "⚪ evergreen"}

st.set_page_config(page_title="Streaming RAG", page_icon="📡", layout="wide")
st.title("📡 Streaming, Real-Time RAG")
st.caption(f"Freshness-aware, multi-tenant retrieval over live sources · API: {API_URL}")

for key, default in [("api_key", ""), ("tenant_name", ""), ("webhook_tokens", {})]:
    st.session_state.setdefault(key, default)


def api(method: str, path: str, quiet: bool = False, **kw):
    headers = {"X-API-Key": st.session_state["api_key"]} if st.session_state["api_key"] else {}
    headers.update(kw.pop("headers", {}))
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=90, **kw)
        if r.status_code >= 400:
            if not quiet:
                st.error(f"{path}: {r.status_code} {r.json().get('detail', r.text) if r.headers.get('content-type', '').startswith('application/json') else r.text}")
            return None
        return r.json()
    except requests.RequestException as e:
        if not quiet:
            st.error(f"Cannot reach the API at {API_URL}: {e}")
        return None


def show_df(df, container=st):
    """Streamlit renamed use_container_width -> width='stretch'. Try the new spelling, fall back to the old
    one, so this file runs on both old and new Streamlit without deprecation noise or breakage."""
    try:
        container.dataframe(df, width="stretch")
    except Exception:
        container.dataframe(df, use_container_width=True)


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.altair_chart(chart, use_container_width=True)


def fmt_ts(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "-"


# ------------------------------------------------------------------ sidebar ----
with st.sidebar:
    st.subheader("Tenant")
    if st.session_state["api_key"]:
        st.success(f"Signed in: **{st.session_state['tenant_name'] or 'tenant'}**")
        if st.button("Sign out"):
            st.session_state.update(api_key="", tenant_name="", webhook_tokens={})
            st.rerun()
    else:
        with st.form("onboard"):
            name = st.text_input("New tenant name", placeholder="Acme Corp")
            admin = st.text_input("Admin key (if the server requires one)", type="password")
            if st.form_submit_button("Create tenant") and name:
                out = api("POST", "/tenants", json={"name": name}, headers={"X-Admin-Key": admin} if admin else {})
                if out:
                    st.session_state.update(api_key=out["api_key"], tenant_name=name)
                    st.warning("Save this API key now; it is shown once:")
                    st.code(out["api_key"], language=None)
        existing = st.text_input("…or paste an existing API key", type="password")
        if st.button("Use key") and existing:
            st.session_state["api_key"] = existing
            st.rerun()

if not st.session_state["api_key"]:
    st.info("Create a tenant or paste an API key in the sidebar to begin.")
    st.stop()

tab_src, tab_ask, tab_fresh, tab_audit = st.tabs(["📡 Live Sources", "💬 Ask", "🕒 Freshness", "🧾 Audit"])

# ------------------------------------------------------------- Live Sources ----
with tab_src:
    st.subheader("Connect a live source")
    c1, c2 = st.columns([1, 2])
    with c1:
        src_type = st.selectbox("Source type", ["simulated", "rss", "webhook", "slack_export", "json_api"],
                                help="'simulated' is a deterministic live feed (prices + news) for demos.")
        name = st.text_input("Name", value=f"{src_type}-feed")
        poll = st.number_input("Poll every (seconds, 0 = manual)", 0, 86400, 0 if src_type == "webhook" else 30)
        ttl = st.number_input("Relevance window / TTL (seconds, 0 = never expires)", 0, 10**7, 3600)
        half = st.number_input("Freshness half-life (seconds, 0 = evergreen)", 0, 10**7, 600)
    with c2:
        config: dict = {}
        if src_type == "rss":
            config["url"] = st.text_input("Feed URL", placeholder="https://example.com/feed.xml")
        elif src_type == "simulated":
            config["seed"] = st.text_input("Seed", value="demo")
        elif src_type == "slack_export":
            st.caption("Paste a Slack or Discord export (JSON list of messages, or {\"messages\": [...]}).")
            raw = st.text_area("Messages JSON", height=140, value='[{"user": "ann", "text": "Deploy window moved to 5pm", "ts": "%d.0001"}]' % time.time())
            channel = st.text_input("Channel", value="ops")
            try:
                import json as _json
                data = _json.loads(raw)
                config = {"messages": data["messages"] if isinstance(data, dict) else data, "channel": channel}
            except Exception:
                st.error("Not valid JSON")
        elif src_type == "json_api":
            config["url"] = st.text_input("Endpoint URL", placeholder="https://api.example.com/news")
            config["items_path"] = st.text_input("Items path (dotted, blank = root list)", value="")
            for k in ("id_field", "title_field", "text_field", "url_field", "time_field"):
                config[k] = st.text_input(k, value={"id_field": "id", "title_field": "title", "text_field": "text",
                                                    "url_field": "url", "time_field": "published_at"}[k])
            st.caption("Secrets: set auth_header + auth_env_var (the NAME of a server env var); tokens are never stored.")
        else:
            st.info("Webhook sources are push-only: external systems POST to `/webhooks/{id}` with the source's token.")
        if st.button("Create source", type="primary"):
            out = api("POST", "/sources", json={"name": name, "type": src_type, "config": config,
                                                "poll_interval_seconds": int(poll), "ttl_seconds": int(ttl) or None,
                                                "half_life_seconds": int(half)})
            if out:
                if out.get("webhook_token"):
                    st.session_state["webhook_tokens"][out["id"]] = out["webhook_token"]
                    st.warning(f"Webhook token (shown once): `{out['webhook_token']}`")
                st.success(f"Created '{out['name']}'")

    st.divider()
    sources = api("GET", "/sources", quiet=True) or []
    st.subheader("Sources")
    if not sources:
        st.caption("None yet.")
    for s in sources:
        with st.container(border=True):
            a, b, c, d = st.columns([3, 2, 2, 2])
            a.markdown(f"**{s['name']}** `{s['type']}`  \nTTL {s['ttl_seconds'] or 'never'}s · half-life {s['half_life_seconds'] or 'none'}s")
            nxt = s["next_run_at"]
            b.caption(f"Last poll: {fmt_ts(s['last_polled_at'])}  \nNext: {fmt_ts(nxt) if nxt else 'manual / push'}")
            if s["last_error"]:
                c.error(f"{s['last_error'][:80]} (x{s['consecutive_failures']})")
            else:
                c.caption("✅ healthy" if s["last_polled_at"] else "not polled yet")
            if s["type"] != "webhook" and d.button("Poll now", key=f"poll_{s['id']}"):
                out = api("POST", f"/sources/{s['id']}/poll")
                if out:
                    st.toast(f"+{out.get('new_items', 0)} new · {out.get('updated_items', 0)} updated · "
                             f"{out.get('duplicate_items', 0)} duplicate · {out.get('chunks_embedded', 0)} chunks embedded")
                    st.rerun()
            if d.button("Delete", key=f"del_{s['id']}"):
                if api("DELETE", f"/sources/{s['id']}"):
                    st.rerun()

    hooks = [s for s in sources if s["type"] == "webhook"]
    if hooks:
        with st.expander("Push an item to a webhook source"):
            target = st.selectbox("Source", hooks, format_func=lambda s: s["name"])
            iid = st.text_input("Item id (reuse an id to UPDATE an item)", value="price:widget")
            title = st.text_input("Title", value="Widget price")
            text = st.text_area("Text", value="Widget is currently priced at $10.00 per unit.")
            when = st.text_input("Published at (ISO or epoch; blank = now)", value="")
            if st.button("Push item"):
                item = {"id": iid, "title": title, "text": text}
                if when:
                    item["published_at"] = when
                out = api("POST", f"/sources/{target['id']}/items", json={"items": [item]})
                if out:
                    st.success(f"new {out['new_items']} · updated {out['updated_items']} · duplicate {out['duplicate_items']} "
                               f"· expired on arrival {out['expired_on_arrival']} · chunks embedded {out['chunks_embedded']} "
                               f"· reused {out['chunks_reused']} · superseded {out['chunks_superseded']}")

    st.subheader("Ingestion log")
    runs = api("GET", "/runs", params={"limit": 25}, quiet=True) or []
    if runs:
        df = pd.DataFrame(runs)
        df["started"] = pd.to_datetime(df["started_at"], unit="s")
        show_df(df[["started", "source_name", "trigger", "status", "fetched", "new_items", "updated_items",
                         "duplicate_items", "duplicate_content", "expired_on_arrival", "chunks_embedded",
                         "chunks_reused", "chunks_superseded", "error"]])
        st.caption("Duplicates and reused chunks are work NOT done: nothing re-embedded, nothing re-written.")

# ---------------------------------------------------------------------- Ask ----
with tab_ask:
    st.subheader("Ask a question")
    question = st.text_input("Question", value="What is the price of Widget X?")
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("Ranking", ["freshness", "relevance", "latest"],
                        help="freshness = similarity × time-decay · relevance = similarity only · latest = newest first")
    top_k = c2.slider("Sources", 1, 10, 5)
    override = c3.number_input("Half-life override (s, 0 = per-source)", 0, 10**7, 0)
    compare = c4.checkbox("Compare with pure relevance ranking")

    if st.button("Ask", type="primary") and question:
        body = {"question": question, "top_k": top_k, "mode": mode}
        if override:
            body["half_life_seconds"] = override
        res = api("POST", "/query", json=body)
        if res:
            d = res["disclosure"]
            (st.warning if d["stale"] else st.info)(f"🕒 {d['text']}")
            st.markdown("### Answer")
            st.write(res["answer"])
            st.caption(f"answer mode: {res['answer_mode']} · {res['latency_ms']} ms · query id `{res['query_id']}`")

            def table(sources):
                return pd.DataFrame([{
                    "freshness": LABEL_ICON.get(h["freshness_label"], h["freshness_label"]), "age": h["age_human"],
                    "similarity": round(h["similarity"], 3), "decay": round(h["decay"], 3), "score": round(h["score"], 3),
                    "source": h["source_name"], "text": h["text"][:110]} for h in sources])

            if compare:
                other = api("POST", "/query/retrieve", json={**body, "mode": "relevance"})
                left, right = st.columns(2)
                left.markdown(f"**{mode} ranking** (used for the answer)")
                show_df(table(res["sources"]), left)
                right.markdown("**pure relevance ranking** (what a static RAG would use)")
                if other:
                    show_df(table(other["sources"]), right)
                st.caption("Same index, same question. Differences in order are exactly what freshness weighting changes.")
            else:
                st.markdown("### Sources")
                show_df(table(res["sources"]))

# ---------------------------------------------------------------- Freshness ----
with tab_fresh:
    st.subheader("What the index holds right now")
    stats = api("GET", "/stats")
    if stats:
        ch, ing = stats["chunks"], stats["ingestion"]
        cols = st.columns(5)
        for col, (lbl, k) in zip(cols, [("Live", "live"), ("Expired (hidden)", "expired"),
                                        ("Superseded (hidden)", "superseded"), ("Purged", "purged"), ("Total ever", "total")]):
            col.metric(lbl, ch[k])
        cols2 = st.columns(3)
        cols2[0].metric("Dedup ratio", f"{ing['dedup_ratio']:.0%}", help="Items skipped as duplicates / items seen")
        cols2[1].metric("Chunks embedded", ing["chunks_embedded"])
        cols2[2].metric("Chunks reused (no re-embed)", ing["chunks_reused"])
        st.caption("Expired and superseded chunks are hidden from search but kept for audit until the retention purge.")

        buckets = ["< 1h", "1-6h", "6-24h", "1-7d", "> 7d"]
        rows = [{"source": s, "age": b, "chunks": n} for s, bs in ch["age_buckets"].items() for b, n in bs.items()]
        if rows:
            st.markdown("**Age of live chunks, by source**")
            show_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                x=alt.X("age:N", sort=buckets, title="age"), y="chunks:Q", color="source:N"))

    st.markdown("**Decay curve**")
    hl = st.slider("Half-life (minutes)", 1, 1440, 10)
    floor = 0.1
    ages = [i * hl / 10 for i in range(0, 61)]
    curve = pd.DataFrame({"age (minutes)": ages, "weight": [max(floor, 0.5 ** (a / hl)) for a in ages]})
    show_chart(alt.Chart(curve).mark_line().encode(x="age (minutes)", y=alt.Y("weight:Q", scale=alt.Scale(domain=[0, 1]))))
    st.caption(f"A chunk's similarity is multiplied by this weight: half at {hl} min, never below {floor} "
               "so an old but highly relevant chunk is down-weighted, not erased.")

# -------------------------------------------------------------------- Audit ----
with tab_audit:
    st.subheader("Replay a past answer")
    queries = api("GET", "/audit/queries", params={"limit": 30}, quiet=True) or []
    if not queries:
        st.caption("Ask a question first; every answer is logged with its exact parameters.")
    else:
        pick = st.selectbox("Past query", queries, format_func=lambda q: f"{q['ts_iso']}  ·  {q['question'][:60]}")
        regen = st.checkbox("Also regenerate the answer text (LLM is not guaranteed identical)")
        if st.button("Replay & verify", type="primary"):
            r = api("GET", f"/audit/{pick['query_id']}", params={"regenerate": regen})
            if r:
                (st.success if r["match"] else st.error)(
                    "✅ Reproduced exactly: same chunks, same order, same scores." if r["match"]
                    else "❌ Could not reproduce exactly.")
                for n in r["notes"]:
                    st.warning(n)
                st.caption(f"Evaluated as of {fmt_ts(r['evaluated_at'])}; the index has changed since, but visibility "
                           "is a function of time, so the past is reconstructable.")
                l, rr = st.columns(2)
                l.markdown("**Original answer**")
                l.write(r["original_answer"])
                l.caption(r["original_disclosure"])
                rr.markdown("**Replayed sources**")
                show_df(pd.DataFrame([{"age": h["age_human"], "score": round(h["score"], 3), "text": h["text"][:90]}
                                           for h in r["replay_sources"]]), rr)
                if regen:
                    st.markdown("**Regenerated answer**")
                    st.write(r.get("regenerated_answer", ""))

    st.divider()
    st.subheader("Time travel: what did the system know at time T?")
    runs = api("GET", "/runs", params={"limit": 15}, quiet=True) or []
    options = ["custom"] + [f"{fmt_ts(r['finished_at'])}" for r in runs if r["status"] == "ok"]
    chosen = st.selectbox("Pick an ingestion time (or custom)", options)
    default_ts = datetime.fromtimestamp(time.time() - 600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    custom = st.text_input("As-of time (ISO-8601 or epoch)", value=default_ts) if chosen == "custom" else None
    tq = st.text_input("Question for the past", value="What is the price of Widget X?")
    if st.button("Answer as of that time"):
        as_of = custom or datetime.strptime(chosen, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc).isoformat()
        res = api("POST", "/query", json={"question": tq, "as_of": as_of})
        if res:
            st.info(f"🕒 {res['disclosure']['text']}")
            st.write(res["answer"])
            show_df(pd.DataFrame([{"published": h["published_iso"], "age at T": h["age_human"],
                                   "score": round(h["score"], 3), "text": h["text"][:100]} for h in res["sources"]]))
            st.caption("Chunks ingested after T, superseded before T, or already expired at T are excluded.")
