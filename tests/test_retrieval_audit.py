"""Time-travel, expiry, isolation and audit replay: the behaviours that make this more than a static RAG."""
from app import query, registry
from app.connectors import Item
from app.ingestion import ingest_items, purge_expired

T0 = 1_700_000_000.0
PRICE_Q = "What is the Widget price?"


def price(ext, text, at):
    return Item(ext, "Widget price", text, "", at)


def ids(res):
    return [h["id"] for h in res["sources"]]


def texts(res):
    return " | ".join(h["text"] for h in res["sources"])


def test_expired_chunks_are_hidden_now_but_visible_in_the_past(tenant, make_source):
    src = make_source(ttl=3600)
    ingest_items(tenant, src, [price("p", "Widget price is 10 dollars", T0)], now=T0)
    inside = query.run_query(tenant, PRICE_Q, now=T0 + 600, generate=False)
    after = query.run_query(tenant, PRICE_Q, now=T0 + 7200, generate=False)
    back_then = query.run_query(tenant, PRICE_Q, as_of=T0 + 600, now=T0 + 7200, generate=False)
    assert len(inside["sources"]) == 1
    assert after["sources"] == [] and "no sources" in after["disclosure"]["text"]
    assert len(back_then["sources"]) == 1 and back_then["disclosure"]["historical"]


def test_supersession_is_time_travel_correct(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [price("p", "Widget price is 10 dollars", T0)], now=T0)
    ingest_items(tenant, src, [price("p", "Widget price is 12 dollars", T0 + 1000)], now=T0 + 1000)
    now_res = query.run_query(tenant, PRICE_Q, now=T0 + 2000, generate=False)
    between = query.run_query(tenant, PRICE_Q, as_of=T0 + 500, now=T0 + 2000, generate=False)
    before_any = query.run_query(tenant, PRICE_Q, as_of=T0 - 500, now=T0 + 2000, generate=False)
    assert "12" in texts(now_res) and "10" not in texts(now_res)
    assert "10" in texts(between) and "12" not in texts(between)
    assert before_any["sources"] == []


def test_knowledge_time_is_respected_for_late_arriving_data(tenant, make_source):
    """Published at T0 but only INGESTED at T0+1000: the system did not know it at T0+500."""
    src = make_source()
    ingest_items(tenant, src, [price("late", "Widget price is 99 dollars", T0)], now=T0 + 1000)
    assert query.run_query(tenant, PRICE_Q, as_of=T0 + 500, now=T0 + 2000, generate=False)["sources"] == []
    assert len(query.run_query(tenant, PRICE_Q, as_of=T0 + 1500, now=T0 + 2000, generate=False)["sources"]) == 1


def test_freshness_mode_ranks_newer_fact_first_and_relevance_mode_does_not_care(tenant, make_source):
    src = make_source(half_life=600)
    ingest_items(tenant, src, [price("a", "Widget price is 10 dollars today", T0 - 7200),
                               price("b", "Widget price is 14 dollars today", T0 - 30)], now=T0)
    fresh = query.run_query(tenant, PRICE_Q, mode="freshness", now=T0, generate=False)
    assert "14" in fresh["sources"][0]["text"]
    assert fresh["sources"][0]["decay"] > fresh["sources"][1]["decay"]
    rel = query.run_query(tenant, PRICE_Q, mode="relevance", now=T0, generate=False)
    assert all(h["score"] == h["similarity"] for h in rel["sources"])


def test_latest_mode_orders_by_publish_time(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [Item("1", "Widget news", "Widget recall announced today", "", T0 - 5000),
                               Item("2", "Widget news", "Widget restock announced today", "", T0 - 50)], now=T0)
    res = query.run_query(tenant, "Widget announced", mode="latest", min_similarity=0.0, now=T0, generate=False)
    published = [h["published_at"] for h in res["sources"]]
    assert published == sorted(published, reverse=True)


def test_source_filter(tenant, make_source):
    a, b = make_source(name="a"), make_source(name="b")
    ingest_items(tenant, a, [Item("1", "T", "alpha shipping delay news", "", T0)], now=T0)
    ingest_items(tenant, b, [Item("2", "T", "beta shipping delay report", "", T0)], now=T0)
    res = query.run_query(tenant, "shipping delay", source_ids=[b["id"]], now=T0, generate=False)
    assert {h["source_id"] for h in res["sources"]} == {b["id"]}


def test_tenants_cannot_see_each_others_data(make_source):
    a = registry.create_tenant("a", "hash-a-1")
    b = registry.create_tenant("b", "hash-b-1")
    ingest_items(a, make_source(tenant_id=a), [price("p", "Widget price is 10 dollars", T0)], now=T0)
    ingest_items(b, make_source(tenant_id=b), [price("q", "Gadget secret roadmap is confidential", T0)], now=T0)
    assert all(h["document_id"].split(":")[0] and "Gadget" not in h["text"]
               for h in query.run_query(a, "Gadget secret roadmap", now=T0 + 1, generate=False)["sources"])
    assert query.retrieve_only(a, "confidential roadmap", now=T0 + 1)["sources"] == [] or all(
        "confidential" not in h["text"] for h in query.retrieve_only(a, "confidential roadmap", now=T0 + 1)["sources"])
    own = query.run_query(b, "Gadget secret roadmap", now=T0 + 1, generate=False)
    assert any("confidential" in h["text"] for h in own["sources"])


def test_vector_search_requires_a_tenant():
    import pytest
    from app import vector_store
    with pytest.raises(ValueError):
        vector_store.search(384, "", [0.0] * 384, 5, T0)


def test_answer_is_extractive_offline_and_carries_a_disclosure(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [price("p", "Widget price is 10 dollars", T0)], now=T0)
    res = query.run_query(tenant, PRICE_Q, now=T0 + 300)
    assert res["answer_mode"] == "extractive"
    assert "newest is 5m old" in res["answer_with_disclosure"]
    assert res["disclosure"]["n_sources"] == 1


def test_audit_replay_reproduces_the_original_answer_despite_later_changes(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [price("p", "Widget price is 10 dollars", T0)], now=T0)
    original = query.run_query(tenant, PRICE_Q, now=T0 + 100, generate=False)
    # The world moves on: the price changes and unrelated data arrives.
    ingest_items(tenant, src, [price("p", "Widget price is 15 dollars", T0 + 5000)], now=T0 + 5000)
    ingest_items(tenant, src, [price("z", "Widget price discount announced today", T0 + 6000)], now=T0 + 6000)
    assert "15" in texts(query.run_query(tenant, PRICE_Q, now=T0 + 7000, generate=False))
    r = query.replay(tenant, original["query_id"])
    assert r["match"] and r["replayed_chunk_ids"] == ids(original) and not r["notes"]


def test_audit_replay_ignores_later_edits_to_source_half_life(tenant, make_source):
    src = make_source(half_life=600)
    ingest_items(tenant, src, [price("a", "Widget price is 10 dollars today", T0 - 3000),
                               price("b", "Widget price is 11 dollars today", T0 - 10)], now=T0)
    original = query.run_query(tenant, PRICE_Q, now=T0 + 1, generate=False)
    registry.update_source(tenant, src["id"], half_life_seconds=999_999)
    assert query.replay(tenant, original["query_id"])["match"]  # uses the LOGGED half-lives


def test_audit_replay_flags_purged_chunks_instead_of_silently_differing(tenant, make_source):
    src = make_source(ttl=1000)
    ingest_items(tenant, src, [price("p", "Widget price is 10 dollars", T0)], now=T0)
    original = query.run_query(tenant, PRICE_Q, now=T0 + 100, generate=False)
    purge_expired(now=T0 + 5000, retention_s=100, tenant_id=tenant)
    r = query.replay(tenant, original["query_id"])
    assert not r["match"] and r["purged_chunk_ids"] and any("purged" in n for n in r["notes"])


def test_replay_is_tenant_scoped(tenant, make_source):
    other = registry.create_tenant("other", "hash-other-1")
    src = make_source()
    ingest_items(tenant, src, [price("p", "Widget price is 10 dollars", T0)], now=T0)
    q = query.run_query(tenant, PRICE_Q, now=T0 + 1, generate=False)
    assert query.replay(other, q["query_id"]) is None
