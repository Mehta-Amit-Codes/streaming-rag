from app import registry
from app.connectors import Item
from app.ingestion import ingest_items, purge_expired, retire_source

T0 = 1_700_000_000.0


def item(ext="a", text="The quick brown fox jumps over the lazy dog", title="Fox", at=T0, **kw):
    return Item(ext, title, text, "", at, **kw)


def test_new_item_is_embedded_and_registered(tenant, make_source):
    src = make_source()
    s = ingest_items(tenant, src, [item()], now=T0)
    assert (s.new_items, s.duplicate_items) == (1, 0) and s.chunks_embedded >= 1
    assert registry.get_item(tenant, f"{src['id']}:a")["version"] == 1


def test_identical_item_is_a_free_duplicate(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [item()], now=T0)
    s = ingest_items(tenant, src, [item()], now=T0 + 60)
    assert s.duplicate_items == 1 and s.chunks_embedded == 0 and s.new_items == 0


def test_cosmetic_changes_do_not_count_as_changes(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [item(text="Price is  TEN dollars")], now=T0)
    s = ingest_items(tenant, src, [item(text="price is ten   dollars")], now=T0 + 5)
    assert s.duplicate_items == 1


def test_changed_item_supersedes_old_version(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [item(text="Widget costs 10 dollars")], now=T0)
    s = ingest_items(tenant, src, [item(text="Widget costs 12 dollars", at=T0 + 100)], now=T0 + 100)
    assert s.updated_items == 1 and s.chunks_superseded >= 1
    assert registry.get_item(tenant, f"{src['id']}:a")["version"] == 2
    stats = registry.chunk_stats(tenant, T0 + 200)
    assert stats["live"] == 1 and stats["superseded"] == 1


def test_partial_update_reembeds_only_changed_chunks(tenant, make_source):
    src = make_source()
    words = [f"w{i}" for i in range(70)]
    original = " ".join(words)
    first = ingest_items(tenant, src, [item(text=original, title="")], now=T0)
    edited = " ".join(words[:-1] + ["CHANGED"])  # only the tail differs
    second = ingest_items(tenant, src, [item(text=edited, title="", at=T0 + 10)], now=T0 + 10)
    assert first.chunks_embedded >= 4
    assert second.chunks_reused >= 3 and 1 <= second.chunks_embedded <= 2
    assert second.chunks_embedded < first.chunks_embedded


def test_same_story_from_two_sources_is_deduplicated(tenant, make_source):
    a, b = make_source(name="feed-a"), make_source(name="feed-b")
    ingest_items(tenant, a, [item(ext="1")], now=T0)
    s = ingest_items(tenant, b, [item(ext="9")], now=T0 + 30)
    assert s.duplicate_content == 1 and s.new_items == 0


def test_item_already_past_its_ttl_is_not_ingested(tenant, make_source):
    src = make_source(ttl=3600)
    s = ingest_items(tenant, src, [item(at=T0 - 7200)], now=T0)
    assert s.expired_on_arrival == 1 and s.new_items == 0
    assert registry.chunk_stats(tenant, T0)["total"] == 0


def test_future_dated_item_is_clamped_to_now(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [item(at=T0 + 10_000)], now=T0)
    row = registry.get_item(tenant, f"{src['id']}:a")
    assert row["published_at"] == T0


def test_retired_source_is_hidden_but_kept_until_purge(tenant, make_source):
    src = make_source()
    ingest_items(tenant, src, [item()], now=T0)
    assert retire_source(tenant, src["id"], now=T0 + 10) >= 1
    s = registry.chunk_stats(tenant, T0 + 20)
    assert s["live"] == 0 and s["superseded"] >= 1 and s["purged"] == 0
    assert purge_expired(now=T0 + 20 + 100, retention_s=50, tenant_id=tenant)["purged"] >= 1
    assert registry.chunk_stats(tenant, T0 + 200)["purged"] >= 1
