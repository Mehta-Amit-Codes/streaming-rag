from app import freshness as fr


def cand(id_, score, published, source="s1"):
    return {"id": id_, "score": score, "payload": {
        "item_key": f"k{id_}", "chunk_index": 0, "text": "t", "source_id": source, "source_name": "S",
        "published_at": published}}


def test_decay_halves_at_half_life_and_respects_floor():
    assert fr.decay(0, 3600) == 1.0
    assert abs(fr.decay(3600, 3600) - 0.5) < 1e-9
    assert fr.decay(36000, 3600) == 0.1           # floor
    assert fr.decay(10 ** 9, 0) == 1.0            # evergreen


def test_labels():
    assert fr.label(100, 3600) == "fresh"
    assert fr.label(3600, 3600) == "aging"
    assert fr.label(10 * 3600, 3600) == "stale"
    assert fr.label(5, 0) == "evergreen"


def test_freshness_mode_prefers_newer_when_relevance_is_equal():
    now = 100_000
    hits = fr.rank([cand("old", 0.8, now - 7200), cand("new", 0.8, now - 60)], at=now, mode="freshness",
                   top_k=2, half_lives={"s1": 3600})
    assert [h["id"] for h in hits] == ["new", "old"]


def test_relevance_mode_ignores_age_and_freshness_is_multiplicative():
    now = 100_000
    cands = [cand("old", 0.9, now - 7200), cand("new", 0.7, now - 60)]
    rel = fr.rank(cands, at=now, mode="relevance", top_k=2, half_lives={"s1": 3600})
    assert rel[0]["id"] == "old"
    fresh = fr.rank(cands, at=now, mode="freshness", top_k=2, half_lives={"s1": 3600})
    assert fresh[0]["id"] == "new"
    junk = fr.rank([cand("fresh-junk", 0.05, now), cand("old-good", 0.9, now - 7200)], at=now,
                   mode="freshness", top_k=2, half_lives={"s1": 3600})
    assert junk[0]["id"] == "old-good"            # fresh irrelevant noise must not win


def test_latest_mode_sorts_by_time_after_similarity_floor():
    now = 100_000
    hits = fr.rank([cand("a", 0.9, now - 500), cand("b", 0.6, now - 10), cand("c", 0.1, now - 1)], at=now,
                   mode="latest", top_k=5, half_lives={"s1": 3600}, min_similarity=0.3)
    assert [h["id"] for h in hits] == ["b", "a"]


def test_half_life_override_and_per_source_half_lives():
    now = 100_000
    c = [cand("x", 0.8, now - 3600, source="fast"), cand("y", 0.8, now - 3600, source="slow")]
    hits = fr.rank(c, at=now, mode="freshness", top_k=2, half_lives={"fast": 600, "slow": 86400})
    assert hits[0]["id"] == "y"
    over = fr.rank(c, at=now, mode="freshness", top_k=2, half_lives={"fast": 600, "slow": 86400},
                   half_life_override=1800)
    assert over[0]["decay"] == over[1]["decay"]


def test_disclosure_reports_ages_staleness_and_empty_results():
    now = 1_700_000_000.0
    hit = fr.rank([cand("a", 0.8, now - 300)], at=now, mode="freshness", top_k=1, half_lives={"s1": 3600})
    d = fr.build_disclosure(hit, now)
    assert "newest is 5m old" in d["text"] and not d["stale"] and d["n_sources"] == 1
    old = fr.rank([cand("a", 0.8, now - 3 * 86400)], at=now, mode="freshness", top_k=1, half_lives={"s1": 3600})
    stale = fr.build_disclosure(old, now)
    assert stale["stale"] and "may be outdated" in stale["text"]
    assert "no sources" in fr.build_disclosure([], now)["text"]
    assert fr.build_disclosure(hit, now, historical=True)["text"].startswith("Historical view")


def test_disclosure_wording_for_brand_new_sources():
    now = 1_700_000_000.0
    hit = fr.rank([cand("a", 0.8, now - 5)], at=now, mode="freshness", top_k=1, half_lives={"s1": 3600})
    text = fr.build_disclosure(hit, now)["text"]
    assert "newest is just now)" in text and "just now old" not in text
