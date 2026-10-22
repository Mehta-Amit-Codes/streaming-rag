import time

import pytest
from fastapi.testclient import TestClient

from app import scheduler, worker
from app.main import app

client = TestClient(app)
NOW_ISO = "2023-11-14T22:13:20Z"


def onboard(name="acme"):
    r = client.post("/tenants", json={"name": name})
    assert r.status_code == 201
    return {"X-API-Key": r.json()["api_key"]}, r.json()["tenant_id"]


def webhook_source(h, ttl=None, half_life=3600):
    r = client.post("/sources", headers=h, json={"name": "hooks", "type": "webhook", "poll_interval_seconds": 0,
                                                 "ttl_seconds": ttl, "half_life_seconds": half_life})
    assert r.status_code == 201, r.text
    return r.json()


def now_iso(offset=0):
    from app.timeutil import iso
    return iso(time.time() + offset)


def test_endpoints_require_an_api_key():
    assert client.post("/query", json={"question": "x"}).status_code == 422          # header missing
    assert client.post("/query", json={"question": "x"}, headers={"X-API-Key": "bogus"}).status_code == 401
    assert client.get("/sources", headers={"X-API-Key": "bogus"}).status_code == 401


def test_admin_key_gates_tenant_creation(monkeypatch):
    from app import auth
    monkeypatch.setattr(auth, "ADMIN_KEY", "root-secret")
    assert client.post("/tenants", json={"name": "x"}).status_code == 401
    assert client.post("/tenants", json={"name": "x"}, headers={"X-Admin-Key": "root-secret"}).status_code == 201


def test_push_then_query_end_to_end_with_freshness_disclosure():
    h, _ = onboard()
    src = webhook_source(h)
    pushed = client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p1", "title": "Widget price", "text": "Widget price is 10 dollars", "published_at": now_iso(-300)}]})
    assert pushed.json()["new_items"] == 1
    again = client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p1", "title": "Widget price", "text": "Widget price is 10 dollars", "published_at": now_iso(-300)}]})
    assert again.json()["duplicate_items"] == 1 and again.json()["chunks_embedded"] == 0
    res = client.post("/query", headers=h, json={"question": "Widget price?"}).json()
    assert res["sources"][0]["freshness_label"] in ("fresh", "aging") and res["disclosure"]["n_sources"] == 1
    assert "newest is" in res["answer_with_disclosure"] and res["query_id"]


def test_webhook_requires_the_per_source_token():
    h, _ = onboard()
    src = webhook_source(h)
    body = {"items": [{"id": "w1", "text": "Webhook delivered item about gadgets", "published_at": now_iso(-10)}]}
    assert client.post(f"/webhooks/{src['id']}", json=body).status_code == 401
    assert client.post(f"/webhooks/{src['id']}", json=body, headers={"X-Webhook-Token": "nope"}).status_code == 401
    assert client.post("/webhooks/does-not-exist", json=body, headers={"X-Webhook-Token": "x"}).status_code == 401
    token = {"X-Webhook-Token": src["webhook_token"]}
    ack = client.post(f"/webhooks/{src['id']}", json=body, headers=token)
    assert ack.status_code == 200 and ack.json()["queued"] is True        # acks immediately: senders time out and retry
    assert client.get("/stats", headers=h).json()["chunks"]["live"] == 1  # ...and the data lands via the queue
    sync = client.post(f"/webhooks/{src['id']}?wait=true", headers=token, json={"items": [
        {"id": "w2", "text": "Second delivery about gizmos", "published_at": now_iso(-5)}]})
    assert sync.json()["new_items"] == 1
    assert "webhook_token" not in client.get("/sources", headers=h).json()[0]  # shown once only


def test_tenants_are_isolated_over_http():
    ha, _ = onboard("a")
    hb, _ = onboard("b")
    sa = webhook_source(ha)
    client.post(f"/sources/{sa['id']}/items", headers=ha, json={"items": [
        {"id": "s", "text": "Project Nightingale launch date is classified", "published_at": now_iso(-5)}]})
    for path in ("/query", "/query/retrieve"):
        r = client.post(path, headers=hb, json={"question": "Project Nightingale launch", "min_similarity": 0})
        assert r.json()["sources"] == []
    assert client.get("/sources", headers=hb).json() == []
    assert client.post(f"/sources/{sa['id']}/poll", headers=hb).status_code == 404
    assert client.post(f"/sources/{sa['id']}/items", headers=hb, json={"items": [{"id": "x", "text": "y"}]}).status_code == 404
    assert client.delete(f"/sources/{sa['id']}", headers=hb).status_code == 404
    assert client.get("/stats", headers=hb).json()["chunks"]["total"] == 0


def test_simulated_source_shows_dedup_and_supersession_across_polls():
    h, _ = onboard()
    r = client.post("/sources", headers=h, json={"name": "sim", "type": "simulated", "config": {"seed": "t"},
                                                 "poll_interval_seconds": 0, "ttl_seconds": 3600})
    sid = r.json()["id"]
    first = client.post(f"/sources/{sid}/poll", headers=h).json()
    second = client.post(f"/sources/{sid}/poll", headers=h).json()
    assert first["status"] == "ok" and first["new_items"] >= 3
    assert second["duplicate_items"] + second["updated_items"] >= 1       # prices re-seen: skipped or superseded
    assert second["chunks_embedded"] < first["chunks_embedded"] + 3
    stats = client.get("/stats", headers=h).json()
    assert stats["chunks"]["live"] >= 3 and stats["ingestion"]["items_seen"] >= 6
    assert len(client.get("/runs", headers=h).json()) == 2


def test_push_only_and_pollable_source_types_are_enforced():
    h, _ = onboard()
    hook = webhook_source(h)
    assert client.post(f"/sources/{hook['id']}/poll", headers=h).status_code == 422
    sim = client.post("/sources", headers=h, json={"name": "s", "type": "simulated", "poll_interval_seconds": 0}).json()
    assert client.post(f"/sources/{sim['id']}/items", headers=h, json={"items": [{"id": "x", "text": "y"}]}).status_code == 422


def test_source_validation_blocks_ssrf_and_traversal():
    h, _ = onboard()
    bad = [{"type": "rss", "config": {"url": "http://169.254.169.254/latest/meta-data"}},
           {"type": "json_api", "config": {"url": "http://127.0.0.1:8000/admin"}},
           {"type": "slack_export", "config": {"path": "../../etc/passwd"}},
           {"type": "rss", "config": {}}, {"type": "carrier-pigeon", "config": {}}]
    for b in bad:
        r = client.post("/sources", headers=h, json={"name": "x", **b})
        assert r.status_code == 422, (b, r.text)


def test_as_of_and_mode_validation_and_time_travel_over_http():
    h, _ = onboard()
    src = webhook_source(h, ttl=3600)
    client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p", "title": "Widget price", "text": "Widget price is 10 dollars", "published_at": now_iso(-100)}]})
    assert client.post("/query", headers=h, json={"question": "Widget", "mode": "bogus"}).status_code == 422
    assert client.post("/query", headers=h, json={"question": "Widget", "as_of": "not a date"}).status_code == 422
    past = client.post("/query", headers=h, json={"question": "Widget price", "as_of": now_iso(-3600)}).json()
    assert past["sources"] == [] and past["disclosure"]["historical"]


def test_audit_endpoints_replay_a_logged_query():
    h, _ = onboard()
    src = webhook_source(h)
    client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p", "title": "Widget price", "text": "Widget price is 10 dollars", "published_at": now_iso(-60)}]})
    qid = client.post("/query", headers=h, json={"question": "Widget price"}).json()["query_id"]
    client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p", "title": "Widget price", "text": "Widget price is 99 dollars", "published_at": now_iso(0)}]})
    assert any(q["query_id"] == qid for q in client.get("/audit/queries", headers=h).json())
    replay = client.get(f"/audit/{qid}", headers=h).json()
    assert replay["match"] is True
    other, _ = onboard("other")
    assert client.get(f"/audit/{qid}", headers=other).status_code == 404


def test_deleting_a_source_hides_its_data_immediately():
    h, _ = onboard()
    src = webhook_source(h)
    client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p", "title": "Widget price", "text": "Widget price is 10 dollars", "published_at": now_iso(-60)}]})
    assert client.delete(f"/sources/{src['id']}", headers=h).json()["chunks_hidden"] >= 1
    assert client.post("/query", headers=h, json={"question": "Widget price"}).json()["sources"] == []


def test_purge_endpoint_and_stats():
    h, _ = onboard()
    src = webhook_source(h, ttl=60)
    client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p", "text": "Short lived flash sale on widgets", "published_at": now_iso(-30)}]})
    time.sleep(0.01)
    out = client.post("/maintenance/purge", headers=h, json={"retention_seconds": 0}).json()
    assert out["purged"] == 0                              # still inside its 60s TTL: nothing to purge
    assert client.get("/stats", headers=h).json()["chunks"]["live"] == 1


def test_scheduler_reconciles_jobs_with_sources():
    h, _ = onboard()
    scheduler._scheduler = None
    scheduler.start()
    try:
        sim = client.post("/sources", headers=h, json={"name": "s", "type": "simulated", "poll_interval_seconds": 60}).json()
        hook = webhook_source(h)
        ids = {j.id for j in scheduler._scheduler.get_jobs()}
        assert f"poll:{sim['id']}" in ids and f"poll:{hook['id']}" not in ids      # push-only sources aren't polled
        client.patch(f"/sources/{sim['id']}", headers=h, json={"enabled": False})
        assert f"poll:{sim['id']}" not in {j.id for j in scheduler._scheduler.get_jobs()}
    finally:
        scheduler.shutdown()


def test_threaded_worker_processes_queued_work_and_survives_a_failing_task():
    q = worker.IngestQueue(threaded=True)
    q.start()
    try:
        def boom():
            raise RuntimeError("bad feed")
        bad, good = q.submit(boom), q.submit(lambda: "ok")
        assert good.wait(5) == "ok" and bad.error and "bad feed" in bad.error
    finally:
        q.stop()


def test_retrieve_endpoint_matches_the_shape_other_projects_expect():
    h, _ = onboard()
    src = webhook_source(h)
    client.post(f"/sources/{src['id']}/items", headers=h, json={"items": [
        {"id": "p", "title": "Widget price", "text": "Widget price is 10 dollars", "published_at": now_iso(-60)}]})
    s = client.post("/query/retrieve", headers=h, json={"question": "Widget price"}).json()["sources"][0]
    assert {"document_id", "chunk_index", "text"} <= set(s)   # what drift detection / eval monitoring read
