import pytest

from app import connectors as cx

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Rates held</title><guid>g-1</guid><link>http://x/1</link>
<description>&lt;p&gt;The bank &lt;b&gt;held&lt;/b&gt; rates.&lt;/p&gt;</description><pubDate>Tue, 14 Nov 2023 22:13:20 GMT</pubDate></item>
<item><title>No date here</title><guid>g-2</guid><description>Undated story</description></item>
</channel></rss>"""


def test_rss_parsing_strips_html_and_flags_missing_dates():
    items = cx.parse_rss(RSS, now=2_000_000_000.0)
    assert items[0].external_id == "g-1" and items[0].text == "The bank held rates."
    assert items[0].published_at == 1_700_000_000.0 and not items[0].published_assumed
    assert items[1].published_assumed and items[1].published_at == 2_000_000_000.0


def test_slack_export_uses_edit_time_and_skips_noise():
    data = [{"type": "message", "user": "ann", "text": "deploy at 5", "ts": "1700000000.0001",
             "edited": {"ts": "1700000500.0"}},
            {"type": "message", "subtype": "channel_join", "user": "bob", "text": "joined", "ts": "1700000001.0"},
            {"type": "message", "user": "cy", "text": "  ", "ts": "1700000002.0"}]
    items = cx.parse_chat_export(data, "ops")
    assert len(items) == 1 and items[0].published_at == 1_700_000_500.0 and items[0].title == "#ops - ann"


def test_discord_export_shape():
    data = {"messages": [{"id": "9", "timestamp": "2023-11-14T22:13:20+00:00", "content": "hello",
                          "author": {"name": "dee"}}]}
    items = cx.parse_chat_export(data, "chat")
    assert items[0].external_id == "9" and items[0].published_at == 1_700_000_000.0


def test_json_api_connector_maps_fields_and_reads_secret_from_env(monkeypatch):
    seen = {}

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"data": {"items": [{"id": 7, "headline": "Hi", "body": "<b>x</b>", "t": 1700000000}]}}

    def fake_get(url, headers=None, **kw):
        seen["headers"] = headers
        return Resp()

    monkeypatch.setattr(cx, "safe_get", fake_get)
    monkeypatch.setenv("MY_TOKEN", "s3cret")
    conn = cx.JSONAPIConnector({"url": "https://api.example.com/x", "items_path": "data.items", "title_field": "headline",
                                "text_field": "body", "time_field": "t", "auth_header": "Authorization",
                                "auth_env_var": "MY_TOKEN"})
    items, _ = conn.fetch({}, 1.0)
    assert items[0].title == "Hi" and items[0].text == "x" and items[0].published_at == 1_700_000_000.0
    assert seen["headers"]["Authorization"] == "s3cret"


@pytest.mark.parametrize("url", ["http://127.0.0.1/feed", "http://169.254.169.254/latest/meta-data",
                                 "http://10.0.0.5/x", "http://192.168.1.1/x", "ftp://example.com/x", "file:///etc/passwd"])
def test_ssrf_guard_blocks_internal_and_non_http_urls(url):
    with pytest.raises(ValueError):
        cx.validate_fetch_url(url)


def test_ssrf_guard_allows_public_ip():
    cx.validate_fetch_url("http://93.184.216.34/feed")


def test_export_path_cannot_escape_exports_dir():
    for bad in ("../../etc/passwd", "/etc/passwd", "a/../../x"):
        with pytest.raises(ValueError):
            cx.resolve_export_path(bad)
    with pytest.raises(ValueError):
        cx.validate_source_config("slack_export", {"path": "../secrets.json"})


def test_source_config_validation():
    with pytest.raises(ValueError):
        cx.validate_source_config("rss", {})
    with pytest.raises(ValueError):
        cx.validate_source_config("nonsense", {})
    cx.validate_source_config("webhook", {})
    cx.validate_source_config("simulated", {})


def test_simulator_is_deterministic_and_prices_have_stable_ids():
    c = cx.SimulatedConnector({"seed": "s"})
    a, st1 = c.fetch({}, 1000.0)
    b, st2 = cx.SimulatedConnector({"seed": "s"}).fetch({}, 1000.0)
    assert [i.text for i in a] == [i.text for i in b]
    again, _ = c.fetch(st1, 2000.0)
    assert {i.external_id for i in a if i.external_id.startswith("price:")} == \
           {i.external_id for i in again if i.external_id.startswith("price:")}


def test_strip_html_handles_single_and_double_escaping_without_eating_plain_text():
    from app.chunking import strip_html
    assert strip_html("&lt;p&gt;The bank &lt;b&gt;held&lt;/b&gt; rates &amp;amp; said more.&lt;/p&gt;") == "The bank held rates & said more."
    assert strip_html("&lt;p&gt;Escaped once by feedparser.&lt;/p&gt;") == "Escaped once by feedparser."
    assert strip_html("<p>Raw <i>tags</i> too</p>") == "Raw tags too"
    assert strip_html("Revenue was 5 &lt; 6 and 7 &gt; 3 last year") == "Revenue was 5 < 6 and 7 > 3 last year"


def test_rss_with_double_escaped_html_yields_clean_text():
    """Regression: many real feeds double-escape their HTML, which used to leave literal <p> tags in chunks."""
    feed = ("<?xml version=\"1.0\"?><rss version=\"2.0\"><channel><title>t</title><item><title>x</title>"
            "<guid>g</guid><description>&amp;lt;p&amp;gt;The bank held rates.&amp;lt;/p&amp;gt;</description>"
            "</item></channel></rss>")
    assert cx.parse_rss(feed, now=1.0)[0].text == "The bank held rates."
