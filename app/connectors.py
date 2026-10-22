"""
Live-source connectors. Each turns some feed into a list of `Item`s.

    fetch(state, now) -> (items, new_state)

`state` is connector-private memory persisted between polls (ETag, simulator tick...).

SECURITY NOTE. In a multi-tenant service, tenants choose what URL/file the SERVER fetches:
  * URLs  -> SSRF: a tenant could point a feed at http://169.254.169.254/ (cloud metadata) or an
            internal admin service. `validate_fetch_url` blocks non-http(s) schemes and any host
            resolving to a private / loopback / link-local / reserved address, and redirects are
            followed manually so every hop is re-validated. (DNS rebinding between check and
            connect is a residual risk; production deployments should add an egress proxy.)
  * Files -> arbitrary file read: chat-export paths are resolved inside EXPORTS_DIR only.
  * Auth  -> API tokens for JSON sources are read from an ENV VAR NAME in the config, never
            stored in the database.
"""
import calendar
import ipaddress
import json
import os
import random
import socket
import warnings
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests

from app.chunking import strip_html
from app.config import ALLOW_PRIVATE_FETCH, EXPORTS_DIR
from app.timeutil import iso, parse_time

SOURCE_TYPES = ("rss", "slack_export", "json_api", "simulated", "webhook")
POLLABLE_TYPES = ("rss", "slack_export", "json_api", "simulated")


@dataclass
class Item:
    external_id: str
    title: str
    text: str
    url: str
    published_at: float
    published_assumed: bool = False  # True when the source gave no date and we fell back to "now"


class NotPollable(Exception):
    pass


# ---------------------------------------------------------------- SSRF guard ----
def validate_fetch_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("URL must be http(s) with a hostname")
    if ALLOW_PRIVATE_FETCH:
        return
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve host '{parsed.hostname}': {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"host resolves to a non-public address ({ip}); blocked")


def safe_get(url: str, headers: dict | None = None, timeout: int = 20, max_redirects: int = 3) -> requests.Response:
    """GET with per-hop SSRF validation (requests would follow redirects to internal hosts blindly)."""
    for _ in range(max_redirects + 1):
        validate_fetch_url(url)
        resp = requests.get(url, headers=headers or {}, timeout=timeout, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])
            continue
        return resp
    raise ValueError("too many redirects")


# ----------------------------------------------------------------------- RSS ----
def parse_rss(content: bytes | str, now: float) -> list[Item]:
    import feedparser
    feed = feedparser.parse(content)
    items = []
    for e in feed.entries:
        # Prefer "updated" over "published": for an edited story, the edit time is the freshness signal.
        with warnings.catch_warnings():  # feedparser warns when updated_parsed falls back to published_parsed
            warnings.simplefilter("ignore", DeprecationWarning)
            parsed = e.get("updated_parsed") or e.get("published_parsed")
        assumed = parsed is None
        ts = float(calendar.timegm(parsed)) if parsed else now
        body = (e.get("content") or [{}])[0].get("value") or e.get("summary", "")
        ext = e.get("id") or e.get("link") or e.get("title")
        if not ext:
            continue
        items.append(Item(str(ext), strip_html(e.get("title", "")), strip_html(body), e.get("link", ""), ts, assumed))
    return items


class RSSConnector:
    def __init__(self, config: dict):
        self.url = config["url"]

    def fetch(self, state: dict, now: float):
        headers = {"User-Agent": "streaming-rag/0.1"}
        # Conditional GET: an unchanged feed costs one tiny 304 instead of a full download + re-parse.
        if state.get("etag"):
            headers["If-None-Match"] = state["etag"]
        if state.get("last_modified"):
            headers["If-Modified-Since"] = state["last_modified"]
        resp = safe_get(self.url, headers)
        if resp.status_code == 304:
            return [], state
        resp.raise_for_status()
        new_state = {**state, "etag": resp.headers.get("ETag"), "last_modified": resp.headers.get("Last-Modified")}
        return parse_rss(resp.content, now), new_state


# ------------------------------------------------------ Slack / Discord export ----
def parse_chat_export(data, channel: str) -> list[Item]:
    msgs = data.get("messages", []) if isinstance(data, dict) else data
    items = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("subtype") in {"channel_join", "channel_leave", "bot_add"}:
            continue
        text = (m.get("text") or m.get("content") or "").strip()
        if not text:
            continue
        if "ts" in m:  # Slack
            ts, ext = float(m["ts"]), str(m["ts"])
            user = m.get("user_name") or m.get("user") or "unknown"
            if isinstance(m.get("edited"), dict) and m["edited"].get("ts"):
                ts = float(m["edited"]["ts"])  # an edit is a newer version of the same item
        elif "timestamp" in m:  # Discord (DiscordChatExporter JSON)
            ts, ext = parse_time(m.get("timestampEdited") or m["timestamp"]), str(m.get("id", m["timestamp"]))
            user = (m.get("author") or {}).get("name", "unknown")
        else:
            continue
        items.append(Item(ext, f"#{channel} - {user}", text, "", ts))
    return items


def resolve_export_path(path: str):
    root = EXPORTS_DIR.resolve()
    target = (root / path).resolve()
    if root != target and root not in target.parents:
        raise ValueError("export path must stay inside the exports directory")
    return target


class ChatExportConnector:
    def __init__(self, config: dict):
        self.config = config

    def fetch(self, state: dict, now: float):
        channel = self.config.get("channel", "general")
        if "messages" in self.config:
            data = self.config["messages"]
        else:
            data = json.loads(resolve_export_path(self.config["path"]).read_text(encoding="utf-8"))
        return parse_chat_export(data, channel), state


# ------------------------------------------------------------ generic JSON API ----
def _dig(obj, path: str):
    for part in [p for p in (path or "").split(".") if p]:
        obj = obj[part]
    return obj


class JSONAPIConnector:
    """Polls a JSON endpoint. config: url, items_path, id_field, title_field, text_field, url_field,
    time_field, and optionally auth_header + auth_env_var (the NAME of an env var holding the secret)."""

    def __init__(self, config: dict):
        self.c = config

    def fetch(self, state: dict, now: float):
        headers = {"User-Agent": "streaming-rag/0.1"}
        if self.c.get("auth_header") and self.c.get("auth_env_var"):
            secret = os.environ.get(self.c["auth_env_var"], "")
            if not secret:
                raise RuntimeError(f"env var {self.c['auth_env_var']} is not set")
            headers[self.c["auth_header"]] = secret
        resp = safe_get(self.c["url"], headers)
        resp.raise_for_status()
        items = []
        for obj in _dig(resp.json(), self.c.get("items_path", "")):
            ext = obj.get(self.c.get("id_field", "id"))
            if ext is None:
                continue
            ts = parse_time(obj.get(self.c.get("time_field", "")))
            items.append(Item(str(ext), str(obj.get(self.c.get("title_field", "title"), "")),
                              strip_html(str(obj.get(self.c.get("text_field", "text"), ""))),
                              str(obj.get(self.c.get("url_field", "url"), "")),
                              ts if ts is not None else now, ts is None))
        return items, state


# ------------------------------------------------------------------ simulator ----
class SimulatedConnector:
    """
    A deterministic live feed for demos and tests (public RSS feeds rarely change on cue).
    Each poll = one "tick":
      * a price item per product with a STABLE external id: when the price moves, the old chunk is
        superseded; when it doesn't, the item is a duplicate and costs nothing to ingest.
      * 1-2 news headlines with new ids, some of which contradict earlier ones.
    """
    PRODUCTS = ["Widget X", "Gadget Pro", "Gizmo Mini"]
    EVENTS = ["is back in stock after a supply delay", "is temporarily sold out",
              "gets a 10 percent discount this week", "is subject to a voluntary safety recall",
              "receives a major firmware update", "is now available in new colors"]

    def __init__(self, config: dict):
        self.seed = str(config.get("seed", "demo"))

    def fetch(self, state: dict, now: float):
        tick = int(state.get("tick", 0)) + 1
        rng = random.Random(f"{self.seed}:{tick}")
        prices = dict(state.get("prices") or {p: round(rng.uniform(8, 20), 2) for p in self.PRODUCTS})
        items = []
        for product in self.PRODUCTS:
            if tick > 1 and rng.random() < 0.6:
                prices[product] = round(max(1.0, prices[product] + rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)), 2)
            slug = product.lower().replace(" ", "-")
            items.append(Item(f"price:{slug}", f"{product} price",
                              f"{product} is currently priced at ${prices[product]:.2f} per unit.",
                              "", now))
        for i in range(rng.randint(1, 2)):
            product = rng.choice(self.PRODUCTS)
            items.append(Item(f"news:{tick}:{i}", f"{product} news",
                              f"{product} {rng.choice(self.EVENTS)}. Reported at {iso(now)}.",
                              "", now - rng.uniform(0, 120)))
        return items, {"tick": tick, "prices": prices}


# -------------------------------------------------------------------- factory ----
def build_connector(source: dict):
    t, cfg = source["type"], source["config"]
    if t == "rss":
        return RSSConnector(cfg)
    if t == "slack_export":
        return ChatExportConnector(cfg)
    if t == "json_api":
        return JSONAPIConnector(cfg)
    if t == "simulated":
        return SimulatedConnector(cfg)
    raise NotPollable(f"source type '{t}' is push-only")


def validate_source_config(type_: str, config: dict) -> None:
    """Fail fast at creation time, not at 3am on the first poll."""
    if type_ not in SOURCE_TYPES:
        raise ValueError(f"type must be one of {SOURCE_TYPES}")
    if type_ in ("rss", "json_api"):
        if not config.get("url"):
            raise ValueError(f"{type_} sources need config.url")
        validate_fetch_url(config["url"])
    if type_ == "slack_export":
        if "messages" not in config and "path" not in config:
            raise ValueError("slack_export needs config.messages (inline) or config.path (inside exports/)")
        if "path" in config:
            resolve_export_path(config["path"])
