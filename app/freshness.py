"""
Freshness as a first-class retrieval signal.

Ranking modes
  relevance  pure similarity (what a static RAG does). Useful as a baseline to SEE the difference.
  freshness  similarity x decay(age). Default. Multiplicative on purpose: an irrelevant-but-new
             chunk stays low, whereas an additive blend would let fresh noise outrank good answers.
  latest     keep chunks above a similarity floor, newest first ("what's the latest on X?").

decay(age) = max(floor, 0.5 ** (age / half_life)). Each source has its own half-life: minutes for
prices, hours for news, effectively none for evergreen docs. The floor stops an old but highly
relevant chunk from being zeroed out entirely.

The disclosure is built by CODE from the retrieved chunks' timestamps, not left to the LLM. The model
is asked to mention dates too, but the guarantee that every answer carries its provenance doesn't
depend on the model remembering to.
"""
from app.config import DECAY_FLOOR, DEFAULT_HALF_LIFE_S, DEFAULT_STALE_AFTER_S
from app.timeutil import humanize, iso

MODES = ("relevance", "freshness", "latest")


def decay(age_s: float, half_life_s: float, floor: float = DECAY_FLOOR) -> float:
    if not half_life_s or half_life_s <= 0:
        return 1.0  # evergreen: no decay
    return max(floor, 0.5 ** (max(0.0, age_s) / half_life_s))


def label(age_s: float, half_life_s: float) -> str:
    if not half_life_s or half_life_s <= 0:
        return "evergreen"
    ratio = max(0.0, age_s) / half_life_s
    return "fresh" if ratio < 0.5 else "aging" if ratio < 2 else "stale"


def rank(candidates: list[dict], *, at: float, mode: str, top_k: int, half_lives: dict[str, float],
         half_life_override: float | None = None, floor: float = DECAY_FLOOR,
         min_similarity: float = 0.0) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    hits = []
    for cand in candidates:
        p = cand["payload"]
        hl = half_life_override if half_life_override else half_lives.get(p["source_id"], DEFAULT_HALF_LIFE_S)
        age = max(0.0, at - p["published_at"])
        d = decay(age, hl, floor)
        sim = cand["score"]
        hits.append({
            "id": cand["id"], "document_id": p["item_key"], "chunk_index": p["chunk_index"],
            "title": p.get("title", ""), "url": p.get("url", ""), "text": p["text"],
            "source_id": p["source_id"], "source_name": p.get("source_name", ""),
            "published_at": p["published_at"], "published_iso": iso(p["published_at"]),
            "published_assumed": bool(p.get("published_assumed", False)),
            "age_seconds": age, "age_human": humanize(age), "freshness_label": label(age, hl),
            "half_life_s": hl, "similarity": sim, "decay": d,
            "score": sim * d if mode == "freshness" else sim, "version": p.get("version", 1),
        })
    if mode == "latest":
        hits = [h for h in hits if h["similarity"] >= min_similarity]
        hits.sort(key=lambda h: (-h["published_at"], -h["similarity"], h["id"]))
    else:
        hits.sort(key=lambda h: (-h["score"], h["id"]))  # id breaks ties so replay is deterministic
    return hits[:top_k]


def _age_phrase(age_s: float) -> str:
    h = humanize(age_s)
    return h if h == "just now" else f"{h} old"


def build_disclosure(hits: list[dict], at: float, historical: bool = False,
                     stale_after_s: float = DEFAULT_STALE_AFTER_S) -> dict:
    prefix = f"Historical view as of {iso(at)}" if historical else f"As of {iso(at)}"
    if not hits:
        return {"text": f"{prefix}: no sources within their relevance window matched this question.",
                "newest_at": None, "oldest_at": None, "newest_age_s": None, "n_sources": 0,
                "stale": True, "historical": historical, "as_of": iso(at), "assumed_dates": 0}
    newest = max(h["published_at"] for h in hits)
    oldest = min(h["published_at"] for h in hits)
    newest_age = max(0.0, at - newest)
    assumed = sum(1 for h in hits if h["published_assumed"])
    n = len(hits)
    span = iso(newest) if newest == oldest else f"{iso(oldest)} to {iso(newest)}"
    text = (f"{prefix}: based on {n} source{'s' if n != 1 else ''} published {span} "
            f"(newest is {_age_phrase(newest_age)}).")
    stale = newest_age > stale_after_s
    if stale:
        text += f" Warning: the most recent relevant information is {_age_phrase(newest_age)} and may be outdated."
    if assumed:
        text += f" Note: {assumed} source{'s' if assumed != 1 else ''} had no publish date, so ingestion time was used."
    return {"text": text, "newest_at": iso(newest), "oldest_at": iso(oldest), "newest_age_s": newest_age,
            "n_sources": n, "stale": stale, "historical": historical, "as_of": iso(at), "assumed_dates": assumed}
