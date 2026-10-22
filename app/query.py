"""
Query + audit.

`retrieve_hits` is the single retrieval path. Live queries, audit replay and the what-if endpoint all
go through it with explicit parameters, so a replay reproduces the original ranking by construction
(same time T, same visibility predicate, same logged half-lives, same tie-breaks).

Time semantics: with as_of=None, T = now. With as_of set, T = as_of for EVERYTHING: which chunks were
visible (published, ingested, not expired, not superseded) and how old they look. That is "what did
this system know at T", not just "what was published before T".
"""
import time

from app import registry, vector_store
from app.config import DECAY_FLOOR, OVERSAMPLE
from app.embedding import get_embedder
from app.freshness import MODES, build_disclosure, rank
from app.llm import generate_answer


def retrieve_hits(tenant_id: str, question: str, *, mode: str, top_k: int, at: float,
                  half_lives: dict, half_life_override: float | None, source_ids: list[str] | None,
                  min_similarity: float, floor: float = DECAY_FLOOR, oversample: int = OVERSAMPLE) -> list[dict]:
    embedder = get_embedder()
    [vec] = embedder.embed([question])
    cands = vector_store.search(embedder.dim, tenant_id, vec, max(top_k * oversample, 20), at, source_ids)
    return rank(cands, at=at, mode=mode, top_k=top_k, half_lives=half_lives,
                half_life_override=half_life_override, floor=floor, min_similarity=min_similarity)


def _params(mode, top_k, at, half_lives, override, source_ids, min_similarity, as_of) -> dict:
    return {"mode": mode, "top_k": top_k, "at": at, "half_lives": half_lives, "half_life_override": override,
            "source_ids": source_ids, "min_similarity": min_similarity, "as_of": as_of,
            "floor": DECAY_FLOOR, "oversample": OVERSAMPLE, "embedder": get_embedder().name}


def run_query(tenant_id: str, question: str, *, top_k: int = 5, mode: str = "freshness",
              as_of: float | None = None, half_life_override: float | None = None,
              source_ids: list[str] | None = None, min_similarity: float = 0.25,
              generate: bool = True, now: float | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    started = time.perf_counter()
    now = now or time.time()
    at = as_of if as_of is not None else now
    half_lives = registry.source_half_lives(tenant_id)
    hits = retrieve_hits(tenant_id, question, mode=mode, top_k=top_k, at=at, half_lives=half_lives,
                         half_life_override=half_life_override, source_ids=source_ids,
                         min_similarity=min_similarity)
    disclosure = build_disclosure(hits, at, historical=as_of is not None)
    answer, answer_mode = generate_answer(question, hits, at) if generate else ("", "skipped")
    query_id = registry.log_query(
        tenant_id, now, question,
        _params(mode, top_k, at, half_lives, half_life_override, source_ids, min_similarity, as_of),
        [h["id"] for h in hits], [h["score"] for h in hits], answer, disclosure["text"])
    return {"query_id": query_id, "answer": answer, "answer_mode": answer_mode, "disclosure": disclosure,
            "answer_with_disclosure": f"{answer}\n\n{disclosure['text']}" if answer else disclosure["text"],
            "sources": hits, "mode": mode, "as_of": disclosure["as_of"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def retrieve_only(tenant_id: str, question: str, *, top_k: int = 5, mode: str = "freshness",
                  as_of: float | None = None, half_life_override: float | None = None,
                  source_ids: list[str] | None = None, min_similarity: float = 0.25,
                  now: float | None = None) -> dict:
    """Retrieval without generation or logging (used for ranking comparisons and by the other projects'
    drift checks / cost layers; same response shape as the multi-tenant service's /query/retrieve)."""
    at = as_of if as_of is not None else (now or time.time())
    hits = retrieve_hits(tenant_id, question, mode=mode, top_k=top_k, at=at,
                         half_lives=registry.source_half_lives(tenant_id), half_life_override=half_life_override,
                         source_ids=source_ids, min_similarity=min_similarity)
    return {"sources": hits, "disclosure": build_disclosure(hits, at, historical=as_of is not None)}


def replay(tenant_id: str, query_id: str, regenerate: bool = False) -> dict | None:
    """Compliance mode: re-run a logged query at ITS original time with ITS original parameters and
    check we get the same chunks in the same order."""
    logged = registry.get_query(tenant_id, query_id)
    if logged is None:
        return None
    p = logged["params"]
    notes = []
    if p.get("embedder") != get_embedder().name:
        notes.append(f"embedder changed since the original query ({p.get('embedder')} -> {get_embedder().name}); "
                     "an exact match is not expected")
    hits = retrieve_hits(tenant_id, logged["question"], mode=p["mode"], top_k=p["top_k"], at=p["at"],
                         half_lives=p["half_lives"], half_life_override=p["half_life_override"],
                         source_ids=p["source_ids"], min_similarity=p["min_similarity"],
                         floor=p.get("floor", DECAY_FLOOR), oversample=p.get("oversample", OVERSAMPLE))
    replay_ids = [h["id"] for h in hits]
    original_ids = logged["chunk_ids"]
    purged = [c["id"] for c in registry.chunks_by_ids(tenant_id, original_ids) if c["purged"]]
    if purged:
        notes.append(f"{len(purged)} chunk(s) used by the original answer were purged after the retention "
                     "window, so it can no longer be reconstructed exactly")
    scores_match = len(hits) == len(logged["scores"]) and all(
        abs(h["score"] - s) < 1e-6 for h, s in zip(hits, logged["scores"]))
    result = {
        "query_id": query_id, "question": logged["question"], "original_time": logged["ts"], "evaluated_at": p["at"],
        "match": replay_ids == original_ids and scores_match, "same_chunks": set(replay_ids) == set(original_ids),
        "original_chunk_ids": original_ids, "replayed_chunk_ids": replay_ids,
        "missing_from_replay": [i for i in original_ids if i not in replay_ids],
        "extra_in_replay": [i for i in replay_ids if i not in original_ids],
        "purged_chunk_ids": purged, "notes": notes, "original_answer": logged["answer"],
        "original_disclosure": logged["disclosure"], "replay_sources": hits,
    }
    if regenerate:
        result["regenerated_answer"] = generate_answer(logged["question"], hits, p["at"])[0]
    return result
