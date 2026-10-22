"""
Pluggable embedders.

  sentence-transformers  real semantic embeddings (default; downloads the model on first use)
  hash                   deterministic feature-hashing embedder: no model, no network, purely
                         lexical. Used by the tests and for offline demos. Do NOT use it for
                         real retrieval quality.

The embedder's name is recorded with every logged query so audit replay can detect
that the model changed (a replay under a different embedder can't be expected to match).
"""
import hashlib
import math
import re
from functools import lru_cache

from app.config import EMBEDDER, EMBEDDING_DIM, EMBEDDING_MODEL


class HashEmbedder:
    name = f"hash-{EMBEDDING_DIM}"
    dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        toks = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dim

        def add(tok: str, weight: float) -> None:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += weight if (h >> 64) & 1 else -weight

        for t in toks:
            add(t, 1.0)
        for a, b in zip(toks, toks[1:]):
            add(f"{a} {b}", 0.5)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_embedder():
    return HashEmbedder() if EMBEDDER == "hash" else STEmbedder(EMBEDDING_MODEL)
