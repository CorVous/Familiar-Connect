"""Hash-based embedder — deterministic, no-deps fallback (M6).

A locality-hashing baseline: tokenise on word boundaries, project each
token onto a stable bucket via BLAKE2b, accumulate into a fixed-dim
float vector, then L2-normalise. Cosine similarity over these vectors
correlates with token overlap — a weak signal compared to a real
neural embedder, but it's deterministic, has no install cost, and
gives the storage + retrieval seams something to validate against.

Production deployments should swap in :class:`FastEmbedEmbedder` (or
another real backend) via ``[providers.embedding].backend``. The
``hash`` backend stays useful as an offline fixture / smoke test.
"""

from __future__ import annotations

import hashlib
import math
import re

# Word-boundary tokenisation matches FTS5's ``unicode61`` shape closely
# enough for "is the BM25 hit also similar by embedding?" sanity checks.
# Drop diacritics + casefold so paraphrase variation in the cue still
# hashes onto the same buckets ("Café" / "cafe" / "CAFE").
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [m.group(0).casefold() for m in _TOKEN_RE.finditer(text)]


class HashEmbedder:
    """Deterministic locality-hashing baseline.

    Stable across processes / runs given the same ``dim`` — BLAKE2b is
    seedless and the projection is purely arithmetic.
    """

    name: str = "hash-v1"

    def __init__(self, *, dim: int = 256) -> None:
        if dim < 8:
            msg = f"HashEmbedder dim must be >= 8, got {dim}"
            raise ValueError(msg)
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dim
            sign = 1.0 if (digest[2] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if not norm:
            return vec
        return [v / norm for v in vec]
