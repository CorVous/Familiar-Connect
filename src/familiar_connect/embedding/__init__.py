"""Embedding seam for semantic recall (M6).

Splits the embedding provider behind a thin :class:`Embedder` Protocol
so retrieval can fuse vector similarity into the existing BM25 +
recency + importance fusion in :class:`RagContextLayer`.

Built-in backends:

* ``hash`` — deterministic locality-hashing fallback, no extra deps.
  Useful for tests + structural validation; gives a weak but stable
  signal on token overlap. Ships in core.
* ``off`` — disables the seam (default). Picks the no-op embedder so
  every call site can stay non-conditional; the projector and the
  RAG layer treat ``None`` and a no-op embedder the same way.

Optional backends register themselves via :func:`register_embedder`
at import time (matches the projector / STT pattern).
"""

from familiar_connect.embedding.factory import (
    create_embedder,
    known_embedders,
    register_embedder,
)
from familiar_connect.embedding.hash import HashEmbedder
from familiar_connect.embedding.protocol import Embedder

__all__ = [
    "Embedder",
    "HashEmbedder",
    "create_embedder",
    "known_embedders",
    "register_embedder",
]
