"""Embedding seam for semantic recall (M6).

Splits embedding provider behind thin :class:`Embedder` Protocol so
retrieval can fuse vector similarity into existing BM25 + recency +
importance fusion in :class:`RagContextLayer`.

Built-in backends:

* ``hash`` — deterministic locality-hashing fallback, no extra deps.
  Useful for tests + structural validation; gives weak but stable
  signal on token overlap. Ships in core.
* ``off`` — disables seam (default). Picks no-op embedder so every
  call site stays non-conditional; projector + RAG layer treat
  ``None`` and a no-op embedder the same way.

Optional backends register via :func:`register_embedder` at import
time (matches projector / STT pattern).
"""

from familiar_connect.embedding.factory import (
    create_embedder,
    known_embedders,
    register_embedder,
)
from familiar_connect.embedding.fastembed import FastEmbedEmbedder
from familiar_connect.embedding.hash import HashEmbedder
from familiar_connect.embedding.protocol import Embedder

__all__ = [
    "Embedder",
    "FastEmbedEmbedder",
    "HashEmbedder",
    "create_embedder",
    "known_embedders",
    "register_embedder",
]
