"""Embedder Protocol — text → fixed-dim vector seam (M6).

Three method-level guarantees the seam relies on:

* :attr:`Embedder.name` — backend label persisted alongside each
  vector. Lets the projector detect a model swap and rebuild without
  destroying audit history.
* :attr:`Embedder.dim` — vector dimensionality. Must be stable across
  calls; the storage layer assumes a single ``(fact_id, model)`` row
  has one fixed-size vector.
* :meth:`Embedder.embed` — batch interface. Backends are free to
  vectorise internally; tests exercise both single-shot and batched
  call shapes.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Stable text → vector seam.

    Backends:

    * :class:`HashEmbedder` — built-in deterministic baseline.
    * ``fastembed`` — optional, opts in via the ``local-embed`` extra.

    Implementations must be safe to call from a background asyncio
    task; CPU-bound work belongs inside ``asyncio.to_thread``.
    """

    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one float vector per input text. Order matches input.

        Empty input returns an empty list. Backends should preserve
        list length even for empty / blank strings (project blank text
        to the zero vector or a stable sentinel; never raise).
        """
        ...
