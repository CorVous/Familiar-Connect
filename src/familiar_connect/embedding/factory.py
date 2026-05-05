"""Embedder backend selector (M6).

Mirrors the projector / STT / turn-detection pattern: a thin registry
keyed on the ``[providers.embedding].backend`` selector. Built-ins
register at import time; third-party backends call
:func:`register_embedder` with a factory.

Built-ins:

* ``off`` — returns ``None``. Disables the seam end to end.
* ``hash`` — :class:`HashEmbedder`. Deterministic, no extra deps.

Optional:

* ``fastembed`` — ONNX-backed sentence embedder. Pulls
  :mod:`fastembed` from the ``local-embed`` extra; raises a
  pointed install error when unavailable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from familiar_connect.embedding.hash import HashEmbedder

if TYPE_CHECKING:
    from familiar_connect.config import EmbeddingConfig
    from familiar_connect.embedding.protocol import Embedder


EmbedderFactory = Callable[["EmbeddingConfig"], "Embedder | None"]


_REGISTRY: dict[str, EmbedderFactory] = {}


def register_embedder(name: str, factory: EmbedderFactory) -> None:
    """Register *factory* under *name*. Re-registration overwrites."""
    _REGISTRY[name] = factory


def known_embedders() -> set[str]:
    """Names registered today (built-ins + any third-party additions)."""
    return set(_REGISTRY)


def create_embedder(config: EmbeddingConfig) -> Embedder | None:
    """Instantiate the embedder selected by ``config.backend``.

    :raises ValueError: when ``config.backend`` is not registered.
    """
    factory = _REGISTRY.get(config.backend)
    if factory is None:
        valid = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"unknown embedding backend {config.backend!r}; valid: {valid}"
        raise ValueError(msg)
    return factory(config)


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def _off_factory(config: EmbeddingConfig) -> Embedder | None:  # noqa: ARG001
    return None


def _hash_factory(config: EmbeddingConfig) -> Embedder:
    return HashEmbedder(dim=config.dim)


register_embedder("off", _off_factory)
register_embedder("hash", _hash_factory)
