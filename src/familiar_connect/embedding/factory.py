"""Embedder backend selector (M6).

Mirrors projector / STT / turn-detection pattern: thin registry keyed
on ``[providers.embedding].backend`` selector. Built-ins register at
import time; third-party backends call :func:`register_embedder` with
a factory.

Built-ins:

* ``off`` — returns ``None``. Disables seam end to end.
* ``hash`` — :class:`HashEmbedder`. Deterministic, no extra deps.
* ``fastembed`` — :class:`FastEmbedEmbedder`. ONNX-backed sentence
  embedder. Factory probes the ``fastembed`` import at load and raises
  if the ``local-embed`` extra is missing — fail fast at startup, not
  mid-turn. Model itself still loads lazily on first ``embed()``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING

from familiar_connect.embedding.fastembed import FastEmbedEmbedder
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
    """Names registered today (built-ins + third-party additions)."""
    return set(_REGISTRY)


def create_embedder(config: EmbeddingConfig) -> Embedder | None:
    """Instantiate embedder selected by ``config.backend``.

    :raises ValueError: ``config.backend`` not registered.
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


def _fastembed_factory(config: EmbeddingConfig) -> Embedder:
    # Fail fast at load: a deploy that selects fastembed but lacks the
    # extra should refuse to start, not crash mid-turn on first embed.
    # Probe the import only (not the ~130 MB model) — startup stays fast,
    # model still loads lazily on first embed().
    if importlib.util.find_spec("fastembed") is None:
        msg = (
            "embedding backend 'fastembed' requires the 'local-embed' extra. "
            "Install with `uv sync --extra local-embed`."
        )
        raise RuntimeError(msg)
    return FastEmbedEmbedder(
        model_name=config.fastembed_model,
        cache_dir=config.fastembed_cache_dir,
    )


register_embedder("off", _off_factory)
register_embedder("hash", _hash_factory)
register_embedder("fastembed", _fastembed_factory)
