"""FastEmbed-backed Embedder (M6 phase 2).

ONNX-compiled sentence-transformer wrapper. Replaces the ``hash``
baseline's token-overlap proxy with real semantic similarity — the
intended production backend for paraphrase-tolerant fact recall.

Lazy load: ``fastembed.TextEmbedding`` is imported on first
``embed()`` call so the seam stays import-safe without the
``local-embed`` extra installed. The first call also pays the
~130 MB BGE-small model download (cached under
``~/.cache/fastembed`` after that).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from familiar_connect import log_style as ls

_logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Known model dimensionalities. Pre-populated so we can advertise
# ``dim`` before the first embed call lands; on a model not listed
# here, ``dim`` stays 0 until the first embed call probes a real
# vector. Worker logging falls back gracefully on either path.
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/e5-small-v2": 384,
    "intfloat/multilingual-e5-small": 384,
}


class FastEmbedEmbedder:
    """ONNX sentence embedder. Lazy model load on first use."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        # ``name`` carries the model name so the (fact_id, model)
        # storage key splits rows by backend version. Upgrading from
        # BGE-small to BGE-base accumulates new rows without
        # corrupting the old similarity space.
        self.name: str = f"fastembed:{model_name}"
        self.dim: int = _KNOWN_DIMS.get(model_name, 0)
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_loaded()
        # ``model.embed`` is synchronous + CPU-bound (or GPU when
        # onnxruntime-gpu is installed). Push to a worker thread so
        # the asyncio loop keeps draining.
        vectors = await asyncio.to_thread(self._embed_sync, texts)
        # Update ``dim`` opportunistically the first time we see a
        # real vector — covers models not in ``_KNOWN_DIMS``.
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Run the model synchronously. Called from a worker thread."""
        # ``model.embed`` yields numpy.ndarray; iterate to a list of
        # floats without forcing a numpy import in the public surface.
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            self._model = await asyncio.to_thread(self._load_model)

    def _load_model(self) -> Any:  # noqa: ANN401 — opaque TextEmbedding
        """Import fastembed and load the model. Blocking."""
        try:
            from fastembed import (  # ty: ignore[unresolved-import]  # noqa: PLC0415
                TextEmbedding,
            )
        except ImportError as exc:
            msg = (
                "FastEmbedEmbedder requires the 'local-embed' extra. "
                "Install with `uv sync --extra local-embed`."
            )
            raise RuntimeError(msg) from exc

        _logger.info(
            f"{ls.tag('FastEmbed', ls.LG)} "
            f"{ls.kv('loading', self._model_name, vc=ls.LW)}"
        )
        kwargs: dict[str, Any] = {"model_name": self._model_name}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        model = TextEmbedding(**kwargs)
        _logger.info(
            f"{ls.tag('FastEmbed', ls.G)} {ls.kv('ready', self._model_name, vc=ls.LW)}"
        )
        return model
