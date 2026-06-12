"""Watermark-driven fact-embedding worker (M6).

For each current fact (``superseded_at IS NULL``) without entry
in ``fact_embeddings`` for active embedder's :attr:`Embedder.name`,
runs embedder, persists vector. Idempotent: model swap re-runs
against new ``model`` key, accumulates rows beside old, preserves
audit history.

Cadence matches :class:`FactExtractor` (15 s) — embeddings cheap
vs fact-extraction LLM call; tracks new facts closely so RAG sees
fresh vectors. Per-row watermark on ``fact_embeddings`` makes
partial progress safe.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span

if TYPE_CHECKING:
    from familiar_connect.embedding.protocol import Embedder
    from familiar_connect.history.async_store import AsyncHistoryStore

_logger = logging.getLogger("familiar_connect.processors.fact_embedding_worker")


class FactEmbeddingWorker:
    """Embeds new facts in batches off ``fact_embeddings``.

    :param batch_size: max facts per tick. Embedders may vectorise
        across batch; per-row upsert means partial batch still advances.
    :param tick_interval_s: idle-loop interval. Sleeps every tick
        whether work found or not — next tick re-checks cheaply.
    """

    name: str = "fact-embedding-worker"

    def __init__(
        self,
        *,
        store: AsyncHistoryStore,
        embedder: Embedder,
        familiar_id: str,
        batch_size: int = 32,
        tick_interval_s: float = 15.0,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._familiar_id = familiar_id
        self._batch_size = max(1, batch_size)
        self._tick_interval_s = tick_interval_s

    async def run(self) -> None:
        """Forever loop; tick on interval. Cancel to stop."""
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — worker must not die
                _logger.warning(
                    f"{ls.tag('FactEmbed', ls.R)} "
                    f"{ls.kv('tick_error', repr(exc), vc=ls.R)}"
                )
            await asyncio.sleep(self._tick_interval_s)

    @span("fact_embedding.tick")
    async def tick(self) -> int:
        """Embed up to ``batch_size`` unembedded facts; return count written."""
        pending = await self._store.unembedded_facts(
            familiar_id=self._familiar_id,
            model=self._embedder.name,
            limit=self._batch_size,
        )
        if not pending:
            return 0
        texts = [f.text for f in pending]
        vectors = await self._embedder.embed(texts)
        if len(vectors) != len(pending):
            # Backend bug — surface loudly, skip batch;
            # watermark retries next tick
            _logger.warning(
                f"{ls.tag('FactEmbed', ls.R)} "
                f"{ls.kv('mismatch', f'{len(vectors)}!={len(pending)}', vc=ls.R)}"
            )
            return 0
        for fact, vec in zip(pending, vectors, strict=True):
            await self._store.set_fact_embedding(
                fact_id=fact.id,
                model=self._embedder.name,
                vector=vec,
            )
        _logger.info(
            f"{ls.tag('FactEmbed', ls.LC)} "
            f"{ls.kv('model', self._embedder.name, vc=ls.LW)} "
            f"{ls.kv('dim', str(self._embedder.dim), vc=ls.LW)} "
            f"{ls.kv('written', str(len(pending)), vc=ls.LC)} "
            f"{ls.kv('latest_id', str(pending[-1].id), vc=ls.LW)}"
        )
        return len(pending)
