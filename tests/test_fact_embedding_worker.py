"""Tests for the M6 FactEmbeddingWorker projector.

Validates the watermark-driven embedding loop: pulls
``unembedded_facts``, calls the embedder once per batch, persists each
vector. Idempotent — re-running the tick on already-embedded rows is
a no-op.
"""

from __future__ import annotations

import pytest

from familiar_connect.embedding import HashEmbedder
from familiar_connect.history.store import HistoryStore
from familiar_connect.processors.fact_embedding_worker import FactEmbeddingWorker


class _CountingEmbedder:
    """Embedder stub recording call counts for batch-size assertions."""

    name: str = "stub-v1"
    dim: int = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(i), 0.0, 0.0, 0.0] for i, _ in enumerate(texts, 1)]


def _store_with_facts(n: int) -> HistoryStore:
    store = HistoryStore(":memory:")
    for i in range(n):
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text=f"fact about thing {i}",
            source_turn_ids=[i],
        )
    return store


class TestFactEmbeddingWorkerTick:
    @pytest.mark.asyncio
    async def test_first_tick_embeds_all_pending_within_batch(self) -> None:
        store = _store_with_facts(3)
        embedder = _CountingEmbedder()
        worker = FactEmbeddingWorker(
            store=store, embedder=embedder, familiar_id="fam", batch_size=10
        )
        written = await worker.tick()
        assert written == 3
        assert len(embedder.calls) == 1
        assert len(embedder.calls[0]) == 3
        # Vectors persisted.
        assert (
            store.latest_embedded_fact_id(familiar_id="fam", model=embedder.name) == 3
        )

    @pytest.mark.asyncio
    async def test_batch_size_caps_per_tick_writes(self) -> None:
        store = _store_with_facts(5)
        embedder = _CountingEmbedder()
        worker = FactEmbeddingWorker(
            store=store, embedder=embedder, familiar_id="fam", batch_size=2
        )
        written = await worker.tick()
        assert written == 2
        assert (
            store.latest_embedded_fact_id(familiar_id="fam", model=embedder.name) == 2
        )

    @pytest.mark.asyncio
    async def test_subsequent_tick_skips_embedded(self) -> None:
        store = _store_with_facts(3)
        embedder = _CountingEmbedder()
        worker = FactEmbeddingWorker(
            store=store, embedder=embedder, familiar_id="fam", batch_size=10
        )
        await worker.tick()
        embedder.calls.clear()
        # Another tick with no new facts: no work, no embedder call.
        written = await worker.tick()
        assert written == 0
        assert embedder.calls == []

    @pytest.mark.asyncio
    async def test_resume_after_partial_progress(self) -> None:
        store = _store_with_facts(5)
        embedder = _CountingEmbedder()
        worker = FactEmbeddingWorker(
            store=store, embedder=embedder, familiar_id="fam", batch_size=2
        )
        # Three ticks of size 2 cover all 5 facts.
        n1 = await worker.tick()
        n2 = await worker.tick()
        n3 = await worker.tick()
        n4 = await worker.tick()
        assert (n1, n2, n3, n4) == (2, 2, 1, 0)
        assert (
            store.latest_embedded_fact_id(familiar_id="fam", model=embedder.name) == 5
        )

    @pytest.mark.asyncio
    async def test_real_hash_embedder_persists_vectors_for_recall(self) -> None:
        store = _store_with_facts(2)
        embedder = HashEmbedder(dim=64)
        worker = FactEmbeddingWorker(store=store, embedder=embedder, familiar_id="fam")
        await worker.tick()
        got = store.get_fact_embeddings(fact_ids=[1, 2], model=embedder.name)
        assert set(got) == {1, 2}
        assert all(len(v) == 64 for v in got.values())


class TestFactEmbeddingWorkerSafety:
    @pytest.mark.asyncio
    async def test_empty_store_tick_is_noop(self) -> None:
        store = _store_with_facts(0)
        embedder = _CountingEmbedder()
        worker = FactEmbeddingWorker(store=store, embedder=embedder, familiar_id="fam")
        assert await worker.tick() == 0

    @pytest.mark.asyncio
    async def test_mismatched_embedder_batch_skips_without_writing(self) -> None:
        """Defensive: a bad backend mustn't corrupt rows or advance state."""

        class _BadEmbedder:
            name = "bad-v1"
            dim = 4

            async def embed(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
                # Returns fewer vectors than requested — backend bug.
                return [[1.0, 0.0, 0.0, 0.0]]

        store = _store_with_facts(3)
        worker = FactEmbeddingWorker(
            store=store, embedder=_BadEmbedder(), familiar_id="fam"
        )
        assert await worker.tick() == 0
        assert store.latest_embedded_fact_id(familiar_id="fam", model="bad-v1") == 0
