"""Red-first tests for EmbeddingRetriever — Retriever protocol.

The retriever embeds the utterance, queries the index, and returns
ranked RetrievedChunks. Tests inject a stub embedding model so CI
never downloads weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from familiar_connect.context.providers.content_search.index.embeddings import (
    EmbeddingIndex,
)
from familiar_connect.context.providers.content_search.retrieval import (
    EmbeddingRetriever,
    RetrievedChunk,
    Retriever,
)
from familiar_connect.memory.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

_DIM = 8


@dataclass
class _StubModel:
    dim: int = _DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            h = hashlib.sha256(text.encode("utf-8")).digest()
            out[i] = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out / norms


@pytest.fixture
def retriever(tmp_path: Path) -> EmbeddingRetriever:
    store = MemoryStore(tmp_path / "memory")
    store.write_file("a.md", "alpha text here")
    store.write_file("b.md", "beta text here")
    model = _StubModel()
    index = EmbeddingIndex(
        tmp_path / "memory" / ".index" / "embeddings.sqlite", dim=_DIM
    )
    index.build_if_stale(store, model)
    return EmbeddingRetriever(index=index, model=model)


class TestProtocolConformance:
    def test_conforms_to_retriever_protocol(
        self, retriever: EmbeddingRetriever
    ) -> None:
        assert isinstance(retriever, Retriever)


class TestQuery:
    @pytest.mark.asyncio
    async def test_returns_ranked_retrieved_chunks(
        self, retriever: EmbeddingRetriever
    ) -> None:
        results = await retriever.query("anything", top_k=2)
        assert len(results) <= 2
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert r.rel_path in {"a.md", "b.md"}
            assert r.text
            assert isinstance(r.score, float)
        # monotonic score ordering (highest first)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_utterance_returns_empty(
        self, retriever: EmbeddingRetriever
    ) -> None:
        results = await retriever.query("", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_exclude_paths_drops_matches(
        self, retriever: EmbeddingRetriever
    ) -> None:
        """Retriever honours an exclude set.

        Lets the caller skip files the deterministic people-lookup
        tier already surfaced.
        """
        results = await retriever.query(
            "alpha text here", top_k=5, exclude_paths={"a.md"}
        )
        assert all(r.rel_path != "a.md" for r in results)


class TestEmptyIndex:
    @pytest.mark.asyncio
    async def test_query_empty_index_returns_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path / "memory")  # no files written
        model = _StubModel()
        index = EmbeddingIndex(
            tmp_path / "memory" / ".index" / "embeddings.sqlite", dim=_DIM
        )
        index.build_if_stale(store, model)
        retriever = EmbeddingRetriever(index=index, model=model)

        results = await retriever.query("anything", top_k=5)
        assert results == []
