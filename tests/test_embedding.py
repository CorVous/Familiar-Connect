"""Tests for the embedding seam (M6).

Covers :class:`Embedder` Protocol implementations and the factory
selector. Backend-specific behaviour for optional backends
(``fastembed``) lives in their own modules; here we only exercise
shipped built-ins (``off``, ``hash``).
"""

from __future__ import annotations

import math

import pytest

from familiar_connect.config import EmbeddingConfig
from familiar_connect.embedding import (
    HashEmbedder,
    create_embedder,
    known_embedders,
)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


class TestHashEmbedder:
    def test_dim_must_be_at_least_8(self) -> None:
        with pytest.raises(ValueError, match=">= 8"):
            HashEmbedder(dim=4)

    def test_name_is_versioned(self) -> None:
        assert HashEmbedder().name == "hash-v1"

    def test_dim_matches_constructor(self) -> None:
        assert HashEmbedder(dim=128).dim == 128

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self) -> None:
        assert await HashEmbedder().embed([]) == []

    @pytest.mark.asyncio
    async def test_blank_text_projects_to_zero_vector(self) -> None:
        [vec] = await HashEmbedder(dim=64).embed([""])
        assert vec == [0.0] * 64

    @pytest.mark.asyncio
    async def test_output_length_matches_input(self) -> None:
        vecs = await HashEmbedder(dim=32).embed(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 32 for v in vecs)

    @pytest.mark.asyncio
    async def test_deterministic_across_calls(self) -> None:
        e = HashEmbedder(dim=64)
        v1 = await e.embed(["hello world"])
        v2 = await e.embed(["hello world"])
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_casefold_invariant(self) -> None:
        """Tokens casefold so cue paraphrasing in case still hashes the same."""
        e = HashEmbedder(dim=64)
        [a, b] = await e.embed(["Café Latte", "café LATTE"])
        assert a == b

    @pytest.mark.asyncio
    async def test_unrelated_texts_score_low(self) -> None:
        e = HashEmbedder(dim=256)
        [a, b] = await e.embed([
            "the capital of france is paris",
            "rust ownership rules",
        ])
        # No shared tokens — should be near zero.
        assert _cos(a, b) < 0.1

    @pytest.mark.asyncio
    async def test_overlapping_tokens_score_higher(self) -> None:
        e = HashEmbedder(dim=256)
        [a, b, c] = await e.embed([
            "alice has a cat named whiskers",
            "alice and the cat whiskers",
            "rust ownership rules",
        ])
        # Overlap pair should rank higher than disjoint pair.
        assert _cos(a, b) > _cos(a, c)


class TestEmbedderFactory:
    def test_known_embedders_includes_builtins(self) -> None:
        assert {"off", "hash"} <= known_embedders()

    def test_off_returns_none(self) -> None:
        assert create_embedder(EmbeddingConfig(backend="off")) is None

    def test_hash_returns_hash_embedder(self) -> None:
        e = create_embedder(EmbeddingConfig(backend="hash", dim=128))
        assert isinstance(e, HashEmbedder)
        assert e.dim == 128

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown embedding backend"):
            create_embedder(EmbeddingConfig(backend="nonexistent"))


class TestEmbeddingConfig:
    def test_default_backend_is_off(self) -> None:
        assert EmbeddingConfig().backend == "off"

    def test_default_dim_is_256(self) -> None:
        assert EmbeddingConfig().dim == 256
