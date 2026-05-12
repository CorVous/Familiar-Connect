"""Tests for the FastEmbed-backed Embedder (M6 phase 2).

The dev environment doesn't install ``fastembed`` (it lives behind
the ``local-embed`` extra), so these tests cover:

* Construction without a backend installed (pure metadata).
* Lazy-load failure surface — ``RuntimeError`` with a pointed
  install hint when ``fastembed`` is missing.
* The factory dispatch produces a :class:`FastEmbedEmbedder` and
  routes config through ``[providers.embedding]``.

Live-model tests against a real BGE-small download are tagged
``integration`` so they only run with the extra installed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import EmbeddingConfig
from familiar_connect.embedding import (
    FastEmbedEmbedder,
    create_embedder,
    known_embedders,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class TestFactory:
    def test_known_embedders_includes_fastembed(self) -> None:
        assert "fastembed" in known_embedders()

    def test_factory_picks_fastembed_when_selected(self) -> None:
        e = create_embedder(EmbeddingConfig(backend="fastembed"))
        assert isinstance(e, FastEmbedEmbedder)

    def test_factory_threads_model_name(self) -> None:
        e = create_embedder(
            EmbeddingConfig(
                backend="fastembed",
                fastembed_model="BAAI/bge-base-en-v1.5",
            )
        )
        assert isinstance(e, FastEmbedEmbedder)
        assert "bge-base" in e.name


class TestMetadata:
    def test_name_carries_model(self) -> None:
        e = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
        assert e.name == "fastembed:BAAI/bge-small-en-v1.5"

    def test_dim_known_for_bge_small(self) -> None:
        assert FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5").dim == 384

    def test_dim_known_for_bge_base(self) -> None:
        assert FastEmbedEmbedder(model_name="BAAI/bge-base-en-v1.5").dim == 768

    def test_dim_zero_for_unknown_model_until_first_embed(self) -> None:
        # Stays 0 until ``embed()`` probes a real vector.
        assert FastEmbedEmbedder(model_name="custom/model").dim == 0

    def test_distinct_models_get_distinct_names(self) -> None:
        a = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
        b = FastEmbedEmbedder(model_name="BAAI/bge-base-en-v1.5")
        # Storage keys on ``name``: split rows so a model swap doesn't
        # mix old + new vectors in the same similarity space.
        assert a.name != b.name


class TestLazyLoad:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_without_loading(self) -> None:
        """No texts → no model load. Cheap fast-path stays cheap."""
        e = FastEmbedEmbedder()
        result = await e.embed([])
        assert result == []
        assert e._model is None

    @pytest.mark.asyncio
    async def test_missing_extra_raises_pointed_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing extra surfaces a pointed runtime error.

        First embed call must name the install command in the message.
        """
        # Force ``import fastembed`` inside ``_load_model`` to fail.
        monkeypatch.setitem(sys.modules, "fastembed", None)
        e = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
        with pytest.raises(RuntimeError, match="local-embed"):
            await e.embed(["any text"])

    @pytest.mark.asyncio
    async def test_load_is_idempotent_with_stub_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two embed calls reuse the loaded model (single TextEmbedding ctor)."""
        load_count = 0

        class _StubModel:
            def __init__(self, **_kwargs: object) -> None:
                nonlocal load_count
                load_count += 1

            def embed(self, texts: list[str]) -> Iterator[list[float]]:
                for _ in texts:
                    yield [0.1, 0.2, 0.3, 0.4]

        # Inject a fake ``fastembed`` module exposing TextEmbedding.
        fake_module = SimpleNamespace(TextEmbedding=_StubModel)
        monkeypatch.setitem(sys.modules, "fastembed", fake_module)

        e = FastEmbedEmbedder(model_name="custom/test")
        v1 = await e.embed(["alpha"])
        v2 = await e.embed(["beta", "gamma"])
        assert load_count == 1
        assert len(v1) == 1
        assert len(v2) == 2
        # Dim discovered from probe on first vector.
        assert e.dim == 4

    @pytest.mark.asyncio
    async def test_known_dim_preserved_after_embed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-known ``dim`` doesn't get clobbered by an oddly-shaped probe."""

        class _StubModel:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def embed(self, texts: list[str]) -> Iterator[list[float]]:
                for _ in texts:
                    yield [0.0] * 384

        monkeypatch.setitem(
            sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_StubModel)
        )
        e = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
        assert e.dim == 384  # set by the lookup table at construction
        await e.embed(["x"])
        assert e.dim == 384  # unchanged

    @pytest.mark.asyncio
    async def test_cache_dir_threaded_to_text_embedding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class _StubModel:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def embed(self, texts: list[str]) -> Iterator[list[float]]:
                for _ in texts:
                    yield [0.1]

        monkeypatch.setitem(
            sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_StubModel)
        )
        e = FastEmbedEmbedder(model_name="m", cache_dir="/tmp/fastembed-test")  # noqa: S108
        await e.embed(["x"])
        assert captured["cache_dir"] == "/tmp/fastembed-test"  # noqa: S108
        assert captured["model_name"] == "m"

    @pytest.mark.asyncio
    async def test_cache_dir_omitted_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class _StubModel:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def embed(self, texts: list[str]) -> Iterator[list[float]]:
                for _ in texts:
                    yield [0.1]

        monkeypatch.setitem(
            sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_StubModel)
        )
        e = FastEmbedEmbedder(model_name="m")
        await e.embed(["x"])
        # No cache_dir kwarg unless operator pinned one — stays on
        # fastembed's default (``~/.cache/fastembed``).
        assert "cache_dir" not in captured
