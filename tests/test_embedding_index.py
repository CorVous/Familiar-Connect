"""Red-first tests for the SQLite-backed embedding index.

Covers build/query/rebuild-on-mtime/vector-roundtrip. A stub
embedding model returns deterministic vectors keyed off chunk text
so CI never downloads ONNX weights.
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
from familiar_connect.memory.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.context.providers.content_search.retrieval import (
        EmbeddingModel,
    )

_DIM = 8


@dataclass
class _StubModel:
    """Deterministic 8-dim embedding.

    Hashes the input text and reads the first ``_DIM`` bytes as the
    vector. Similar texts produce similar vectors only by accident
    — for retrieval-quality tests the caller crafts inputs with
    known vector relationships.
    """

    dim: int = _DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            h = hashlib.sha256(text.encode("utf-8")).digest()
            out[i] = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
        # L2-normalize so downstream dot product == cosine similarity
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out / norms


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    return tmp_path / "memory" / ".index" / "embeddings.sqlite"


@pytest.fixture
def model() -> EmbeddingModel:
    return _StubModel()


class TestBuildEmpty:
    def test_build_on_empty_store(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)
        # empty store → no chunks, no documents, but file exists (so the
        # next startup doesn't re-scan)
        assert index_path.exists()
        assert idx.query(model.embed(["hello"])[0], top_k=5) == []
        idx.close()


class TestBuildAndQuery:
    def test_build_produces_queryable_chunks(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        store.write_file("people/alice.md", "Alice likes ska music.")
        store.write_file("people/bob.md", "Bob the builder fixes things.")

        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)

        # query with the exact body of alice's chunk — hash-based stub
        # matches exactly on identical input
        q = model.embed(["Alice likes ska music."])[0]
        # but the stored chunk text has the heading prepended, so the
        # retriever won't get an exact match; query against something
        # indexed under both paths to assert non-emptiness.
        results = idx.query(q, top_k=5)
        assert len(results) >= 1
        rel_paths = {r.rel_path for r in results}
        assert rel_paths <= {"people/alice.md", "people/bob.md"}
        idx.close()

    def test_top_k_respected(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        for i in range(5):
            store.write_file(f"notes/{i}.md", f"note number {i} here")
        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)
        q = model.embed(["anything"])[0]
        results = idx.query(q, top_k=2)
        assert len(results) == 2
        idx.close()

    def test_top_result_for_exact_match(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        """Query = a chunk's stored embedding input → that chunk is top."""
        store.write_file("a.md", "unique phrase about penguins")
        store.write_file("b.md", "a totally unrelated sentence about foxes")
        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)

        # embed the exact same text used as chunk body of a.md (small
        # file so the chunk text === original)
        q = model.embed(["unique phrase about penguins"])[0]
        results = idx.query(q, top_k=2)
        assert results[0].rel_path == "a.md"
        assert results[0].score >= results[1].score
        idx.close()


class TestRebuildFreshness:
    def test_unchanged_mtime_is_noop(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        store.write_file("note.md", "hello")
        idx = EmbeddingIndex(index_path, dim=_DIM)
        stats1 = idx.build_if_stale(store, model)
        assert stats1.rebuilt_paths == ["note.md"]
        # second build with no file changes → no rebuild
        stats2 = idx.build_if_stale(store, model)
        assert stats2.rebuilt_paths == []
        idx.close()

    def test_changed_mtime_triggers_rebuild(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        store.write_file("note.md", "original text about ska")
        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)

        # overwrite; mtime changes, content changes
        store.write_file("note.md", "totally new content about jazz")
        stats = idx.build_if_stale(store, model)
        assert stats.rebuilt_paths == ["note.md"]

        # old chunks are gone; query for old content finds new content
        results = idx.query(model.embed(["totally new content about jazz"])[0], top_k=5)
        assert len(results) == 1
        assert "jazz" in results[0].text
        idx.close()

    def test_deleted_file_removed_from_index(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        store.write_file("a.md", "apple")
        store.write_file("b.md", "banana")
        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)

        # manually delete b.md from disk (MemoryStore has no delete API;
        # the indexer has to handle files vanishing)
        (store.root / "b.md").unlink()

        stats = idx.build_if_stale(store, model)
        assert stats.removed_paths == ["b.md"]
        results = idx.query(model.embed(["banana"])[0], top_k=5)
        assert all(r.rel_path != "b.md" for r in results)
        idx.close()


class TestVectorRoundtrip:
    def test_float32_blob_roundtrip(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        """Vectors survive the SQLite BLOB roundtrip bit-for-bit."""
        store.write_file("x.md", "some content")
        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)

        # close + reopen — simulates process restart
        idx.close()
        idx2 = EmbeddingIndex(index_path, dim=_DIM)

        # query with the same stub input → identical vector → top score
        # very close to 1.0 (L2-normalized self-dot)
        q = model.embed(["some content"])[0]
        results = idx2.query(q, top_k=1)
        assert len(results) == 1
        # self-similarity of an L2-normalized unit vector is 1.0;
        # roundtrip error would drop this below 0.999
        assert results[0].score >= 0.999
        idx2.close()


class TestIndexIgnoresOwnDirectory:
    def test_index_dir_not_embedded(
        self, store: MemoryStore, index_path: Path, model: EmbeddingModel
    ) -> None:
        """The index must not scan its own .index/ directory."""
        store.write_file("note.md", "hello")
        idx = EmbeddingIndex(index_path, dim=_DIM)
        idx.build_if_stale(store, model)
        # second build — nothing under .index/ should be treated as
        # source markdown
        stats = idx.build_if_stale(store, model)
        assert stats.rebuilt_paths == []
        idx.close()
