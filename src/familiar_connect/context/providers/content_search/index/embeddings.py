"""SQLite-backed embedding index for ContentSearchProvider tier 2.

Stores one row per chunk with the embedding vector as a
little-endian float32 ``BLOB``. L2-normalised at write time so
query-time similarity is a plain dot product.

``build_if_stale`` is the single entry point the runtime uses: it
walks the store's ``**/*.md`` (skipping ``.index/``), compares
``mtime`` against ``documents``, and re-embeds only changed files.

Query-time correctness does not depend on the index existing — a
missing or stale index degrades to zero hits, and the deterministic
people-lookup tier is still the correctness floor.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from familiar_connect.context.providers.content_search.retrieval import (
    RetrievedChunk,
    chunk_markdown,
)
from familiar_connect.memory.store import MemoryStoreError

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.context.providers.content_search.retrieval import (
        EmbeddingModel,
    )
    from familiar_connect.memory.store import MemoryStore


_logger = logging.getLogger(__name__)

_INDEX_DIR = ".index"
_GITIGNORE_FILENAME = ".gitignore"
_GITIGNORE_CONTENT = "# regenerable cache — see docs/architecture/memory.md\n.index/\n"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    rel_path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    chunk_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_path TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_DIM_KEY = "dim"
_BATCH_SIZE = 32


@dataclass
class BuildStats:
    """Summary returned by ``build_if_stale`` for logging / tests."""

    rebuilt_paths: list[str] = field(default_factory=list)
    removed_paths: list[str] = field(default_factory=list)
    chunks_written: int = 0


class EmbeddingIndex:
    """SQLite-backed vector store over a familiar's memory directory."""

    def __init__(self, path: Path, *, dim: int) -> None:
        self._path = path
        self._dim = dim
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_gitignore()
        # check_same_thread=False: the async retriever runs queries
        # via asyncio.to_thread, and the background indexer also runs
        # off the main loop. Our usage is serialised (builds happen at
        # startup, queries afterwards; a future concurrent-writer
        # design should add an explicit lock).
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._verify_or_stamp_dim(dim)
        # Lazy-loaded vector matrix for query time.
        self._cache_vectors: np.ndarray | None = None
        self._cache_meta: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_if_stale(
        self,
        store: MemoryStore,
        model: EmbeddingModel,
    ) -> BuildStats:
        """Re-embed any file whose ``mtime`` has changed; drop vanished files."""
        current = _scan_markdown(store)
        stored = self._stored_mtimes()

        to_remove = [p for p in stored if p not in current]
        to_rebuild = [
            p for p, mt in current.items() if p not in stored or stored[p] != mt
        ]

        stats = BuildStats()

        for rel_path in to_remove:
            self._remove_document(rel_path)
            stats.removed_paths.append(rel_path)

        for rel_path in to_rebuild:
            try:
                text = store.read_file(rel_path)
            except MemoryStoreError:
                # file vanished between scan and read — treat as remove
                self._remove_document(rel_path)
                stats.removed_paths.append(rel_path)
                continue
            written = self._reindex_file(
                rel_path=rel_path,
                mtime=current[rel_path],
                text=text,
                model=model,
            )
            stats.rebuilt_paths.append(rel_path)
            stats.chunks_written += written

        if to_remove or to_rebuild:
            self._invalidate_cache()
        return stats

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return top-K chunks by cosine similarity.

        *query_vector* must be L2-normalised and match the index
        dimension. Returns ``[]`` for an empty index or empty query.
        """
        if top_k <= 0:
            return []
        vectors, meta = self._load_cache()
        if vectors.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            msg = f"query vector dim {q.shape[0]} != index dim {self._dim}"
            raise ValueError(msg)
        scores = vectors @ q
        k = min(top_k, scores.shape[0])
        # argpartition is O(n) vs O(n log n) for argsort, worthwhile once
        # corpora get into the thousands.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out: list[RetrievedChunk] = []
        for i in top_idx:
            rel_path, heading_path, text = meta[i]
            out.append(
                RetrievedChunk(
                    rel_path=rel_path,
                    heading_path=heading_path,
                    text=text,
                    score=float(scores[i]),
                )
            )
        return out

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_gitignore(self) -> None:
        gitignore = self._path.parent / _GITIGNORE_FILENAME
        if gitignore.exists():
            return
        gitignore.write_text(_GITIGNORE_CONTENT, encoding="utf-8")

    def _verify_or_stamp_dim(self, dim: int) -> None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_DIM_KEY,)
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                (_DIM_KEY, str(dim)),
            )
            self._conn.commit()
            return
        stored = int(row[0])
        if stored != dim:
            msg = (
                f"index at {self._path} was built with dim={stored}, "
                f"but caller provided dim={dim}. Delete the index and rebuild."
            )
            raise ValueError(msg)

    def _stored_mtimes(self) -> dict[str, float]:
        return {
            row[0]: row[1]
            for row in self._conn.execute("SELECT rel_path, mtime FROM documents")
        }

    def _remove_document(self, rel_path: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE rel_path = ?", (rel_path,))
            self._conn.execute("DELETE FROM documents WHERE rel_path = ?", (rel_path,))

    def _reindex_file(
        self,
        *,
        rel_path: str,
        mtime: float,
        text: str,
        model: EmbeddingModel,
    ) -> int:
        chunks = chunk_markdown(text, rel_path=rel_path, mtime=mtime)
        if not chunks:
            # still record the document so we don't re-read next cycle
            with self._conn:
                self._conn.execute("DELETE FROM chunks WHERE rel_path = ?", (rel_path,))
                self._conn.execute(
                    "DELETE FROM documents WHERE rel_path = ?", (rel_path,)
                )
                self._conn.execute(
                    "INSERT INTO documents (rel_path, mtime, chunk_count) "
                    "VALUES (?, ?, 0)",
                    (rel_path, mtime),
                )
            return 0

        # embed in batches to amortise ONNX session overhead
        vectors_parts: list[np.ndarray] = []
        for i in range(0, len(chunks), _BATCH_SIZE):
            batch_texts = [c.text for c in chunks[i : i + _BATCH_SIZE]]
            vectors_parts.append(model.embed(batch_texts))
        vectors = np.concatenate(vectors_parts, axis=0).astype(np.float32)
        # guard: L2-normalise here too in case the model didn't
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        vectors /= norms
        if vectors.shape != (len(chunks), self._dim):
            msg = (
                f"embedding shape {vectors.shape} != expected "
                f"({len(chunks)}, {self._dim})"
            )
            raise ValueError(msg)

        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE rel_path = ?", (rel_path,))
            self._conn.execute("DELETE FROM documents WHERE rel_path = ?", (rel_path,))
            self._conn.execute(
                "INSERT INTO documents (rel_path, mtime, chunk_count) VALUES (?, ?, ?)",
                (rel_path, mtime, len(chunks)),
            )
            self._conn.executemany(
                "INSERT INTO chunks ("
                "rel_path, heading_path, char_start, char_end, "
                "token_count, text, vector) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.rel_path,
                        c.heading_path,
                        c.char_start,
                        c.char_end,
                        c.token_count,
                        c.text,
                        vectors[i].tobytes(),
                    )
                    for i, c in enumerate(chunks)
                ],
            )
        return len(chunks)

    def _invalidate_cache(self) -> None:
        self._cache_vectors = None
        self._cache_meta = []

    def _load_cache(self) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
        if self._cache_vectors is not None:
            return self._cache_vectors, self._cache_meta
        with closing(
            self._conn.execute(
                "SELECT rel_path, heading_path, text, vector FROM chunks ORDER BY id"
            )
        ) as cur:
            rows = cur.fetchall()
        if not rows:
            self._cache_vectors = np.zeros((0, self._dim), dtype=np.float32)
            self._cache_meta = []
            return self._cache_vectors, self._cache_meta
        meta = [(r[0], r[1], r[2]) for r in rows]
        vectors = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        if vectors.shape[1] != self._dim:
            msg = f"stored vector dim {vectors.shape[1]} != index dim {self._dim}"
            raise ValueError(msg)
        self._cache_vectors = vectors
        self._cache_meta = meta
        return self._cache_vectors, self._cache_meta


# ---------------------------------------------------------------------------
# Store scan
# ---------------------------------------------------------------------------


def _scan_markdown(store: MemoryStore) -> dict[str, float]:
    """Return ``{rel_path: mtime}`` for every ``*.md`` outside .index/."""
    try:
        matches = store.glob("**/*.md")
    except MemoryStoreError:
        return {}
    out: dict[str, float] = {}
    for rel in matches:
        if rel.startswith(_INDEX_DIR + "/") or rel == _INDEX_DIR:
            continue
        try:
            stat = (store.root / rel).stat()
        except OSError:
            continue
        out[rel] = stat.st_mtime
    return out
