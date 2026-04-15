"""Retriever protocol + markdown chunking + fastembed-backed retriever.

``chunk_markdown`` turns one Markdown file into retrieval-ready
``PreparedChunk``s — per-H2 sections with the H1 + heading
denormalized into the embedding input, small files as a single
whole-file chunk, oversized sections via sliding windows.

``EmbeddingRetriever`` embeds the utterance, queries an
``EmbeddingIndex`` by cosine similarity, and returns the top-K as
``RetrievedChunk``s.

``FastEmbedModel`` is the default production ``EmbeddingModel``
adapter; tests inject stubs so CI never downloads ONNX weights.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from familiar_connect.context.providers.content_search.index.embeddings import (
        EmbeddingIndex,
    )


_logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
"""384-dim, ~68 MB quantized ONNX. See plan doc for rationale."""

DEFAULT_EMBEDDING_DIM = 384
"""Dimension of DEFAULT_EMBEDDING_MODEL. Tests use a smaller stub dim."""

SMALL_FILE_WORD_THRESHOLD = 300
"""Below this word count, the file becomes a single chunk — no H2 split."""

HARD_CHUNK_TOKEN_CAP = 1200
"""Any single chunk at or below this token count is kept as-is. Above it,
sliding-window fragmentation kicks in."""

_SLIDING_WINDOW_TOKENS = 800
_SLIDING_STRIDE_TOKENS = 600
_CHARS_PER_TOKEN = 4  # matches context.budget.estimate_tokens convention

_H1_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2_SPLIT = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedChunk:
    """One chunk ready for indexing.

    ``text`` is the string fed to the embedder — heading path
    prepended to the section body. ``char_start``/``char_end`` refer
    to the original file text, not to ``text``.
    """

    rel_path: str
    heading_path: str
    text: str
    char_start: int
    char_end: int
    token_count: int
    mtime: float


@dataclass(frozen=True)
class RetrievedChunk:
    """One result from ``Retriever.query``."""

    rel_path: str
    heading_path: str
    text: str
    score: float


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EmbeddingModel(Protocol):
    """Produces unit-length float32 embeddings.

    Implementations: ``FastEmbedModel`` (production) or a test stub.
    """

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # shape (n, dim)
        """Return an L2-normalised ``(len(texts), dim)`` float32 array."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Returns top-K chunks relevant to an utterance."""

    async def query(
        self,
        utterance: str,
        top_k: int,
        *,
        exclude_paths: set[str] | None = None,
    ) -> list[RetrievedChunk]: ...


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_markdown(
    text: str,
    *,
    rel_path: str,
    mtime: float,
) -> list[PreparedChunk]:
    """Split *text* into retrieval-ready chunks.

    Rules (see plan doc for rationale):

    - empty / whitespace-only → no chunks
    - word count < SMALL_FILE_WORD_THRESHOLD → one whole-file chunk
    - otherwise → per-H2 section, H1 + heading prepended to the
      embedding input; preamble before first H2 is its own chunk
    - any section over HARD_CHUNK_TOKEN_CAP tokens → sliding windows
      of window/stride tokens with overlap
    """
    if not text.strip():
        return []

    h1_title = _extract_h1(text)

    # small files bypass all the structure logic
    if len(text.split()) < SMALL_FILE_WORD_THRESHOLD:
        return [
            _make_chunk(
                rel_path=rel_path,
                heading_path=h1_title or "",
                body=text,
                char_start=0,
                char_end=len(text),
                mtime=mtime,
                prepend_heading=False,
            )
        ]

    sections = _split_sections(text, h1_title)
    chunks: list[PreparedChunk] = []
    for section in sections:
        chunks.extend(_chunk_section(section, rel_path=rel_path, mtime=mtime))
    return chunks


# ---------------------------------------------------------------------------
# Chunking internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Section:
    heading_path: str
    body: str
    char_start: int
    char_end: int


def _extract_h1(text: str) -> str:
    m = _H1_PATTERN.search(text)
    return m.group(1).strip() if m else ""


def _split_sections(text: str, h1_title: str) -> list[_Section]:
    """Split *text* into per-H2 sections.

    The preamble before the first H2 is its own section tagged with
    just the H1 (or empty if there's no H1).
    """
    sections: list[_Section] = []
    matches = list(_H2_SPLIT.finditer(text))
    if not matches:
        return [
            _Section(
                heading_path=h1_title,
                body=text,
                char_start=0,
                char_end=len(text),
            )
        ]

    # preamble before the first ## heading — skip if it's only the H1 line
    preamble_end = matches[0].start()
    preamble_text = text[:preamble_end]
    if _H1_PATTERN.sub("", preamble_text, count=1).strip():
        sections.append(
            _Section(
                heading_path=h1_title,
                body=preamble_text,
                char_start=0,
                char_end=preamble_end,
            )
        )

    # one section per H2
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading_path = f"{h1_title} > {heading}" if h1_title else heading
        # include the ## line itself so the chunk round-trips cleanly
        # as a standalone markdown fragment
        sections.append(
            _Section(
                heading_path=heading_path,
                body=text[m.start() : body_end],
                char_start=m.start(),
                char_end=body_end,
            )
        )
    return sections


def _chunk_section(
    section: _Section,
    *,
    rel_path: str,
    mtime: float,
) -> list[PreparedChunk]:
    token_count = _estimate_tokens(section.body)
    if token_count <= HARD_CHUNK_TOKEN_CAP:
        return [
            _make_chunk(
                rel_path=rel_path,
                heading_path=section.heading_path,
                body=section.body,
                char_start=section.char_start,
                char_end=section.char_end,
                mtime=mtime,
                prepend_heading=True,
            )
        ]
    # sliding window — in *char* space but sized from token estimate
    window_chars = _SLIDING_WINDOW_TOKENS * _CHARS_PER_TOKEN
    stride_chars = _SLIDING_STRIDE_TOKENS * _CHARS_PER_TOKEN
    body = section.body
    out: list[PreparedChunk] = []
    offset = 0
    while offset < len(body):
        end = min(offset + window_chars, len(body))
        piece = body[offset:end]
        out.append(
            _make_chunk(
                rel_path=rel_path,
                heading_path=section.heading_path,
                body=piece,
                char_start=section.char_start + offset,
                char_end=section.char_start + end,
                mtime=mtime,
                prepend_heading=True,
            )
        )
        if end >= len(body):
            break
        offset += stride_chars
    return out


def _make_chunk(
    *,
    rel_path: str,
    heading_path: str,
    body: str,
    char_start: int,
    char_end: int,
    mtime: float,
    prepend_heading: bool,
) -> PreparedChunk:
    text = f"{heading_path}\n\n{body}" if prepend_heading and heading_path else body
    return PreparedChunk(
        rel_path=rel_path,
        heading_path=heading_path,
        text=text,
        char_start=char_start,
        char_end=char_end,
        token_count=_estimate_tokens(text),
        mtime=mtime,
    )


def _estimate_tokens(text: str) -> int:
    # matches familiar_connect.context.budget.estimate_tokens
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# EmbeddingRetriever
# ---------------------------------------------------------------------------


class EmbeddingRetriever:
    """Retriever backed by an ``EmbeddingIndex`` + ``EmbeddingModel``."""

    def __init__(
        self,
        *,
        index: EmbeddingIndex,
        model: EmbeddingModel,
    ) -> None:
        self._index = index
        self._model = model

    async def query(
        self,
        utterance: str,
        top_k: int,
        *,
        exclude_paths: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not utterance.strip():
            return []
        # Embedding + dot product are CPU-bound — offload so the
        # pipeline's event loop isn't blocked.
        vector = await asyncio.to_thread(self._embed_query, utterance)
        results = await asyncio.to_thread(
            self._index.query, vector, top_k if not exclude_paths else top_k * 2
        )
        if exclude_paths:
            results = [r for r in results if r.rel_path not in exclude_paths]
        return results[:top_k]

    def _embed_query(self, utterance: str) -> np.ndarray:
        arr = self._model.embed([utterance])
        return np.asarray(arr[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# FastEmbed adapter (production)
# ---------------------------------------------------------------------------


class FastEmbedModel:
    """``EmbeddingModel`` backed by ``fastembed.TextEmbedding``.

    Lazy-imports fastembed so unit tests that only use stubs don't
    pay the onnxruntime startup cost.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        # Deferred import — heavy module, many transitive deps.
        from fastembed import TextEmbedding  # noqa: PLC0415

        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = np.stack(list(self._model.embed(texts))).astype(np.float32)
        # bge models already output near-unit vectors, but normalise
        # defensively so dot product == cosine for any swapped model.
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr / norms
