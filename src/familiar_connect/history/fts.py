"""Tantivy-backed full-text index for turns and facts.

Outside Turso — pyturso 0.5.1 wheels lack FTS, and keeping FTS
external means reads don't queue behind SQL writes. One
:class:`FtsIndex` per indexed table; on-disk under
``data/familiars/<id>/fts/<name>/``, in-memory when ``path`` is
``None`` (tests).

Analyzer matches prior SQLite FTS5 ``unicode61 remove_diacritics 2``
plus Python-side stopword pass:

* simple tokenizer (whitespace + punctuation split)
* lowercase
* ascii_fold (combining marks → base char; ``café`` ⇒ ``cafe``)
* custom_stopword(_FTS_STOPWORDS) — small high-confidence English list

Searcher returns ``[(row_id, bm25_score)]``; caller joins back to
relational table.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from threading import RLock

import tantivy

from familiar_connect import log_style as ls

PathLike = str | Path

_logger = logging.getLogger(__name__)

# tantivy `_writer.commit()` occasionally hits Windows file locks held
# briefly by Defender/AV on freshly-written segment files; back off and
# retry rather than letting one .term collision tear down the bot.
# Delays are seconds; total worst-case wait ≈ 0.75s before re-raise.
_COMMIT_RETRY_DELAYS: tuple[float, ...] = (0.05, 0.2, 0.5)

# tantivy raises a plain ValueError whose message wraps a Rust `IoError`
# of `PermissionDenied`. match on the substring rather than the exact
# format (Rust's Debug repr can drift between tantivy versions).
_LOCK_SIGNATURES: tuple[str, ...] = (
    "PermissionDenied",
    "Access is denied",
    "os error 5",
)


def _is_transient_lock_error(exc: ValueError) -> bool:
    msg = str(exc)
    return any(sig in msg for sig in _LOCK_SIGNATURES)


# Drop common English stopwords before FTS matching. without this,
# chat cues like "hey do you know about X" dilute BM25 and produce
# noisy hits on filler. same list as pre-Turso FTS5 path.
_FTS_STOPWORDS: tuple[str, ...] = (
    "a",
    "about",
    "an",
    "and",
    "any",
    "anything",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hey",
    "hi",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "know",
    "lol",
    "me",
    "my",
    "no",
    "not",
    "of",
    "ok",
    "on",
    "or",
    "our",
    "out",
    "she",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "yes",
    "you",
    "your",
    "yours",
)

_ANALYZER_NAME = "familiar_en"


def _build_analyzer() -> tantivy.TextAnalyzer:
    # Stemmer covers old `fox*` prefix-match trick for plurals
    # (fox/foxes, bear/bears) without dragging in unrelated prefixes
    # (foxhound). remove_long caps token length so URLs + garbage
    # can't bloat index — matches unicode61's 64-char default.
    return (
        tantivy
        .TextAnalyzerBuilder(tantivy.Tokenizer.simple())
        .filter(tantivy.Filter.remove_long(64))
        .filter(tantivy.Filter.lowercase())
        .filter(tantivy.Filter.ascii_fold())
        .filter(tantivy.Filter.custom_stopword(list(_FTS_STOPWORDS)))
        .filter(tantivy.Filter.stemmer("english"))
        .build()
    )


def _build_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_integer_field("row_id", stored=True, indexed=True, fast=True)
    sb.add_text_field("content", stored=False, tokenizer_name=_ANALYZER_NAME)
    return sb.build()


class FtsIndex:
    """Tantivy index over (row_id, content) for one relational table.

    Thread-safe — tantivy writer/searcher are thread-safe; short
    critical section guards the writer handle. One instance per
    indexed entity (``turns``, ``facts``).

    ``path=None`` → in-memory (tests with :memory: store).
    """

    def __init__(self, path: PathLike | None) -> None:
        schema = _build_schema()
        if path is None:
            self._path: Path | None = None
            self._index = tantivy.Index(schema)
        else:
            self._path = Path(path)
            self._path.mkdir(parents=True, exist_ok=True)
            self._index = tantivy.Index(schema, path=str(self._path), reuse=True)
        self._index.register_tokenizer(_ANALYZER_NAME, _build_analyzer())
        self._lock = RLock()
        # One persistent writer per index — cheaper than open/close per write.
        self._writer = self._index.writer(heap_size=15_000_000, num_threads=1)

    def _commit_writer(self) -> None:
        """Direct ``_writer.commit()`` — extracted so :meth:`_commit` can retry."""
        self._writer.commit()

    def _commit(self) -> None:
        """Commit writer batch; retry transient Windows AV file locks.

        Tantivy on Windows surfaces antivirus segment-scan races as
        ``ValueError: ... PermissionDenied ...`` from
        ``_writer.commit()``. Back off briefly and retry; only Lock-
        shaped errors retry, everything else raises immediately.
        """
        for attempt, delay in enumerate(_COMMIT_RETRY_DELAYS):
            try:
                self._commit_writer()
            except ValueError as exc:
                if not _is_transient_lock_error(exc):
                    raise
                _logger.warning(
                    f"{ls.tag('FTS', ls.Y)} "
                    f"{ls.kv('commit_retry', str(attempt + 1), vc=ls.LY)} "
                    f"{ls.kv('delay_s', f'{delay:.2f}', vc=ls.LY)} "
                    f"{ls.kv('err', ls.trunc(str(exc), 120), vc=ls.LY)}"
                )
                time.sleep(delay)
            else:
                return
        # Final attempt — let any error propagate to caller.
        self._commit_writer()

    def add(self, row_id: int, content: str) -> None:
        """Index one document; commits immediately so reads see it."""
        with self._lock:
            doc = tantivy.Document()
            doc.add_integer("row_id", int(row_id))
            doc.add_text("content", content)
            # Upsert — delete any prior doc with same row_id.
            self._writer.delete_documents("row_id", int(row_id))
            self._writer.add_document(doc)
            self._commit()
        self._index.reload()

    def add_many(self, rows: list[tuple[int, str]]) -> None:
        """Bulk-index; one commit for the batch — cheaper for migrations."""
        if not rows:
            return
        with self._lock:
            for row_id, content in rows:
                self._writer.delete_documents("row_id", int(row_id))
                doc = tantivy.Document()
                doc.add_integer("row_id", int(row_id))
                doc.add_text("content", content)
                self._writer.add_document(doc)
            self._commit()
        self._index.reload()

    def delete(self, row_id: int) -> None:
        with self._lock:
            self._writer.delete_documents("row_id", int(row_id))
            self._commit()
        self._index.reload()

    def clear(self) -> None:
        """Drop every document. Used by rebuild paths."""
        with self._lock:
            self._writer.delete_all_documents()
            self._commit()
        self._index.reload()

    def search(self, query: str, *, limit: int) -> list[tuple[int, float]]:
        """Return ``[(row_id, score)]`` for the query.

        Empty / stopword-only query returns ``[]`` — tantivy strips
        stopwords, leaving zero parsed terms. Operator semantics:
        disjunctive by default (OR), so multi-token cues match on any
        substantive token (parity with old ``_build_fts_match``).
        """
        if limit <= 0 or not query or not query.strip():
            return []
        # tantivy raises ValueError on syntax/field errors — treat as empty
        try:
            parsed = self._index.parse_query(query, default_field_names=["content"])
        except ValueError:
            return []
        searcher = self._index.searcher()
        hits = searcher.search(parsed, limit=limit).hits
        out: list[tuple[int, float]] = []
        for score, addr in hits:
            doc = searcher.doc(addr)
            row_id = int(doc["row_id"][0])
            out.append((row_id, float(score)))
        return out

    def is_empty(self) -> bool:
        """Return True when index has zero documents.

        :class:`HistoryStore` uses this to detect first-run after the
        SQLite→Turso migration and trigger a bulk reindex.
        """
        searcher = self._index.searcher()
        return searcher.num_docs == 0

    def close(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._writer.commit()
        self._writer.wait_merging_threads()
