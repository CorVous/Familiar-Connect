"""Tantivy-backed full-text index for turns and facts.

Lives outside Turso so we get FTS today (pyturso 0.5.1 wheels don't
ship the FTS module) and so FTS reads don't queue behind SQL writes.
One :class:`FtsIndex` instance per indexed table; on-disk index dir
under ``data/familiars/<id>/fts/<name>/``, or in-memory when ``path``
is ``None`` (tests).

Analyzer matches the previous SQLite FTS5 ``unicode61
remove_diacritics 2`` plus the Python-side stopword pass:

* simple tokenizer (whitespace + punctuation split)
* lowercase
* ascii_fold (combining marks → base char; ``café`` ⇒ ``cafe``)
* custom_stopword(_FTS_STOPWORDS) — small high-confidence English list

Searcher returns ``[(row_id, bm25_score)]``; caller joins back to the
relational table.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from threading import RLock

import tantivy

PathLike = str | Path


# Drop common English stopwords before FTS matching. Without this,
# casual chat cues like "hey do you know about X" dilute BM25 scoring
# and produce noisy hits on conversational filler. Same list as the
# pre-Turso FTS5 path used; kept small and high-confidence.
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
    # stemmer covers the old `fox*` prefix-match trick for plurals
    # (fox/foxes, bear/bears, etc.) without dragging in unrelated
    # prefixes (foxhound). remove_long caps token length so URLs and
    # garbage tokens can't bloat the index — matches the spirit of
    # unicode61's 64-char default.
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

    Thread-safe — tantivy's writer/searcher are thread-safe, plus a
    short critical section guards the writer handle. Use one instance
    per indexed entity (``turns``, ``facts``).

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
        # one persistent writer per index — cheaper than open/close per write
        self._writer = self._index.writer(heap_size=15_000_000, num_threads=1)

    def add(self, row_id: int, content: str) -> None:
        """Index one document. Commits immediately so reads see it."""
        with self._lock:
            doc = tantivy.Document()
            doc.add_integer("row_id", int(row_id))
            doc.add_text("content", content)
            # treat add as upsert — delete any prior doc with same row_id
            self._writer.delete_documents("row_id", int(row_id))
            self._writer.add_document(doc)
            self._writer.commit()
        self._index.reload()

    def add_many(self, rows: list[tuple[int, str]]) -> None:
        """Bulk-index. One commit for the batch; cheaper for migrations."""
        if not rows:
            return
        with self._lock:
            for row_id, content in rows:
                self._writer.delete_documents("row_id", int(row_id))
                doc = tantivy.Document()
                doc.add_integer("row_id", int(row_id))
                doc.add_text("content", content)
                self._writer.add_document(doc)
            self._writer.commit()
        self._index.reload()

    def delete(self, row_id: int) -> None:
        with self._lock:
            self._writer.delete_documents("row_id", int(row_id))
            self._writer.commit()
        self._index.reload()

    def clear(self) -> None:
        """Drop every document. Used by rebuild paths."""
        with self._lock:
            self._writer.delete_all_documents()
            self._writer.commit()
        self._index.reload()

    def search(self, query: str, *, limit: int) -> list[tuple[int, float]]:
        """Return ``[(row_id, score)]`` for the query.

        Empty / stopword-only query returns ``[]`` — tantivy's analyzer
        strips stopwords, so the parsed query has zero terms in that
        case. Operator semantics: disjunctive by default (OR), so
        multi-token cues match on any substantive token (parity with
        the old ``_build_fts_match`` behaviour).
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
        """Cheap check — true when the index has zero documents.

        Used by :class:`HistoryStore` to detect first-run after the
        SQLite→Turso migration and trigger a bulk reindex.
        """
        searcher = self._index.searcher()
        return searcher.num_docs == 0

    def close(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._writer.commit()
        self._writer.wait_merging_threads()
