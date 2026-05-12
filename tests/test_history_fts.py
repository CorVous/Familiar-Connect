"""Tests for the ``fts_turns`` FTS5 virtual table on :class:`HistoryStore`.

Side-index over ``turns.content``, rebuildable. Used by
:class:`RagContextLayer` to retrieve relevant historical turns.
"""

from __future__ import annotations

from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author


def _store_with_turns() -> HistoryStore:
    store = HistoryStore(":memory:")
    alice = Author(
        platform="discord", user_id="1", username="alice", display_name="Alice"
    )
    for i, text in enumerate([
        "The fox jumped over the moon.",
        "I like strawberry jam.",
        "The quick brown fox.",
        "Rainy day in Seattle.",
        "Anything with foxes is cool.",
    ]):
        store.append_turn(
            familiar_id="fam",
            channel_id=100 + (i % 2),
            role="user",
            content=text,
            author=alice,
        )
    return store


class TestFtsSearch:
    def test_finds_matching_turns(self) -> None:
        store = _store_with_turns()
        results = store.search_turns(familiar_id="fam", query="fox", limit=10)
        assert results
        assert all("fox" in r.content.lower() for r in results)
        # three "fox" turns above
        assert len(results) == 3

    def test_scopes_to_channel_when_requested(self) -> None:
        store = _store_with_turns()
        results = store.search_turns(
            familiar_id="fam", channel_id=100, query="fox", limit=10
        )
        # channel 100 gets turns 0, 2, 4 — all fox mentions
        assert len(results) == 3
        assert all(r.channel_id == 100 for r in results)

        results_101 = store.search_turns(
            familiar_id="fam", channel_id=101, query="fox", limit=10
        )
        # channel 101 gets turns 1 (jam) and 3 (rainy) — no fox
        assert results_101 == []

    def test_scopes_to_familiar(self) -> None:
        store = _store_with_turns()
        store.append_turn(
            familiar_id="other",
            channel_id=100,
            role="user",
            content="A unicorn appeared on the other familiar.",
            author=None,
        )
        # ``unicorn`` is unique to the other-familiar row; matching
        # tokens are OR'd so any non-stopword would do.
        results = store.search_turns(familiar_id="fam", query="unicorn", limit=10)
        # only the other-familiar row contains "unicorn", and that row
        # is scoped out — so the result must be empty.
        assert results == []

    def test_respects_limit(self) -> None:
        store = _store_with_turns()
        results = store.search_turns(familiar_id="fam", query="fox", limit=1)
        assert len(results) == 1

    def test_deterministic_order(self) -> None:
        """BM25 ranking is deterministic for the same query/data."""
        store = _store_with_turns()
        r1 = store.search_turns(familiar_id="fam", query="fox", limit=10)
        r2 = store.search_turns(familiar_id="fam", query="fox", limit=10)
        assert [r.id for r in r1] == [r.id for r in r2]

    def test_rebuild_from_scratch(self) -> None:
        """FTS index can be dropped and re-populated from ``turns``."""
        store = _store_with_turns()
        # sanity check: index populated
        assert store.search_turns(familiar_id="fam", query="fox", limit=10)
        store.rebuild_fts()
        # still returns the same results after rebuild
        results = store.search_turns(familiar_id="fam", query="fox", limit=10)
        assert len(results) == 3

    def test_empty_query_returns_nothing(self) -> None:
        store = _store_with_turns()
        assert store.search_turns(familiar_id="fam", query="", limit=10) == []

    def test_unknown_term_returns_nothing(self) -> None:
        store = _store_with_turns()
        assert store.search_turns(familiar_id="fam", query="zyzzyx", limit=10) == []

    def test_punctuation_in_query_is_tolerated(self) -> None:
        """FTS5 queries with punctuation should not raise."""
        store = _store_with_turns()
        # Should not raise; may return empty.
        store.search_turns(familiar_id="fam", query="fox?", limit=10)

    def test_latest_indexed_id_tracks_writes(self) -> None:
        store = _store_with_turns()
        latest = store.latest_fts_id(familiar_id="fam")
        assert latest == store.latest_id(familiar_id="fam")

    def test_chat_cue_with_stopwords_still_recalls_substantive_match(self) -> None:
        """A casual chat cue with mostly stopwords still surfaces hits.

        Tokens are OR'd, not AND'd, and English stopwords are dropped
        before matching — otherwise multi-word chat cues return nothing
        (FTS5's implicit-AND semantics destroys recall on free text).
        """
        store = _store_with_turns()
        cue = "Hey, do you know anything about the fox?"
        results = store.search_turns(familiar_id="fam", query=cue, limit=10)
        assert results
        # All "fox" turns should surface — `fox*` is the substantive token.
        assert all("fox" in r.content.lower() for r in results)

    def test_query_of_only_stopwords_returns_nothing(self) -> None:
        """Empty token set after stopword filtering means no FTS query."""
        store = _store_with_turns()
        results = store.search_turns(
            familiar_id="fam", query="hi the a do you", limit=10
        )
        assert results == []

    def test_search_turns_respects_max_id(self) -> None:
        """``max_id`` excludes turns with id > max_id (inclusive bound).

        Used by ``RagContextLayer`` to keep RAG from re-surfacing turns
        that ``RecentHistoryLayer`` already includes verbatim.
        """
        store = _store_with_turns()
        full = store.search_turns(familiar_id="fam", query="fox", limit=10)
        max_id = min(r.id for r in full)
        bounded = store.search_turns(
            familiar_id="fam", query="fox", limit=10, max_id=max_id
        )
        assert bounded
        assert all(r.id <= max_id for r in bounded)
        # bounded is a strict subset of full
        assert len(bounded) < len(full)
