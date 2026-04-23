"""Tests for facts table + FTS over facts."""

from __future__ import annotations

from familiar_connect.history.store import Fact, HistoryStore


def _store_with_turns_and_facts() -> HistoryStore:
    store = HistoryStore(":memory:")
    for i in range(5):
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content=f"turn text {i}",
            author=None,
        )
    store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria likes strawberries.",
        source_turn_ids=[1, 2],
    )
    store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Boris works night shifts on Tuesdays.",
        source_turn_ids=[3, 4],
    )
    return store


class TestFactStore:
    def test_append_returns_fact_with_provenance(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="A fact.",
            source_turn_ids=[7, 9, 11],
        )
        assert isinstance(fact, Fact)
        assert fact.source_turn_ids == (7, 9, 11)
        assert fact.text == "A fact."
        assert fact.channel_id == 1
        assert fact.id > 0

    def test_recent_facts_ordered_newest_first(self) -> None:
        store = _store_with_turns_and_facts()
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert len(recents) == 2
        assert recents[0].text.startswith("Boris")
        assert recents[1].text.startswith("Aria")

    def test_search_facts_finds_by_content(self) -> None:
        store = _store_with_turns_and_facts()
        found = store.search_facts(familiar_id="fam", query="strawb", limit=5)
        # Prefix tokenization makes "strawb" a prefix of "strawberries".
        assert len(found) == 1
        assert "strawberries" in found[0].text

    def test_search_respects_familiar(self) -> None:
        store = _store_with_turns_and_facts()
        store.append_fact(
            familiar_id="other",
            channel_id=1,
            text="Other familiar knows strawberries too.",
            source_turn_ids=[1],
        )
        found = store.search_facts(familiar_id="fam", query="strawb", limit=10)
        # Only the "fam" fact returned
        assert len(found) == 1
        assert found[0].familiar_id == "fam"

    def test_latest_fact_id(self) -> None:
        store = _store_with_turns_and_facts()
        assert store.latest_fact_id(familiar_id="fam") == 2
        assert store.latest_fact_id(familiar_id="nobody") == 0

    def test_empty_query_returns_nothing(self) -> None:
        store = _store_with_turns_and_facts()
        assert store.search_facts(familiar_id="fam", query="", limit=10) == []

    def test_source_turn_ids_roundtrip(self) -> None:
        store = _store_with_turns_and_facts()
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].source_turn_ids == (3, 4)
        assert recents[1].source_turn_ids == (1, 2)
