"""Tests for facts table + FTS over facts."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from familiar_connect.history.store import Fact, FactSubject, HistoryStore

if TYPE_CHECKING:
    from pathlib import Path


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


class TestFactSubjects:
    """Subject-key metadata: best-effort link from fact text to canonical_keys.

    The extractor's identification of *who* a fact is about is a soft
    annotation, not authoritative — mic-sharing, relayed quotes, and
    plain ambiguity all break a 1:1 mapping. Storage must preserve
    whatever the extractor emits (zero, one, or many subjects per
    fact) and surface it back unchanged.
    """

    def test_append_with_subjects_roundtrip(self) -> None:
        store = HistoryStore(":memory:")
        subjects = (
            FactSubject(canonical_key="discord:123", display_at_write="Cass"),
            FactSubject(canonical_key="discord:456", display_at_write="Aria"),
        )
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass and Aria are roommates.",
            source_turn_ids=[1],
            subjects=subjects,
        )
        assert fact.subjects == subjects
        # And again on re-read.
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].subjects == subjects

    def test_subjects_default_empty(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="A fact with no identified subjects.",
            source_turn_ids=[1],
        )
        assert fact.subjects == ()

    def test_search_facts_returns_subjects(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Cass likes pho.",
            source_turn_ids=[1],
            subjects=(
                FactSubject(canonical_key="discord:123", display_at_write="Cass"),
            ),
        )
        found = store.search_facts(familiar_id="fam", query="pho", limit=5)
        assert len(found) == 1
        assert found[0].subjects == (
            FactSubject(canonical_key="discord:123", display_at_write="Cass"),
        )

    def test_migration_adds_subjects_column(self, tmp_path: Path) -> None:
        """Existing installs without ``subjects_json`` get the column added.

        Pre-existing rows return ``subjects=()`` — no backfill, the
        feature is forward-only.
        """
        db_path = tmp_path / "history.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE facts (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id       TEXT    NOT NULL,
                channel_id        INTEGER,
                text              TEXT    NOT NULL,
                source_turn_ids   TEXT    NOT NULL,
                created_at        TEXT    NOT NULL,
                superseded_at     TEXT,
                superseded_by     INTEGER
            );
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id TEXT NOT NULL,
                channel_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            INSERT INTO facts (familiar_id, channel_id, text,
                               source_turn_ids, created_at)
            VALUES ('fam', 1, 'Legacy fact.', '[1]', '2026-04-26T00:00:00+00:00');
            """
        )
        conn.commit()
        conn.close()

        store = HistoryStore(db_path)
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].text == "Legacy fact."
        assert facts[0].subjects == ()


class TestFactSupersession:
    """Version chains: facts that go stale are superseded, not overwritten.

    A new fact replaces the old; the old keeps its row with
    ``superseded_at`` set and ``superseded_by`` pointing at the new
    row. Reads default to current (non-superseded), but the history
    is preserved for audit and contradiction inspection.
    """

    def test_new_facts_are_current_by_default(self) -> None:
        store = _store_with_turns_and_facts()
        facts = store.recent_facts(familiar_id="fam", limit=10)
        for f in facts:
            assert f.superseded_at is None
            assert f.superseded_by is None

    def test_supersede_marks_old_and_links_new(self) -> None:
        store = _store_with_turns_and_facts()
        # Replace the strawberry fact with an updated one.
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        # Reload via recent (include superseded so we can inspect the old row)
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        old = next(f for f in all_facts if f.id == 1)
        assert old.superseded_at is not None
        assert old.superseded_by == new.id

    def test_recent_facts_excludes_superseded_by_default(self) -> None:
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        current = store.recent_facts(familiar_id="fam", limit=10)
        texts = {f.text for f in current}
        assert "Aria likes strawberries." not in texts
        assert "Aria is allergic to strawberries now." in texts

    def test_recent_facts_includes_superseded_when_asked(self) -> None:
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        all_facts = store.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        texts = {f.text for f in all_facts}
        assert "Aria likes strawberries." in texts
        assert "Aria is allergic to strawberries now." in texts

    def test_search_facts_excludes_superseded(self) -> None:
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries now.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        found = store.search_facts(familiar_id="fam", query="strawb", limit=10)
        texts = {f.text for f in found}
        assert "Aria likes strawberries." not in texts
        assert "Aria is allergic to strawberries now." in texts

    def test_supersede_idempotent_raises_on_already_superseded(self) -> None:
        """Supersession is one-shot — re-superseding signals a bug upstream."""
        store = _store_with_turns_and_facts()
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="updated.",
            source_turn_ids=[5],
        )
        store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)
        with pytest.raises(ValueError, match="already superseded"):
            store.supersede_fact(familiar_id="fam", old_id=1, new_id=new.id)

    def test_supersede_unknown_id_raises(self) -> None:
        store = _store_with_turns_and_facts()
        with pytest.raises(ValueError, match="unknown fact"):
            store.supersede_fact(familiar_id="fam", old_id=999, new_id=1)

    def test_migration_adds_supersede_columns_to_pre_existing_table(
        self, tmp_path: Path
    ) -> None:
        """Existing installs created ``facts`` without supersede columns.

        Opening a HistoryStore against such a DB must add them
        idempotently — no data loss, defaults to current.
        """
        db_path = tmp_path / "history.db"
        # Hand-build the pre-supersession facts schema.
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE facts (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id       TEXT    NOT NULL,
                channel_id        INTEGER,
                text              TEXT    NOT NULL,
                source_turn_ids   TEXT    NOT NULL,
                created_at        TEXT    NOT NULL
            );
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id TEXT NOT NULL,
                channel_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            INSERT INTO facts (familiar_id, channel_id, text,
                               source_turn_ids, created_at)
            VALUES ('fam', 1, 'Old fact.', '[1]', '2026-04-26T00:00:00+00:00');
            """
        )
        conn.commit()
        conn.close()

        # Re-open via HistoryStore — migration should add the columns.
        store = HistoryStore(db_path)
        facts = store.recent_facts(familiar_id="fam", limit=10)
        assert len(facts) == 1
        assert facts[0].text == "Old fact."
        assert facts[0].superseded_at is None
        assert facts[0].superseded_by is None
