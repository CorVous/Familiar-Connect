"""Tests for facts table + FTS over facts."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
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


class TestFactBiTemporal:
    """Bi-temporal validity: ``valid_from`` / ``valid_to`` (M1 — world-time).

    ``superseded_at`` captures *when we recorded the change* (system-
    time). ``valid_from`` / ``valid_to`` capture *when the fact applied
    in the world* (world-time). Default reads stay "current truth";
    ``as_of=...`` unlocks audit queries against world-time.
    """

    def test_append_defaults_valid_from_to_now_when_no_source_turns(self) -> None:
        store = HistoryStore(":memory:")
        before = datetime.now(tz=UTC)
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="A fact.",
            source_turn_ids=[],
        )
        after = datetime.now(tz=UTC)
        assert fact.valid_from is not None
        assert before <= fact.valid_from <= after
        assert fact.valid_to is None

    def test_append_with_explicit_valid_from(self) -> None:
        store = HistoryStore(":memory:")
        when = datetime(2025, 6, 1, tzinfo=UTC)
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria moved to Berlin.",
            source_turn_ids=[1],
            valid_from=when,
        )
        assert fact.valid_from == when
        assert fact.valid_to is None
        # And on re-read.
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].valid_from == when

    def test_append_with_explicit_valid_to(self) -> None:
        store = HistoryStore(":memory:")
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme in 2025.",
            source_turn_ids=[1],
            valid_from=start,
            valid_to=end,
        )
        assert fact.valid_from == start
        assert fact.valid_to == end

    def test_recent_facts_default_excludes_expired(self) -> None:
        """A fact whose ``valid_to`` is in the past is not "current truth"."""
        store = HistoryStore(":memory:")
        past = datetime.now(tz=UTC) - timedelta(days=30)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            valid_from=past - timedelta(days=365),
            valid_to=past,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            valid_from=past,
        )
        current = store.recent_facts(familiar_id="fam", limit=10)
        texts = {f.text for f in current}
        assert texts == {"Aria works at Globex."}

    def test_as_of_returns_world_time_slice(self) -> None:
        store = HistoryStore(":memory:")
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        t3 = datetime(2025, 9, 1, tzinfo=UTC)
        # Fact applied Jan-Jun.
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            valid_from=t1,
            valid_to=t2,
        )
        # Fact applies from Jun onward.
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            valid_from=t2,
        )
        # As of March: only Acme.
        march = datetime(2025, 3, 1, tzinfo=UTC)
        slice_march = store.recent_facts(familiar_id="fam", limit=10, as_of=march)
        assert {f.text for f in slice_march} == {"Aria worked at Acme."}
        # As of September: only Globex.
        slice_sept = store.recent_facts(familiar_id="fam", limit=10, as_of=t3)
        assert {f.text for f in slice_sept} == {"Aria works at Globex."}

    def test_as_of_includes_superseded_for_audit(self) -> None:
        """``as_of`` is an audit query — superseded rows participate."""
        store = HistoryStore(":memory:")
        old_time = datetime(2025, 1, 1, tzinfo=UTC)
        old = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes strawberries.",
            source_turn_ids=[1],
            valid_from=old_time,
        )
        new = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to strawberries.",
            source_turn_ids=[2],
            valid_from=datetime(2025, 6, 1, tzinfo=UTC),
        )
        store.supersede_fact(familiar_id="fam", old_id=old.id, new_id=new.id)
        # As of February the world believed Aria liked strawberries.
        feb = datetime(2025, 2, 1, tzinfo=UTC)
        slice_feb = store.recent_facts(familiar_id="fam", limit=10, as_of=feb)
        texts = {f.text for f in slice_feb}
        assert "Aria likes strawberries." in texts

    def test_search_facts_supports_as_of(self) -> None:
        store = HistoryStore(":memory:")
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            valid_from=t1,
            valid_to=t2,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            valid_from=t2,
        )
        march = datetime(2025, 3, 1, tzinfo=UTC)
        found = store.search_facts(
            familiar_id="fam", query="Aria", limit=10, as_of=march
        )
        assert {f.text for f in found} == {"Aria worked at Acme."}

    def test_facts_for_subject_supports_as_of(self) -> None:
        store = HistoryStore(":memory:")
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        subject = (FactSubject(canonical_key="discord:1", display_at_write="Aria"),)
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria worked at Acme.",
            source_turn_ids=[1],
            subjects=subject,
            valid_from=t1,
            valid_to=t2,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria works at Globex.",
            source_turn_ids=[2],
            subjects=subject,
            valid_from=t2,
        )
        march = datetime(2025, 3, 1, tzinfo=UTC)
        slice_march = store.facts_for_subject(
            familiar_id="fam", canonical_key="discord:1", as_of=march
        )
        assert {f.text for f in slice_march} == {"Aria worked at Acme."}

    def test_migration_adds_validity_columns(self, tmp_path: Path) -> None:
        """Pre-existing ``facts`` tables get ``valid_from``/``valid_to`` added.

        Pre-existing rows return ``valid_from = None`` and ``valid_to =
        None`` — no backfill, the feature is forward-only. Default
        reads still surface them (NULL ``valid_to`` ⇒ still applies).
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
                superseded_by     INTEGER,
                subjects_json     TEXT
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
        assert facts[0].valid_from is None
        assert facts[0].valid_to is None


class TestFactImportance:
    """Importance score (M2): 1-10 hint for retrieval ranking.

    Persisted alongside each fact; consumed by ``RagContextLayer``.
    Optional — legacy rows and extractions that omit it read back as
    ``None`` and are treated as the neutral midpoint at rank time.
    """

    def test_append_with_importance_roundtrip(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria is allergic to peanuts.",
            source_turn_ids=[1],
            importance=9,
        )
        assert fact.importance == 9
        recents = store.recent_facts(familiar_id="fam", limit=10)
        assert recents[0].importance == 9

    def test_importance_defaults_none(self) -> None:
        store = HistoryStore(":memory:")
        fact = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Casual aside.",
            source_turn_ids=[1],
        )
        assert fact.importance is None

    def test_importance_clamped_to_range(self) -> None:
        """Out-of-range values clamp to [1, 10] so the LLM can't poison ranks."""
        store = HistoryStore(":memory:")
        low = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="too low.",
            source_turn_ids=[1],
            importance=0,
        )
        high = store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="too high.",
            source_turn_ids=[2],
            importance=42,
        )
        assert low.importance == 1
        assert high.importance == 10

    def test_search_facts_scored_returns_bm25(self) -> None:
        store = HistoryStore(":memory:")
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes strawberries.",
            source_turn_ids=[1],
            importance=8,
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Boris likes strawberries too.",
            source_turn_ids=[2],
            importance=2,
        )
        scored = store.search_facts_scored(familiar_id="fam", query="strawb", limit=5)
        assert len(scored) == 2
        # bm25 is a real number; both rows should score (FTS5 returns negative
        # numbers, lower = better — exposed verbatim so callers can rerank).
        for fact, score in scored:
            assert isinstance(fact, Fact)
            assert isinstance(score, float)
        # Order matches default search_facts (BM25 ascending = best first).
        assert scored[0][1] <= scored[1][1]

    def test_migration_adds_importance_column(self, tmp_path: Path) -> None:
        """Pre-existing ``facts`` tables get ``importance`` added.

        Legacy rows return ``importance=None``; no backfill — forward-only.
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
                superseded_by     INTEGER,
                subjects_json     TEXT,
                valid_from        TEXT,
                valid_to          TEXT
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
        assert facts[0].importance is None
