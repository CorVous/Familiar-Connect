"""Red-first tests for the SQLite-backed HistoryStore.

The HistoryStore is the persistent record of every conversational
turn the bot sees, plus a per-``familiar_id`` cache of rolling
summaries built from older turns by a cheap side-model.
HistoryProvider reads from it; the bot's text-session and voice-
session loops write to it.

Familiar-Connect runs exactly one familiar per install — see
``docs/architecture/configuration-model.md``. ``familiar_id`` still
rides through the API so tests can exercise multiple familiars
against a single store; in production the bot always passes the one
currently-active character id.

Covers familiar_connect.history.store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import (
    ActivityRecord,
    ChannelUnread,
    HistoryStore,
    HistoryTurn,
    SummaryEntry,
    WatermarkEntry,
)
from familiar_connect.identity import Author

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_CHANNEL = 200
_FAMILIAR = "aria"

_ALICE = Author(platform="discord", user_id="1", username="alice", display_name="Alice")


def _store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "history.db")


def _seed(store: HistoryStore, n: int) -> list[HistoryTurn]:
    """Append *n* alternating user/assistant turns and return them."""
    out: list[HistoryTurn] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        author = _ALICE if role == "user" else None
        out.append(
            store.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role=role,
                content=f"turn {i}",
                author=author,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Construction & schema
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_database_file(self, tmp_path: Path) -> None:
        path = tmp_path / "history.db"
        assert not path.exists()
        HistoryStore(path)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        HistoryStore(str(tmp_path / "history.db"))

    def test_creates_intermediate_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "data" / "familiars" / "aria" / "history.db"
        HistoryStore(nested)
        assert nested.exists()

    def test_in_memory_database_for_tests(self) -> None:
        """Passing ``:memory:`` gives an ephemeral DB."""
        s = HistoryStore(":memory:")
        s.append_turn(
            channel_id=1,
            familiar_id="x",
            role="user",
            content="hello",
        )
        assert s.count(familiar_id="x", channel_id=1) == 1


# ---------------------------------------------------------------------------
# append_turn
# ---------------------------------------------------------------------------


class TestAppendTurn:
    def test_returns_history_turn_with_id(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="hello",
            author=_ALICE,
        )
        assert isinstance(turn, HistoryTurn)
        assert turn.id > 0
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.author == _ALICE
        assert isinstance(turn.timestamp, datetime)

    def test_author_round_trips_through_select(self, tmp_path: Path) -> None:
        """Author fields are persisted via the four author_* columns."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="hello",
            author=_ALICE,
        )
        [turn] = s.recent(channel_id=_CHANNEL, familiar_id=_FAMILIAR, limit=1)
        assert turn.author is not None
        assert turn.author.canonical_key == "discord:1"
        assert turn.author.display_name == "Alice"
        assert turn.author.username == "alice"

    def test_assistant_turn_has_no_author(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="assistant",
            content="hi back",
        )
        assert turn.author is None

    def test_optional_guild_id_observability(self, tmp_path: Path) -> None:
        """guild_id is observability-only — accepted but never partitioning."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="with guild",
            guild_id=999,
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="without guild",
        )
        # Both turns are in the same channel partition regardless of guild.
        turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
        )
        assert [t.content for t in turns] == ["with guild", "without guild"]

    def test_ids_are_monotonically_increasing(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 5)
        ids = [t.id for t in turns]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)

    def test_persistent_across_reopens(self, tmp_path: Path) -> None:
        path = tmp_path / "history.db"
        s = HistoryStore(path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="persisted",
        )
        s.close()

        reopened = HistoryStore(path)
        turns = reopened.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
        )
        assert len(turns) == 1
        assert turns[0].content == "persisted"


# ---------------------------------------------------------------------------
# recent
# ---------------------------------------------------------------------------


class TestRecent:
    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert (
            s.recent(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                limit=10,
            )
            == []
        )

    def test_returns_chronological_order_oldest_first(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 5)
        turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
        )
        assert [t.content for t in turns] == [
            "turn 0",
            "turn 1",
            "turn 2",
            "turn 3",
            "turn 4",
        ]

    def test_limit_returns_only_the_latest_n(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 5)
        turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=3,
        )
        # Last 3, in chronological order.
        assert [t.content for t in turns] == ["turn 2", "turn 3", "turn 4"]

    def test_isolated_per_channel(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch1",
        )
        s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch2",
        )
        ch1 = s.recent(channel_id=200, familiar_id=_FAMILIAR, limit=10)
        ch2 = s.recent(channel_id=300, familiar_id=_FAMILIAR, limit=10)
        assert [t.content for t in ch1] == ["ch1"]
        assert [t.content for t in ch2] == ["ch2"]

    def test_isolated_per_familiar(self, tmp_path: Path) -> None:
        """A single store may host multiple familiars; each sees only its own turns."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id="aria",
            role="user",
            content="for-aria",
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id="bob",
            role="user",
            content="for-bob",
        )
        aria = s.recent(
            channel_id=_CHANNEL,
            familiar_id="aria",
            limit=10,
        )
        assert [t.content for t in aria] == ["for-aria"]


# ---------------------------------------------------------------------------
# recent_distinct_authors (per channel — powers the user-fetch step)
# ---------------------------------------------------------------------------


_BOB = Author(platform="discord", user_id="2", username="bob", display_name="Bob")
_CAROL = Author(platform="discord", user_id="3", username="carol", display_name="Carol")
_DAVE = Author(platform="discord", user_id="4", username="dave", display_name="Dave")


class TestRecentDistinctAuthors:
    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert (
            s.recent_distinct_authors(
                familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=5
            )
            == []
        )

    def test_most_recent_first_distinct(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        for author in (_ALICE, _BOB, _ALICE, _CAROL):
            s.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role="user",
                content="x",
                author=author,
            )
        authors = s.recent_distinct_authors(
            familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=5
        )
        # most-recent unique first; Alice dedupes to her latest slot
        assert [a.canonical_key for a in authors] == [
            _CAROL.canonical_key,
            _ALICE.canonical_key,
            _BOB.canonical_key,
        ]

    def test_respects_limit(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        for author in (_ALICE, _BOB, _CAROL, _DAVE):
            s.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role="user",
                content="x",
                author=author,
            )
        authors = s.recent_distinct_authors(
            familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=2
        )
        assert [a.canonical_key for a in authors] == [
            _DAVE.canonical_key,
            _CAROL.canonical_key,
        ]

    def test_skips_assistant_turns(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="hi",
            author=_ALICE,
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="assistant",
            content="hi back",
        )
        authors = s.recent_distinct_authors(
            familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=5
        )
        assert [a.canonical_key for a in authors] == [_ALICE.canonical_key]

    def test_scoped_per_channel(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="x",
            author=_ALICE,
        )
        s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="x",
            author=_BOB,
        )
        ch200 = s.recent_distinct_authors(
            familiar_id=_FAMILIAR, channel_id=200, limit=5
        )
        ch300 = s.recent_distinct_authors(
            familiar_id=_FAMILIAR, channel_id=300, limit=5
        )
        assert [a.canonical_key for a in ch200] == [_ALICE.canonical_key]
        assert [a.canonical_key for a in ch300] == [_BOB.canonical_key]

    def test_zero_limit_returns_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="x",
            author=_ALICE,
        )
        assert (
            s.recent_distinct_authors(
                familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=0
            )
            == []
        )


# ---------------------------------------------------------------------------
# older_than (global per familiar — not partitioned by channel)
# ---------------------------------------------------------------------------


class TestOlderThan:
    def test_returns_turns_with_id_at_or_below(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 5)
        cut_id = turns[2].id  # the third turn

        older = s.older_than(
            familiar_id=_FAMILIAR,
            max_id=cut_id,
        )
        assert [t.content for t in older] == ["turn 0", "turn 1", "turn 2"]

    def test_empty_when_max_id_below_everything(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 3)
        older = s.older_than(
            familiar_id=_FAMILIAR,
            max_id=0,
        )
        assert older == []

    def test_global_across_channels(self, tmp_path: Path) -> None:
        """older_than is per familiar, not per channel."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch200",
        )
        s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch300",
        )
        older = s.older_than(
            familiar_id=_FAMILIAR,
            max_id=10,
        )
        contents = [t.content for t in older]
        assert "ch200" in contents
        assert "ch300" in contents


# ---------------------------------------------------------------------------
# latest_id (global per familiar — used by HistoryProvider as the watermark)
# ---------------------------------------------------------------------------


class TestLatestId:
    def test_none_when_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.latest_id(familiar_id=_FAMILIAR) is None

    def test_returns_max_id_globally(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 5)
        latest = s.latest_id(familiar_id=_FAMILIAR)
        assert latest is not None
        assert latest > 0

    def test_global_across_channels(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="a",
        )
        last = s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
        )
        latest = s.latest_id(familiar_id=_FAMILIAR)
        assert latest == last.id


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    def test_zero_when_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.count(familiar_id=_FAMILIAR, channel_id=_CHANNEL) == 0

    def test_grows_with_appends(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 7)
        assert s.count(familiar_id=_FAMILIAR, channel_id=_CHANNEL) == 7

    def test_global_when_channel_id_omitted(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="a",
        )
        s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
        )
        assert s.count(familiar_id=_FAMILIAR) == 2
        assert s.count(familiar_id=_FAMILIAR, channel_id=200) == 1


# ---------------------------------------------------------------------------
# Summary cache (global per familiar — not partitioned by channel)
# ---------------------------------------------------------------------------


class TestSummaryCache:
    def test_get_summary_missing_returns_none(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.get_summary(familiar_id=_FAMILIAR) is None

    def test_put_then_get_round_trip(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            familiar_id=_FAMILIAR,
            last_summarised_id=42,
            summary_text="they argued about ska",
        )
        entry = s.get_summary(familiar_id=_FAMILIAR)
        assert isinstance(entry, SummaryEntry)
        assert entry.last_summarised_id == 42
        assert entry.summary_text == "they argued about ska"
        assert isinstance(entry.created_at, datetime)

    def test_put_summary_overwrites_existing(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            familiar_id=_FAMILIAR,
            last_summarised_id=10,
            summary_text="old",
        )
        s.put_summary(
            familiar_id=_FAMILIAR,
            last_summarised_id=15,
            summary_text="new",
        )
        entry = s.get_summary(familiar_id=_FAMILIAR)
        assert entry is not None
        assert entry.last_summarised_id == 15
        assert entry.summary_text == "new"

    def test_summary_isolated_per_familiar(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            familiar_id="aria",
            last_summarised_id=5,
            summary_text="aria summary",
        )
        assert s.get_summary(familiar_id="bob") is None

    def test_summary_persists_across_reopens(self, tmp_path: Path) -> None:
        path = tmp_path / "history.db"
        s = HistoryStore(path)
        s.put_summary(
            familiar_id=_FAMILIAR,
            last_summarised_id=99,
            summary_text="persisted",
        )
        s.close()

        reopened = HistoryStore(path)
        entry = reopened.get_summary(familiar_id=_FAMILIAR)
        assert entry is not None
        assert entry.last_summarised_id == 99
        assert entry.summary_text == "persisted"

    def test_put_get_summary_roundtrips_last_consumed_at(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            familiar_id=_FAMILIAR,
            last_summarised_id=7,
            summary_text="x",
            last_consumed_at="2026-06-13T10:00:00+00:00",
        )
        entry = s.get_summary(familiar_id=_FAMILIAR)
        assert entry is not None
        assert entry.last_consumed_at == "2026-06-13T10:00:00+00:00"

    def test_get_summary_old_row_has_null_last_consumed_at(
        self, tmp_path: Path
    ) -> None:
        s = _store(tmp_path)
        s.put_summary(
            familiar_id=_FAMILIAR,
            last_summarised_id=3,
            summary_text="legacy",
        )
        entry = s.get_summary(familiar_id=_FAMILIAR)
        assert entry is not None
        assert entry.last_consumed_at is None


# ---------------------------------------------------------------------------
# Consumed cross-channel stream (focus-stream summary source)
# ---------------------------------------------------------------------------


class TestConsumedTurnsAfter:
    def _old(self, n: int) -> datetime:
        """Deterministic past timestamp, n seconds before a fixed epoch."""
        return datetime(2026, 6, 13, 9, 0, 0, tzinfo=UTC) + timedelta(seconds=n)

    def test_empty_cursor_returns_all_consumed(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 5)
        out = s.consumed_turns_after(
            familiar_id=_FAMILIAR,
            after_consumed_at="",
            after_id=0,
            limit=10,
        )
        assert [t.id for t in out] == [1, 2, 3, 4, 5]

    def test_respects_limit(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 10)
        out = s.consumed_turns_after(
            familiar_id=_FAMILIAR,
            after_consumed_at="",
            after_id=0,
            limit=3,
        )
        assert [t.id for t in out] == [1, 2, 3]

    def test_composite_cursor_advances_to_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 4)
        last = turns[-1]
        assert last.consumed_at is not None
        out = s.consumed_turns_after(
            familiar_id=_FAMILIAR,
            after_consumed_at=last.consumed_at.isoformat(),
            after_id=last.id,
            limit=10,
        )
        assert out == []

    def test_excludes_staged(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="staged",
            consumed=False,
        )
        out = s.consumed_turns_after(
            familiar_id=_FAMILIAR,
            after_consumed_at="",
            after_id=0,
            limit=10,
        )
        assert out == []

    def test_includes_late_promoted_low_id(self, tmp_path: Path) -> None:
        """Old-id staged turn promoted after watermark is NOT skipped.

        The headline correctness guarantee: id-based watermark would miss
        it; consumed_at-based does not.
        """
        s = _store(tmp_path)
        # id=1: staged turn, old arrived_at, consumed_at NULL for now
        s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="dormant-channel msg",
            consumed=False,
            arrived_at=self._old(0),
        )
        # id=2,3: consumed turns in the focused channel
        b = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
        )
        c = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="assistant",
            content="c",
        )
        assert c.consumed_at is not None
        # watermark sits past c (higher id, but its consumed_at is "now")
        watermark = (c.consumed_at.isoformat(), c.id)
        # focus shifts to channel 300 -> promote staged turn (consumed_at=NOW)
        promoted = s.promote_staged_turns(familiar_id=_FAMILIAR, channel_id=300)
        assert promoted == 1
        out = s.consumed_turns_after(
            familiar_id=_FAMILIAR,
            after_consumed_at=watermark[0],
            after_id=watermark[1],
            limit=10,
        )
        assert [t.id for t in out] == [1]  # low id, but newly consumed
        _ = b  # appended for id sequencing


# ---------------------------------------------------------------------------
# Migration backfill scope
# ---------------------------------------------------------------------------


class TestMigrationBackfillScope:
    def test_staged_turn_stays_staged_after_reopen(self, tmp_path: Path) -> None:
        """``consumed_at`` backfill is one-time, not every open.

        A deliberately-staged turn (focus model) must survive a restart;
        the legacy ``NULL -> consumed`` backfill must not re-promote it.
        """
        path = tmp_path / "history.db"
        s = HistoryStore(path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="staged",
            consumed=False,
        )
        assert turn.consumed_at is None
        s.close()

        reopened = HistoryStore(path)
        row = reopened._conn.execute(
            "SELECT consumed_at FROM turns WHERE id = ?", (turn.id,)
        ).fetchone()
        assert row["consumed_at"] is None  # still staged, not promoted


# ---------------------------------------------------------------------------
# Mode column on turns
# ---------------------------------------------------------------------------


class TestModeColumn:
    def test_append_turn_persists_mode(self, tmp_path: Path) -> None:
        """Mode should be stored as a string column on each turn row."""
        s = _store(tmp_path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="hello",
            mode="full_rp",
        )
        # Verify via raw SQL that the mode column exists and has the right value.
        row = s._conn.execute(
            "SELECT mode FROM turns WHERE id = ?", (turn.id,)
        ).fetchone()
        assert row is not None
        assert row["mode"] == "full_rp"

    def test_append_turn_without_mode_stores_null(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="legacy",
        )
        row = s._conn.execute(
            "SELECT mode FROM turns WHERE id = ?", (turn.id,)
        ).fetchone()
        assert row is not None
        assert row["mode"] is None

    def test_recent_filters_by_mode(self, tmp_path: Path) -> None:
        """When mode is passed to recent(), only matching turns are returned."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="rp turn",
            mode="full_rp",
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="voice turn",
            mode="imitate_voice",
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="another rp",
            mode="full_rp",
        )

        rp_turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
            mode="full_rp",
        )
        assert [t.content for t in rp_turns] == ["rp turn", "another rp"]

        voice_turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
            mode="imitate_voice",
        )
        assert [t.content for t in voice_turns] == ["voice turn"]

    def test_recent_without_mode_returns_all(self, tmp_path: Path) -> None:
        """Without a mode filter, recent() returns all turns."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="rp",
            mode="full_rp",
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="voice",
            mode="imitate_voice",
        )
        turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
        )
        assert len(turns) == 2

    def test_migration_idempotent_when_alter_fails_with_no_such_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Migration tolerates Turso reporting "no such table" on ALTER.

        Observed in the wild on pyturso 0.5.1 (Windows): ``sqlite_master``
        + ``PRAGMA table_info`` agree the table exists, but ``ALTER TABLE
        … ADD COLUMN`` raises ``Parse error: no such table``. The
        migration must swallow that and let ``_SCHEMA`` create / repair
        the table on the same init.
        """
        import turso  # noqa: PLC0415

        from familiar_connect.history.store import HistoryStore  # noqa: PLC0415

        db_path = tmp_path / "history.db"

        # Hand-build the legacy state directly: ``turns`` exists, and
        # ``accounts`` exists without the new ``pronouns`` / ``bio``
        # columns. This is what the migration is designed to upgrade.
        c = turso.connect(str(db_path), experimental_features="index_method")
        c.executescript(
            """
            CREATE TABLE turns (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id            TEXT    NOT NULL,
                channel_id             INTEGER NOT NULL,
                guild_id               INTEGER,
                role                   TEXT    NOT NULL,
                author_platform        TEXT,
                author_user_id         TEXT,
                author_username        TEXT,
                author_display_name    TEXT,
                content                TEXT    NOT NULL,
                timestamp              TEXT    NOT NULL,
                mode                   TEXT,
                platform_message_id    TEXT,
                reply_to_message_id    TEXT,
                tool_calls_json        TEXT,
                tool_call_id           TEXT
            );
            CREATE TABLE accounts (
                canonical_key TEXT PRIMARY KEY,
                platform      TEXT NOT NULL,
                user_id       TEXT NOT NULL,
                username      TEXT,
                global_name   TEXT,
                last_seen_at  TEXT NOT NULL
            );
            """
        )
        c.commit()
        c.close()

        # Make every ``ALTER TABLE accounts ADD COLUMN`` blow up with
        # the user-reported parse error, even though ``accounts`` is in
        # ``sqlite_master``.
        real_execute = turso.Cursor.execute

        def fake_execute(
            cursor: turso.Cursor,
            sql: str,
            params: Sequence[Any] | Mapping[str, Any] = (),
        ) -> turso.Cursor:
            if "ALTER TABLE accounts ADD COLUMN" in sql:
                msg = "Parse error: no such table: accounts"
                raise turso.DatabaseError(msg)
            return real_execute(cursor, sql, params)

        monkeypatch.setattr(turso.Cursor, "execute", fake_execute)

        # Re-opening must not raise. The post-migration ``_SCHEMA`` pass
        # leaves the DB in a usable state for the rest of the bot.
        s2 = HistoryStore(db_path)
        s2.close()


# ---------------------------------------------------------------------------
# pings_bot column on turns
# ---------------------------------------------------------------------------


class TestPingsBotColumn:
    def test_append_turn_persists_pings_bot_true(self, tmp_path: Path) -> None:
        """``pings_bot=True`` is stored as ``1`` and set on the return."""
        s = _store(tmp_path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="hey @bot",
            author=_ALICE,
            pings_bot=True,
        )
        assert turn.pings_bot is True
        row = s._conn.execute(
            "SELECT pings_bot FROM turns WHERE id = ?", (turn.id,)
        ).fetchone()
        assert row["pings_bot"] == 1

    def test_pings_bot_round_trips_through_select(self, tmp_path: Path) -> None:
        """A reloaded turn reports ``pings_bot is True`` (cast back to bool)."""
        s = _store(tmp_path)
        appended = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="hey @bot",
            author=_ALICE,
            pings_bot=True,
        )
        reloaded = s.recent(channel_id=_CHANNEL, familiar_id=_FAMILIAR, limit=1)[0]
        assert reloaded.id == appended.id
        assert reloaded.pings_bot is True

    def test_append_turn_defaults_pings_bot_false(self, tmp_path: Path) -> None:
        """Omitting ``pings_bot`` stores ``0`` and reloads as ``False``."""
        s = _store(tmp_path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="hello",
        )
        assert turn.pings_bot is False
        row = s._conn.execute(
            "SELECT pings_bot FROM turns WHERE id = ?", (turn.id,)
        ).fetchone()
        assert row["pings_bot"] == 0
        reloaded = s.recent(channel_id=_CHANNEL, familiar_id=_FAMILIAR, limit=1)[0]
        assert reloaded.pings_bot is False

    def test_legacy_db_gains_pings_bot_column_with_default_zero(
        self, tmp_path: Path
    ) -> None:
        """A pre-``pings_bot`` DB gains the column; legacy rows read ``0``.

        Mirrors the ``arrived_at``/``consumed_at`` migration: hand-build a
        legacy ``turns`` table lacking the column, insert a row, then open
        through :class:`HistoryStore`. The migration adds ``pings_bot``
        with ``DEFAULT 0`` so the legacy row reads ``False``, and a second
        open is a no-op.
        """
        import turso  # noqa: PLC0415

        from familiar_connect.history.store import HistoryStore  # noqa: PLC0415

        db_path = tmp_path / "history.db"
        c = turso.connect(str(db_path), experimental_features="index_method")
        c.executescript(
            """
            CREATE TABLE turns (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id            TEXT    NOT NULL,
                channel_id             INTEGER NOT NULL,
                guild_id               INTEGER,
                role                   TEXT    NOT NULL,
                author_platform        TEXT,
                author_user_id         TEXT,
                author_username        TEXT,
                author_display_name    TEXT,
                content                TEXT    NOT NULL,
                timestamp              TEXT    NOT NULL,
                mode                   TEXT,
                platform_message_id    TEXT,
                reply_to_message_id    TEXT,
                tool_calls_json        TEXT,
                tool_call_id           TEXT
            );
            INSERT INTO turns (familiar_id, channel_id, role, content, timestamp)
            VALUES ('aria', 200, 'user', 'legacy ping', '2026-06-13T09:00:00+00:00');
            """
        )
        c.commit()
        c.close()

        reopened = HistoryStore(db_path)
        row = reopened._conn.execute(
            "SELECT pings_bot FROM turns WHERE id = 1"
        ).fetchone()
        assert row["pings_bot"] == 0
        reloaded = reopened.recent(channel_id=200, familiar_id="aria", limit=1)[0]
        assert reloaded.pings_bot is False
        reopened.close()

        # Second open must not raise (idempotent ALTER).
        again = HistoryStore(db_path)
        again.close()


# ---------------------------------------------------------------------------
# Per-channel summary scope
# ---------------------------------------------------------------------------


class TestPerChannelSummary:
    def test_summary_scoped_to_channel(self, tmp_path: Path) -> None:
        """Two channels can have independent summaries."""
        s = _store(tmp_path)
        s.put_summary(
            familiar_id=_FAMILIAR,
            channel_id=100,
            last_summarised_id=10,
            summary_text="channel 100 summary",
        )
        s.put_summary(
            familiar_id=_FAMILIAR,
            channel_id=200,
            last_summarised_id=20,
            summary_text="channel 200 summary",
        )
        entry_100 = s.get_summary(familiar_id=_FAMILIAR, channel_id=100)
        entry_200 = s.get_summary(familiar_id=_FAMILIAR, channel_id=200)
        assert entry_100 is not None
        assert entry_100.summary_text == "channel 100 summary"
        assert entry_200 is not None
        assert entry_200.summary_text == "channel 200 summary"

    def test_get_summary_missing_channel_returns_none(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            familiar_id=_FAMILIAR,
            channel_id=100,
            last_summarised_id=10,
            summary_text="exists",
        )
        assert s.get_summary(familiar_id=_FAMILIAR, channel_id=999) is None


class TestOlderThanPerChannel:
    def test_older_than_scoped_to_channel(self, tmp_path: Path) -> None:
        """older_than with channel_id only returns turns from that channel."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=100,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch100",
        )
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch200",
        )
        older = s.older_than(
            familiar_id=_FAMILIAR,
            max_id=999,
            channel_id=100,
        )
        assert [t.content for t in older] == ["ch100"]


class TestLatestIdPerChannel:
    def test_latest_id_scoped_to_channel(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=100,
            familiar_id=_FAMILIAR,
            role="user",
            content="a",
        )
        last_200 = s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
        )
        # Global latest should be b
        assert s.latest_id(familiar_id=_FAMILIAR) == last_200.id
        # Channel-scoped latest for 100 should be a (lower id)
        latest_100 = s.latest_id(familiar_id=_FAMILIAR, channel_id=100)
        assert latest_100 is not None
        assert latest_100 < last_200.id


# ---------------------------------------------------------------------------
# Cross-context support: distinct_other_channels + cache
# ---------------------------------------------------------------------------


class TestDistinctOtherChannels:
    def test_returns_other_channels_with_mode(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=100,
            familiar_id=_FAMILIAR,
            role="user",
            content="a",
            mode="full_rp",
        )
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
            mode="text_conversation_rp",
        )
        s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="c",
            mode="imitate_voice",
        )
        others = s.distinct_other_channels(
            familiar_id=_FAMILIAR,
            exclude_channel_id=100,
        )
        channel_ids = {o.channel_id for o in others}
        assert channel_ids == {200, 300}
        # Should carry mode info.
        modes = {o.channel_id: o.mode for o in others}
        assert modes[200] == "text_conversation_rp"
        assert modes[300] == "imitate_voice"

    def test_empty_when_no_other_channels(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=100,
            familiar_id=_FAMILIAR,
            role="user",
            content="only",
            mode="full_rp",
        )
        others = s.distinct_other_channels(
            familiar_id=_FAMILIAR,
            exclude_channel_id=100,
        )
        assert others == []


class TestCrossContextCache:
    def test_put_and_get_round_trip(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_cross_context(
            familiar_id=_FAMILIAR,
            viewer_mode="full_rp",
            source_channel_id=200,
            source_last_id=42,
            summary_text="Meanwhile in chat...",
        )
        entry = s.get_cross_context(
            familiar_id=_FAMILIAR,
            viewer_mode="full_rp",
            source_channel_id=200,
        )
        assert entry is not None
        assert entry.source_last_id == 42
        assert entry.summary_text == "Meanwhile in chat..."

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert (
            s.get_cross_context(
                familiar_id=_FAMILIAR,
                viewer_mode="full_rp",
                source_channel_id=999,
            )
            is None
        )

    def test_upsert_overwrites(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_cross_context(
            familiar_id=_FAMILIAR,
            viewer_mode="full_rp",
            source_channel_id=200,
            source_last_id=10,
            summary_text="old",
        )
        s.put_cross_context(
            familiar_id=_FAMILIAR,
            viewer_mode="full_rp",
            source_channel_id=200,
            source_last_id=20,
            summary_text="new",
        )
        entry = s.get_cross_context(
            familiar_id=_FAMILIAR,
            viewer_mode="full_rp",
            source_channel_id=200,
        )
        assert entry is not None
        assert entry.source_last_id == 20
        assert entry.summary_text == "new"


# ---------------------------------------------------------------------------
# Memory-writer watermark
# ---------------------------------------------------------------------------


class TestMemoryWriterWatermark:
    def test_get_watermark_none_when_unset(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.get_writer_watermark(familiar_id=_FAMILIAR) is None

    def test_put_and_get_watermark(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_writer_watermark(familiar_id=_FAMILIAR, last_written_id=42)
        entry = s.get_writer_watermark(familiar_id=_FAMILIAR)
        assert isinstance(entry, WatermarkEntry)
        assert entry.last_written_id == 42

    def test_put_watermark_upsert(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_writer_watermark(familiar_id=_FAMILIAR, last_written_id=10)
        s.put_writer_watermark(familiar_id=_FAMILIAR, last_written_id=50)
        entry = s.get_writer_watermark(familiar_id=_FAMILIAR)
        assert entry is not None
        assert entry.last_written_id == 50

    def test_turns_since_watermark(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 10)
        # Set watermark at turn 5 — should get turns 6..10
        s.put_writer_watermark(familiar_id=_FAMILIAR, last_written_id=turns[4].id)
        since = s.turns_since_watermark(familiar_id=_FAMILIAR)
        assert len(since) == 5
        assert since[0].id == turns[5].id
        assert since[-1].id == turns[9].id

    def test_turns_since_watermark_no_watermark_returns_all(
        self, tmp_path: Path
    ) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 6)
        since = s.turns_since_watermark(familiar_id=_FAMILIAR)
        assert len(since) == 6
        assert since[0].id == turns[0].id


# ---------------------------------------------------------------------------
# recent — before_id paging
# ---------------------------------------------------------------------------


class TestRecentBeforeId:
    def test_filters_ids_at_or_above(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 5)
        got = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
            before_id=turns[3].id,
        )
        assert [t.content for t in got] == ["turn 0", "turn 1", "turn 2"]

    def test_combines_with_limit(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 5)
        got = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=2,
            before_id=turns[4].id,
        )
        assert [t.content for t in got] == ["turn 2", "turn 3"]

    def test_none_returns_all(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 3)
        got = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
            before_id=None,
        )
        assert len(got) == 3

    def test_combines_with_mode(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="rp early",
            mode="full_rp",
        )
        cut = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="voice",
            mode="imitate_voice",
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="rp late",
            mode="full_rp",
        )
        got = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
            mode="full_rp",
            before_id=cut.id,
        )
        assert [t.content for t in got] == ["rp early"]


# ---------------------------------------------------------------------------
# recent_cross_channel — respect_archive (archive watermark)
# ---------------------------------------------------------------------------


class TestRecentCrossChannelRespectArchive:
    """Archive filter rides the cross-channel window query.

    Semantics: latest N consumed turns, minus archived ones — the
    window shrinks, never backfills past the watermark.
    """

    @staticmethod
    def _turn(s: HistoryStore, *, channel_id: int, content: str) -> HistoryTurn:
        return s.append_turn(
            channel_id=channel_id,
            familiar_id=_FAMILIAR,
            role="user",
            content=content,
            author=_ALICE,
        )

    def test_no_watermark_row_no_filter(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 3)
        got = s.recent_cross_channel(
            familiar_id=_FAMILIAR,
            limit=10,
            respect_archive=True,
        )
        assert len(got) == 3

    def test_hides_turns_at_or_below_watermark(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 5)
        s.set_archive_watermark(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=turns[2].id,
        )
        got = s.recent_cross_channel(
            familiar_id=_FAMILIAR,
            limit=10,
            respect_archive=True,
        )
        assert [t.content for t in got] == ["turn 3", "turn 4"]

    def test_default_ignores_watermark(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 5)
        s.set_archive_watermark(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=turns[4].id,
        )
        got = s.recent_cross_channel(familiar_id=_FAMILIAR, limit=10)
        assert len(got) == 5

    def test_watermark_scoped_per_channel(self, tmp_path: Path) -> None:
        """Watermark on channel A hides only A's pre-watermark turns."""
        s = _store(tmp_path)
        chan_a, chan_b = 100, 101
        a1 = self._turn(s, channel_id=chan_a, content="a before")
        self._turn(s, channel_id=chan_b, content="b before")
        s.set_archive_watermark(familiar_id=_FAMILIAR, channel_id=chan_a, turn_id=a1.id)
        self._turn(s, channel_id=chan_a, content="a after")
        self._turn(s, channel_id=chan_b, content="b after")
        got = s.recent_cross_channel(
            familiar_id=_FAMILIAR,
            limit=10,
            respect_archive=True,
        )
        assert [t.content for t in got] == ["b before", "a after", "b after"]

    def test_window_shrinks_instead_of_backfilling(self, tmp_path: Path) -> None:
        """Latest N minus archived — never latest N unarchived."""
        s = _store(tmp_path)
        turns = _seed(s, 6)
        s.set_archive_watermark(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=turns[3].id,
        )
        # latest 4 = turns 2..5; 2 and 3 archived ⇒ 2 survivors, no
        # backfill from turns 0..1
        got = s.recent_cross_channel(
            familiar_id=_FAMILIAR,
            limit=4,
            respect_archive=True,
        )
        assert [t.content for t in got] == ["turn 4", "turn 5"]


# ---------------------------------------------------------------------------
# promote_staged_turns_since — return-from-absence promotion
# ---------------------------------------------------------------------------


class TestPromoteStagedTurnsSince:
    """Cross-channel promotion scoped above the departure turn id.

    Pre-absence staged turns (id <= after_turn_id) keep their
    attentional semantics; everything staged during the absence is
    consumed at return.
    """

    @staticmethod
    def _stage(s: HistoryStore, *, channel_id: int, content: str) -> HistoryTurn:
        return s.stage_turn(
            channel_id=channel_id,
            familiar_id=_FAMILIAR,
            role="user",
            content=content,
            author=_ALICE,
        )

    def test_promotes_staged_turns_across_channels(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        departure = s.append_turn(
            channel_id=100,
            familiar_id=_FAMILIAR,
            role="user",
            content="before",
        )
        self._stage(s, channel_id=100, content="a during")
        self._stage(s, channel_id=101, content="b during")
        n = s.promote_staged_turns_since(
            familiar_id=_FAMILIAR, after_turn_id=departure.id
        )
        assert n == 2
        assert s.count_staged(familiar_id=_FAMILIAR, channel_id=100) == 0
        assert s.count_staged(familiar_id=_FAMILIAR, channel_id=101) == 0

    def test_leaves_staged_at_or_below_id_untouched(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        pre_absence = self._stage(s, channel_id=100, content="never attended")
        during = self._stage(s, channel_id=101, content="during")
        n = s.promote_staged_turns_since(
            familiar_id=_FAMILIAR, after_turn_id=pre_absence.id
        )
        assert n == 1
        assert s.count_staged(familiar_id=_FAMILIAR, channel_id=100) == 1
        assert s.count_staged(familiar_id=_FAMILIAR, channel_id=101) == 0
        # boundary: turn at exactly after_turn_id stays staged
        n2 = s.promote_staged_turns_since(
            familiar_id=_FAMILIAR, after_turn_id=during.id
        )
        assert n2 == 0

    def test_leaves_consumed_turns_untouched(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        consumed = s.append_turn(
            channel_id=100,
            familiar_id=_FAMILIAR,
            role="user",
            content="already consumed",
        )
        before = s._conn.execute(
            "SELECT consumed_at FROM turns WHERE id = ?",
            (consumed.id,),
        ).fetchone()["consumed_at"]
        n = s.promote_staged_turns_since(familiar_id=_FAMILIAR, after_turn_id=0)
        assert n == 0
        after = s._conn.execute(
            "SELECT consumed_at FROM turns WHERE id = ?",
            (consumed.id,),
        ).fetchone()["consumed_at"]
        assert after == before

    def test_promoted_turns_appear_in_recent_cross_channel(
        self, tmp_path: Path
    ) -> None:
        s = _store(tmp_path)
        departure = s.append_turn(
            channel_id=100,
            familiar_id=_FAMILIAR,
            role="assistant",
            content="departure",
        )
        self._stage(s, channel_id=101, content="chatter during absence")
        got = s.recent_cross_channel(familiar_id=_FAMILIAR, limit=10)
        assert [t.content for t in got] == ["departure"]
        s.promote_staged_turns_since(familiar_id=_FAMILIAR, after_turn_id=departure.id)
        got = s.recent_cross_channel(familiar_id=_FAMILIAR, limit=10)
        assert [t.content for t in got] == ["departure", "chatter during absence"]


# ---------------------------------------------------------------------------
# turns_around
# ---------------------------------------------------------------------------


class TestTurnsAround:
    def test_window_centred_on_anchor(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 11)
        got = s.turns_around(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=turns[5].id,
            before=2,
            after=2,
        )
        assert [t.content for t in got] == [
            "turn 3",
            "turn 4",
            "turn 5",
            "turn 6",
            "turn 7",
        ]

    def test_clips_at_history_edges(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 3)
        got = s.turns_around(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=turns[0].id,
            before=5,
            after=5,
        )
        assert [t.content for t in got] == ["turn 0", "turn 1", "turn 2"]

    def test_defaults_five_each_side(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turns = _seed(s, 20)
        got = s.turns_around(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=turns[10].id,
        )
        assert [t.id for t in got] == [t.id for t in turns[5:16]]

    def test_partitioned_per_channel(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        anchor = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="here",
        )
        s.append_turn(
            channel_id=999,
            familiar_id=_FAMILIAR,
            role="user",
            content="elsewhere",
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="here too",
        )
        got = s.turns_around(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=anchor.id,
        )
        assert [t.content for t in got] == ["here", "here too"]


# ---------------------------------------------------------------------------
# channel_archive_watermark
# ---------------------------------------------------------------------------


class TestArchiveWatermark:
    def test_get_none_when_unset(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert (
            s.get_archive_watermark(familiar_id=_FAMILIAR, channel_id=_CHANNEL) is None
        )

    def test_set_then_get_round_trip(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.set_archive_watermark(familiar_id=_FAMILIAR, channel_id=_CHANNEL, turn_id=42)
        assert s.get_archive_watermark(familiar_id=_FAMILIAR, channel_id=_CHANNEL) == 42

    def test_upsert_overwrites(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.set_archive_watermark(familiar_id=_FAMILIAR, channel_id=_CHANNEL, turn_id=10)
        s.set_archive_watermark(familiar_id=_FAMILIAR, channel_id=_CHANNEL, turn_id=50)
        assert s.get_archive_watermark(familiar_id=_FAMILIAR, channel_id=_CHANNEL) == 50

    def test_scoped_per_channel(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.set_archive_watermark(familiar_id=_FAMILIAR, channel_id=100, turn_id=7)
        assert s.get_archive_watermark(familiar_id=_FAMILIAR, channel_id=200) is None

    def test_scoped_per_familiar(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.set_archive_watermark(familiar_id="aria", channel_id=_CHANNEL, turn_id=7)
        assert s.get_archive_watermark(familiar_id="bob", channel_id=_CHANNEL) is None


# ---------------------------------------------------------------------------
# activities — append-only log; active row = actual_return_at IS NULL
# ---------------------------------------------------------------------------


_T0 = datetime(2026, 6, 12, 10, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(minutes=30)


def _create(s: HistoryStore, *, familiar_id: str = _FAMILIAR) -> int:
    return s.create_activity(
        familiar_id=familiar_id,
        type_id="walk",
        label="on a walk",
        started_at=_T0,
        planned_return_at=_T1,
        note="creek",
    )


class TestActivities:
    def test_create_returns_positive_id(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert _create(s) > 0

    def test_active_none_when_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.active_activity(familiar_id=_FAMILIAR) is None

    def test_create_then_active_round_trip(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        activity_id = _create(s)
        rec = s.active_activity(familiar_id=_FAMILIAR)
        assert isinstance(rec, ActivityRecord)
        assert rec.id == activity_id
        assert rec.familiar_id == _FAMILIAR
        assert rec.type_id == "walk"
        assert rec.label == "on a walk"
        assert rec.started_at == _T0
        assert rec.planned_return_at == _T1
        assert rec.note == "creek"
        assert rec.status is None
        assert rec.actual_return_at is None
        assert rec.experience_text is None

    def test_note_optional(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.create_activity(
            familiar_id=_FAMILIAR,
            type_id="walk",
            label="on a walk",
            started_at=_T0,
            planned_return_at=_T1,
            note=None,
        )
        rec = s.active_activity(familiar_id=_FAMILIAR)
        assert rec is not None
        assert rec.note is None

    def test_finish_clears_active(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        activity_id = _create(s)
        s.finish_activity(
            activity_id=activity_id,
            status="completed",
            actual_return_at=_T1,
            experience_text="saw a heron",
        )
        assert s.active_activity(familiar_id=_FAMILIAR) is None

    def test_finish_persists_fields(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        activity_id = _create(s)
        s.finish_activity(
            activity_id=activity_id,
            status="cut_short",
            actual_return_at=_T1,
            experience_text="got pinged",
        )
        row = s._conn.execute(
            "SELECT status, actual_return_at, experience_text"
            " FROM activities WHERE id = ?",
            (activity_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "cut_short"
        assert row["actual_return_at"] == _T1.isoformat()
        assert row["experience_text"] == "got pinged"

    def test_set_activity_experience_persists_on_active_row(
        self, tmp_path: Path
    ) -> None:
        """Persist prose mid-activity (row stays active — no return stamp)."""
        s = _store(tmp_path)
        activity_id = _create(s)
        s.set_activity_experience(
            activity_id=activity_id, experience_text="a dream of rain"
        )
        rec = s.active_activity(familiar_id=_FAMILIAR)
        assert rec is not None
        assert rec.experience_text == "a dream of rain"

    def test_finish_rejects_bad_status(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        activity_id = _create(s)
        with pytest.raises(ValueError, match="status"):
            s.finish_activity(
                activity_id=activity_id,
                status="abandoned",
                actual_return_at=_T1,
                experience_text=None,
            )

    def test_append_only_history_kept(self, tmp_path: Path) -> None:
        """Finished rows stay; new activity becomes the active one."""
        s = _store(tmp_path)
        first = _create(s)
        s.finish_activity(
            activity_id=first,
            status="completed",
            actual_return_at=_T1,
            experience_text=None,
        )
        second = _create(s)
        rec = s.active_activity(familiar_id=_FAMILIAR)
        assert rec is not None
        assert rec.id == second
        row = s._conn.execute(
            "SELECT COUNT(*) AS n FROM activities WHERE familiar_id = ?",
            (_FAMILIAR,),
        ).fetchone()
        assert int(row["n"]) == 2

    def test_active_isolated_per_familiar(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _create(s, familiar_id="aria")
        assert s.active_activity(familiar_id="bob") is None

    def test_latest_activity_none_when_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.latest_activity(familiar_id=_FAMILIAR, type_id="sleep") is None

    def test_latest_activity_filters_type_and_returns_newest(
        self, tmp_path: Path
    ) -> None:
        s = _store(tmp_path)
        first = _create(s)  # type_id "walk"
        s.finish_activity(
            activity_id=first,
            status="completed",
            actual_return_at=_T1,
            experience_text=None,
        )
        second = _create(s)
        rec = s.latest_activity(familiar_id=_FAMILIAR, type_id="walk")
        assert rec is not None
        assert rec.id == second
        assert s.latest_activity(familiar_id=_FAMILIAR, type_id="sleep") is None

    def test_latest_activity_includes_finished_rows(self, tmp_path: Path) -> None:
        """Window guard checks STARTED, not active — finished rows count."""
        s = _store(tmp_path)
        only = _create(s)
        s.finish_activity(
            activity_id=only,
            status="completed",
            actual_return_at=_T1,
            experience_text=None,
        )
        rec = s.latest_activity(familiar_id=_FAMILIAR, type_id="walk")
        assert rec is not None
        assert rec.id == only
        assert rec.status == "completed"

    def test_latest_activity_isolated_per_familiar(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _create(s, familiar_id="aria")
        assert s.latest_activity(familiar_id="bob", type_id="walk") is None


# ---------------------------------------------------------------------------
# AsyncHistoryStore passthrough for new methods
# ---------------------------------------------------------------------------


class TestAsyncStoreActivityWrappers:
    @pytest.mark.asyncio
    async def test_activity_round_trip_async(self) -> None:
        store = AsyncHistoryStore(HistoryStore(":memory:"))
        activity_id = await store.create_activity(
            familiar_id=_FAMILIAR,
            type_id="walk",
            label="on a walk",
            started_at=_T0,
            planned_return_at=_T1,
            note=None,
        )
        rec = await store.active_activity(familiar_id=_FAMILIAR)
        assert rec is not None
        assert rec.id == activity_id
        await store.finish_activity(
            activity_id=activity_id,
            status="completed",
            actual_return_at=_T1,
            experience_text=None,
        )
        assert await store.active_activity(familiar_id=_FAMILIAR) is None
        store.close()

    @pytest.mark.asyncio
    async def test_archive_watermark_and_windows_async(self) -> None:
        inner = HistoryStore(":memory:")
        store = AsyncHistoryStore(inner)
        turns = [
            inner.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role="user",
                content=f"turn {i}",
            )
            for i in range(5)
        ]
        await store.set_archive_watermark(
            familiar_id=_FAMILIAR, channel_id=_CHANNEL, turn_id=turns[1].id
        )
        assert (
            await store.get_archive_watermark(
                familiar_id=_FAMILIAR, channel_id=_CHANNEL
            )
            == turns[1].id
        )
        got = await store.recent_cross_channel(
            familiar_id=_FAMILIAR,
            limit=10,
            respect_archive=True,
        )
        assert [t.content for t in got] == ["turn 2", "turn 3", "turn 4"]
        window = await store.turns_around(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            turn_id=turns[2].id,
            before=1,
            after=1,
        )
        assert [t.content for t in window] == ["turn 1", "turn 2", "turn 3"]
        store.close()

    @pytest.mark.asyncio
    async def test_promote_staged_turns_since_async(self) -> None:
        inner = HistoryStore(":memory:")
        store = AsyncHistoryStore(inner)
        departure = inner.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="assistant",
            content="departure",
        )
        inner.stage_turn(
            channel_id=_CHANNEL + 1,
            familiar_id=_FAMILIAR,
            role="user",
            content="during",
        )
        n = await store.promote_staged_turns_since(
            familiar_id=_FAMILIAR, after_turn_id=departure.id
        )
        assert n == 1
        store.close()


class TestStagedChannelsPingTally:
    def test_tally_counts_unread_and_ping_subset(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        for i in range(3):
            s.append_turn(
                channel_id=10,
                familiar_id=_FAMILIAR,
                role="user",
                content=f"a{i}",
                consumed=False,
                pings_bot=(i == 0),
            )
        s.append_turn(
            channel_id=20,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
            consumed=False,
            pings_bot=False,
        )
        assert s.staged_channels(familiar_id=_FAMILIAR) == {
            10: ChannelUnread(3, 1),
            20: ChannelUnread(1, 0),
        }
