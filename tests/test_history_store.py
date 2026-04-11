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

from datetime import datetime
from typing import TYPE_CHECKING

from familiar_connect.config import ChannelMode
from familiar_connect.history.store import (
    HistoryStore,
    HistoryTurn,
    SummaryEntry,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


_CHANNEL = 200
_FAMILIAR = "aria"


def _store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "history.db")


def _seed(store: HistoryStore, n: int) -> list[HistoryTurn]:
    """Append *n* alternating user/assistant turns and return them."""
    out: list[HistoryTurn] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        speaker = "Alice" if role == "user" else None
        out.append(
            store.append_turn(
                channel_id=_CHANNEL,
                familiar_id=_FAMILIAR,
                role=role,
                content=f"turn {i}",
                speaker=speaker,
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
            speaker="Alice",
        )
        assert isinstance(turn, HistoryTurn)
        assert turn.id > 0
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.speaker == "Alice"
        assert isinstance(turn.timestamp, datetime)

    def test_assistant_turn_has_no_speaker(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turn = s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="assistant",
            content="hi back",
        )
        assert turn.speaker is None

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
            mode=ChannelMode.full_rp,
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
            mode=ChannelMode.full_rp,
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="voice turn",
            mode=ChannelMode.imitate_voice,
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="another rp",
            mode=ChannelMode.full_rp,
        )

        rp_turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
            mode=ChannelMode.full_rp,
        )
        assert [t.content for t in rp_turns] == ["rp turn", "another rp"]

        voice_turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
            mode=ChannelMode.imitate_voice,
        )
        assert [t.content for t in voice_turns] == ["voice turn"]

    def test_migration_adds_mode_column_to_existing_db(self, tmp_path: Path) -> None:
        """Opening a DB created before the mode column migrates it."""
        import sqlite3  # noqa: PLC0415

        db_path = tmp_path / "legacy.db"
        # Create a DB with the old schema (no mode column).
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE turns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id TEXT NOT NULL,
                channel_id  INTEGER NOT NULL,
                guild_id    INTEGER,
                role        TEXT NOT NULL,
                speaker     TEXT,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO turns (familiar_id, channel_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (_FAMILIAR, _CHANNEL, "user", "old turn", "2025-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        # Re-open with HistoryStore — migration should add the mode column.
        s = HistoryStore(db_path)
        turns = s.recent(channel_id=_CHANNEL, familiar_id=_FAMILIAR, limit=10)
        assert len(turns) == 1
        assert turns[0].content == "old turn"
        # New turns can use mode.
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="new turn",
            mode=ChannelMode.full_rp,
        )
        row = s._conn.execute(
            "SELECT mode FROM turns WHERE content = 'new turn'"
        ).fetchone()
        assert row["mode"] == "full_rp"

    def test_recent_without_mode_returns_all(self, tmp_path: Path) -> None:
        """Without a mode filter, recent() returns all turns."""
        s = _store(tmp_path)
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="rp",
            mode=ChannelMode.full_rp,
        )
        s.append_turn(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="voice",
            mode=ChannelMode.imitate_voice,
        )
        turns = s.recent(
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
        )
        assert len(turns) == 2


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
            mode=ChannelMode.full_rp,
        )
        s.append_turn(
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
            mode=ChannelMode.text_conversation_rp,
        )
        s.append_turn(
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="c",
            mode=ChannelMode.imitate_voice,
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
            mode=ChannelMode.full_rp,
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
