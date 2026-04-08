"""Red-first tests for the SQLite-backed HistoryStore.

The HistoryStore is the persistent record of every conversational
turn the bot sees, plus a per-(owner_user_id, familiar_id) cache of
rolling summaries built from older turns by a cheap side-model.
HistoryProvider reads from it; the bot's text-session and voice-
session loops will write to it (step 7 of
future-features/context-management.md).

Familiars are owned by Discord users, not guilds — see
``future-features/configuration-levels.md`` for the ownership model.
The recent conversation window is partitioned per channel, but the
rolling summary is global per familiar.

Covers familiar_connect.history.store.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

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


_OWNER = 42
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
                owner_user_id=_OWNER,
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
        nested = tmp_path / "data" / "users" / "1" / "history.db"
        HistoryStore(nested)
        assert nested.exists()

    def test_in_memory_database_for_tests(self) -> None:
        """Passing ``:memory:`` (or None) gives an ephemeral DB."""
        s = HistoryStore(":memory:")
        s.append_turn(
            owner_user_id=1,
            channel_id=1,
            familiar_id="x",
            role="user",
            content="hello",
        )
        assert s.count(owner_user_id=1, familiar_id="x", channel_id=1) == 1


# ---------------------------------------------------------------------------
# append_turn
# ---------------------------------------------------------------------------


class TestAppendTurn:
    def test_returns_history_turn_with_id(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        turn = s.append_turn(
            owner_user_id=_OWNER,
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
            owner_user_id=_OWNER,
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
            owner_user_id=_OWNER,
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="with guild",
            guild_id=999,
        )
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="without guild",
        )
        # Both turns are in the same channel partition regardless of guild.
        turns = s.recent(
            owner_user_id=_OWNER,
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
            owner_user_id=_OWNER,
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="persisted",
        )
        s.close()

        reopened = HistoryStore(path)
        turns = reopened.recent(
            owner_user_id=_OWNER,
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
                owner_user_id=_OWNER,
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
            owner_user_id=_OWNER,
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
            owner_user_id=_OWNER,
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=3,
        )
        # Last 3, in chronological order.
        assert [t.content for t in turns] == ["turn 2", "turn 3", "turn 4"]

    def test_isolated_per_channel(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch1",
        )
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch2",
        )
        ch1 = s.recent(
            owner_user_id=_OWNER, channel_id=200, familiar_id=_FAMILIAR, limit=10
        )
        ch2 = s.recent(
            owner_user_id=_OWNER, channel_id=300, familiar_id=_FAMILIAR, limit=10
        )
        assert [t.content for t in ch1] == ["ch1"]
        assert [t.content for t in ch2] == ["ch2"]

    def test_isolated_per_owner(self, tmp_path: Path) -> None:
        """Two users with same-named familiars never see each other's turns."""
        s = _store(tmp_path)
        s.append_turn(
            owner_user_id=1,
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="for-owner-1",
        )
        s.append_turn(
            owner_user_id=2,
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            role="user",
            content="for-owner-2",
        )
        owner_1 = s.recent(
            owner_user_id=1,
            channel_id=_CHANNEL,
            familiar_id=_FAMILIAR,
            limit=10,
        )
        assert [t.content for t in owner_1] == ["for-owner-1"]

    def test_isolated_per_familiar(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=_CHANNEL,
            familiar_id="aria",
            role="user",
            content="for-aria",
        )
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=_CHANNEL,
            familiar_id="bob",
            role="user",
            content="for-bob",
        )
        aria = s.recent(
            owner_user_id=_OWNER,
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
            owner_user_id=_OWNER,
            familiar_id=_FAMILIAR,
            max_id=cut_id,
        )
        assert [t.content for t in older] == ["turn 0", "turn 1", "turn 2"]

    def test_empty_when_max_id_below_everything(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 3)
        older = s.older_than(
            owner_user_id=_OWNER,
            familiar_id=_FAMILIAR,
            max_id=0,
        )
        assert older == []

    def test_global_across_channels(self, tmp_path: Path) -> None:
        """older_than is per familiar, not per channel."""
        s = _store(tmp_path)
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch200",
        )
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="ch300",
        )
        older = s.older_than(
            owner_user_id=_OWNER,
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
        assert s.latest_id(owner_user_id=_OWNER, familiar_id=_FAMILIAR) is None

    def test_returns_max_id_globally(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 5)
        latest = s.latest_id(owner_user_id=_OWNER, familiar_id=_FAMILIAR)
        assert latest is not None
        assert latest > 0

    def test_global_across_channels(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="a",
        )
        last = s.append_turn(
            owner_user_id=_OWNER,
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
        )
        latest = s.latest_id(owner_user_id=_OWNER, familiar_id=_FAMILIAR)
        assert latest == last.id


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    def test_zero_when_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert (
            s.count(owner_user_id=_OWNER, familiar_id=_FAMILIAR, channel_id=_CHANNEL)
            == 0
        )

    def test_grows_with_appends(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        _seed(s, 7)
        assert (
            s.count(owner_user_id=_OWNER, familiar_id=_FAMILIAR, channel_id=_CHANNEL)
            == 7
        )

    def test_global_when_channel_id_omitted(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=200,
            familiar_id=_FAMILIAR,
            role="user",
            content="a",
        )
        s.append_turn(
            owner_user_id=_OWNER,
            channel_id=300,
            familiar_id=_FAMILIAR,
            role="user",
            content="b",
        )
        assert s.count(owner_user_id=_OWNER, familiar_id=_FAMILIAR) == 2
        assert s.count(owner_user_id=_OWNER, familiar_id=_FAMILIAR, channel_id=200) == 1


# ---------------------------------------------------------------------------
# Summary cache (global per familiar — not partitioned by channel)
# ---------------------------------------------------------------------------


class TestSummaryCache:
    def test_get_summary_missing_returns_none(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        assert s.get_summary(owner_user_id=_OWNER, familiar_id=_FAMILIAR) is None

    def test_put_then_get_round_trip(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            owner_user_id=_OWNER,
            familiar_id=_FAMILIAR,
            last_summarised_id=42,
            summary_text="they argued about ska",
        )
        entry = s.get_summary(owner_user_id=_OWNER, familiar_id=_FAMILIAR)
        assert isinstance(entry, SummaryEntry)
        assert entry.last_summarised_id == 42
        assert entry.summary_text == "they argued about ska"
        assert isinstance(entry.created_at, datetime)

    def test_put_summary_overwrites_existing(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            owner_user_id=_OWNER,
            familiar_id=_FAMILIAR,
            last_summarised_id=10,
            summary_text="old",
        )
        s.put_summary(
            owner_user_id=_OWNER,
            familiar_id=_FAMILIAR,
            last_summarised_id=15,
            summary_text="new",
        )
        entry = s.get_summary(owner_user_id=_OWNER, familiar_id=_FAMILIAR)
        assert entry is not None
        assert entry.last_summarised_id == 15
        assert entry.summary_text == "new"

    def test_summary_isolated_per_owner(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.put_summary(
            owner_user_id=1,
            familiar_id=_FAMILIAR,
            last_summarised_id=5,
            summary_text="owner-1 summary",
        )
        assert s.get_summary(owner_user_id=2, familiar_id=_FAMILIAR) is None

    def test_summary_persists_across_reopens(self, tmp_path: Path) -> None:
        path = tmp_path / "history.db"
        s = HistoryStore(path)
        s.put_summary(
            owner_user_id=_OWNER,
            familiar_id=_FAMILIAR,
            last_summarised_id=99,
            summary_text="persisted",
        )
        s.close()

        reopened = HistoryStore(path)
        entry = reopened.get_summary(owner_user_id=_OWNER, familiar_id=_FAMILIAR)
        assert entry is not None
        assert entry.last_summarised_id == 99
        assert entry.summary_text == "persisted"
