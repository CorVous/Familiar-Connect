"""Tests for platform-message-id + reply linkage + turn_mentions.

The bot tracks Discord-side message ids alongside our internal turn
ids so it can:

- thread its replies back to the originating message
- resolve who got pinged in a given turn
- render reply markers in the prompt's recent history
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author

if TYPE_CHECKING:
    from pathlib import Path


def _author(user_id: str, name: str = "Cass") -> Author:
    return Author(
        platform="discord",
        user_id=user_id,
        username=name.lower(),
        display_name=name,
    )


class TestPlatformMessageIdOnTurns:
    def test_append_turn_persists_platform_message_id(self) -> None:
        store = HistoryStore(":memory:")
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hi",
            author=_author("111"),
            platform_message_id="9999000111",
        )
        # Round-trip via store helper
        looked = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="9999000111"
        )
        assert looked is not None
        assert looked.id == turn.id

    def test_append_turn_persists_reply_to_message_id(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="parent",
            author=_author("111"),
            platform_message_id="aaa",
        )
        store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="child reply",
            author=_author("222", "Bob"),
            platform_message_id="bbb",
            reply_to_message_id="aaa",
        )
        # The child turn carries the parent's platform message id.
        row = store._conn.execute(
            "SELECT platform_message_id, reply_to_message_id "
            "FROM turns WHERE platform_message_id = ?",
            ("bbb",),
        ).fetchone()
        assert row["reply_to_message_id"] == "aaa"

    def test_lookup_turn_by_platform_message_id_returns_none_when_missing(
        self,
    ) -> None:
        store = HistoryStore(":memory:")
        assert (
            store.lookup_turn_by_platform_message_id(
                familiar_id="fam", platform_message_id="ghost"
            )
            is None
        )

    def test_lookup_scoped_to_familiar(self) -> None:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="famA",
            channel_id=1,
            role="user",
            content="A",
            author=_author("111"),
            platform_message_id="shared",
        )
        store.append_turn(
            familiar_id="famB",
            channel_id=1,
            role="user",
            content="B",
            author=_author("111"),
            platform_message_id="shared",
        )
        a = store.lookup_turn_by_platform_message_id(
            familiar_id="famA", platform_message_id="shared"
        )
        b = store.lookup_turn_by_platform_message_id(
            familiar_id="famB", platform_message_id="shared"
        )
        assert a is not None
        assert b is not None
        assert a.id != b.id


class TestTurnMentions:
    def test_record_mentions_for_turn(self) -> None:
        store = HistoryStore(":memory:")
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hey @aria",
            author=_author("111"),
        )
        store.record_mentions(
            turn_id=turn.id,
            canonical_keys=["discord:222", "discord:333"],
        )
        mentions = store.mentions_for_turn(turn_id=turn.id)
        assert mentions == ("discord:222", "discord:333")

    def test_record_mentions_dedupes(self) -> None:
        store = HistoryStore(":memory:")
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="hey",
            author=_author("111"),
        )
        store.record_mentions(
            turn_id=turn.id,
            canonical_keys=["discord:222", "discord:222"],
        )
        mentions = store.mentions_for_turn(turn_id=turn.id)
        assert mentions == ("discord:222",)

    def test_mentions_for_turn_empty_when_none_recorded(self) -> None:
        store = HistoryStore(":memory:")
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="no pings",
            author=_author("111"),
        )
        assert store.mentions_for_turn(turn_id=turn.id) == ()

    def test_record_mentions_empty_list_is_noop(self) -> None:
        store = HistoryStore(":memory:")
        turn = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="no pings",
            author=_author("111"),
        )
        store.record_mentions(turn_id=turn.id, canonical_keys=[])
        assert store.mentions_for_turn(turn_id=turn.id) == ()


class TestMigrations:
    def test_migration_adds_columns_to_existing_turns_table(
        self, tmp_path: Path
    ) -> None:
        """Older DB without platform_message_id / reply_to_message_id."""
        db_path = tmp_path / "history.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id TEXT NOT NULL,
                channel_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            INSERT INTO turns (familiar_id, channel_id, role, content, timestamp)
            VALUES ('fam', 1, 'user', 'old', '2026-04-26T00:00:00+00:00');
            """
        )
        conn.commit()
        conn.close()

        store = HistoryStore(db_path)
        # New columns + helpers exist; lookup of legacy turn returns None
        # (no platform_message_id stored on it).
        assert (
            store.lookup_turn_by_platform_message_id(
                familiar_id="fam", platform_message_id="anything"
            )
            is None
        )
        # New writes work.
        new = store.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="user",
            content="new",
            author=_author("111"),
            platform_message_id="new-msg-id",
        )
        looked = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="new-msg-id"
        )
        assert looked is not None
        assert looked.id == new.id
