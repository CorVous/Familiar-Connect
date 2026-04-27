"""Tests for the accounts + per-guild-nicks identity tables.

Discord identity has four facets — id, username, global display
("real name"), per-guild nick. The store keeps a stable identity
table (``accounts``) alongside a per-guild override table
(``account_guild_nicks``); ``resolve_label`` walks them in
preference order. See ``docs/architecture/context-pipeline.md``.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author

if TYPE_CHECKING:
    from pathlib import Path


class TestAccountsUpsert:
    def test_upsert_creates_new_row(self) -> None:
        store = HistoryStore(":memory:")
        author = Author(
            platform="discord",
            user_id="111",
            username="cass_login",
            display_name="Cass",
            global_name="Cassidy",
        )
        store.upsert_account(author)
        row = store._conn.execute(
            "SELECT canonical_key, platform, user_id, username, global_name "
            "FROM accounts WHERE canonical_key = ?",
            ("discord:111",),
        ).fetchone()
        assert row is not None
        assert row["canonical_key"] == "discord:111"
        assert row["platform"] == "discord"
        assert row["user_id"] == "111"
        assert row["username"] == "cass_login"
        assert row["global_name"] == "Cassidy"

    def test_upsert_updates_existing_row(self) -> None:
        store = HistoryStore(":memory:")
        store.upsert_account(
            Author(
                platform="discord",
                user_id="111",
                username="old_login",
                display_name="OldName",
                global_name="OldGlobal",
            )
        )
        store.upsert_account(
            Author(
                platform="discord",
                user_id="111",
                username="new_login",
                display_name="NewName",
                global_name="NewGlobal",
            )
        )
        row = store._conn.execute(
            "SELECT username, global_name FROM accounts WHERE canonical_key = ?",
            ("discord:111",),
        ).fetchone()
        assert row["username"] == "new_login"
        assert row["global_name"] == "NewGlobal"


class TestGuildNicksUpsert:
    def test_upsert_records_per_guild_nick(self) -> None:
        store = HistoryStore(":memory:")
        store.upsert_account(
            Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Aria",
                global_name="Cassidy",
                guild_nick="Aria",
            )
        )
        store.upsert_guild_nick(canonical_key="discord:111", guild_id=42, nick="Aria")
        row = store._conn.execute(
            "SELECT canonical_key, guild_id, nick "
            "FROM account_guild_nicks WHERE canonical_key = ? AND guild_id = ?",
            ("discord:111", 42),
        ).fetchone()
        assert row is not None
        assert row["nick"] == "Aria"

    def test_upsert_overwrites_nick(self) -> None:
        store = HistoryStore(":memory:")
        store.upsert_guild_nick(canonical_key="discord:111", guild_id=42, nick="Aria")
        store.upsert_guild_nick(
            canonical_key="discord:111", guild_id=42, nick="AriaPrime"
        )
        row = store._conn.execute(
            "SELECT nick FROM account_guild_nicks "
            "WHERE canonical_key = ? AND guild_id = ?",
            ("discord:111", 42),
        ).fetchone()
        assert row["nick"] == "AriaPrime"

    def test_upsert_with_null_nick_clears_per_guild_override(self) -> None:
        """A user removing their guild-specific nick should be reflected."""
        store = HistoryStore(":memory:")
        store.upsert_guild_nick(canonical_key="discord:111", guild_id=42, nick="Aria")
        store.upsert_guild_nick(canonical_key="discord:111", guild_id=42, nick=None)
        row = store._conn.execute(
            "SELECT nick FROM account_guild_nicks "
            "WHERE canonical_key = ? AND guild_id = ?",
            ("discord:111", 42),
        ).fetchone()
        assert row is not None
        assert row["nick"] is None


class TestResolveLabel:
    """Preference order: guild nick → global_name → username → user_id."""

    def test_prefers_guild_nick_when_present(self) -> None:
        store = HistoryStore(":memory:")
        store.upsert_account(
            Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Aria",
                global_name="Cassidy",
            )
        )
        store.upsert_guild_nick(canonical_key="discord:111", guild_id=42, nick="Aria")
        assert store.resolve_label(canonical_key="discord:111", guild_id=42) == "Aria"

    def test_falls_back_to_global_name_in_other_guild(self) -> None:
        store = HistoryStore(":memory:")
        store.upsert_account(
            Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Cassidy",
                global_name="Cassidy",
            )
        )
        store.upsert_guild_nick(canonical_key="discord:111", guild_id=42, nick="Aria")
        # Different guild — no per-guild nick row, fall back to global_name
        assert (
            store.resolve_label(canonical_key="discord:111", guild_id=99) == "Cassidy"
        )

    def test_falls_back_to_username_when_no_global_name(self) -> None:
        store = HistoryStore(":memory:")
        store.upsert_account(
            Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="cass_login",
                global_name=None,
            )
        )
        assert (
            store.resolve_label(canonical_key="discord:111", guild_id=42)
            == "cass_login"
        )

    def test_falls_back_to_user_id_when_no_account_row(self) -> None:
        """Unknown canonical_key still produces a label, never raises."""
        store = HistoryStore(":memory:")
        assert store.resolve_label(canonical_key="discord:999", guild_id=42) == "999"

    def test_handles_missing_guild_id(self) -> None:
        """guild_id=None → skip the per-guild lookup, go straight to account."""
        store = HistoryStore(":memory:")
        store.upsert_account(
            Author(
                platform="discord",
                user_id="111",
                username="cass_login",
                display_name="Cassidy",
                global_name="Cassidy",
            )
        )
        assert (
            store.resolve_label(canonical_key="discord:111", guild_id=None) == "Cassidy"
        )


class TestMigrations:
    def test_migration_adds_accounts_and_guild_nicks(self, tmp_path: Path) -> None:
        """Old DBs without these tables get them on open."""
        db_path = tmp_path / "history.db"
        # Pre-create a turns table only (no accounts).
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
            """
        )
        conn.commit()
        conn.close()

        store = HistoryStore(db_path)
        # New tables should exist and be empty.
        store.upsert_account(
            Author(
                platform="discord",
                user_id="1",
                username="ada",
                display_name="Ada",
            )
        )
        row = store._conn.execute(
            "SELECT canonical_key FROM accounts WHERE canonical_key = ?",
            ("discord:1",),
        ).fetchone()
        assert row is not None
