"""Persistent subscription registry backed by TOML sidecar.

Multi-channel, multi-kind. Every mutation persists — survives restart.
Human-editable on disk.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class SubscriptionKind(Enum):
    """Text or voice — distinct rows even when channel hosts both."""

    text = "text"
    voice = "voice"


@dataclass(frozen=True)
class Subscription:
    """Single persistent subscription row."""

    channel_id: int
    kind: SubscriptionKind
    guild_id: int | None = None


class SubscriptionRegistry:
    """In-memory set backed by TOML sidecar.

    Loads on construction; mutations rewrite whole file (tens of rows
    at most).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rows: dict[tuple[int, SubscriptionKind], Subscription] = {}
        self._ephemeral: set[tuple[int, SubscriptionKind]] = set()
        self._load()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all(self) -> Iterable[Subscription]:
        """Every registered subscription."""
        return list(self._rows.values())

    def get(
        self,
        *,
        channel_id: int,
        kind: SubscriptionKind,
    ) -> Subscription | None:
        return self._rows.get((channel_id, kind))

    def kind_for(self, channel_id: int) -> SubscriptionKind | None:
        """Return subscription kind for channel_id, or None if not subscribed."""
        for kind in SubscriptionKind:
            sub = self.get(channel_id=channel_id, kind=kind)
            if sub is not None:
                return kind
        return None

    def voice_in_guild(self, guild_id: int) -> Subscription | None:
        """Voice subscription in ``guild_id``, if any.

        At most one per guild (``discord.VoiceClient`` constraint).
        """
        for sub in self._rows.values():
            if sub.kind is SubscriptionKind.voice and sub.guild_id == guild_id:
                return sub
        return None

    # ------------------------------------------------------------------
    # Mutations (each writes whole file)
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        channel_id: int,
        kind: SubscriptionKind,
        guild_id: int | None,
        persist: bool = True,
    ) -> Subscription:
        """Add or replace ``(channel_id, kind)``; idempotent.

        Re-add updates ``guild_id``. With ``persist=False`` the row is
        registered in memory only and never written to the sidecar — even
        when a later persisted mutation rewrites the file. A subsequent
        ``persist=True`` add of the same key promotes it to persisted.
        """
        key = (channel_id, kind)
        sub = Subscription(channel_id=channel_id, kind=kind, guild_id=guild_id)
        self._rows[key] = sub
        if persist:
            self._ephemeral.discard(key)
            self._save()
        else:
            self._ephemeral.add(key)
        return sub

    def remove(self, *, channel_id: int, kind: SubscriptionKind) -> None:
        """Remove ``(channel_id, kind)``; no-op if absent."""
        key = (channel_id, kind)
        if self._rows.pop(key, None) is not None:
            self._ephemeral.discard(key)
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("rb") as f:
            data = tomllib.load(f)
        raw_rows = data.get("subscription", [])
        if not isinstance(raw_rows, list):
            return
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            row_dict = cast("dict[str, object]", row)
            channel_id_raw = row_dict.get("channel_id")
            kind_raw = row_dict.get("kind")
            guild_id_raw = row_dict.get("guild_id")
            if not isinstance(channel_id_raw, int) or not isinstance(kind_raw, str):
                continue
            try:
                kind = SubscriptionKind(kind_raw)
            except ValueError:
                continue
            guild_id = guild_id_raw if isinstance(guild_id_raw, int) else None
            sub = Subscription(
                channel_id=channel_id_raw,
                kind=kind,
                guild_id=guild_id,
            )
            self._rows[channel_id_raw, kind] = sub

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Persistent subscription registry.\n"
            "# Managed by /subscribe-* slash commands; "
            "safe to hand-edit while the bot is stopped.\n"
        )
        lines: list[str] = [header]
        persisted = (
            sub
            for key, sub in self._rows.items()
            if key not in self._ephemeral
        )
        for sub in sorted(
            persisted,
            key=lambda s: (s.channel_id, s.kind.value),
        ):
            row_lines = [
                "[[subscription]]",
                f"channel_id = {sub.channel_id}",
                f'kind = "{sub.kind.value}"',
            ]
            if sub.guild_id is not None:
                row_lines.append(f"guild_id = {sub.guild_id}")
            row_lines.append("")
            lines.append("\n".join(row_lines))
        self._path.write_text("\n".join(lines))
