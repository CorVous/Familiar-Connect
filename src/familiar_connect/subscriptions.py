"""Persistent subscription registry backed by a TOML sidecar.

Multi-channel, multi-kind. Mutations persist on every write so
subscriptions survive restarts. Human-editable on disk.
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
    """Text or voice — distinct rows even when a channel hosts both."""

    text = "text"
    voice = "voice"


@dataclass(frozen=True)
class Subscription:
    """Single persistent subscription row."""

    channel_id: int
    kind: SubscriptionKind
    guild_id: int | None = None


class SubscriptionRegistry:
    """In-memory subscription set backed by a TOML sidecar.

    Loads on construction; every mutation rewrites the whole file
    (tens of rows at most).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rows: dict[tuple[int, SubscriptionKind], Subscription] = {}
        self._load()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all(self) -> Iterable[Subscription]:
        """Yield every registered subscription."""
        return list(self._rows.values())

    def get(
        self,
        *,
        channel_id: int,
        kind: SubscriptionKind,
    ) -> Subscription | None:
        """Return the subscription for ``(channel_id, kind)``, or ``None``."""
        return self._rows.get((channel_id, kind))

    def voice_in_guild(self, guild_id: int) -> Subscription | None:
        """Return voice subscription in *guild_id*, if any.

        At most one voice row per guild (``discord.VoiceClient`` constraint).
        """
        for sub in self._rows.values():
            if sub.kind is SubscriptionKind.voice and sub.guild_id == guild_id:
                return sub
        return None

    # ------------------------------------------------------------------
    # Mutations (each one writes the whole file)
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        channel_id: int,
        kind: SubscriptionKind,
        guild_id: int | None,
    ) -> Subscription:
        """Add or replace subscription for ``(channel_id, kind)``.

        Idempotent; re-adding updates ``guild_id``.
        """
        sub = Subscription(channel_id=channel_id, kind=kind, guild_id=guild_id)
        self._rows[channel_id, kind] = sub
        self._save()
        return sub

    def remove(self, *, channel_id: int, kind: SubscriptionKind) -> None:
        """Remove subscription for ``(channel_id, kind)``; no-op if absent."""
        if self._rows.pop((channel_id, kind), None) is not None:
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
        for sub in sorted(
            self._rows.values(),
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
