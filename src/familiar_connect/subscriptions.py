"""Persistent subscription registry.

The single-slot ``TextSession`` that step 0 shipped with is replaced
by a multi-channel, multi-kind :class:`SubscriptionRegistry`. Each
``/subscribe-text`` or ``/subscribe-my-voice`` slash command adds a
row; each ``/unsubscribe-*`` removes one. Mutations are persisted to
a TOML sidecar on every write so the bot's listening set survives
restarts without the operator having to re-subscribe everywhere.

The file format is deliberately human-editable — the whole point of
the file-on-disk config decision (see
``docs/architecture/configuration-model.md``) is that an admin can
``$EDITOR data/familiars/<id>/subscriptions.toml`` to sanity-check
or prune rows without stopping the bot.
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
    """Which feed a subscription is for.

    Text and voice have different latency budgets and different
    chattiness requirements, so they are distinct rows even when a
    single channel hosts both.
    """

    text = "text"
    voice = "voice"


@dataclass(frozen=True)
class Subscription:
    """A single persistent subscription row.

    :param channel_id: Discord channel id (text or voice).
    :param kind: Which feed — text or voice.
    :param guild_id: Discord guild id the channel lives in, or
        ``None`` if the bot doesn't have one recorded (legacy rows
        or non-Discord sources).
    """

    channel_id: int
    kind: SubscriptionKind
    guild_id: int | None = None


class SubscriptionRegistry:
    """In-memory view of subscriptions, backed by a TOML sidecar.

    Constructing the registry loads the file (if it exists) and
    populates the in-memory set. Every subsequent mutation rewrites
    the whole file — the expected volume is tens of rows at most, so
    atomic rewrites are cheaper than incremental patches.
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
        """Return the voice subscription in *guild_id*, if any.

        The bot can hold at most one voice connection at a time
        (``discord.VoiceClient`` constraint) so there is at most one
        voice row per guild.
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
        """Add or replace the subscription for ``(channel_id, kind)``.

        Idempotent: re-adding an already-present row simply updates
        the stored ``guild_id`` (which can legitimately change if
        Discord moves the channel).
        """
        sub = Subscription(channel_id=channel_id, kind=kind, guild_id=guild_id)
        self._rows[channel_id, kind] = sub
        self._save()
        return sub

    def remove(self, *, channel_id: int, kind: SubscriptionKind) -> None:
        """Remove the subscription for ``(channel_id, kind)`` if present.

        Silently a no-op when the key is absent.
        """
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
