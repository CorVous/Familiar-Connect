"""Single-writer history persistence processor.

Consumes ``discord.text`` events and writes user turns to the
:class:`HistoryStore`. Centralising writes here keeps the SQLite
connection owned by one task and makes dedup-by-``event_id`` simple.

See plan § Design.5 *Side-indices off watermarks* and plan § Design.6
*SQLite write path under load*.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT

if TYPE_CHECKING:
    from familiar_connect.bus.envelope import Event
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.identity import Author

_logger = logging.getLogger("familiar_connect.processors.history_writer")


class HistoryWriter:
    """Persist turn-generating events into :class:`HistoryStore`."""

    name: str = "history-writer"
    topics: tuple[str, ...] = (TOPIC_DISCORD_TEXT,)

    def __init__(self, *, store: HistoryStore, familiar_id: str) -> None:
        self._store = store
        self._familiar_id = familiar_id
        # in-process dedup set; survives a single run. Acceptable because
        # the bus itself doesn't republish on retry today.
        self._seen: set[str] = set()

    async def handle(self, event: Event, bus: EventBus) -> None:  # noqa: ARG002
        if event.topic != TOPIC_DISCORD_TEXT:
            return
        if event.event_id in self._seen:
            return

        payload = event.payload
        if not isinstance(payload, dict):
            return
        if payload.get("familiar_id") != self._familiar_id:
            return

        channel_id = payload.get("channel_id")
        content = payload.get("content") or ""
        if not isinstance(channel_id, int) or not content:
            return

        author: Author | None = payload.get("author")
        guild_id = payload.get("guild_id")

        self._seen.add(event.event_id)
        self._store.append_turn(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            role="user",
            content=content,
            author=author,
            guild_id=guild_id if isinstance(guild_id, int) else None,
        )
        _logger.debug(
            f"{ls.tag('History', ls.LC)} "
            f"{ls.kv('append', event.event_id, vc=ls.LC)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LY)} "
            f"{ls.kv('text', ls.trunc(content, 80), vc=ls.LW)}"
        )
