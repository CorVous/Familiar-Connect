"""Alarm waker processor.

Listens on :data:`TOPIC_ALARM_FIRED`, republishes a synthetic
``discord.text``-shaped event so existing :class:`TextResponder`
picks it up and produces a follow-up reply.

MVP behavior:

* text-origin alarms → publish synthetic ``discord.text`` event with
  ``content = "[alarm fired: {reason}]"``, ``author=None``.
* voice-origin alarms → fall back to text by publishing same shape
  using the alarm's recorded ``channel_id``. Real Discord voice and
  text channels have distinct ids — production wiring needs a
  per-channel mapping (out of scope for MVP).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_ALARM_FIRED, TOPIC_DISCORD_TEXT

if TYPE_CHECKING:
    from familiar_connect.bus.protocols import EventBus

_logger = logging.getLogger(__name__)


class AlarmWaker:
    """Translate ``alarm.fired`` into a synthetic text-channel turn."""

    name: str = "alarm-waker"
    topics: tuple[str, ...] = (TOPIC_ALARM_FIRED,)

    def __init__(self, *, familiar_id: str) -> None:
        self._familiar_id = familiar_id

    async def handle(self, event: Event, bus: EventBus) -> None:
        if event.topic != TOPIC_ALARM_FIRED:
            return
        payload = event.payload
        if not isinstance(payload, dict):
            return

        channel_id = payload.get("channel_id")
        reason = payload.get("reason") or ""
        if not isinstance(channel_id, int):
            return

        kind = payload.get("channel_kind") or "text"
        if kind not in {"text", "voice"}:
            _logger.warning(
                f"{ls.tag('AlarmWaker', ls.LY)} "
                f"{ls.kv('unknown_channel_kind', str(kind), vc=ls.LY)}"
            )
            return

        if kind == "voice":
            _logger.info(
                f"{ls.tag('AlarmWaker', ls.LM)} "
                f"{ls.kv('voice_fallback_to_text', str(channel_id), vc=ls.LM)} "
                f"{ls.kv('reason', ls.trunc(reason, limit=80), vc=ls.LW)}"
            )

        synth_event_id = uuid.uuid4().hex
        synth_payload = {
            "familiar_id": self._familiar_id,
            "channel_id": channel_id,
            "content": f"[alarm fired: {reason}]",
            "author": None,
            "guild_id": None,
            "message_id": None,
            "reply_to_message_id": None,
            "mentions": (),
        }
        await bus.publish(
            Event(
                event_id=synth_event_id,
                turn_id=f"wake-{payload.get('alarm_id', synth_event_id)}",
                session_id=str(channel_id),
                parent_event_ids=(event.event_id,),
                topic=TOPIC_DISCORD_TEXT,
                timestamp=datetime.now(tz=UTC),
                sequence_number=0,
                payload=synth_payload,
            )
        )
