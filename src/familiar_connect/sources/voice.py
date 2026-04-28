"""Voice source: Deepgram transcription queue → bus events.

Publishes four topics per utterance:

- ``voice.activity.start`` — first sign of speech; drives
  :class:`TurnRouter` cancel-prior-scope (plan § Design.3).
- ``voice.transcript.partial`` — interim Deepgram results.
- ``voice.transcript.final`` — final transcript; triggers the
  :class:`VoiceResponder` to assemble + reply.
- ``voice.activity.end`` — utterance closed; responder can commit.

All four events in one utterance share ``turn_id``. The next
utterance gets a fresh ``turn_id``. Drop-oldest at the audio boundary
is owned by the recording-sink/transcriber upstream — this source
only sees post-transcription text and publishes with ``BLOCK`` policy
via the bus default.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import (
    TOPIC_VOICE_ACTIVITY_END,
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
    TOPIC_VOICE_TRANSCRIPT_PARTIAL,
)

if TYPE_CHECKING:
    import asyncio

    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.transcription import TranscriptionResult


class VoiceSource:
    """Drains a Deepgram transcription queue onto the bus.

    Discord delivers per-SSRC audio so each user_id has an independent
    transcript stream. State machine is keyed by user_id (with ``None``
    as the legacy single-stream key for unattributed results) so two
    speakers talking concurrently get distinct turn_ids — a single
    state slot would drop the second speaker's ``activity.start``.
    """

    name: str = "voice"

    def __init__(
        self,
        *,
        bus: EventBus,
        familiar_id: str,
        voice_channel_id: int,
        queue: asyncio.Queue[TranscriptionResult],
    ) -> None:
        self._bus = bus
        self._familiar_id = familiar_id
        self._channel_id = voice_channel_id
        self._queue = queue
        self._seq = 0
        # per-user state machine: user_id → current_turn_id (None when idle).
        # ``None`` user_id is the legacy unattributed slot — kept so older
        # mixed-stream paths still work.
        self._turn_ids: dict[int | None, str] = {}

    async def run(self) -> None:
        """Forever loop: drain queue, publish. Cancel to stop."""
        while True:
            result = await self._queue.get()
            await self._handle(result)

    async def _handle(self, result: TranscriptionResult) -> None:
        user_id = result.user_id
        turn_id = self._turn_ids.get(user_id)
        if turn_id is None:
            turn_id = f"voice-{uuid4().hex[:12]}"
            self._turn_ids[user_id] = turn_id
            await self._publish(
                TOPIC_VOICE_ACTIVITY_START, turn_id=turn_id, payload=None
            )

        if result.is_final:
            await self._publish(
                TOPIC_VOICE_TRANSCRIPT_FINAL,
                turn_id=turn_id,
                payload={
                    "text": result.text,
                    "confidence": result.confidence,
                    "start": result.start,
                    "end": result.end,
                    "speaker": result.speaker,
                    "user_id": result.user_id,
                },
            )
            await self._publish(TOPIC_VOICE_ACTIVITY_END, turn_id=turn_id, payload=None)
            self._turn_ids.pop(user_id, None)
        else:
            await self._publish(
                TOPIC_VOICE_TRANSCRIPT_PARTIAL,
                turn_id=turn_id,
                payload={
                    "text": result.text,
                    "confidence": result.confidence,
                    "user_id": result.user_id,
                },
            )

    async def _publish(self, topic: str, *, turn_id: str, payload: object) -> Event:
        self._seq += 1
        event_id = f"voice-{self._seq:08d}"
        ev = Event(
            event_id=event_id,
            turn_id=turn_id,
            session_id=f"voice:{self._channel_id}",
            parent_event_ids=(),
            topic=topic,
            timestamp=datetime.now(tz=UTC),
            sequence_number=self._seq,
            payload=payload,
        )
        await self._bus.publish(ev)
        return ev
