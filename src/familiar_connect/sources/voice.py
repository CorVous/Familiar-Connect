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
    """Drains a Deepgram transcription queue onto the bus."""

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
        # state machine: None → "speaking" → None (after final)
        self._current_turn_id: str | None = None

    async def run(self) -> None:
        """Forever loop: drain queue, publish. Cancel to stop."""
        while True:
            result = await self._queue.get()
            await self._handle(result)

    async def _handle(self, result: TranscriptionResult) -> None:
        if self._current_turn_id is None:
            # fresh utterance
            self._current_turn_id = f"voice-{uuid4().hex[:12]}"
            await self._publish(TOPIC_VOICE_ACTIVITY_START, payload=None)

        if result.is_final:
            await self._publish(
                TOPIC_VOICE_TRANSCRIPT_FINAL,
                payload={
                    "text": result.text,
                    "confidence": result.confidence,
                    "start": result.start,
                    "end": result.end,
                    "speaker": result.speaker,
                },
            )
            await self._publish(TOPIC_VOICE_ACTIVITY_END, payload=None)
            self._current_turn_id = None
        else:
            await self._publish(
                TOPIC_VOICE_TRANSCRIPT_PARTIAL,
                payload={
                    "text": result.text,
                    "confidence": result.confidence,
                },
            )

    async def _publish(self, topic: str, *, payload: object) -> Event:
        self._seq += 1
        turn_id = self._current_turn_id
        assert turn_id is not None  # noqa: S101 — invariant: set before publish
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
