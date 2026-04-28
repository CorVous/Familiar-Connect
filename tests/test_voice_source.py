"""Tests for :class:`familiar_connect.sources.voice.VoiceSource`.

Queue→bus adapter for Deepgram transcription events. Emits
``voice.activity.start`` at utterance start, ``voice.transcript.partial``
for interim results, ``voice.transcript.final`` + ``voice.activity.end``
at utterance close.
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.bus import InProcessEventBus
from familiar_connect.bus.topics import (
    TOPIC_VOICE_ACTIVITY_END,
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
    TOPIC_VOICE_TRANSCRIPT_PARTIAL,
)
from familiar_connect.sources.voice import VoiceSource
from familiar_connect.transcription import TranscriptionResult


def _final(
    text: str = "hello world", *, user_id: int | None = None
) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        is_final=True,
        start=0.0,
        end=1.0,
        confidence=0.9,
        user_id=user_id,
    )


def _partial(text: str = "hel", *, user_id: int | None = None) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        is_final=False,
        start=0.0,
        end=0.3,
        confidence=0.5,
        user_id=user_id,
    )


class TestVoiceSource:
    @pytest.mark.asyncio
    async def test_final_only_emits_start_transcript_end(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus,
            familiar_id="fam",
            voice_channel_id=123,
            queue=queue,
        )

        collected: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_ACTIVITY_END,
                TOPIC_VOICE_TRANSCRIPT_FINAL,
            )):
                collected.append(ev)
                if len(collected) >= 3:
                    return

        consumer = asyncio.create_task(consume())
        producer = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        await queue.put(_final("hello"))

        await asyncio.wait_for(consumer, timeout=1.0)
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        await bus.shutdown()

        topics_emitted = [ev.topic for ev in collected]
        assert topics_emitted == [
            TOPIC_VOICE_ACTIVITY_START,
            TOPIC_VOICE_TRANSCRIPT_FINAL,
            TOPIC_VOICE_ACTIVITY_END,
        ]
        final_ev = collected[1]
        assert final_ev.payload["text"] == "hello"
        assert final_ev.session_id == "voice:123"
        # All three events in one utterance must share the same turn_id
        assert collected[0].turn_id == final_ev.turn_id == collected[2].turn_id

    @pytest.mark.asyncio
    async def test_partial_emits_start_and_partial_but_not_end(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )
        got: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_TRANSCRIPT_PARTIAL,
            )):
                got.append(ev)
                if len(got) >= 2:
                    return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        await queue.put(_partial("hel"))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        assert [e.topic for e in got] == [
            TOPIC_VOICE_ACTIVITY_START,
            TOPIC_VOICE_TRANSCRIPT_PARTIAL,
        ]

    @pytest.mark.asyncio
    async def test_only_one_activity_start_per_utterance(self) -> None:
        """Two partials + a final must only emit ONE activity.start."""
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )
        got: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_ACTIVITY_END,
                TOPIC_VOICE_TRANSCRIPT_PARTIAL,
                TOPIC_VOICE_TRANSCRIPT_FINAL,
            )):
                got.append(ev)
                if ev.topic == TOPIC_VOICE_ACTIVITY_END:
                    return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        await queue.put(_partial("hel"))
        await queue.put(_partial("hello"))
        await queue.put(_final("hello world"))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        start_count = sum(1 for e in got if e.topic == TOPIC_VOICE_ACTIVITY_START)
        assert start_count == 1
        # All events in this utterance share the same turn_id
        turn_ids = {e.turn_id for e in got}
        assert len(turn_ids) == 1

    @pytest.mark.asyncio
    async def test_new_utterance_after_final_gets_fresh_turn_id(self) -> None:
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )
        finals: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_VOICE_TRANSCRIPT_FINAL,)):
                finals.append(ev)
                if len(finals) >= 2:
                    return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        await queue.put(_final("first"))
        await queue.put(_final("second"))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        assert finals[0].turn_id != finals[1].turn_id

    @pytest.mark.asyncio
    async def test_final_payload_carries_user_id(self) -> None:
        """Discord user_id from per-user fan-in surfaces in the bus payload."""
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )
        finals: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_VOICE_TRANSCRIPT_FINAL,)):
                finals.append(ev)
                return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        await queue.put(_final("hello", user_id=42))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        assert finals[0].payload["user_id"] == 42

    @pytest.mark.asyncio
    async def test_concurrent_speakers_get_independent_turn_ids(self) -> None:
        """Two users speaking interleaved must each open their own turn.

        Mixed-stream design (single turn_id state) drops the second
        speaker's ``activity.start`` and orphans their final under the
        first speaker's turn. With per-user state each user_id has its
        own state machine.
        """
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )
        events: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_ACTIVITY_END,
                TOPIC_VOICE_TRANSCRIPT_FINAL,
            )):
                events.append(ev)
                if sum(1 for e in events if e.topic == TOPIC_VOICE_ACTIVITY_END) >= 2:
                    return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        # Alice starts; Bob starts before Alice finals; both final.
        await queue.put(_partial("hel", user_id=101))
        await queue.put(_partial("hi", user_id=202))
        await queue.put(_final("hello", user_id=101))
        await queue.put(_final("hi there", user_id=202))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        starts = [e for e in events if e.topic == TOPIC_VOICE_ACTIVITY_START]
        assert len(starts) == 2
        # Each speaker gets a distinct turn_id
        assert starts[0].turn_id != starts[1].turn_id
        # Each final lands on its own speaker's turn_id
        finals = [e for e in events if e.topic == TOPIC_VOICE_TRANSCRIPT_FINAL]
        finals_by_user = {e.payload["user_id"]: e for e in finals}
        assert finals_by_user[101].turn_id != finals_by_user[202].turn_id
