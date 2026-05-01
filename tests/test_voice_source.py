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
from familiar_connect.diagnostics.collector import (
    get_span_collector,
    reset_span_collector,
)
from familiar_connect.diagnostics.voice_budget import (
    PHASE_STT_FINAL,
    PHASE_VAD_END,
    get_voice_budget_recorder,
    reset_voice_budget_recorder,
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


class TestVoiceBudget:
    """``VoiceSource`` stamps ``stt_final`` for the budget recorder.

    See :mod:`familiar_connect.diagnostics.voice_budget`. The recorder
    needs ``stt_final`` to anchor downstream gap spans.
    """

    @pytest.mark.asyncio
    async def test_final_records_stt_final_phase(self) -> None:
        reset_voice_budget_recorder()
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )

        events: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_VOICE_TRANSCRIPT_FINAL,)):
                events.append(ev)
                return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        await queue.put(_final("hi"))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        rec = get_voice_budget_recorder()
        # internal access — reasonable for a unit test
        assert events[0].turn_id in rec._turns
        assert PHASE_STT_FINAL in rec._turns[events[0].turn_id]

    @pytest.mark.asyncio
    async def test_pending_vad_end_stamped_on_next_result(self) -> None:
        """Local endpointer fires before transcript final → vad_end buffered.

        ``record_vad_end(user_id, t)`` parks a perf-counter timestamp;
        the next transcription result for that user_id stamps it on
        the freshly-minted (or current) turn so ``voice.vad_to_stt``
        emits with the right delta.
        """
        reset_voice_budget_recorder()
        reset_span_collector()
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )

        events: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_VOICE_TRANSCRIPT_FINAL,)):
                events.append(ev)
                return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        # Endpointer fires first; final arrives later.
        source.record_vad_end(user_id=42, t=100.000)
        await queue.put(_final("hi", user_id=42))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        rec = get_voice_budget_recorder()
        turn_id = events[0].turn_id
        phases = rec._turns[turn_id]
        assert PHASE_VAD_END in phases
        assert PHASE_STT_FINAL in phases
        # Buffered ``t=100.0`` must reach the recorder unchanged.
        assert phases[PHASE_VAD_END] == pytest.approx(100.0)
        # Adjacent gap span emits.
        names = [r.name for r in get_span_collector().all()]
        assert "voice.vad_to_stt" in names

    @pytest.mark.asyncio
    async def test_vad_end_only_applies_to_matching_user(self) -> None:
        """Buffered timestamp keyed by user_id — wrong speaker ignores it."""
        reset_voice_budget_recorder()
        reset_span_collector()
        bus = InProcessEventBus()
        await bus.start()
        queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        source = VoiceSource(
            bus=bus, familiar_id="fam", voice_channel_id=123, queue=queue
        )

        events: list = []

        async def consume() -> None:
            async for ev in bus.subscribe((TOPIC_VOICE_TRANSCRIPT_FINAL,)):
                events.append(ev)
                return

        c = asyncio.create_task(consume())
        p = asyncio.create_task(source.run())
        await asyncio.sleep(0)
        # Endpointer fired for user 101; user 202 happens to final first.
        source.record_vad_end(user_id=101, t=99.0)
        await queue.put(_final("other speaker", user_id=202))
        await asyncio.wait_for(c, timeout=1.0)
        p.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p
        await bus.shutdown()

        rec = get_voice_budget_recorder()
        turn_id = events[0].turn_id
        # 202's turn must NOT carry 101's vad_end stamp.
        assert PHASE_VAD_END not in rec._turns[turn_id]
