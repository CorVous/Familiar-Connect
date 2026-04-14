"""Tests for the voice lull debounce monitor."""

from __future__ import annotations

import asyncio
import logging

import pytest

from familiar_connect.transcription import TranscriptionResult
from familiar_connect.voice_lull import VoiceActivityEvent, VoiceLullMonitor


def _final(text: str, start: float = 0.0, end: float = 1.0) -> TranscriptionResult:
    return TranscriptionResult(text=text, is_final=True, start=start, end=end)


def _interim(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, is_final=False, start=0.0, end=0.3)


class TestVoiceLullMonitor:
    @pytest.mark.asyncio
    async def test_fires_after_speech_end(self) -> None:
        """Dispatch fires lull_timeout after the user's Deepgram UtteranceEnd."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("hello world"))
        monitor.on_speech_end(42)
        await asyncio.sleep(0.2)

        assert len(calls) == 1
        user_id, result = calls[0]
        assert user_id == 42
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_speech_start_cancels_pending_lull(self) -> None:
        """Resumed speech before the timer fires cancels it; finals merge."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.1,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("hello"))
        monitor.on_speech_end(42)
        # timer armed; user resumes before it fires
        await asyncio.sleep(0.05)
        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("world"))
        await asyncio.sleep(0.05)
        assert len(calls) == 0
        # user finally stops
        monitor.on_speech_end(42)
        await asyncio.sleep(0.2)

        assert len(calls) == 1
        _, result = calls[0]
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_merges_multiple_finals_within_one_utterance(self) -> None:
        """Finals emitted mid-utterance merge into a single dispatch."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("one"))
        monitor.on_transcript(42, _final("two"))
        monitor.on_transcript(42, _final("three"))
        monitor.on_speech_end(42)
        await asyncio.sleep(0.2)

        assert len(calls) == 1
        _, result = calls[0]
        assert result.text == "one two three"

    @pytest.mark.asyncio
    async def test_concurrent_speech_holds_pending_lull(self) -> None:
        """Bob's continued speech events re-arm the lull.

        Any Deepgram activity (VAD pulse or final) re-arms the lull
        timer. While Bob keeps emitting ``SpeechStarted``, Alice's
        buffered final stays pending — the timer is continually pushed
        forward. Only when Bob's stream goes quiet for ``lull_timeout``
        does the merged dispatch fire.
        """
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(1)
        monitor.on_speech_start(2)
        monitor.on_transcript(1, _final("alice done"))
        monitor.on_speech_end(1)
        # Bob continues speaking — simulate Deepgram emitting more VAD
        # on his stream (each cancels pending lull)
        for _ in range(3):
            await asyncio.sleep(0.04)
            monitor.on_speech_start(2)
        assert len(calls) == 0

        # Bob finally stops → lull arms → dispatch
        monitor.on_speech_end(2)
        await asyncio.sleep(0.2)
        assert len(calls) == 1
        assert calls[0][1].text == "alice done"

    @pytest.mark.asyncio
    async def test_ignores_interim_transcripts(self) -> None:
        """Only final transcripts are buffered; interims are ignored."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _interim("hel"))
        monitor.on_transcript(42, _interim("hello"))
        monitor.on_speech_end(42)
        await asyncio.sleep(0.2)

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_fires_with_last_speaker_user_id(self) -> None:
        """Merged utterance is attributed to the user of the last final."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(1)
        monitor.on_transcript(1, _final("alice"))
        monitor.on_speech_end(1)
        monitor.on_speech_start(2)
        monitor.on_transcript(2, _final("bob"))
        monitor.on_speech_end(2)
        await asyncio.sleep(0.25)

        assert len(calls) == 1
        user_id, _ = calls[0]
        assert user_id == 2

    @pytest.mark.asyncio
    async def test_separate_utterances_fire_separately(self) -> None:
        """Two utterances separated by > lull_timeout fire as two callbacks."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("first"))
        monitor.on_speech_end(42)
        await asyncio.sleep(0.2)

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("second"))
        monitor.on_speech_end(42)
        await asyncio.sleep(0.2)

        assert len(calls) == 2
        assert calls[0][1].text == "first"
        assert calls[1][1].text == "second"

    @pytest.mark.asyncio
    async def test_speech_end_with_no_finals_does_not_fire(self) -> None:
        """VAD end without any finals (e.g. failed transcription) does nothing."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_speech_end(42)
        await asyncio.sleep(0.2)

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_clear_cancels_pending(self) -> None:
        """clear() cancels pending timer and drops buffered state."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("hi"))
        monitor.on_speech_end(42)
        monitor.clear()
        await asyncio.sleep(0.2)

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_monitor(self) -> None:
        """An exception in on_utterance_complete is logged, not propagated."""
        calls: list[int] = []

        async def _on_done(  # noqa: RUF029
            user_id: int,
            result: TranscriptionResult,  # noqa: ARG001
        ) -> None:
            calls.append(user_id)
            if len(calls) == 1:
                msg = "boom"
                raise RuntimeError(msg)

        monitor = VoiceLullMonitor(
            lull_timeout=0.05,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(1)
        monitor.on_transcript(1, _final("first"))
        monitor.on_speech_end(1)
        await asyncio.sleep(0.2)

        monitor.on_speech_start(2)
        monitor.on_transcript(2, _final("second"))
        monitor.on_speech_end(2)
        await asyncio.sleep(0.2)

        # both attempts happened despite the first raising
        assert calls == [1, 2]

    @pytest.mark.asyncio
    async def test_fire_lull_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """_fire_lull emits an INFO log when the lull timer expires."""

        async def _on_done(user_id: int, result: TranscriptionResult) -> None:
            pass

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("hello"))
        monitor.on_speech_end(42)
        with caplog.at_level(logging.INFO, logger="familiar_connect.voice_lull"):
            await asyncio.sleep(0.2)

        assert any(
            r.levelno == logging.INFO and "voice lull expired" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_final_arms_lull_without_speech_end(self) -> None:
        """Final arrival arms the lull; no `on_speech_end` needed.

        Under ``interim_results=false`` Deepgram emits finals directly
        but no ``UtteranceEnd`` (so ``on_speech_end`` never fires).
        The lull is armed on final arrival; when no further VAD
        activity arrives within ``lull_timeout``, the buffered final
        dispatches.
        """
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.05,
            on_utterance_complete=_on_done,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("hello"))
        # no speech_end arrives (Deepgram UtteranceEnd stuck)
        await asyncio.sleep(0.2)

        assert len(calls) == 1
        assert calls[0][1].text == "hello"


class TestVoiceActivityEvents:
    @pytest.mark.asyncio
    async def test_started_fires_on_speech_start(self) -> None:
        """on_speech_start emits a `started` event synchronously."""
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_speech_start(42)
        assert events == [(42, VoiceActivityEvent.started)]

    @pytest.mark.asyncio
    async def test_started_idempotent(self) -> None:
        """Repeated speech_start for the same user does not re-emit."""
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        for _ in range(5):
            monitor.on_speech_start(42)
        assert events == [(42, VoiceActivityEvent.started)]

    @pytest.mark.asyncio
    async def test_ended_fires_on_speech_end(self) -> None:
        """on_speech_end emits an `ended` event synchronously."""
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_speech_start(42)
        monitor.on_speech_end(42)
        assert events == [
            (42, VoiceActivityEvent.started),
            (42, VoiceActivityEvent.ended),
        ]

    @pytest.mark.asyncio
    async def test_ended_fires_on_final_for_speaking_user(self) -> None:
        """Final arrival for a user in ``_speaking`` emits ``.ended``.

        Deepgram with ``interim_results=false`` doesn't emit ``UtteranceEnd``,
        so ``on_speech_end`` never fires. Final arrival is the authoritative
        "user paused" signal and must drive ``.ended`` emission so the
        interruption detector keeps getting both transitions.
        """
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_speech_start(42)
        monitor.on_transcript(42, _final("hello"))
        assert events == [
            (42, VoiceActivityEvent.started),
            (42, VoiceActivityEvent.ended),
        ]

    @pytest.mark.asyncio
    async def test_speech_end_without_start_is_noop(self) -> None:
        """speech_end for a user not in _speaking does not emit or crash."""
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_speech_end(42)
        assert events == []

    @pytest.mark.asyncio
    async def test_separate_users_get_separate_events(self) -> None:
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_speech_start(1)
        monitor.on_speech_start(2)
        monitor.on_speech_end(1)
        monitor.on_speech_end(2)

        starts = [(uid, ev) for uid, ev in events if ev is VoiceActivityEvent.started]
        ends = [(uid, ev) for uid, ev in events if ev is VoiceActivityEvent.ended]
        assert sorted(starts) == [
            (1, VoiceActivityEvent.started),
            (2, VoiceActivityEvent.started),
        ]
        assert sorted(ends) == [
            (1, VoiceActivityEvent.ended),
            (2, VoiceActivityEvent.ended),
        ]

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_monitor(self) -> None:
        """A misbehaving subscriber must not crash the lull state machine."""

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            del user_id, event
            msg = "boom"
            raise RuntimeError(msg)

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        # Exception must be swallowed; monitor still tracks the user.
        monitor.on_speech_start(42)
        monitor.on_speech_end(42)

    @pytest.mark.asyncio
    async def test_no_callback_is_optional(self) -> None:
        """on_voice_activity is optional; default is None."""

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
        )
        monitor.on_speech_start(42)
        monitor.on_speech_end(42)

    @pytest.mark.asyncio
    async def test_logs_speech_started_and_ended(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            on_utterance_complete=_on_done,
        )

        with caplog.at_level(logging.INFO, logger="familiar_connect.voice_lull"):
            monitor.on_speech_start(42)
            monitor.on_speech_end(42)

        msgs = [r.message for r in caplog.records]
        assert any("event=started" in m and "user=42" in m for m in msgs)
        assert any("event=ended" in m and "user=42" in m for m in msgs)
