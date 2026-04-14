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
    async def test_fires_after_silence(self) -> None:
        """Monitor fires on_utterance_complete after lull_timeout of silence."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        monitor.on_transcript(42, _final("hello world"))
        # No more audio frames — user silence watchdog expires, then lull fires.
        await asyncio.sleep(0.2)

        assert len(calls) == 1
        user_id, result = calls[0]
        assert user_id == 42
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_late_final_after_silence_still_fires(self) -> None:
        """Real Deepgram timing: silence watchdog fires BEFORE the final.

        Regression: previously the buffered final waited for the next
        speech-then-silence cycle, causing indefinite delivery delay.
        """
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        # let user-silence watchdog fire first (Discord frames stopped)
        await asyncio.sleep(0.05)
        # now the late Deepgram final arrives
        monitor.on_transcript(42, _final("hello world"))
        await asyncio.sleep(0.15)

        assert len(calls) == 1
        user_id, result = calls[0]
        assert user_id == 42
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_audio_resets_lull(self) -> None:
        """A new audio frame cancels a pending lull timer."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.1,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        monitor.on_transcript(42, _final("hello"))
        # Let silence watchdog fire and lull timer start.
        await asyncio.sleep(0.05)
        # New audio frame before lull fires: should cancel the lull timer.
        monitor.on_audio(42)
        monitor.on_transcript(42, _final("world"))
        await asyncio.sleep(0.05)

        # Should not have fired yet — user is speaking again.
        assert len(calls) == 0

        # Let everything finally settle.
        await asyncio.sleep(0.2)
        assert len(calls) == 1
        _, result = calls[0]
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_merges_multiple_finals(self) -> None:
        """Multiple buffered finals are merged into a single utterance."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        monitor.on_transcript(42, _final("one"))
        monitor.on_audio(42)
        monitor.on_transcript(42, _final("two"))
        monitor.on_audio(42)
        monitor.on_transcript(42, _final("three"))
        await asyncio.sleep(0.25)

        assert len(calls) == 1
        _, result = calls[0]
        assert result.text == "one two three"

    @pytest.mark.asyncio
    async def test_does_not_fire_while_another_user_speaks(self) -> None:
        """If any user is still transmitting audio, the lull does not start."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.05,
            on_utterance_complete=_on_done,
        )

        # Alice finishes, but Bob keeps talking (keeps sending audio frames).
        monitor.on_audio(1)
        monitor.on_transcript(1, _final("alice done"))

        # Bob keeps his silence watchdog alive with frequent frames.
        async def _keep_bob_speaking() -> None:
            for _ in range(8):
                monitor.on_audio(2)
                await asyncio.sleep(0.02)

        await _keep_bob_speaking()

        # Lull should NOT have fired because Bob was always speaking.
        assert len(calls) == 0

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
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        monitor.on_transcript(42, _interim("hel"))
        monitor.on_transcript(42, _interim("hello"))
        await asyncio.sleep(0.2)

        # Nothing to fire — no finals were ever received.
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_fires_with_last_speaker_user_id(self) -> None:
        """The merged utterance is attributed to the user of the last final."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(1)
        monitor.on_transcript(1, _final("alice"))
        monitor.on_audio(2)
        monitor.on_transcript(2, _final("bob"))
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
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        monitor.on_transcript(42, _final("first"))
        await asyncio.sleep(0.25)

        monitor.on_audio(42)
        monitor.on_transcript(42, _final("second"))
        await asyncio.sleep(0.25)

        assert len(calls) == 2
        assert calls[0][1].text == "first"
        assert calls[1][1].text == "second"

    @pytest.mark.asyncio
    async def test_fires_only_when_finals_buffered(self) -> None:
        """Silence with no finals (e.g. failed transcription) does not fire."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        # Audio arrived but no transcript ever did.
        monitor.on_audio(42)
        await asyncio.sleep(0.25)

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_clear_cancels_pending(self) -> None:
        """clear() cancels pending timers and drops buffered finals."""
        calls: list[tuple[int, TranscriptionResult]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            calls.append((user_id, result))

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        monitor.on_transcript(42, _final("hi"))
        monitor.clear()
        await asyncio.sleep(0.25)

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
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(1)
        monitor.on_transcript(1, _final("first"))
        await asyncio.sleep(0.2)

        monitor.on_audio(2)
        monitor.on_transcript(2, _final("second"))
        await asyncio.sleep(0.2)

        # Both attempts happened despite the first raising.
        assert calls == [1, 2]

    @pytest.mark.asyncio
    async def test_fire_lull_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """_fire_lull emits an INFO log when the lull timer expires."""

        async def _on_done(user_id: int, result: TranscriptionResult) -> None:
            pass

        monitor = VoiceLullMonitor(
            lull_timeout=0.08,
            user_silence_s=0.02,
            on_utterance_complete=_on_done,
        )

        monitor.on_audio(42)
        monitor.on_transcript(42, _final("hello"))
        with caplog.at_level(logging.INFO, logger="familiar_connect.voice_lull"):
            await asyncio.sleep(0.2)

        assert any(
            r.levelno == logging.INFO and "voice lull expired" in r.message
            for r in caplog.records
        )


class TestVoiceActivityEvents:
    @pytest.mark.asyncio
    async def test_started_fires_on_first_audio_frame(self) -> None:
        """First audio frame after silence emits a `started` event."""
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            user_silence_s=0.05,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_audio(42)
        # No await between first call and assertion: started fires synchronously.
        assert events == [(42, VoiceActivityEvent.started)]

    @pytest.mark.asyncio
    async def test_started_does_not_fire_on_repeated_frames(self) -> None:
        """While a user keeps speaking, no extra `started` events fire."""
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            user_silence_s=0.1,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        for _ in range(5):
            monitor.on_audio(42)
            await asyncio.sleep(0.01)

        # Exactly one started event despite five frames.
        assert events == [(42, VoiceActivityEvent.started)]

    @pytest.mark.asyncio
    async def test_ended_fires_after_silence_window(self) -> None:
        """`ended` event fires when the per-user silence watchdog expires."""
        events: list[tuple[int, VoiceActivityEvent]] = []

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        def _on_activity(user_id: int, event: VoiceActivityEvent) -> None:
            events.append((user_id, event))

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            user_silence_s=0.03,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_audio(42)
        await asyncio.sleep(0.1)

        assert events == [
            (42, VoiceActivityEvent.started),
            (42, VoiceActivityEvent.ended),
        ]

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
            user_silence_s=0.03,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        monitor.on_audio(1)
        monitor.on_audio(2)
        await asyncio.sleep(0.1)

        # Two starts (one per user), then two ends as their watchdogs fire.
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
            user_silence_s=0.03,
            on_utterance_complete=_on_done,
            on_voice_activity=_on_activity,
        )

        # Exception must be swallowed; monitor still tracks the user.
        monitor.on_audio(42)
        await asyncio.sleep(0.1)  # let silence watchdog fire too
        # Should reach this line without raising.

    @pytest.mark.asyncio
    async def test_no_callback_is_optional(self) -> None:
        """on_voice_activity is optional; default is None."""

        async def _on_done(  # noqa: RUF029
            user_id: int, result: TranscriptionResult
        ) -> None:
            del user_id, result

        monitor = VoiceLullMonitor(
            lull_timeout=0.5,
            user_silence_s=0.03,
            on_utterance_complete=_on_done,
        )
        # Must not raise even though no subscriber is wired.
        monitor.on_audio(42)
        await asyncio.sleep(0.1)

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
            user_silence_s=0.03,
            on_utterance_complete=_on_done,
        )

        with caplog.at_level(logging.INFO, logger="familiar_connect.voice_lull"):
            monitor.on_audio(42)
            await asyncio.sleep(0.1)

        msgs = [r.message for r in caplog.records]
        assert any("event=started" in m and "user=42" in m for m in msgs)
        assert any("event=ended" in m and "user=42" in m for m in msgs)
