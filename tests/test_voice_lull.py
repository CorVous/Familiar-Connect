"""Tests for the voice lull debounce monitor."""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.transcription import TranscriptionResult
from familiar_connect.voice_lull import VoiceLullMonitor


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
