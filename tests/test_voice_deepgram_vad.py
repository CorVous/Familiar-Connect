"""Tests for DeepgramVoiceActivityDetector.

Exercises edge-triggered speech-start / speech-end logic driven by
Deepgram TranscriptionResult arrivals (interims and finals).
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.transcription import TranscriptionResult
from familiar_connect.voice.deepgram_vad import (
    DEFAULT_SILENCE_TIMEOUT_MS,
    DeepgramVoiceActivityDetector,
)


def _interim(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, is_final=False, start=0.0, end=0.2)


def _final(text: str) -> TranscriptionResult:
    return TranscriptionResult(text=text, is_final=True, start=0.0, end=0.5)


def _empty_interim() -> TranscriptionResult:
    return TranscriptionResult(text="", is_final=False, start=0.0, end=0.0)


class TestDeepgramVoiceActivityDetector:
    def _make(
        self,
        starts: list[int],
        ends: list[int],
        silence_timeout_ms: int = 80,
    ) -> DeepgramVoiceActivityDetector:
        return DeepgramVoiceActivityDetector(
            on_speech_start=starts.append,
            on_speech_end=ends.append,
            silence_timeout_ms=silence_timeout_ms,
        )

    # ------------------------------------------------------------------
    # speech_start
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_first_interim_fires_speech_start(self) -> None:
        """First non-empty interim fires on_speech_start once."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends)

        det.feed_transcript(42, _interim("hello"))

        assert starts == [42]
        assert ends == []

    @pytest.mark.asyncio
    async def test_speech_start_fires_only_once(self) -> None:
        """Multiple interims while speaking do not re-fire on_speech_start."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends)

        det.feed_transcript(42, _interim("hello"))
        det.feed_transcript(42, _interim("hello world"))
        det.feed_transcript(42, _interim("hello world how"))

        assert starts == [42]

    @pytest.mark.asyncio
    async def test_first_final_fires_speech_start(self) -> None:
        """A cold final (no prior interim) still triggers on_speech_start."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends)

        det.feed_transcript(42, _final("hello"))

        assert starts == [42]

    @pytest.mark.asyncio
    async def test_empty_interim_is_noop(self) -> None:
        """Empty-text TranscriptionResult triggers no callbacks."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends)

        det.feed_transcript(42, _empty_interim())

        assert starts == []
        assert ends == []

    # ------------------------------------------------------------------
    # speech_end via watchdog
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_watchdog_fires_speech_end(self) -> None:
        """on_speech_end fires after silence_timeout_ms of no feeds."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends, silence_timeout_ms=60)

        det.feed_transcript(42, _interim("hi"))
        assert ends == []

        await asyncio.sleep(0.15)
        assert ends == [42]

    @pytest.mark.asyncio
    async def test_watchdog_rearms_on_subsequent_interim(self) -> None:
        """Each new interim resets the watchdog; end fires after the last one."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends, silence_timeout_ms=60)

        det.feed_transcript(42, _interim("hi"))
        await asyncio.sleep(0.03)
        det.feed_transcript(42, _interim("hi there"))
        await asyncio.sleep(0.03)
        # 60 ms has not elapsed since the second interim
        assert ends == []

        await asyncio.sleep(0.08)
        assert ends == [42]

    @pytest.mark.asyncio
    async def test_final_rearms_watchdog(self) -> None:
        """Final resets the watchdog just like an interim."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends, silence_timeout_ms=60)

        det.feed_transcript(42, _interim("hi"))
        await asyncio.sleep(0.03)
        det.feed_transcript(42, _final("hi there"))
        await asyncio.sleep(0.03)
        assert ends == []

        await asyncio.sleep(0.08)
        assert ends == [42]

    @pytest.mark.asyncio
    async def test_speech_end_fires_once(self) -> None:
        """Watchdog fires on_speech_end exactly once per utterance."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends, silence_timeout_ms=60)

        det.feed_transcript(42, _interim("hi"))
        await asyncio.sleep(0.2)

        assert len(ends) == 1

    # ------------------------------------------------------------------
    # multi-user isolation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_users_are_independent(self) -> None:
        """User A's silence does not end user B."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends, silence_timeout_ms=100)

        det.feed_transcript(1, _interim("alice"))
        det.feed_transcript(2, _interim("bob"))

        # re-arm user 2 before the watchdog fires (at 50 ms, < 100 ms timeout)
        await asyncio.sleep(0.05)
        det.feed_transcript(2, _interim("still talking"))

        # now at 130 ms from start: user 1 watchdog has fired (100 ms elapsed),
        # user 2 watchdog re-armed at 50 ms so fires at 150 ms — not yet
        await asyncio.sleep(0.08)
        assert 1 in ends
        assert 2 not in ends

        await asyncio.sleep(0.1)
        assert 2 in ends

    @pytest.mark.asyncio
    async def test_second_utterance_after_end(self) -> None:
        """After speech_end, a new interim starts a fresh utterance."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends, silence_timeout_ms=60)

        det.feed_transcript(42, _interim("first"))
        await asyncio.sleep(0.15)
        assert ends == [42]

        # new utterance
        det.feed_transcript(42, _interim("second"))
        assert starts == [42, 42]

        await asyncio.sleep(0.15)
        assert ends == [42, 42]

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reset_cancels_watchdog(self) -> None:
        """reset() drops state; pending watchdog does not fire."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends, silence_timeout_ms=60)

        det.feed_transcript(42, _interim("hi"))
        det.reset()
        await asyncio.sleep(0.15)

        assert ends == []

    @pytest.mark.asyncio
    async def test_reset_clears_speaking_state(self) -> None:
        """After reset, a new interim fires on_speech_start again."""
        starts: list[int] = []
        ends: list[int] = []
        det = self._make(starts, ends)

        det.feed_transcript(42, _interim("first"))
        det.reset()
        det.feed_transcript(42, _interim("second"))

        assert starts == [42, 42]

    # ------------------------------------------------------------------
    # constant
    # ------------------------------------------------------------------

    def test_default_silence_timeout_ms(self) -> None:
        """DEFAULT_SILENCE_TIMEOUT_MS is 700."""
        assert DEFAULT_SILENCE_TIMEOUT_MS == 700
