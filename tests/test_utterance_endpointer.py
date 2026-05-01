"""Tests for the Silero+SmartTurn utterance endpointer state machine.

The endpointer is the core of V1 phase 2: per-user 48 kHz PCM in,
``is-this-turn-complete?`` decisions out. ONNX I/O is mocked so the
suite stays fast and doesn't require model files.

Trace shape:

- Feed ``IDLE`` chunks → no detector calls.
- Feed ``SPEECH`` chunks then ``SILENCE_MS`` of silence → ``SmartTurn``
  fires; if its verdict is ``complete`` the callback runs and state
  resets.
- ``incomplete`` verdict → callback held, more speech + silence
  required before the classifier runs again.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from familiar_connect.voice.turn_detection import UtteranceEndpointer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# 32 ms at 16 kHz mono int16 — Silero v5 chunk size.
VAD_CHUNK_SAMPLES = 512
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * 2

# 48 kHz feed = 3x 16 kHz; one 32 ms VAD chunk corresponds to
# ``VAD_CHUNK_SAMPLES * 3`` input samples.
INPUT_CHUNK_SAMPLES = VAD_CHUNK_SAMPLES * 3
INPUT_CHUNK_BYTES = INPUT_CHUNK_SAMPLES * 2


def _input_chunk(amplitude: int = 0) -> bytes:
    """One 32-ms-equivalent block of 48 kHz mono int16 PCM."""
    return struct.pack(f"<{INPUT_CHUNK_SAMPLES}h", *([amplitude] * INPUT_CHUNK_SAMPLES))


def _make_vad(speech_pattern: list[bool]) -> MagicMock:
    """Mock Silero VAD whose ``is_speech`` walks the pattern.

    Once the pattern runs out, repeats the last value — useful for
    "tail off into silence" tests without listing every chunk.
    ``reset()`` rewinds the pattern iterator so test scenarios that
    reset the endpointer don't bleed the previous pattern state.
    """
    vad = MagicMock()
    state = {"i": 0}

    def _is_speech(_chunk: bytes) -> bool:
        idx = min(state["i"], len(speech_pattern) - 1)
        state["i"] += 1
        return speech_pattern[idx]

    def _reset() -> None:
        state["i"] = 0

    vad.is_speech = MagicMock(side_effect=_is_speech)
    vad.reset = MagicMock(side_effect=_reset)
    return vad


def _make_smart_turn(verdicts: list[bool]) -> MagicMock:
    """Mock SmartTurnDetector. Each ``is_complete`` consumes one verdict."""
    st = MagicMock()
    state = {"i": 0}

    def _is_complete(_audio: bytes) -> bool:
        verdict = verdicts[state["i"]]
        state["i"] += 1
        return verdict

    st.is_complete = MagicMock(side_effect=_is_complete)
    return st


def _capture_callback(
    bucket: list[bytes],
) -> Callable[[bytes], Awaitable[None]]:
    """Closure that appends each turn-complete payload to ``bucket``."""

    async def _cb(audio: bytes) -> None:  # noqa: RUF029 — append is sync; signature must be async
        bucket.append(audio)

    return _cb


class TestSilenceOnly:
    @pytest.mark.asyncio
    async def test_no_classification_when_no_speech(self) -> None:
        """Pure silence → VAD runs, SmartTurn never fires, callback silent."""
        calls: list[bytes] = []
        vad = _make_vad([False])
        st = _make_smart_turn([])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
        )
        for _ in range(10):
            await ep.feed_audio(_input_chunk(0))

        assert vad.is_speech.call_count >= 1
        st.is_complete.assert_not_called()
        assert calls == []


class TestCompleteUtterance:
    @pytest.mark.asyncio
    async def test_speech_then_silence_classifies_and_fires_callback(self) -> None:
        """Speech burst followed by silence_ms silence → SmartTurn fires."""
        calls: list[bytes] = []
        # 5 chunks speech, then 8 chunks silence (256 ms — past 200 ms cap)
        pattern = [True] * 5 + [False] * 8
        vad = _make_vad(pattern)
        st = _make_smart_turn([True])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=32,
        )

        for _ in range(len(pattern)):
            await ep.feed_audio(_input_chunk(1000))

        st.is_complete.assert_called_once()
        assert len(calls) == 1
        # buffer handed to classifier — non-empty, int16-aligned
        assert len(calls[0]) > 0
        assert len(calls[0]) % 2 == 0


class TestIncompleteThenComplete:
    @pytest.mark.asyncio
    async def test_incomplete_holds_callback_until_next_pause(self) -> None:
        """``incomplete`` → no fire; resumed speech + new silence → reclassify."""
        calls: list[bytes] = []
        # 3 speech, 8 silence (classify→incomplete),
        # 3 speech (resume), 8 silence (classify again).
        pattern = [True] * 3 + [False] * 8 + [True] * 3 + [False] * 8
        vad = _make_vad(pattern)
        st = _make_smart_turn([False, True])  # incomplete then complete
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=32,
        )

        for _ in range(len(pattern)):
            await ep.feed_audio(_input_chunk(1500))

        assert st.is_complete.call_count == 2
        assert len(calls) == 1


class TestExtendedSilenceAfterIncomplete:
    @pytest.mark.asyncio
    async def test_no_reclassification_during_continued_silence(self) -> None:
        """After incomplete, more silence alone shouldn't refire SmartTurn."""
        calls: list[bytes] = []
        pattern = [True] * 3 + [False] * 16
        vad = _make_vad(pattern)
        st = _make_smart_turn([False])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=32,
        )
        for _ in range(len(pattern)):
            await ep.feed_audio(_input_chunk(1500))

        st.is_complete.assert_called_once()
        assert calls == []


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_drops_buffer_and_state(self) -> None:
        calls: list[bytes] = []
        vad = _make_vad([True] * 3 + [False] * 8)
        st = _make_smart_turn([True])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=32,
        )
        for _ in range(2):
            await ep.feed_audio(_input_chunk(1500))
        ep.reset()
        for _ in range(8):
            await ep.feed_audio(_input_chunk(0))

        st.is_complete.assert_not_called()
        assert calls == []


class TestSubchunkFraming:
    @pytest.mark.asyncio
    async def test_partial_chunks_buffer_until_full_vad_window(self) -> None:
        """Sub-32ms feeds accumulate; VAD only runs once a full chunk lands."""
        vad = _make_vad([False])
        st = _make_smart_turn([])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback([]),
            silence_ms=200,
        )
        half = INPUT_CHUNK_BYTES // 2
        await ep.feed_audio(b"\x00" * half)
        vad.is_speech.assert_not_called()
        await ep.feed_audio(b"\x00" * (INPUT_CHUNK_BYTES - half))
        assert vad.is_speech.call_count == 1
