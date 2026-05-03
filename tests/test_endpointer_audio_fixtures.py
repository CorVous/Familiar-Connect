"""Audio-fixture integration tests for ``UtteranceEndpointer``.

Where ``test_utterance_endpointer.py`` drives the state machine with
canned VAD/SmartTurn return values, this module feeds **synthesised
48 kHz mono int16 PCM** through the real ``Resampler48to16`` + framer
+ state machine. VAD and SmartTurn are still stubbed (no ONNX/native
deps in CI) but the VAD stub thresholds on actual frame energy, so
the audio fixture genuinely drives ``IDLE → SPEAKING → silence-after-
speech → classify`` transitions.

Three end-to-end patterns covered:

- **complete-sentence** — speech burst + long silence → SmartTurn
  returns complete → callback fires once with the buffered utterance.
- **mid-thought** — speech with an in-utterance pause shorter than
  ``silence_ms`` → SmartTurn never fires during the gap; only the
  trailing long silence triggers classification.
- **filler** — speech + long silence with a SmartTurn ``incomplete``
  verdict (filler word) → callback held; resumed speech + a fresh
  silence streak with ``complete`` verdict → callback fires with
  audio spanning both bursts.
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from familiar_connect.voice.audio import Resampler48to16
from familiar_connect.voice.turn_detection import UtteranceEndpointer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# ---------------------------------------------------------------------------
# Audio fixture builders
# ---------------------------------------------------------------------------

INPUT_SR = 48_000  # Discord-side sample rate fed to the endpointer
VAD_SR = 16_000  # rate after the 3:1 decimator
VAD_CHUNK_MS = 16.0  # TEN-VAD's 256-sample hop at 16 kHz

# 16 kHz post-resample → 16 ms frame = 256 samples = 512 bytes
VAD_FRAME_SAMPLES_16K = 256
VAD_FRAME_BYTES_16K = VAD_FRAME_SAMPLES_16K * 2

# Per 16 ms VAD frame, the resampler consumed 768 input samples.
INPUT_SAMPLES_PER_VAD_FRAME = VAD_FRAME_SAMPLES_16K * 3

# Energy threshold (RMS) the VAD stub uses to flag a 16 ms frame as
# speech. The synthesised speech tone sits comfortably above; pure
# zeros sit at zero. Calibrated for the 6000-amplitude tone below.
VAD_ENERGY_THRESHOLD = 200.0

# Speech tone — a 220 Hz sine at modest amplitude. Mid-band frequency
# survives the 3:1 boxcar decimation cleanly so the post-resample
# frame still has plenty of energy.
SPEECH_FREQ_HZ = 220.0
SPEECH_AMPLITUDE = 6000


def _silence_pcm(duration_ms: float) -> bytes:
    """``duration_ms`` of 48 kHz mono int16 zeros."""
    n_samples = int(duration_ms / 1000.0 * INPUT_SR)
    return b"\x00\x00" * n_samples


def _speech_pcm(duration_ms: float, *, phase_offset: float = 0.0) -> bytes:
    """``duration_ms`` of 48 kHz mono int16 sine at :data:`SPEECH_FREQ_HZ`.

    ``phase_offset`` lets a follow-on burst pick up where a prior one
    stopped, avoiding a discontinuity click at the join.
    """
    n_samples = int(duration_ms / 1000.0 * INPUT_SR)
    omega = 2.0 * math.pi * SPEECH_FREQ_HZ / INPUT_SR
    samples = [
        int(SPEECH_AMPLITUDE * math.sin(omega * i + phase_offset))
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *samples)


def _build_fixture(segments: list[tuple[str, float]]) -> bytes:
    """Stitch ``[(kind, ms), ...]`` segments into a single PCM blob.

    ``kind`` is ``"speech"`` or ``"silence"``. Sine phase advances
    across speech segments so successive bursts join smoothly.
    """
    out = bytearray()
    phase = 0.0
    for kind, ms in segments:
        if kind == "silence":
            out.extend(_silence_pcm(ms))
        elif kind == "speech":
            out.extend(_speech_pcm(ms, phase_offset=phase))
            # advance phase by elapsed time so the next burst is continuous
            phase += 2.0 * math.pi * SPEECH_FREQ_HZ * (ms / 1000.0)
            phase %= 2.0 * math.pi
        else:
            msg = f"unknown segment kind: {kind!r}"
            raise ValueError(msg)
    return bytes(out)


# ---------------------------------------------------------------------------
# VAD + SmartTurn stubs
# ---------------------------------------------------------------------------


def _frame_rms(frame: bytes) -> float:
    """Root-mean-square energy of a 16 kHz int16 frame."""
    n = len(frame) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", frame)
    sq = sum(s * s for s in samples)
    return math.sqrt(sq / n)


def _make_energy_vad(threshold: float = VAD_ENERGY_THRESHOLD) -> MagicMock:
    """Mock TEN-VAD that thresholds on actual post-resample frame energy.

    Real audio bytes flow through the resampler + framer, so the
    fixture drives the state machine the way live audio would.
    """
    vad = MagicMock()
    vad.is_speech = MagicMock(side_effect=lambda frame: _frame_rms(frame) >= threshold)
    vad.reset = MagicMock()
    return vad


def _make_smart_turn(verdicts: list[bool]) -> MagicMock:
    """SmartTurn stub — pops one verdict per ``is_complete`` call.

    Captures every audio buffer the endpointer hands to the classifier
    on ``st.is_complete.call_args_list`` so tests can assert on the
    buffered utterance length.
    """
    st = MagicMock()
    state = {"i": 0}

    def _is_complete(_audio: bytes) -> bool:
        verdict = verdicts[state["i"]]
        state["i"] += 1
        return verdict

    st.is_complete = MagicMock(side_effect=_is_complete)
    return st


def _capture_callback(bucket: list[bytes]) -> Callable[[bytes], Awaitable[None]]:
    async def _cb(audio: bytes) -> None:  # noqa: RUF029 — append is sync; signature must be async
        bucket.append(audio)

    return _cb


async def _feed_in_chunks(
    ep: UtteranceEndpointer, pcm: bytes, *, chunk_ms: float = 20.0
) -> None:
    """Feed ``pcm`` to the endpointer in ``chunk_ms`` slices.

    20 ms matches Discord's voice frame size. Sub-frame boundaries
    exercise the resampler/framer carry path that ``feed_audio``
    relies on for partial chunks.
    """
    chunk_samples = int(chunk_ms / 1000.0 * INPUT_SR)
    chunk_bytes = chunk_samples * 2
    for off in range(0, len(pcm), chunk_bytes):
        await ep.feed_audio(pcm[off : off + chunk_bytes])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompleteSentenceFixture:
    """Speech → long silence → SmartTurn complete → callback fires."""

    @pytest.mark.asyncio
    async def test_complete_sentence_fires_callback_once(self) -> None:
        calls: list[bytes] = []
        vad = _make_energy_vad()
        st = _make_smart_turn([True])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=64,
        )
        pcm = _build_fixture([
            ("silence", 80),  # leading idle (dropped from buffer)
            ("speech", 600),  # ~600 ms utterance
            ("silence", 400),  # well past 200 ms classify threshold
        ])

        await _feed_in_chunks(ep, pcm)

        assert st.is_complete.call_count == 1
        assert len(calls) == 1
        # buffer fed to SmartTurn equals buffer surfaced to the callback
        (buffered,) = st.is_complete.call_args.args
        assert buffered == calls[0]
        # int16-aligned, non-empty, and at least the speech segment's worth.
        # 600 ms at 16 kHz mono int16 = 19200 bytes; allow ≥80% to absorb the
        # leading 100 ms speech_start latch and resampler carry.
        assert len(calls[0]) % 2 == 0
        assert len(calls[0]) >= int(0.6 * 600 / 1000 * VAD_SR * 2)


class TestMidThoughtFixture:
    """In-utterance pause shorter than ``silence_ms`` must NOT classify.

    Captures the bug class where a short breath mid-sentence races the
    silence streak past the threshold and forces a SmartTurn call on a
    half-finished utterance.
    """

    @pytest.mark.asyncio
    async def test_short_mid_utterance_pause_does_not_trigger_classification(
        self,
    ) -> None:
        calls: list[bytes] = []
        vad = _make_energy_vad()
        # Single verdict: only the *trailing* silence should classify.
        st = _make_smart_turn([True])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=64,
        )
        pcm = _build_fixture([
            ("speech", 400),
            ("silence", 96),  # ~6 VAD frames — well below 200 ms threshold
            ("speech", 400),
            ("silence", 400),  # long trailing silence — fires classify
        ])

        await _feed_in_chunks(ep, pcm)

        # Exactly one classify on the trailing pause; the mid-utterance
        # pause did not race the silence streak past the threshold.
        assert st.is_complete.call_count == 1
        assert len(calls) == 1
        # Buffer should span both speech bursts — not just the second one.
        # Two 400 ms bursts = 800 ms; at 16 kHz mono int16 = 25 600 bytes.
        # Tolerate framer/resampler carry by checking ≥ 60 % of that.
        min_bytes = int(0.6 * 800 / 1000 * VAD_SR * 2)
        assert len(calls[0]) >= min_bytes

    @pytest.mark.asyncio
    async def test_pause_just_below_threshold_keeps_speaking_state(self) -> None:
        """Edge case: silence streak reaches ~⅔ of the threshold then resumes."""
        calls: list[bytes] = []
        vad = _make_energy_vad()
        st = _make_smart_turn([True])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=64,
        )
        # 144 ms gap = 9 VAD frames < 12-frame (200 ms) threshold.
        pcm = _build_fixture([
            ("speech", 300),
            ("silence", 144),
            ("speech", 300),
            ("silence", 400),
        ])

        await _feed_in_chunks(ep, pcm)

        st.is_complete.assert_called_once()
        assert len(calls) == 1


class TestFillerFixture:
    """Filler word: incomplete verdict, then resumed speech + complete."""

    @pytest.mark.asyncio
    async def test_filler_holds_callback_until_resumed_speech_classifies_complete(
        self,
    ) -> None:
        calls: list[bytes] = []
        vad = _make_energy_vad()
        # incomplete (filler) → complete (true end of turn)
        st = _make_smart_turn([False, True])
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=64,
        )
        pcm = _build_fixture([
            ("speech", 300),  # "uh"
            ("silence", 320),  # past silence_ms — first classify
            ("speech", 500),  # resumed thought
            ("silence", 400),  # past silence_ms — second classify
        ])

        await _feed_in_chunks(ep, pcm)

        # Two classify calls: filler (incomplete) and resumed (complete).
        assert st.is_complete.call_count == 2
        # Only one callback — the second, complete verdict.
        assert len(calls) == 1
        # The complete-verdict buffer must span both bursts; the
        # endpointer keeps the buffer through the incomplete hold.
        first_call_bytes = len(st.is_complete.call_args_list[0].args[0])
        second_call_bytes = len(st.is_complete.call_args_list[1].args[0])
        assert second_call_bytes > first_call_bytes

    @pytest.mark.asyncio
    async def test_filler_then_extended_silence_does_not_refire(self) -> None:
        """After ``incomplete``, more silence alone should not re-classify."""
        calls: list[bytes] = []
        vad = _make_energy_vad()
        st = _make_smart_turn([False])  # only one verdict will be consumed
        ep = UtteranceEndpointer(
            vad=vad,
            smart_turn=st,
            on_turn_complete=_capture_callback(calls),
            silence_ms=200,
            speech_start_ms=64,
        )
        pcm = _build_fixture([
            ("speech", 300),
            ("silence", 800),  # long tail past the threshold + lots more
        ])

        await _feed_in_chunks(ep, pcm)

        # SmartTurn fires exactly once — silence after an ``incomplete``
        # verdict does not retrigger classification without fresh speech.
        st.is_complete.assert_called_once()
        assert calls == []


class TestFixtureBuilders:
    """Sanity-check the fixture helpers themselves.

    Flaky audio builders would silently mask an endpointer regression
    by failing the wrong condition; pin their invariants here.
    """

    def test_silence_fixture_has_zero_rms(self) -> None:
        pcm = _silence_pcm(20.0)
        # 20 ms at 48 kHz mono int16 → 1920 bytes
        assert len(pcm) == int(0.020 * INPUT_SR) * 2
        assert _frame_rms(pcm) < 1e-9

    def test_speech_fixture_passes_energy_threshold(self) -> None:
        # post-resample energy of a 220 Hz tone must clear VAD threshold
        resampler = Resampler48to16()
        resampled = resampler.feed(_speech_pcm(64.0))
        # take one full 16 ms frame and check it'd flag as speech
        frame = resampled[:VAD_FRAME_BYTES_16K]
        assert len(frame) == VAD_FRAME_BYTES_16K
        assert _frame_rms(frame) >= VAD_ENERGY_THRESHOLD
