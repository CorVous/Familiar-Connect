"""TEN VAD based per-user speech-activity detector.

Runs [TEN VAD](https://github.com/TEN-framework/ten-vad) locally on the
48 kHz mono PCM stream produced by :class:`RecordingSink`, emitting
edge-triggered ``on_speech_start`` / ``on_speech_end`` callbacks for each
Discord user.

The detector replaces Deepgram's hosted VAD as the source of voice-
activity signals driving :class:`VoiceLullMonitor` and
:class:`InterruptionDetector`. Deepgram still provides transcript text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import soxr
from ten_vad import TenVad

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger(__name__)


DEFAULT_HOP_SAMPLES: int = 256
"""TEN VAD hop size at 16 kHz (16 ms/frame)."""

DEFAULT_HANGOVER_MS: int = 300
"""Silence before firing ``on_speech_end`` after last speech frame."""

DEFAULT_THRESHOLD: float = 0.7
"""Speech probability cut. TEN VAD's own default (0.5) flags pure tones
and loud white noise; 0.7 keeps real speech (typ. 0.7-0.95) while
rejecting non-speech onsets."""

DEFAULT_MIN_SPEECH_FRAMES: int = 3
"""Consecutive above-threshold hops required before ``speech_start``
fires. Rejects single-frame spikes on noise onsets (keyboard clicks,
door thuds) without meaningful latency cost (3 x 16 ms = 48 ms)."""

_DISCORD_SAMPLE_RATE: int = 48000
"""Sink delivers mono int16 at Discord's native rate."""

_VAD_SAMPLE_RATE: int = 16000
"""TEN VAD requires 16 kHz mono int16 input."""


class _UserState:
    """Per-user resampler + VAD + streaming accumulator."""

    def __init__(
        self,
        hop_samples: int,
        hangover_frames: int,
        threshold: float,
    ) -> None:
        self.vad = TenVad(hop_size=hop_samples, threshold=threshold)
        self.resampler = soxr.ResampleStream(
            _DISCORD_SAMPLE_RATE,
            _VAD_SAMPLE_RATE,
            1,
            dtype="int16",
        )
        self.buffer = np.empty(0, dtype=np.int16)
        self.hop_samples = hop_samples
        self.hangover_frames = hangover_frames
        self.speaking = False
        self.silence_run = 0
        self.speech_run = 0


class TenVadDetector:
    """Per-user TEN VAD driver with edge-triggered speech callbacks."""

    def __init__(
        self,
        *,
        on_speech_start: Callable[[int], None],
        on_speech_end: Callable[[int], None],
        hop_samples: int = DEFAULT_HOP_SAMPLES,
        hangover_ms: int = DEFAULT_HANGOVER_MS,
        threshold: float = DEFAULT_THRESHOLD,
        min_speech_frames: int = DEFAULT_MIN_SPEECH_FRAMES,
    ) -> None:
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._hop_samples = hop_samples
        # hop duration in ms at 16 kHz
        hop_ms = hop_samples * 1000 // _VAD_SAMPLE_RATE
        self._hangover_frames = max(1, hangover_ms // max(1, hop_ms))
        self._threshold = threshold
        self._min_speech_frames = max(1, min_speech_frames)
        self._users: dict[int, _UserState] = {}

    def feed(self, user_id: int, pcm48_mono: bytes) -> None:
        """Process one chunk of 48 kHz mono int16 PCM for *user_id*.

        :raises ValueError: If byte length is not a multiple of 2.
        """
        if len(pcm48_mono) % 2 != 0:
            msg = "PCM int16 byte length must be even"
            raise ValueError(msg)

        state = self._users.get(user_id)
        if state is None:
            state = _UserState(
                self._hop_samples,
                self._hangover_frames,
                self._threshold,
            )
            self._users[user_id] = state

        samples48 = np.frombuffer(pcm48_mono, dtype=np.int16)
        out16 = state.resampler.resample_chunk(samples48)
        if out16.size == 0:
            return

        # concat leftover + fresh output, process full hops
        buf = np.concatenate([state.buffer, out16]) if state.buffer.size else out16
        n_hops = buf.size // state.hop_samples
        consumed = n_hops * state.hop_samples
        for i in range(n_hops):
            frame = buf[i * state.hop_samples : (i + 1) * state.hop_samples]
            prob, _flag = state.vad.process(frame)
            self._step_state_machine(
                user_id,
                state,
                is_speech=prob >= self._threshold,
            )
        state.buffer = buf[consumed:].copy()

    def _step_state_machine(
        self,
        user_id: int,
        state: _UserState,
        *,
        is_speech: bool,
    ) -> None:
        """Advance one hop of VAD output through the edge-detector.

        ``speech_start`` requires ``min_speech_frames`` consecutive
        above-threshold hops (attack debounce). ``speech_end`` requires
        ``hangover_frames`` consecutive below-threshold hops.
        """
        if is_speech:
            state.silence_run = 0
            if state.speaking:
                return
            state.speech_run += 1
            if state.speech_run >= self._min_speech_frames:
                state.speaking = True
                state.speech_run = 0
                _logger.debug("ten_vad speech_start user=%s", user_id)
                self._on_speech_start(user_id)
            return
        # non-speech frame: drop attack counter, advance hangover if speaking
        state.speech_run = 0
        if state.speaking:
            state.silence_run += 1
            if state.silence_run >= state.hangover_frames:
                state.speaking = False
                state.silence_run = 0
                _logger.debug("ten_vad speech_end user=%s", user_id)
                self._on_speech_end(user_id)

    def reset(self) -> None:
        """Drop all per-user state (call on channel disconnect)."""
        self._users.clear()
