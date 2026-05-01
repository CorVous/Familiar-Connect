"""Utterance endpointer — Silero VAD + Smart Turn over a 48 kHz PCM stream.

Drives V1 phase 2: per-user audio in, "turn complete" callback out.
Owns three building blocks:

- :class:`Resampler48to16` — 3:1 decimation to Silero's native rate.
- :class:`SileroVAD` — per-32 ms is-speech probability.
- :class:`SmartTurnDetector` — semantic completion classifier.

State machine (per user):

- ``IDLE`` → speech burst detected → ``SPEAKING``
- ``SPEAKING`` → silence streak ≥ ``silence_ms`` → run SmartTurn
    - ``complete`` → fire callback, reset to ``IDLE``
    - ``incomplete`` → ``POST_INCOMPLETE``
- ``POST_INCOMPLETE`` → fresh speech → ``SPEAKING`` again
- ``POST_INCOMPLETE`` → continued silence → no reclassification

The classifier is invoked on the silence-after-speech edge only;
extra silence after an ``incomplete`` verdict doesn't refire it.
A subsequent speech-then-silence cycle does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.voice.audio import Resampler48to16

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.voice.turn_detection.silero_vad import SileroVAD
    from familiar_connect.voice.turn_detection.smart_turn import SmartTurnDetector


# Silero v5 native frame at 16 kHz: 512 int16 samples = 1024 bytes = 32 ms.
_VAD_CHUNK_SAMPLES: int = 512
_VAD_CHUNK_BYTES: int = _VAD_CHUNK_SAMPLES * 2
_VAD_CHUNK_MS: float = (_VAD_CHUNK_SAMPLES / 16000.0) * 1000.0


class UtteranceEndpointer:
    """Per-user local turn-detection state machine.

    Caller feeds 48 kHz mono int16 PCM via :meth:`feed_audio` (any
    chunk length); endpointer resamples, frames into 32 ms VAD windows,
    and on a silence-after-speech edge runs Smart Turn over the buffered
    16 kHz utterance audio. ``on_turn_complete`` is awaited with the
    buffered audio whenever Smart Turn classifies ``complete``.
    """

    def __init__(
        self,
        *,
        vad: SileroVAD,
        smart_turn: SmartTurnDetector,
        on_turn_complete: Callable[[bytes], Awaitable[None]],
        silence_ms: int = 200,
        speech_start_ms: int = 100,
    ) -> None:
        self._vad = vad
        self._smart_turn = smart_turn
        self._on_complete = on_turn_complete
        self._silence_chunks_threshold = max(1, int(silence_ms / _VAD_CHUNK_MS))
        self._speech_chunks_threshold = max(1, int(speech_start_ms / _VAD_CHUNK_MS))

        self._resampler = Resampler48to16()
        # remainder bytes from a partial VAD frame — accumulate next feed
        self._frame_carry: bytearray = bytearray()
        self._utterance: bytearray = bytearray()

        # state flags
        self._speaking: bool = False
        self._post_incomplete: bool = False
        self._speech_streak: int = 0
        self._silence_streak: int = 0

    def reset(self) -> None:
        """Drop all buffered audio + VAD/streak state. Resampler resets too."""
        self._resampler.reset()
        self._frame_carry.clear()
        self._utterance.clear()
        self._speaking = False
        self._post_incomplete = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._vad.reset()

    async def feed_audio(self, pcm_48k: bytes) -> None:
        """Resample, frame into 32 ms VAD windows, advance the state machine."""
        if not pcm_48k:
            return
        resampled = self._resampler.feed(pcm_48k)
        if not resampled:
            return
        self._frame_carry.extend(resampled)
        # consume as many full VAD frames as the carry holds
        while len(self._frame_carry) >= _VAD_CHUNK_BYTES:
            frame = bytes(self._frame_carry[:_VAD_CHUNK_BYTES])
            del self._frame_carry[:_VAD_CHUNK_BYTES]
            await self._on_vad_frame(frame)

    async def _on_vad_frame(self, frame: bytes) -> None:
        """Single 32 ms decision step."""
        self._utterance.extend(frame)
        is_speech = self._vad.is_speech(frame)

        if is_speech:
            self._silence_streak = 0
            if not self._speaking:
                self._speech_streak += 1
                if self._speech_streak >= self._speech_chunks_threshold:
                    self._speaking = True
                    self._post_incomplete = False
            return

        # silence frame
        self._speech_streak = 0
        if not self._speaking:
            # idle silence — drop the buffer to bound memory; nothing to classify
            self._utterance.clear()
            return

        self._silence_streak += 1
        if self._silence_streak < self._silence_chunks_threshold:
            return

        # silence threshold hit after speech → classify (unless we already did
        # and got `incomplete` and haven't seen new speech since).
        if self._post_incomplete:
            return

        verdict = self._smart_turn.is_complete(bytes(self._utterance))
        if verdict:
            audio = bytes(self._utterance)
            self._utterance.clear()
            self._speaking = False
            self._post_incomplete = False
            self._silence_streak = 0
            self._vad.reset()
            await self._on_complete(audio)
        else:
            # keep buffer; await fresh speech, then a fresh silence streak
            self._post_incomplete = True
            self._speaking = False
