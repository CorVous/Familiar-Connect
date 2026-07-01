"""Utterance endpointer — TEN-VAD + Smart Turn over a 48 kHz PCM stream.

Drives V1 phase 2: per-user audio in, "turn complete" callback out.
Three building blocks:

- :class:`Resampler48to16` — 3:1 decimation to TEN-VAD native rate.
- :class:`TenVAD` — per-16 ms is-speech probability.
- :class:`SmartTurnDetector` — semantic completion classifier.

State machine (per user):

- ``IDLE`` → speech burst detected → ``SPEAKING``
- ``SPEAKING`` → silence streak ≥ ``silence_ms`` → run SmartTurn
    - ``complete`` → fire callback, reset to ``IDLE``
    - ``incomplete`` → ``POST_INCOMPLETE``
- ``POST_INCOMPLETE`` → fresh speech → ``SPEAKING`` again
- ``POST_INCOMPLETE`` → continued silence → no reclassification

Classifier invoked on the silence-after-speech edge only; extra
silence after an ``incomplete`` verdict doesn't refire it. Subsequent
speech-then-silence cycle does.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from familiar_connect.voice.audio import Resampler48to16

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.voice.turn_detection.smart_turn import SmartTurnDetector
    from familiar_connect.voice.turn_detection.ten_vad import TenVAD


# TEN-VAD native frame at 16 kHz: 256 int16 samples = 512 bytes = 16 ms.
_VAD_CHUNK_SAMPLES: int = 256
_VAD_CHUNK_BYTES: int = _VAD_CHUNK_SAMPLES * 2
_VAD_CHUNK_MS: float = (_VAD_CHUNK_SAMPLES / 16000.0) * 1000.0


class UtteranceEndpointer:
    """Per-user local turn-detection state machine.

    Caller feeds 48 kHz mono int16 PCM via :meth:`feed_audio` (any
    chunk length); endpointer resamples, frames into 16 ms VAD
    windows, and on silence-after-speech edge runs Smart Turn over
    buffered 16 kHz utterance audio. ``on_turn_complete`` awaited
    with buffered audio whenever Smart Turn classifies ``complete``.
    """

    def __init__(
        self,
        *,
        vad: TenVAD,
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
        # Remainder bytes from partial VAD frame — accumulate next feed
        self._frame_carry: bytearray = bytearray()
        self._utterance: bytearray = bytearray()

        # State flags
        self._speaking: bool = False
        self._post_incomplete: bool = False
        self._speech_streak: int = 0
        self._silence_streak: int = 0

    def reset(self) -> None:
        """Drop buffered audio + VAD/streak state; resets resampler too."""
        self._resampler.reset()
        self._frame_carry.clear()
        self._utterance.clear()
        self._speaking = False
        self._post_incomplete = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._vad.reset()

    async def force_complete_if_pending(self) -> bool:
        """Emit buffered speech as a complete turn — external idle fallback.

        The state machine only advances on :meth:`feed_audio` frames, but
        Discord's client VAD halts RTP during silence, so trailing silence
        delivers no frames to re-trigger classification. Two stranding
        cases follow: a Smart Turn ``incomplete`` misfire (we sit in
        ``POST_INCOMPLETE`` awaiting fresh speech that ends the call), or a
        burst that stopped before the silence streak reached ``silence_ms``
        (still ``SPEAKING``, never classified). Either way the buffered
        audio waits for the speaker's *next* utterance to flush.

        The audio pump calls this after an idle gap: if speech is buffered,
        fire ``on_turn_complete`` with it and reset rather than hold the
        transcript indefinitely. Audio has stopped, so the turn is treated
        as over without re-running Smart Turn.

        Fires on the *state* (``SPEAKING`` or ``POST_INCOMPLETE``), not on
        buffered bytes: after an ``incomplete`` verdict the memory-bounding
        idle-clear can drain ``_utterance`` while the turn is still stranded,
        and ``on_turn_complete`` consumers key off the turn ending — the STT
        finalize — not the audio payload.

        Returns ``True`` when a turn was emitted, ``False`` when nothing was
        pending (pure idle, or already drained by normal classification).
        """
        if not (self._speaking or self._post_incomplete):
            return False
        audio = bytes(self._utterance)
        self._utterance.clear()
        self._speaking = False
        self._post_incomplete = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._vad.reset()
        await self._on_complete(audio)
        return True

    async def feed_audio(self, pcm_48k: bytes) -> None:
        """Resample, frame into 16 ms VAD windows, advance state machine."""
        if not pcm_48k:
            return
        resampled = self._resampler.feed(pcm_48k)
        if not resampled:
            return
        self._frame_carry.extend(resampled)
        # Consume as many full VAD frames as carry holds
        while len(self._frame_carry) >= _VAD_CHUNK_BYTES:
            frame = bytes(self._frame_carry[:_VAD_CHUNK_BYTES])
            del self._frame_carry[:_VAD_CHUNK_BYTES]
            await self._on_vad_frame(frame)

    async def _on_vad_frame(self, frame: bytes) -> None:
        """Single 16 ms decision step."""
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

        # Silence frame
        self._speech_streak = 0
        if not self._speaking:
            # Idle silence — drop the buffer to bound memory; nothing to classify
            self._utterance.clear()
            return

        self._silence_streak += 1
        if self._silence_streak < self._silence_chunks_threshold:
            return

        # Silence threshold hit after speech → classify (unless already
        # did and got `incomplete` with no new speech since).
        if self._post_incomplete:
            return

        # SmartTurn ONNX runs wav2vec2 over up to 16 s of audio;
        # dispatch off-loop so slow call can't stall Deepgram
        # keepalives or Discord voice heartbeat (10 s watchdog).
        verdict = await asyncio.to_thread(
            self._smart_turn.is_complete, bytes(self._utterance)
        )
        if verdict:
            audio = bytes(self._utterance)
            self._utterance.clear()
            self._speaking = False
            self._post_incomplete = False
            self._silence_streak = 0
            self._vad.reset()
            await self._on_complete(audio)
        else:
            # Keep buffer; await fresh speech, then fresh silence streak
            self._post_incomplete = True
            self._speaking = False
