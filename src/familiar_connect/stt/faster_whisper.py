"""FasterWhisper (CTranslate2) transcriber backend.

`faster_whisper.WhisperModel` running locally. Buffer-and-finalize
semantics, same shape as :mod:`familiar_connect.stt.parakeet`:
``send_audio`` appends 48 kHz Discord PCM (resampled to 16 kHz mono);
``finalize`` runs inference on the buffered audio and emits a single
``TranscriptionResult`` with ``is_final=True``.

Pair with ``[providers.turn_detection].strategy = "ten+smart_turn"`` —
Whisper has no built-in endpointer in this wiring (we don't lean on its
VAD; the local turn detector decides when to fire ``finalize()``).

Numpy is required at import time; ``faster_whisper`` itself is loaded
lazily on first ``start()``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover — only hit without the extra
    msg = (
        "FasterWhisperTranscriber requires the 'local-stt-whisper' extra. "
        "Install with `uv sync --extra local-stt-whisper`."
    )
    raise RuntimeError(msg) from exc

from familiar_connect import log_style as ls
from familiar_connect.config import FasterWhisperSTTConfig
from familiar_connect.stt.protocol import TranscriptionEvent, TranscriptionResult
from familiar_connect.voice.audio import Resampler48to16

if TYPE_CHECKING:
    from typing import Self

_logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "small"
DEFAULT_COMPUTE_TYPE = "auto"
TARGET_SAMPLE_RATE = 16_000
INT16_MAX = 32768.0


class FasterWhisperTranscriber:
    """Local CTranslate2-backed Whisper transcriber."""

    # bot-side per-user idle window. read by ``bot._start_voice_intake`` to
    # spawn the idle watchdog. parity with sibling backends.
    _IDLE_CLOSE_S: float = 30.0

    def __init__(
        self: Self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "auto",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str = "en",
        sample_rate: int = 48_000,
        channels: int = 1,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        # WhisperModel handle — opaque ``Any`` so the module imports
        # without faster_whisper installed (lazy-loaded in ``start()``).
        self._model: Any = None
        self._resampler = Resampler48to16()
        self._buffer = bytearray()
        self._output: asyncio.Queue[TranscriptionEvent] | None = None
        self._finalize_lock = asyncio.Lock()

    def clone(self: Self) -> FasterWhisperTranscriber:
        """Fresh per-user instance; loaded model handle is shared."""
        c = FasterWhisperTranscriber(
            model_size=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        c._model = self._model
        c._IDLE_CLOSE_S = self._IDLE_CLOSE_S
        return c

    def _load_model(self: Self) -> Any:  # noqa: ANN401 — opaque WhisperModel
        """Import faster_whisper and load the CT2 model. Blocking."""
        try:
            from faster_whisper import (  # ty: ignore[unresolved-import]  # noqa: PLC0415
                WhisperModel,
            )
        except ImportError as exc:
            msg = (
                "Loading FasterWhisper requires the 'local-stt-whisper' extra. "
                "Install with `uv sync --extra local-stt-whisper`."
            )
            raise RuntimeError(msg) from exc

        _logger.info(
            f"{ls.tag('🐦 Whisper', ls.LG)} "
            f"{ls.kv('loading', self.model_size, vc=ls.LW)} "
            f"{ls.kv('device', self.device, vc=ls.LW)} "
            f"{ls.kv('compute', self.compute_type, vc=ls.LW)}"
        )
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    async def start(self: Self, output: asyncio.Queue[TranscriptionEvent]) -> None:
        """Stash output queue; lazy-load the model on first call."""
        self._output = output
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)
            _logger.info(
                f"{ls.tag('🐦 Whisper', ls.G)} "
                f"{ls.kv('ready', self.model_size, vc=ls.LW)}"
            )

    async def send_audio(self: Self, data: bytes) -> None:
        """Resample 48 kHz → 16 kHz int16 PCM and append to the buffer."""
        pcm16 = self._resampler.feed(data)
        if pcm16:
            self._buffer.extend(pcm16)

    async def finalize(self: Self) -> None:
        """Run inference on buffered audio and emit one final transcript."""
        async with self._finalize_lock:
            if not self._buffer or self._output is None or self._model is None:
                return
            audio = (
                np.frombuffer(bytes(self._buffer), dtype=np.int16).astype(np.float32)
                / INT16_MAX
            )
            duration = len(audio) / TARGET_SAMPLE_RATE
            self._buffer.clear()

            text = await asyncio.to_thread(self._transcribe, audio)
            if not text:
                return
            await self._output.put(
                TranscriptionResult(
                    text=text,
                    is_final=True,
                    start=0.0,
                    end=duration,
                )
            )

    def _transcribe(self: Self, audio: Any) -> str:  # noqa: ANN401 — np.ndarray
        """Run Whisper inference; concatenate segment texts."""
        if self._model is None:
            return ""
        segments, _info = self._model.transcribe(audio, language=self.language)
        # ``segments`` is a generator — must be consumed to drive the model.
        text = "".join(seg.text for seg in segments)
        return text.strip()

    async def stop(self: Self) -> None:
        """Reset per-instance buffer + resampler. Model handle persists."""
        self._buffer.clear()
        self._resampler.reset()
        self._output = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_faster_whisper_transcriber(
    config: FasterWhisperSTTConfig | None = None,
) -> FasterWhisperTranscriber:
    """Build :class:`FasterWhisperTranscriber` from typed *config*."""
    cfg = config or FasterWhisperSTTConfig()
    t = FasterWhisperTranscriber(
        model_size=cfg.model_size,
        device=cfg.device,
        compute_type=cfg.compute_type,
        language=cfg.language,
    )
    t._IDLE_CLOSE_S = cfg.idle_close_s  # noqa: SLF001
    return t
