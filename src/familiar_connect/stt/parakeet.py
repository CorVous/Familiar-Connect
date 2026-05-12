"""Parakeet-TDT 0.6B v3 transcriber backend.

NeMo `EncDecRNNTBPEModel` running locally. Buffer-and-finalize semantics:
``send_audio`` appends 48 kHz Discord PCM (resampled to 16 kHz mono) to a
per-instance bytearray; ``finalize`` runs inference on the buffered audio
and emits a single ``TranscriptionResult`` with ``is_final=True``.

Pair with ``[providers.turn_detection].strategy = "ten+smart_turn"`` —
Parakeet has no internal endpointer. Without a local turn detector, the
buffer never gets flushed and no transcripts surface.

Numpy is required at import time (matches ``smart_turn`` pattern); NeMo
itself is loaded lazily on first ``start()``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover — only hit without the extra
    msg = (
        "ParakeetTranscriber requires the 'local-stt' extra. Install with "
        "`uv sync --extra local-stt`."
    )
    raise RuntimeError(msg) from exc

from familiar_connect import log_style as ls
from familiar_connect.config import ParakeetSTTConfig
from familiar_connect.stt.protocol import TranscriptionEvent, TranscriptionResult
from familiar_connect.voice.audio import Resampler48to16

if TYPE_CHECKING:
    from typing import Self

_logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SAMPLE_RATE = 16_000
INT16_MAX = 32768.0


class ParakeetTranscriber:
    """Local Parakeet-TDT transcriber."""

    # bot-side per-user idle window. read by ``bot._start_voice_intake`` to
    # spawn the idle watchdog. parity with ``DeepgramTranscriber._IDLE_CLOSE_S``.
    _IDLE_CLOSE_S: float = 30.0

    def __init__(
        self: Self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        sample_rate: int = 48_000,
        channels: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        # NeMo ASRModel handle — opaque ``Any`` so the module imports
        # without nemo installed (lazy-loaded in ``start()``).
        self._model: Any = None
        self._resampler = Resampler48to16()
        self._buffer = bytearray()
        self._output: asyncio.Queue[TranscriptionEvent] | None = None
        # serialise overlapping finalize calls (idle watchdog + endpointer)
        self._finalize_lock = asyncio.Lock()

    def clone(self: Self) -> ParakeetTranscriber:
        """Fresh per-user instance; loaded model handle is shared."""
        c = ParakeetTranscriber(
            model_name=self.model_name,
            device=self.device,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        # share the heavy model — load once per process
        c._model = self._model
        c._IDLE_CLOSE_S = self._IDLE_CLOSE_S
        return c

    def _load_model(self: Self) -> Any:  # noqa: ANN401 — opaque NeMo handle
        """Import NeMo and load the ASR model. Blocking — call in a thread."""
        try:
            import nemo.collections.asr as nemo_asr  # ty: ignore[unresolved-import]  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "Loading Parakeet requires the 'local-stt' extra. Install with "
                "`uv sync --extra local-stt`."
            )
            raise RuntimeError(msg) from exc

        _logger.info(
            f"{ls.tag('🧠 Parakeet', ls.LG)} "
            f"{ls.kv('loading', self.model_name, vc=ls.LW)} "
            f"{ls.kv('device', self.device, vc=ls.LW)}"
        )
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        if self.device in {"cuda", "cpu"}:
            model = model.to(self.device)
        return model

    async def start(self: Self, output: asyncio.Queue[TranscriptionEvent]) -> None:
        """Stash output queue; lazy-load the model on first call.

        Subsequent ``start`` calls (per-user clones share the model) are
        cheap — model handle is already populated by ``clone()``.
        """
        self._output = output
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)
            _logger.info(
                f"{ls.tag('🧠 Parakeet', ls.G)} "
                f"{ls.kv('ready', self.model_name, vc=ls.LW)}"
            )

    async def send_audio(self: Self, data: bytes) -> None:
        """Resample 48 kHz → 16 kHz int16 PCM and append to the buffer."""
        pcm16 = self._resampler.feed(data)
        if pcm16:
            self._buffer.extend(pcm16)

    async def finalize(self: Self) -> None:
        """Run inference on buffered audio and emit one final transcript.

        No-op when the buffer is empty (idle watchdog, double-finalize
        from overlapping endpointer + reconnect, etc.). Lock guards
        against concurrent calls draining the same buffer twice.
        """
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
        """Run NeMo inference on a single utterance; return stripped text."""
        if self._model is None:
            return ""
        result = self._model.transcribe([audio])
        if not result:
            return ""
        first = result[0]
        # NeMo 1.x returns list[str]; 2.x returns list[Hypothesis] with .text.
        text = first.text if hasattr(first, "text") else first
        return str(text).strip()

    async def stop(self: Self) -> None:
        """Reset per-instance buffer + resampler. Model handle persists."""
        self._buffer.clear()
        self._resampler.reset()
        self._output = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_parakeet_transcriber(
    config: ParakeetSTTConfig | None = None,
) -> ParakeetTranscriber:
    """Build :class:`ParakeetTranscriber` from typed *config*."""
    cfg = config or ParakeetSTTConfig()
    t = ParakeetTranscriber(model_name=cfg.model_name, device=cfg.device)
    t._IDLE_CLOSE_S = cfg.idle_close_s  # noqa: SLF001
    return t
