"""TEN-VAD wrapper — voice activity detection on 16 kHz mono PCM.

Wraps Agora's TEN-VAD (Apache 2.0 + extras). Native shared library
ships with the ``ten-vad`` Python package; the ONNX model is bundled
inside the wheel, so callers don't supply a model path.

Feed 16 ms (256-sample) or 10 ms (160-sample) frames of 16 kHz mono
int16 PCM; get back ``(probability, flag)``. The flag is the model's
binary verdict at the configured threshold; we read the probability
and re-threshold here so the same instance can be re-tuned without
rebuilding the native handle.

Stateful: the underlying C handle accumulates across :meth:`process`
calls. :meth:`reset` recreates it so per-utterance state doesn't leak.

Lazy-imports ``ten_vad`` so projects without the ``local-turn`` extra
never pay the runtime cost.
"""

from __future__ import annotations

try:
    import numpy as np
    from ten_vad import TenVad as _TenVadNative
except ImportError as exc:  # pragma: no cover — only hit without the extra
    msg = (
        "TenVAD requires the 'local-turn' extra. Install with "
        "`uv sync --extra local-turn`."
    )
    raise RuntimeError(msg) from exc

# TEN-VAD native hop sizes at 16 kHz: 160 (10 ms) or 256 (16 ms).
_VALID_HOP_SIZES_16K: tuple[int, ...] = (160, 256)


class TenVAD:
    """In-process voice activity detector (Agora TEN-VAD)."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        hop_size: int = 256,
        threshold: float = 0.5,
    ) -> None:
        if sample_rate != 16000:
            msg = f"sample_rate must be 16000 Hz; got {sample_rate}"
            raise ValueError(msg)
        if hop_size not in _VALID_HOP_SIZES_16K:
            valid = ", ".join(str(c) for c in _VALID_HOP_SIZES_16K)
            msg = f"hop_size {hop_size} not supported at 16 kHz; valid: {valid}"
            raise ValueError(msg)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.threshold = threshold
        self._vad = _TenVadNative(hop_size=hop_size, threshold=threshold)

    def reset(self) -> None:
        """Recreate the native handle — call between utterances.

        TEN-VAD's C handle exposes no public reset; rebuild to drop
        accumulated state.
        """
        self._vad = _TenVadNative(hop_size=self.hop_size, threshold=self.threshold)

    def speech_probability(self, pcm_chunk: bytes) -> float:
        """Run inference; return ``[0, 1]`` speech probability."""
        audio = np.frombuffer(pcm_chunk, dtype=np.int16)
        if audio.shape[0] != self.hop_size:
            msg = f"expected {self.hop_size} samples, got {audio.shape[0]}"
            raise ValueError(msg)
        prob, _flag = self._vad.process(audio)
        return float(prob)

    def is_speech(self, pcm_chunk: bytes) -> bool:
        return self.speech_probability(pcm_chunk) >= self.threshold
