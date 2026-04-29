"""Silero VAD wrapper — voice activity detection on 16 kHz mono PCM.

Stateful: maintains the model's hidden state across calls so probability
estimates respect conversational context. Reset between utterances via
:meth:`reset`. Designed around Silero v5's ONNX I/O shape:

- ``input``: ``float32[1, 512]`` (32 ms at 16 kHz)
- ``state``: ``float32[2, 1, 128]`` (kept on the wrapper, fed back each call)
- ``sr``: ``int64`` scalar tensor

Lazy-imports onnxruntime so projects that don't install the
``local-turn`` extra never pay the runtime cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover — only hit without the extra
    msg = (
        "SileroVAD requires the 'local-turn' extra. Install with "
        "`uv sync --extra local-turn`."
    )
    raise RuntimeError(msg) from exc

if TYPE_CHECKING:
    from pathlib import Path

# Silero v5 accepts only these chunk sizes at 16 kHz.
_VALID_CHUNK_SIZES_16K: tuple[int, ...] = (256, 512)
_STATE_SHAPE: tuple[int, ...] = (2, 1, 128)


class SileroVAD:
    """In-process voice activity detector (Silero v5 ONNX)."""

    def __init__(
        self,
        model_path: Path,
        *,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        threshold: float = 0.5,
    ) -> None:
        if sample_rate == 16000 and chunk_size not in _VALID_CHUNK_SIZES_16K:
            valid = ", ".join(str(c) for c in _VALID_CHUNK_SIZES_16K)
            msg = f"chunk_size {chunk_size} not supported at 16 kHz; valid: {valid}"
            raise ValueError(msg)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self._session = ort.InferenceSession(str(model_path))
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._sr = np.array(sample_rate, dtype=np.int64)

    def reset(self) -> None:
        """Zero the hidden state — call between utterances."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)

    def speech_probability(self, pcm_chunk: bytes) -> float:
        """Run inference; return ``[0, 1]`` speech probability."""
        audio = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.shape[0] != self.chunk_size:
            msg = f"expected {self.chunk_size} samples, got {audio.shape[0]}"
            raise ValueError(msg)
        prob, new_state = self._session.run(
            None,
            {
                "input": audio[np.newaxis, :],
                "state": self._state,
                "sr": self._sr,
            },
        )
        # ONNX's typed union covers SparseTensor; the Silero v5 graph
        # always emits dense ndarrays for both outputs.
        self._state = cast("np.ndarray", new_state)
        return float(cast("np.ndarray", prob)[0][0])

    def is_speech(self, pcm_chunk: bytes) -> bool:
        return self.speech_probability(pcm_chunk) >= self.threshold
