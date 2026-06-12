"""Smart Turn v3 wrapper — semantic utterance-completion classifier.

Pipecat's `Smart Turn v3 <https://github.com/pipecat-ai/smart-turn>`_
is a small wav2vec2-derived model classifying whether the current
audio buffer ends on a turn-complete boundary. Where TEN-VAD answers
"is anyone speaking right now?", Smart Turn answers "did the speaker
actually finish?". Trained on filler-word audio that STT routinely
drops — why it beats transcription-based endpointing.

Stateless — feed buffered utterance audio after VAD reports silence;
classifier returns completion probability.

Output handles both common export shapes:

- 2-class logits ``[incomplete, complete]`` → softmax, take class 1
- single sigmoid logit ``[complete_score]`` → 1/(1+exp(-x))

Lazy-imports onnxruntime so runtime cost paid only when ``local-turn``
extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

_IMPORT_ERROR: Exception | None
if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort

    _IMPORT_ERROR = None
else:
    try:
        import numpy as np
        import onnxruntime as ort

        _IMPORT_ERROR = None
    except ImportError as exc:  # pragma: no cover — only hit without the extra
        np = None
        ort = None
        _IMPORT_ERROR = exc

if TYPE_CHECKING:
    from pathlib import Path


class SmartTurnDetector:
    """Semantic turn-completion classifier (Pipecat Smart Turn v3 ONNX)."""

    def __init__(
        self,
        model_path: Path,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        max_duration_s: float = 16.0,
    ) -> None:
        if _IMPORT_ERROR is not None:
            msg = (
                "SmartTurnDetector requires the 'local-turn' extra. "
                "Install with `uv sync --extra local-turn`."
            )
            raise RuntimeError(msg) from _IMPORT_ERROR
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._max_samples = int(max_duration_s * sample_rate)
        self._session = ort.InferenceSession(str(model_path))
        # First input name on graph — Pipecat exports use ``input_values``
        # (Wav2Vec2 convention) but stay resilient
        self._input_name = self._session.get_inputs()[0].name

    def completion_probability(self, pcm_audio: bytes) -> float:
        """Run classifier; return ``[0, 1]`` is-complete probability."""
        audio = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.shape[0] > self._max_samples:
            # Keep most recent window — turn-end semantics live there
            audio = audio[-self._max_samples :]
        outputs = self._session.run(
            None,
            {self._input_name: audio[np.newaxis, :]},
        )
        # ONNX typed union includes SparseTensor; classifier exports
        # always emit dense ndarray for the logits head
        logits = cast("np.ndarray", outputs[0])
        last_dim = logits.shape[-1]
        if last_dim == 2:
            # Softmax over [incomplete, complete]; numerically stable
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=-1, keepdims=True)
            return float(probs[0, 1])
        if last_dim == 1:
            return float(1.0 / (1.0 + np.exp(-logits[0, 0])))
        msg = f"unsupported logits shape {logits.shape}; expected last dim 1 or 2"
        raise ValueError(msg)

    def is_complete(self, pcm_audio: bytes) -> bool:
        return self.completion_probability(pcm_audio) >= self.threshold
