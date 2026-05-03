"""STT backend selector — dispatches on ``[providers.stt].backend``.

V3 phase 1: only ``deepgram`` is implemented. Parakeet / FasterWhisper
land in later phases and will register additional dispatch arms here.

Env-override precedence mirrors :mod:`voice.turn_detection.factory`:
``STT_BACKEND`` wins over the TOML value, so container deployments can
flip backends without rebuilding the image.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from familiar_connect.stt.deepgram import (
    DeepgramTranscriber,
    create_transcriber_from_env,
)

if TYPE_CHECKING:
    from familiar_connect.config import STTConfig
    from familiar_connect.stt.protocol import Transcriber


_KNOWN_BACKENDS: frozenset[str] = frozenset({"deepgram", "parakeet", "faster_whisper"})


def _resolve_backend(toml_backend: str) -> str:
    """``STT_BACKEND`` env wins over TOML (container override)."""
    return os.environ.get("STT_BACKEND") or toml_backend


def create_transcriber(config: STTConfig) -> Transcriber:
    """Build a :class:`Transcriber` from *config*.

    :raises ValueError: backend unknown, or selected backend's required
        env (e.g. ``DEEPGRAM_API_KEY``) is missing, or the backend's
        optional extra is not installed. Caller is expected to log +
        degrade — see ``commands/run.py``.
    """
    backend = _resolve_backend(config.backend)
    if backend not in _KNOWN_BACKENDS:
        msg = (
            f"Unknown STT backend {backend!r}. "
            f"Known: {sorted(_KNOWN_BACKENDS)}. "
            f"Set [providers.stt].backend in character.toml or STT_BACKEND env."
        )
        raise ValueError(msg)

    if backend == "deepgram":
        return _create_deepgram(config)
    if backend == "parakeet":
        return _create_parakeet(config)
    if backend == "faster_whisper":
        return _create_faster_whisper(config)

    # unreachable while every _KNOWN_BACKENDS entry has a dispatch arm
    msg = f"backend {backend!r} accepted but not dispatched"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def _create_deepgram(config: STTConfig) -> DeepgramTranscriber:
    return create_transcriber_from_env(config.deepgram)


def _create_parakeet(config: STTConfig) -> Transcriber:
    """Lazy-import Parakeet so the heavy numpy/NeMo deps stay optional."""
    try:
        from familiar_connect.stt.parakeet import (  # noqa: PLC0415
            create_parakeet_from_env,
        )
    except RuntimeError as exc:
        # numpy missing → ``ParakeetTranscriber`` module raises on import.
        # Re-raise as ValueError so ``run.py`` catches + warns uniformly.
        raise ValueError(str(exc)) from exc
    return create_parakeet_from_env(config.parakeet)


def _create_faster_whisper(config: STTConfig) -> Transcriber:
    """Lazy-import FasterWhisper so its CT2 deps stay optional."""
    try:
        from familiar_connect.stt.faster_whisper import (  # noqa: PLC0415
            create_faster_whisper_from_env,
        )
    except RuntimeError as exc:
        # numpy missing → module raises on import; surface as ValueError.
        raise ValueError(str(exc)) from exc
    return create_faster_whisper_from_env(config.faster_whisper)
