"""STT backend selector — dispatches on ``[providers.stt].backend``.

Each per-backend factory takes its typed ``[providers.stt.<backend>]``
sub-table; only secrets (e.g. ``DEEPGRAM_API_KEY``) come from env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.stt.deepgram import (
    DeepgramTranscriber,
    create_deepgram_transcriber,
)

if TYPE_CHECKING:
    from familiar_connect.config import STTConfig
    from familiar_connect.stt.protocol import Transcriber


_KNOWN_BACKENDS: frozenset[str] = frozenset({"deepgram", "parakeet", "faster_whisper"})


def create_transcriber(config: STTConfig) -> Transcriber:
    """Build a :class:`Transcriber` from *config*.

    :raises ValueError: backend unknown, or the selected backend's
        required secret (e.g. ``DEEPGRAM_API_KEY``) is missing, or its
        optional extra is not installed. Caller is expected to log +
        degrade — see ``commands/run.py``.
    """
    backend = config.backend
    if backend not in _KNOWN_BACKENDS:
        msg = (
            f"Unknown STT backend {backend!r}. "
            f"Known: {sorted(_KNOWN_BACKENDS)}. "
            f"Set [providers.stt].backend in character.toml."
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
    return create_deepgram_transcriber(config.deepgram)


def _create_parakeet(config: STTConfig) -> Transcriber:
    """Lazy-import Parakeet so the heavy numpy/NeMo deps stay optional."""
    try:
        from familiar_connect.stt.parakeet import (  # noqa: PLC0415
            create_parakeet_transcriber,
        )
    except RuntimeError as exc:
        # numpy missing → ``ParakeetTranscriber`` module raises on import.
        # Re-raise as ValueError so ``run.py`` catches + warns uniformly.
        raise ValueError(str(exc)) from exc
    return create_parakeet_transcriber(config.parakeet)


def _create_faster_whisper(config: STTConfig) -> Transcriber:
    """Lazy-import FasterWhisper so its CT2 deps stay optional."""
    try:
        from familiar_connect.stt.faster_whisper import (  # noqa: PLC0415
            create_faster_whisper_transcriber,
        )
    except RuntimeError as exc:
        # numpy missing → module raises on import; surface as ValueError.
        raise ValueError(str(exc)) from exc
    return create_faster_whisper_transcriber(config.faster_whisper)
