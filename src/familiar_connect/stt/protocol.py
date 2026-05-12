"""Transcriber Protocol + shared transcription result types.

V3 phase 1: lifts the implicit shape of :class:`DeepgramTranscriber` into a
``Protocol`` so local-model backends (FasterWhisper, Parakeet) drop in
behind ``[providers.stt].backend`` without code-path changes downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from familiar_connect.llm import Message

if TYPE_CHECKING:
    import asyncio
    from typing import Self


@dataclass
class TranscriptionResult:
    """Single streaming transcription result.

    user_id: Discord user id when known; stamped by per-user fan-in
    in :func:`bot._start_voice_intake`. Per-SSRC audio gives attribution
    for free, no provider diarization needed.
    speaker: provider-side diarization label (when enabled); unused by
    Discord pipeline.
    """

    text: str
    is_final: bool
    start: float
    end: float
    confidence: float = 0.0
    speaker: int | None = None
    user_id: int | None = None

    def to_message(self: Self, speaker_names: dict[int, str] | None = None) -> Message:
        """Convert to LLM Message; prefix content with ``[Voice]``.

        ``speaker_names`` may key by ``user_id`` (Discord) or by provider
        ``speaker`` label — ``user_id`` wins.
        """
        name = "Voice"
        if speaker_names is not None:
            if self.user_id is not None and self.user_id in speaker_names:
                name = speaker_names[self.user_id]
            elif self.speaker is not None and self.speaker in speaker_names:
                name = speaker_names[self.speaker]
        return Message(role="user", content=f"[Voice] {self.text}", name=name)


# transcriber drives transcription text + VAD edges:
# interims → DeepgramVoiceActivityDetector (speech-start / speech-end)
# finals → VoiceLullMonitor (conversational-lull detection)
TranscriptionEvent = TranscriptionResult


@runtime_checkable
class Transcriber(Protocol):
    """Streaming PCM → :class:`TranscriptionResult` queue surface.

    Implementations: :class:`stt.deepgram.DeepgramTranscriber` (today);
    Parakeet / FasterWhisper backends in later V3 phases.

    Lifecycle: ``clone()`` per user/channel, ``await start(queue)`` once,
    ``await send_audio(pcm)`` per chunk, ``await finalize()`` to flush
    pending segment, ``await stop()`` to tear down.
    """

    def clone(self: Self) -> Transcriber:
        """Fresh independent instance with same config."""
        ...

    async def start(self: Self, output: asyncio.Queue[TranscriptionEvent]) -> None:
        """Begin transcription, pushing :class:`TranscriptionResult`s onto *output*."""
        ...

    async def send_audio(self: Self, data: bytes) -> None:
        """Feed one PCM chunk (linear16, sample rate set by impl)."""
        ...

    async def finalize(self: Self) -> None:
        """Force the impl to flush any pending segment as a final.

        No-op when nothing to flush. Local endpointer (V1) calls this on
        turn-complete verdict to short-circuit Deepgram's silence wait.
        """
        ...

    async def stop(self: Self) -> None:
        """Tear down. Idempotent."""
        ...
