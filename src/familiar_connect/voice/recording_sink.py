"""Custom py-cord Sink that bridges threaded audio capture to asyncio."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from discord.sinks import Filters, Sink

from familiar_connect.voice.audio import stereo_to_mono

if TYPE_CHECKING:
    import asyncio
    from typing import Self

_logger = logging.getLogger(__name__)


class RecordingSink(Sink):
    """A Sink that converts stereo PCM to mono and queues it for transcription.

    py-cord's ``VoiceClient.start_recording`` calls :meth:`write` from a
    background thread. This sink bridges to the asyncio event loop using
    :meth:`loop.call_soon_threadsafe` so the transcription pipeline can
    consume audio from an :class:`asyncio.Queue`.
    """

    def __init__(
        self: Self,
        *,
        loop: asyncio.AbstractEventLoop,
        audio_queue: asyncio.Queue[tuple[int, bytes]],
        filters: dict[str, object] | None = None,
    ) -> None:
        super().__init__(filters=filters)
        self._loop = loop
        self._audio_queue = audio_queue

    @staticmethod
    def stereo_to_mono(data: bytes) -> bytes:
        """Delegate to the audio module for testability."""
        return stereo_to_mono(data)

    @Filters.container
    def write(self: Self, data: bytes, user: int) -> None:
        """Convert stereo PCM to mono and push ``(user, mono)`` to the queue.

        Called from py-cord's recording thread — must not use ``await``.
        The tuple allows the asyncio-side audio router to attribute audio
        to the correct Discord user.
        """
        mono = self.stereo_to_mono(data)
        _logger.debug(
            "[Sink] user=%d stereo=%d bytes → mono=%d bytes",
            user,
            len(data),
            len(mono),
        )
        self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, (user, mono))

    def cleanup(self: Self) -> None:
        """Signal that recording has finished."""
        self.finished = True

    def format_audio(self: Self, audio: object) -> None:
        """No-op — this sink does not write audio to files."""
