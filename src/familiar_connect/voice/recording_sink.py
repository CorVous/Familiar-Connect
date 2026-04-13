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
    """Stereo-to-mono conversion + thread-safe queue bridge for transcription.

    py-cord calls :meth:`write` from a background thread;
    ``call_soon_threadsafe`` pushes to the asyncio queue.
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
        """Delegate to audio module for testability."""
        return stereo_to_mono(data)

    @Filters.container
    def write(self: Self, data: bytes, user: int) -> None:
        """Convert stereo to mono, push ``(user, mono)`` to queue.

        Called from py-cord's recording thread — must not ``await``.
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
        """Signal recording finished."""
        self.finished = True

    def format_audio(self: Self, audio: object) -> None:
        """No-op — sink does not write audio to files."""
