"""TTSPlayer Protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from familiar_connect.bus.envelope import TurnScope


class TTSPlayer(Protocol):
    """Synthesize text and play it. Cancellable mid-speech."""

    async def speak(self, text: str, *, scope: TurnScope) -> None:
        """Speak ``text`` until complete or ``scope`` is cancelled.

        Implementations should check ``scope.is_cancelled()`` at the
        finest granularity their API permits (per word-timestamp
        ideally; per audio-chunk at worst).
        """
        ...

    async def stop(self) -> None:
        """Flush any in-flight audio immediately."""
        ...
