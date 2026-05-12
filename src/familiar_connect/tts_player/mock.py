"""In-process mock TTS player for tests.

Simulates playback by sleeping a fraction of a word duration per
tick; checks :class:`TurnScope.is_cancelled` between ticks so
barge-in tests can measure how promptly cancellation cuts speech
short. Records the played duration for assertions.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.bus.envelope import TurnScope


class MockTTSPlayer:
    """Record-what-you-played TTS stand-in."""

    def __init__(self, *, ms_per_word: int = 200, poll_ms: int = 5) -> None:
        self._ms_per_word = ms_per_word
        self._poll_ms = poll_ms
        self.calls: list[tuple[str, bool]] = []  # (text, was_cancelled_or_stopped)
        self.total_played_ms: int = 0
        self._stop_event = asyncio.Event()

    async def speak(self, text: str, *, scope: TurnScope) -> None:
        words = text.split()
        budget_ms = len(words) * self._ms_per_word
        played_ms = 0
        cancelled_or_stopped = False

        # Reset the per-call stop gate on re-entry.
        self._stop_event = asyncio.Event()

        while played_ms < budget_ms:
            if scope.is_cancelled() or self._stop_event.is_set():
                cancelled_or_stopped = True
                break
            step = min(self._poll_ms, budget_ms - played_ms)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=step / 1000.0,
                )
            except TimeoutError:
                played_ms += step
                continue
            # stop_event fired during the wait
            cancelled_or_stopped = True
            break

        self.total_played_ms += played_ms
        self.calls.append((text, cancelled_or_stopped))

    async def stop(self) -> None:
        self._stop_event.set()
