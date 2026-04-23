"""Logging-only TTS player.

Phase-2 production default — logs what it *would* speak but does not
produce audio. Discord voice-channel playback is a follow-up phase;
until then the responder's output is visible in the log stream so the
pipeline can be exercised end-to-end without hitting a Cartesia/Azure
API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls

if TYPE_CHECKING:
    from familiar_connect.bus.envelope import TurnScope

_logger = logging.getLogger("familiar_connect.tts_player.logging")


class LoggingTTSPlayer:
    """No-audio TTS stand-in.

    Honours cancellation semantics so the barge-in loop behaves the
    same as with a real player.
    """

    def __init__(self, *, ms_per_word: int = 200, poll_ms: int = 20) -> None:
        self._ms_per_word = ms_per_word
        self._poll_ms = poll_ms
        self._stop_event = asyncio.Event()

    async def speak(self, text: str, *, scope: TurnScope) -> None:
        words = text.split()
        budget_ms = len(words) * self._ms_per_word
        played_ms = 0
        _logger.info(
            f"{ls.tag('🔊 Say', ls.G)} "
            f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
            f"{ls.kv('words', str(len(words)), vc=ls.LY)} "
            f"{ls.kv('text', ls.trunc(text, 200), vc=ls.LW)}"
        )
        self._stop_event = asyncio.Event()

        while played_ms < budget_ms:
            if scope.is_cancelled() or self._stop_event.is_set():
                _logger.info(
                    f"{ls.tag('🔊 Cut', ls.Y)} "
                    f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
                    f"{ls.kv('played_ms', str(played_ms), vc=ls.LY)}"
                )
                return
            step = min(self._poll_ms, budget_ms - played_ms)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=step / 1000.0,
                )
            except TimeoutError:
                played_ms += step
                continue
            return

    async def stop(self) -> None:
        self._stop_event.set()
