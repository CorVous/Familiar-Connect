"""Deepgram interim-word voice-activity detector.

Drives per-user ``on_speech_start`` / ``on_speech_end`` edges from
Deepgram :class:`~familiar_connect.transcription.TranscriptionResult`
arrivals (interim + final) instead of a local ML VAD model.

On first non-empty result: emits ``on_speech_start`` and arms a silence
watchdog. Each subsequent result resets the watchdog. When the watchdog
fires after ``silence_timeout_ms`` of no new results: emits
``on_speech_end`` and clears per-user state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls

if TYPE_CHECKING:
    from collections.abc import Callable

    from familiar_connect.transcription import TranscriptionResult

_logger = logging.getLogger(__name__)

DEFAULT_SILENCE_TIMEOUT_MS: int = 700
"""Silence window before firing ``on_speech_end``.

Chosen between normal inter-interim gaps (~100-300 ms) and the
conversational lull timeout (default 5 s). Robust to brief pauses
within an utterance while still detecting genuine stops well before
``VoiceLullMonitor`` fires.
"""


@dataclass
class _UserState:
    speaking: bool = False
    handle: asyncio.TimerHandle | None = field(default=None, repr=False)


class DeepgramVoiceActivityDetector:
    """Per-user VAD driven by Deepgram transcript arrivals.

    Replaces TEN VAD as the source of speech-start / speech-end edges
    that feed :class:`~familiar_connect.voice_lull.VoiceLullMonitor` and
    :class:`~familiar_connect.voice.interruption.InterruptionDetector`.
    """

    def __init__(
        self,
        *,
        on_speech_start: Callable[[int], None],
        on_speech_end: Callable[[int], None],
        silence_timeout_ms: int = DEFAULT_SILENCE_TIMEOUT_MS,
    ) -> None:
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._silence_timeout_s = silence_timeout_ms / 1000.0
        self._users: dict[int, _UserState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_transcript(self, user_id: int, result: TranscriptionResult) -> None:
        """Ingest a Deepgram result for *user_id*.

        Fires ``on_speech_start`` on first non-empty result while the
        user is not already speaking. Always re-arms the silence watchdog.
        Empty-text results are discarded.
        """
        if not result.text:
            return

        state = self._users.get(user_id)
        if state is None:
            state = _UserState()
            self._users[user_id] = state

        if not state.speaking:
            state.speaking = True
            _logger.info(
                f"{ls.tag('🗣️ DGVAD', ls.LG)} "
                f"{ls.kv('event', 'started', vc=ls.LG)} "
                f"{ls.kv('user', str(user_id), vc=ls.LC)}"
            )
            self._on_speech_start(user_id)

        self._arm_watchdog(user_id, state)

    def reset(self) -> None:
        """Cancel all pending watchdogs and drop per-user state.

        Call on voice-channel disconnect. Do NOT call on Deepgram
        reconnect — state should survive reconnects.
        """
        for state in self._users.values():
            if state.handle is not None:
                state.handle.cancel()
                state.handle = None
        self._users.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _arm_watchdog(self, user_id: int, state: _UserState) -> None:
        if state.handle is not None:
            state.handle.cancel()
        loop = asyncio.get_running_loop()
        state.handle = loop.call_later(
            self._silence_timeout_s,
            self._on_watchdog,
            user_id,
        )

    def _on_watchdog(self, user_id: int) -> None:
        state = self._users.get(user_id)
        if state is None or not state.speaking:
            return
        state.speaking = False
        state.handle = None
        _logger.info(
            f"{ls.tag('🗣️ DGVAD', ls.Y)} "
            f"{ls.kv('event', 'ended', vc=ls.Y)} "
            f"{ls.kv('user', str(user_id), vc=ls.LC)}"
        )
        self._on_speech_end(user_id)
