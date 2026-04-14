"""Voice lull monitor — debounce voice utterances until speaker pauses.

Buffers per-speaker transcripts until channel-wide silence exceeds
``lull_timeout``; merged utterance then fires to response pipeline.
Silence detected via Discord audio frames (no Deepgram VAD).
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING

from familiar_connect.transcription import TranscriptionResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_logger = logging.getLogger(__name__)


class VoiceActivityEvent(Enum):
    """Per-user speaking transitions from Discord audio frames."""

    started = "started"
    """User began speaking."""

    ended = "ended"
    """User stopped speaking (silence watchdog fired)."""


class VoiceLullMonitor:
    """Debounce voice utterances by watching Discord audio activity."""

    def __init__(
        self,
        *,
        lull_timeout: float,
        user_silence_s: float,
        on_utterance_complete: Callable[[int, TranscriptionResult], Awaitable[None]],
        on_voice_activity: Callable[[int, VoiceActivityEvent], None] | None = None,
    ) -> None:
        self._lull_timeout = lull_timeout
        self._user_silence_s = user_silence_s
        self._on_utterance_complete = on_utterance_complete
        self._on_voice_activity = on_voice_activity

        # per-user silence watchdogs; presence = currently speaking
        self._speaking: dict[int, asyncio.TimerHandle] = {}
        # buffered finals since last utterance fired
        self._finals: list[tuple[int, TranscriptionResult]] = []
        # pending lull timer
        self._lull_handle: asyncio.TimerHandle | None = None
        # strong refs to in-flight callback tasks (prevent GC)
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Public API — called from the voice pipeline
    # ------------------------------------------------------------------

    def on_audio(self, user_id: int) -> None:
        """Handle inbound audio frame; cancels lull, resets silence watchdog."""
        self._cancel_lull()
        self._reset_user_silence(user_id)

    def on_transcript(self, user_id: int, result: TranscriptionResult) -> None:
        """Buffer final transcription results; interims ignored.

        Arm lull immediately if channel idle — handles late finals that
        arrive after ``_on_user_silent`` already ran with empty ``_finals``.
        Otherwise defer to ``_on_user_silent``; finals received mid-speech
        still wait for silence via its ``not self._speaking`` guard.
        """
        if result.is_final and result.text:
            self._finals.append((user_id, result))
            if not self._speaking:
                self._arm_lull()

    def clear(self) -> None:
        """Cancel pending timers and drop buffered finals."""
        self._cancel_lull()
        for handle in self._speaking.values():
            handle.cancel()
        self._speaking.clear()
        self._finals.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cancel_lull(self) -> None:
        if self._lull_handle is not None:
            self._lull_handle.cancel()
            self._lull_handle = None

    def _reset_user_silence(self, user_id: int) -> None:
        existing = self._speaking.get(user_id)
        if existing is not None:
            existing.cancel()
        else:
            # silent → speaking transition
            _logger.info("voice activity user=%s event=started", user_id)
            self._emit_activity(user_id, VoiceActivityEvent.started)
        loop = asyncio.get_event_loop()
        self._speaking[user_id] = loop.call_later(
            self._user_silence_s,
            self._on_user_silent,
            user_id,
        )

    def _on_user_silent(self, user_id: int) -> None:
        """Sync callback: user's silence watchdog fired."""
        self._speaking.pop(user_id, None)
        _logger.info("voice activity user=%s event=ended", user_id)
        self._emit_activity(user_id, VoiceActivityEvent.ended)
        # if everyone silent and finals buffered, arm lull
        if not self._speaking and self._finals:
            self._arm_lull()

    def _emit_activity(self, user_id: int, event: VoiceActivityEvent) -> None:
        """Forward voice-activity event to optional subscriber."""
        if self._on_voice_activity is None:
            return
        try:
            self._on_voice_activity(user_id, event)
        except Exception:
            _logger.exception(
                "on_voice_activity callback failed for user=%s event=%s",
                user_id,
                event.value,
            )

    def _arm_lull(self) -> None:
        self._cancel_lull()
        loop = asyncio.get_event_loop()
        self._lull_handle = loop.call_later(
            self._lull_timeout,
            self._fire_lull,
        )

    def _fire_lull(self) -> None:
        """Sync callback: lull timer fired, dispatch the merged utterance."""
        self._lull_handle = None
        if not self._finals:
            return
        finals = self._finals
        self._finals = []
        _logger.info("voice lull expired: %d finals buffered", len(finals))

        merged_text = " ".join(r.text for _, r in finals).strip()
        if not merged_text:
            return

        last_user_id, last_result = finals[-1]
        first_result = finals[0][1]

        merged = TranscriptionResult(
            text=merged_text,
            is_final=True,
            start=first_result.start,
            end=last_result.end,
            confidence=last_result.confidence,
            speaker=last_result.speaker,
        )

        loop = asyncio.get_event_loop()
        task = loop.create_task(self._run_callback(last_user_id, merged))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_callback(self, user_id: int, merged: TranscriptionResult) -> None:
        try:
            await self._on_utterance_complete(user_id, merged)
        except Exception:
            _logger.exception("voice lull callback failed")
