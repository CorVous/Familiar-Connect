"""Voice lull monitor — debounce voice utterances via Deepgram VAD events.

Buffers per-speaker Deepgram finals. When all users are silent per
Deepgram's VAD (``UtteranceEnd``) and finals are buffered, a lull timer
of ``lull_timeout`` seconds arms. If any user resumes speech
(``SpeechStarted``) before it fires, the timer cancels and the buffer
keeps accumulating. When the timer fires, finals merge into a single
utterance and dispatch to the response pipeline.

Signals come exclusively from Deepgram VAD, not Discord audio frames.
Frame arrival is unreliable as a speech-state signal (client-side VAD
behaviour varies; background noise keeps frames flowing through real
silence). Deepgram's VAD is authoritative.
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
    """Per-user speaking transitions from Deepgram VAD."""

    started = "started"
    """User began speaking (Deepgram ``SpeechStarted``)."""

    ended = "ended"
    """User stopped speaking (Deepgram ``UtteranceEnd``)."""


class VoiceLullMonitor:
    """Debounce voice utterances using Deepgram VAD events."""

    def __init__(
        self,
        *,
        lull_timeout: float,
        on_utterance_complete: Callable[[int, TranscriptionResult], Awaitable[None]],
        on_voice_activity: Callable[[int, VoiceActivityEvent], None] | None = None,
    ) -> None:
        self._lull_timeout = lull_timeout
        self._on_utterance_complete = on_utterance_complete
        self._on_voice_activity = on_voice_activity

        # users currently speaking per Deepgram VAD
        self._speaking: set[int] = set()
        # buffered finals since last utterance fired
        self._finals: list[tuple[int, TranscriptionResult]] = []
        # pending lull timer
        self._lull_handle: asyncio.TimerHandle | None = None
        # strong refs to in-flight callback tasks (prevent GC)
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Public API — called from the voice pipeline
    # ------------------------------------------------------------------

    def on_speech_start(self, user_id: int) -> None:
        """Deepgram ``SpeechStarted`` — user began speaking."""
        # cancel any pending lull; resumed speech extends the utterance
        self._cancel_lull()
        if user_id in self._speaking:
            return
        self._speaking.add(user_id)
        _logger.info("voice activity user=%s event=started", user_id)
        self._emit_activity(user_id, VoiceActivityEvent.started)

    def on_speech_end(self, user_id: int) -> None:
        """Deepgram ``UtteranceEnd`` — user finished an utterance."""
        if user_id not in self._speaking:
            return
        self._speaking.discard(user_id)
        _logger.info("voice activity user=%s event=ended", user_id)
        self._emit_activity(user_id, VoiceActivityEvent.ended)
        # channel idle + finals buffered → arm lull timer
        if not self._speaking and self._finals:
            self._arm_lull()

    def on_transcript(self, user_id: int, result: TranscriptionResult) -> None:
        """Buffer finals; arm safety-net lull.

        Deepgram's ``UtteranceEnd`` can lag by seconds when audio
        frames keep flowing through real silence (client-side VAD
        variance, background noise above Discord's threshold).
        Arming on each final guarantees dispatch within
        ``lull_timeout`` of the last transcript even if
        ``UtteranceEnd`` never arrives. A subsequent
        ``on_speech_start`` still cancels (resumed speech extends
        the utterance).
        """
        if result.is_final and result.text:
            self._finals.append((user_id, result))
            self._arm_lull()

    def clear(self) -> None:
        """Cancel pending timer and drop buffered state."""
        self._cancel_lull()
        self._speaking.clear()
        self._finals.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cancel_lull(self) -> None:
        if self._lull_handle is not None:
            self._lull_handle.cancel()
            self._lull_handle = None

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
