"""Voice lull monitor — conversational-silence detector over Deepgram finals.

Buffers per-speaker Deepgram finals. Each incoming final (re-)arms the
``lull_timeout`` timer. ``SpeechStarted`` from Deepgram VAD cancels a
pending timer (resumed speech extends the turn). When the timer
expires, buffered finals merge into a single utterance and dispatch to
the response pipeline as the conversational-lull endpoint.

The lull is client-side wall-clock time since the last Deepgram event —
no dependency on ``UtteranceEnd``. Deepgram's ``UtteranceEnd`` is
unreliable under continuous audio (Discord keeps streaming near-silent
frames; Deepgram holds the event until the next speech flushes it).
Running with ``interim_results=false`` makes each ``Results(is_final)``
a complete endpointed segment; arming on those is the prompt trigger.

``VoiceActivityEvent.started`` is emitted on ``SpeechStarted``;
``VoiceActivityEvent.ended`` is emitted on final arrival for a user
currently in ``_speaking`` (final = user paused for this segment).
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
        """Deepgram ``SpeechStarted`` — user began speaking.

        Re-arms the lull timer. Any Deepgram activity (VAD pulse or
        final) pushes the lull out by ``lull_timeout``; when activity
        stops for that long, the timer fires and dispatches any
        buffered finals. No-op fire if nothing is buffered.
        """
        if user_id not in self._speaking:
            self._speaking.add(user_id)
            _logger.info("voice activity user=%s event=started", user_id)
            self._emit_activity(user_id, VoiceActivityEvent.started)
        self._arm_lull()

    def on_speech_end(self, user_id: int) -> None:
        """Deepgram ``UtteranceEnd`` — user finished an utterance.

        Defensive only: with ``interim_results=false`` Deepgram does not
        emit ``UtteranceEnd``. Kept so the pipeline surface stays stable
        and any future server-side behaviour change is absorbed safely.
        """
        if user_id not in self._speaking:
            return
        self._speaking.discard(user_id)
        _logger.info("voice activity user=%s event=ended", user_id)
        self._emit_activity(user_id, VoiceActivityEvent.ended)

    def on_transcript(self, user_id: int, result: TranscriptionResult) -> None:
        """Buffer final; (re-)arm the conversational lull.

        Each final resets the ``lull_timeout`` timer. Interim results
        are ignored (normally absent under ``interim_results=false``).
        If the user is still marked speaking from a prior
        ``SpeechStarted``, a final means they endpointed — emit
        ``VoiceActivityEvent.ended`` and drop them from ``_speaking``.
        """
        if not (result.is_final and result.text):
            return
        self._finals.append((user_id, result))
        self._arm_lull()
        if user_id in self._speaking:
            self._speaking.discard(user_id)
            _logger.info("voice activity user=%s event=ended", user_id)
            self._emit_activity(user_id, VoiceActivityEvent.ended)

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
