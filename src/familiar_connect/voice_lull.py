"""Voice lull monitor — debounce voice utterances until the speaker pauses.

The voice lull is a silence-based debounce. Transcripts are buffered per
speaker until every user in the channel has been silent for
``lull_timeout`` seconds; only then is the accumulated utterance sent to
the response pipeline. This prevents the bot from replying to every
fragmentary mid-sentence final transcript from Deepgram.

Silence is detected Discord-natively: the voice pipeline hands this
monitor ``on_audio(user_id)`` for every inbound audio frame. If no
frames arrive for a user within ``user_silence_s`` seconds, that user is
considered silent. When the set of speaking users becomes empty and at
least one final transcript has been buffered, the lull timer starts;
when it expires without any new audio, the merged utterance fires.
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
    """Per-user speaking transitions derived from Discord audio frames.

    Surfaced by :class:`VoiceLullMonitor` so consumers (today: the
    :class:`InterruptionDetector` in
    :mod:`familiar_connect.voice.interruption`) can react to voice
    activity without being a parallel consumer of audio frames.
    """

    started = "started"
    """First audio frame after a quiet period — user began speaking."""

    ended = "ended"
    """Per-user silence watchdog fired — user stopped speaking."""


class VoiceLullMonitor:
    """Debounce voice utterances by watching Discord audio activity.

    :param lull_timeout: Seconds of channel-wide silence after which a
        buffered utterance is fired to ``on_utterance_complete``.
    :param user_silence_s: Per-user inactivity window. If no audio frames
        arrive for a user within this interval, the user is considered
        silent. Should be much shorter than ``lull_timeout``.
    :param on_utterance_complete: Async callback invoked with
        ``(primary_user_id, merged_result)`` once the lull fires. The
        merged result is a synthesized :class:`TranscriptionResult`
        whose text is the space-joined concatenation of all buffered
        finals; ``primary_user_id`` is the speaker of the most recent
        final.
    :param on_voice_activity: Optional sync callback invoked with
        ``(user_id, event)`` whenever a user transitions into or out
        of speaking. Derived from ``on_audio`` arrivals and the
        per-user silence watchdog. Used by the
        :class:`InterruptionDetector` to react to mid-response speech
        without subscribing to raw audio frames itself.
    """

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

        # Per-user silence watchdogs. A handle in this dict means the
        # user is currently speaking (as far as we can tell from audio
        # frame arrivals).
        self._speaking: dict[int, asyncio.TimerHandle] = {}

        # Buffered finals since the last utterance fired.
        self._finals: list[tuple[int, TranscriptionResult]] = []

        # Pending lull timer. Set only when every user is silent and
        # at least one final is buffered.
        self._lull_handle: asyncio.TimerHandle | None = None

        # Strong refs to in-flight callback tasks so they're not GC'd.
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Public API — called from the voice pipeline
    # ------------------------------------------------------------------

    def on_audio(self, user_id: int) -> None:
        """Handle an inbound audio frame for *user_id*.

        Any frame cancels the pending lull timer (someone is speaking)
        and resets that user's silence watchdog.
        """
        self._cancel_lull()
        self._reset_user_silence(user_id)

    def on_transcript(self, user_id: int, result: TranscriptionResult) -> None:
        """Handle a transcription result for *user_id*.

        Finals are buffered; interims are ignored (they're already logged
        by the transcript logger and add nothing for the response path).
        """
        if result.is_final and result.text:
            self._finals.append((user_id, result))

    def clear(self) -> None:
        """Cancel pending timers and drop buffered finals.

        Called when the voice session ends.
        """
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
            # Transition silent → speaking. Surface as a voice-activity
            # event for the InterruptionDetector and log so operators
            # can see speech-start latency in the bot logs.
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
        # If everyone is silent and we have something to say, arm the lull.
        if not self._speaking and self._finals:
            self._arm_lull()

    def _emit_activity(self, user_id: int, event: VoiceActivityEvent) -> None:
        """Forward a voice-activity event to the optional subscriber.

        Errors in the subscriber are caught + logged so a misbehaving
        consumer can't break the lull state machine.
        """
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
