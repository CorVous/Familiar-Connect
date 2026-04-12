"""Voice lull collator — gates LLM dispatch on Discord SPEAKING events.

Deepgram segments a single human utterance into multiple ``is_final``
transcripts. Forwarding each one directly to the voice response handler
produces one LLM turn per fragment, so a user saying "uh… hey Aria,
what's the weather?" may trigger two or three replies for one thought.

This collator buffers ``is_final`` text per user and dispatches only
after :attr:`lull_timeout` seconds of continuous Discord silence
(``SPEAKING=False``). It also covers the race where the lull timer
fires while Deepgram has yet to emit the ``is_final`` for the most
recent audio burst: a bounded wait (:attr:`dispatch_timeout`) blocks
on an :class:`asyncio.Event` that the ``on_final`` handler signals.

State is entirely per-user. Discord SPEAKING=True is a hard reset that
cancels the lull timer and any in-flight wait task so a fresh burst
never inherits state from the previous one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.transcription import TranscriptionResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_logger = logging.getLogger(__name__)


DEFAULT_LULL_TIMEOUT: float = 2.0
DEFAULT_DISPATCH_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# Per-user state
# ---------------------------------------------------------------------------


@dataclass
class _UserState:
    """Per-user collator state.

    :param pending_texts: Buffered ``is_final`` transcript strings
        accumulated since the last dispatch.
    :param audio_pending: ``True`` once SPEAKING=True has fired since
        the last ``is_final`` — signals "Deepgram still owes us a
        transcript for the current burst".
    :param lull_timer: Handle to the pending lull callback, scheduled
        at SPEAKING=False and cancelled on SPEAKING=True.
    :param wait_task: In-flight :meth:`_wait_and_dispatch` task, if
        the lull fired while ``audio_pending`` was True.
    :param waiting_event: :class:`asyncio.Event` signalled by
        ``on_final`` to unblock a running wait task.
    """

    pending_texts: list[str] = field(default_factory=list)
    audio_pending: bool = False
    lull_timer: asyncio.TimerHandle | None = None
    wait_task: asyncio.Task[None] | None = None
    waiting_event: asyncio.Event | None = None


# ---------------------------------------------------------------------------
# VoiceLullCollator
# ---------------------------------------------------------------------------


class VoiceLullCollator:
    """Collate per-user Deepgram transcripts across a SPEAKING burst.

    :param downstream: Async callback invoked with
        ``(user_id, fused_result)`` once a user's burst has been
        collated and the lull has elapsed. The existing voice response
        handler slots in here.
    :param lull_timeout: Seconds of continuous SPEAKING=False required
        before dispatch.
    :param dispatch_timeout: Upper bound on the wait for a trailing
        ``is_final`` after the lull fires with ``audio_pending=True``.
    """

    def __init__(
        self,
        downstream: Callable[[int, TranscriptionResult], Awaitable[None]],
        *,
        lull_timeout: float = DEFAULT_LULL_TIMEOUT,
        dispatch_timeout: float = DEFAULT_DISPATCH_TIMEOUT,
    ) -> None:
        self._downstream = downstream
        self._lull_timeout = lull_timeout
        self._dispatch_timeout = dispatch_timeout
        self._users: dict[int, _UserState] = {}
        # Strong refs to dispatch tasks so they aren't GC'd mid-flight.
        self._dispatch_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_speaking(self, user_id: int, speaking: bool) -> None:  # noqa: FBT001 — the caller is py-cord's voice WS hook, which receives the speaking bit positionally
        """Handle a Discord SPEAKING opcode for *user_id*.

        ``speaking=True`` is a hard reset: it cancels any pending lull
        timer and any in-flight wait task, and flags the user as having
        audio in flight. ``speaking=False`` (re)starts the lull timer
        so the collator dispatches after :attr:`lull_timeout` seconds
        of continuous silence.
        """
        state = self._users.setdefault(user_id, _UserState())
        if speaking:
            state.audio_pending = True
            self._cancel_lull_timer(state)
            self._cancel_wait_task(state)
        else:
            self._cancel_lull_timer(state)
            loop = asyncio.get_event_loop()
            state.lull_timer = loop.call_later(
                self._lull_timeout,
                self._on_lull_fired,
                user_id,
            )

    def on_final(self, user_id: int, result: TranscriptionResult) -> None:
        """Handle a Deepgram ``is_final`` transcript for *user_id*.

        Appends the text to the per-user buffer, clears the
        ``audio_pending`` flag (Deepgram has delivered), and wakes any
        :meth:`_wait_and_dispatch` coroutine currently blocked waiting
        for a trailing transcript.
        """
        state = self._users.setdefault(user_id, _UserState())
        if result.text:
            state.pending_texts.append(result.text)
        state.audio_pending = False
        if state.waiting_event is not None:
            state.waiting_event.set()

    async def close(self) -> None:
        """Cancel all pending timers and wait tasks.

        Called by :func:`voice_pipeline.stop_pipeline` during teardown.
        Safe to call multiple times.
        """
        # Snapshot wait tasks before cancellation so we can await them
        # below (``_cancel_wait_task`` nulls the field).
        tasks = [s.wait_task for s in self._users.values() if s.wait_task is not None]
        for state in self._users.values():
            self._cancel_lull_timer(state)
            self._cancel_wait_task(state)
        # Drain the cancelled wait tasks so they fully unwind before
        # ``stop_pipeline`` returns.
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_lull_timer(self, state: _UserState) -> None:
        if state.lull_timer is not None:
            state.lull_timer.cancel()
            state.lull_timer = None

    def _cancel_wait_task(self, state: _UserState) -> None:
        if state.wait_task is not None and not state.wait_task.done():
            state.wait_task.cancel()
        state.wait_task = None
        state.waiting_event = None

    def _on_lull_fired(self, user_id: int) -> None:
        """Sync callback from ``call_later`` — route to the async branch."""
        state = self._users.get(user_id)
        if state is None:
            return
        state.lull_timer = None

        if state.pending_texts:
            # Deepgram already delivered — dispatch immediately.
            self._spawn_dispatch(user_id, state)
            return
        if state.audio_pending:
            # Wait for the trailing is_final (bounded).
            loop = asyncio.get_event_loop()
            state.wait_task = loop.create_task(self._wait_and_dispatch(user_id))
            return
        # Nothing to do.

    async def _wait_and_dispatch(self, user_id: int) -> None:
        """Block on the waiting event, then dispatch whatever is buffered."""
        state = self._users.get(user_id)
        if state is None:
            return
        state.waiting_event = asyncio.Event()
        try:
            await asyncio.wait_for(
                state.waiting_event.wait(),
                timeout=self._dispatch_timeout,
            )
        except TimeoutError:
            _logger.warning(
                "VoiceLullCollator: dispatch_timeout (%.1fs) elapsed for"
                " user=%d with no trailing is_final",
                self._dispatch_timeout,
                user_id,
            )
        except asyncio.CancelledError:
            # Hard reset via SPEAKING=True or close() — no dispatch.
            state.waiting_event = None
            state.wait_task = None
            raise
        state.waiting_event = None
        state.wait_task = None
        if state.pending_texts:
            await self._dispatch_now(user_id, state)

    def _spawn_dispatch(self, user_id: int, state: _UserState) -> None:
        """Dispatch immediately as a background task (lull callback is sync)."""
        loop = asyncio.get_event_loop()
        task = loop.create_task(self._dispatch_now(user_id, state))
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)

    async def _dispatch_now(self, user_id: int, state: _UserState) -> None:
        """Join buffered texts and invoke the downstream handler."""
        if not state.pending_texts:
            return
        fused_text = " ".join(state.pending_texts)
        state.pending_texts.clear()
        fused = TranscriptionResult(
            text=fused_text,
            is_final=True,
            start=0.0,
            end=0.0,
        )
        try:
            await self._downstream(user_id, fused)
        except Exception:
            _logger.exception(
                "VoiceLullCollator downstream handler failed for user=%d",
                user_id,
            )
