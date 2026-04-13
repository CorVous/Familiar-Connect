"""Scheduler for the post-session memory writer.

Manages two triggers that fire the :class:`MemoryWriter`:

1. **Turn-count threshold** — after N new turns since the last write.
2. **Idle timeout** — after M seconds of silence across all channels.

Both triggers check the same watermark in :class:`HistoryStore` to
avoid double-writes. A flush method allows explicit invocation on
unsubscribe events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.memory.writer import MemoryWriter, MemoryWriterResult

_logger = logging.getLogger(__name__)


class MemoryWriterScheduler:
    """Gates when the :class:`MemoryWriter` runs.

    :param writer: The :class:`MemoryWriter` to invoke.
    :param history_store: Used to check watermark vs latest turn id.
    :param familiar_id: The active familiar's id.
    :param turn_threshold: Run the writer after this many new turns.
    :param idle_timeout: Run the writer after this many seconds of silence.
    """

    def __init__(
        self,
        *,
        writer: MemoryWriter,
        history_store: HistoryStore,
        familiar_id: str,
        turn_threshold: int = 50,
        idle_timeout: float = 1800.0,
    ) -> None:
        self._writer = writer
        self._history_store = history_store
        self._familiar_id = familiar_id
        self._turn_threshold = turn_threshold
        self._idle_timeout = idle_timeout
        self._lock = asyncio.Lock()
        self._idle_handle: asyncio.TimerHandle | None = None
        self._idle_tasks: set[asyncio.Task[None]] = set()
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the idle timer. Called once at bot startup."""
        self._started = True
        self._reset_idle_timer()

    def stop(self) -> None:
        """Cancel pending timers. Called on shutdown."""
        self._started = False
        self._cancel_idle_timer()

    async def notify_turn(self) -> None:
        """Check turn threshold and reset idle timer after a new turn."""
        self._reset_idle_timer()

        unsummarized = self._unsummarized_count()
        if unsummarized >= self._turn_threshold:
            await self._try_run()

    async def flush(self) -> MemoryWriterResult | None:
        """Force an immediate write if there are unsummarized turns.

        Called on unsubscribe events. Returns the result if a write
        happened, or ``None`` if there was nothing to write.
        """
        if self._unsummarized_count() <= 0:
            return None
        return await self._try_run()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unsummarized_count(self) -> int:
        """Return the number of turns since the last watermark."""
        latest = self._history_store.latest_id(familiar_id=self._familiar_id)
        if latest is None:
            return 0
        wm = self._history_store.get_writer_watermark(
            familiar_id=self._familiar_id,
        )
        last_written = wm.last_written_id if wm is not None else 0
        return latest - last_written

    async def _try_run(self) -> MemoryWriterResult | None:
        """Acquire the lock and run the writer. Returns None if locked."""
        if self._lock.locked():
            return None
        async with self._lock:
            try:
                return await self._writer.run()
            except Exception:
                _logger.exception("memory writer failed")
                return None

    def _reset_idle_timer(self) -> None:
        """Cancel existing idle timer and start a new one."""
        self._cancel_idle_timer()
        if not self._started:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._idle_handle = loop.call_later(
            self._idle_timeout,
            self._schedule_idle_run,
        )

    def _cancel_idle_timer(self) -> None:
        if self._idle_handle is not None:
            self._idle_handle.cancel()
            self._idle_handle = None

    def _schedule_idle_run(self) -> None:
        """Sync callback from call_later — schedules the async writer."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._idle_triggered())
        self._idle_tasks.add(task)
        task.add_done_callback(self._idle_tasks.discard)

    async def _idle_triggered(self) -> None:
        """Run the writer if there are unsummarized turns."""
        if self._unsummarized_count() > 0:
            await self._try_run()
