"""Red-first tests for the MemoryWriterScheduler.

The scheduler manages two triggers (turn-count threshold and idle
timeout) that fire the MemoryWriter.

Covers familiar_connect.memory.scheduler.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from familiar_connect.history.store import HistoryStore
from familiar_connect.memory.scheduler import MemoryWriterScheduler
from familiar_connect.memory.store import MemoryStore
from familiar_connect.memory.writer import MemoryWriter

if TYPE_CHECKING:
    from pathlib import Path

from tests.conftest import FakeLLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAMILIAR = "aria"
_CHANNEL = 100


def _make_scheduler(
    tmp_path: Path,
    *,
    turn_threshold: int = 5,
    idle_timeout: float = 0.1,
    llm_replies: list[str] | None = None,
) -> tuple[MemoryWriterScheduler, MemoryWriter, HistoryStore, MemoryStore]:
    mem_root = tmp_path / "memory"
    mem_root.mkdir(exist_ok=True)
    mem_store = MemoryStore(mem_root)
    hist_store = HistoryStore(tmp_path / "history.db")
    llm = FakeLLMClient(replies=llm_replies or [])
    writer = MemoryWriter(
        memory_store=mem_store,
        history_store=hist_store,
        llm_client=llm,
        familiar_id=_FAMILIAR,
    )
    scheduler = MemoryWriterScheduler(
        writer=writer,
        history_store=hist_store,
        familiar_id=_FAMILIAR,
        turn_threshold=turn_threshold,
        idle_timeout=idle_timeout,
    )
    return scheduler, writer, hist_store, mem_store


def _seed(hist_store: HistoryStore, n: int) -> None:
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        speaker = "Alice" if role == "user" else None
        hist_store.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role=role,
            content=f"turn {i}",
            speaker=speaker,
        )


_MINIMAL_OUTPUT = """\
===SESSION_SUMMARY===
A brief session.
===END_SESSION_SUMMARY===

===PEOPLE===
===END_PEOPLE===

===TOPICS===
===END_TOPICS==="""


# ---------------------------------------------------------------------------
# Turn-count threshold tests
# ---------------------------------------------------------------------------


class TestTurnThreshold:
    def test_threshold_triggers_write(self, tmp_path: Path) -> None:
        scheduler, writer, hist_store, _ = _make_scheduler(
            tmp_path, turn_threshold=5, llm_replies=[_MINIMAL_OUTPUT]
        )
        _seed(hist_store, 5)

        with patch.object(writer, "run", wraps=writer.run) as mock_run:
            asyncio.run(scheduler.notify_turn())
            assert mock_run.call_count == 1

    def test_below_threshold_no_write(self, tmp_path: Path) -> None:
        scheduler, writer, hist_store, _ = _make_scheduler(
            tmp_path, turn_threshold=10, llm_replies=[_MINIMAL_OUTPUT]
        )
        _seed(hist_store, 5)

        with patch.object(writer, "run", wraps=writer.run) as mock_run:
            asyncio.run(scheduler.notify_turn())
            assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# Flush tests
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_runs_writer(self, tmp_path: Path) -> None:
        scheduler, _writer, hist_store, _ = _make_scheduler(
            tmp_path, turn_threshold=999, llm_replies=[_MINIMAL_OUTPUT]
        )
        _seed(hist_store, 3)

        result = asyncio.run(scheduler.flush())
        assert result is not None
        assert result.turns_summarized == 3

    def test_flush_noop_when_no_unsummarized(self, tmp_path: Path) -> None:
        scheduler, _, _, _ = _make_scheduler(tmp_path, turn_threshold=999)
        result = asyncio.run(scheduler.flush())
        assert result is None


# ---------------------------------------------------------------------------
# Idle timeout tests
# ---------------------------------------------------------------------------


class TestIdleTimeout:
    @pytest.mark.asyncio
    async def test_idle_timeout_triggers_write(self, tmp_path: Path) -> None:
        scheduler, writer, hist_store, _ = _make_scheduler(
            tmp_path,
            turn_threshold=999,
            idle_timeout=0.05,
            llm_replies=[_MINIMAL_OUTPUT],
        )
        _seed(hist_store, 3)
        scheduler.start()

        with patch.object(writer, "run", wraps=writer.run) as mock_run:
            # Reset the idle timer by calling notify_turn
            await scheduler.notify_turn()
            # Wait longer than idle_timeout
            await asyncio.sleep(0.15)
            assert mock_run.call_count >= 1

        scheduler.stop()

    @pytest.mark.asyncio
    async def test_idle_timer_reset_on_new_turn(self, tmp_path: Path) -> None:
        scheduler, writer, hist_store, _ = _make_scheduler(
            tmp_path,
            turn_threshold=999,
            idle_timeout=0.1,
            llm_replies=[_MINIMAL_OUTPUT],
        )
        _seed(hist_store, 3)
        scheduler.start()

        with patch.object(writer, "run", wraps=writer.run) as mock_run:
            # First notify - starts timer
            await scheduler.notify_turn()
            # Wait half the timeout
            await asyncio.sleep(0.05)
            # Notify again - should reset timer
            await scheduler.notify_turn()
            # Wait half the timeout again - shouldn't have fired yet
            await asyncio.sleep(0.05)
            assert mock_run.call_count == 0

        scheduler.stop()


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_runs_serialized(self, tmp_path: Path) -> None:
        scheduler, writer, hist_store, _ = _make_scheduler(
            tmp_path, turn_threshold=3, llm_replies=[_MINIMAL_OUTPUT, _MINIMAL_OUTPUT]
        )
        _seed(hist_store, 10)

        call_count = 0
        original_run = writer.run

        async def slow_run():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return await original_run()

        with patch.object(writer, "run", side_effect=slow_run):
            # Trigger two runs simultaneously
            async def _double():
                return await asyncio.gather(
                    scheduler.flush(),
                    scheduler.flush(),
                )

            results = asyncio.run(_double())
            # One should have gotten the lock, the other returns None
            non_none = [r for r in results if r is not None]
            assert len(non_none) <= 1


# ---------------------------------------------------------------------------
# Stop tests
# ---------------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_timer(self, tmp_path: Path) -> None:
        scheduler, writer, hist_store, _ = _make_scheduler(
            tmp_path,
            turn_threshold=999,
            idle_timeout=0.05,
            llm_replies=[_MINIMAL_OUTPUT],
        )
        _seed(hist_store, 3)
        scheduler.start()

        with patch.object(writer, "run", wraps=writer.run) as mock_run:
            await scheduler.notify_turn()
            scheduler.stop()
            # Wait longer than idle_timeout
            await asyncio.sleep(0.1)
            assert mock_run.call_count == 0
