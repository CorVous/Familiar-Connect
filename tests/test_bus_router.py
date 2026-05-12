"""Tests for :class:`familiar_connect.bus.router.TurnRouter`."""

from __future__ import annotations

import asyncio

import pytest

from familiar_connect.bus.router import TurnRouter


class TestBeginTurn:
    @pytest.mark.asyncio
    async def test_first_turn_returns_new_scope(self) -> None:
        router = TurnRouter()
        scope = router.begin_turn(session_id="chan-1", turn_id="t-1")
        assert scope.turn_id == "t-1"
        assert scope.session_id == "chan-1"
        assert scope.is_cancelled() is False

    @pytest.mark.asyncio
    async def test_second_turn_cancels_first_same_session(self) -> None:
        router = TurnRouter()
        first = router.begin_turn(session_id="chan-1", turn_id="t-1")
        second = router.begin_turn(session_id="chan-1", turn_id="t-2")
        assert first.is_cancelled() is True
        assert second.is_cancelled() is False
        assert router.active_scope("chan-1") is second

    @pytest.mark.asyncio
    async def test_turns_in_different_sessions_are_independent(self) -> None:
        router = TurnRouter()
        a = router.begin_turn(session_id="chan-1", turn_id="a")
        b = router.begin_turn(session_id="chan-2", turn_id="b")
        assert a.is_cancelled() is False
        assert b.is_cancelled() is False
        # cancelling one doesn't disturb the other
        a.cancel()
        assert b.is_cancelled() is False

    @pytest.mark.asyncio
    async def test_cancel_propagates_within_50ms(self) -> None:
        router = TurnRouter()
        scope = router.begin_turn(session_id="s", turn_id="t-1")

        async def worker() -> float:
            started = asyncio.get_running_loop().time()
            await scope.wait_cancelled()
            return asyncio.get_running_loop().time() - started

        task = asyncio.create_task(worker())
        await asyncio.sleep(0)
        router.begin_turn(session_id="s", turn_id="t-2")  # cancels t-1
        elapsed = await asyncio.wait_for(task, timeout=1.0)
        assert elapsed < 0.05


class TestEndTurn:
    @pytest.mark.asyncio
    async def test_end_turn_clears_active(self) -> None:
        router = TurnRouter()
        scope = router.begin_turn(session_id="s", turn_id="t-1")
        router.end_turn(scope)
        assert router.active_scope("s") is None

    @pytest.mark.asyncio
    async def test_end_turn_is_idempotent(self) -> None:
        router = TurnRouter()
        scope = router.begin_turn(session_id="s", turn_id="t-1")
        router.end_turn(scope)
        router.end_turn(scope)  # must not raise
        assert router.active_scope("s") is None

    @pytest.mark.asyncio
    async def test_end_stale_scope_doesnt_clear_newer(self) -> None:
        router = TurnRouter()
        old = router.begin_turn(session_id="s", turn_id="t-1")
        new = router.begin_turn(session_id="s", turn_id="t-2")
        router.end_turn(old)
        assert router.active_scope("s") is new


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_all_active_turns(self) -> None:
        router = TurnRouter()
        a = router.begin_turn(session_id="s1", turn_id="a")
        b = router.begin_turn(session_id="s2", turn_id="b")
        router.shutdown()
        assert a.is_cancelled()
        assert b.is_cancelled()
