"""Tests for attentional-stream tools: shift_focus, silent, read_channel.

TDD red-first: covers all specified behaviors before implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_connect.llm import LLMClient, LLMDelta, Message
from familiar_connect.tools.builtins import (
    build_read_channel_tool,
    build_shift_focus_tool,
    build_silent_tool,
    build_text_registry,
    build_voice_registry,
)
from familiar_connect.tools.loop import agentic_loop
from familiar_connect.tools.read_channel import _read_channel_handler
from familiar_connect.tools.read_channel import (
    build_read_channel_tool as _build_read_channel_tool_direct,
)
from familiar_connect.tools.registry import Tool, ToolContext, ToolRegistry
from familiar_connect.tools.shift_focus import _shift_focus_handler
from familiar_connect.tools.shift_focus import (
    build_shift_focus_tool as _build_shift_focus_tool_direct,
)
from familiar_connect.tools.silent import SILENT_RESULT, _silent_handler
from familiar_connect.tools.silent import (
    build_silent_tool as _build_silent_tool_direct,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.focus import FocusManager
    from familiar_connect.history.async_store import AsyncHistoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    focus_manager: FocusManager | None = None,
    store: AsyncHistoryStore | None = None,
    channel_id: int = 42,
    channel_kind: str = "text",
) -> ToolContext:
    return ToolContext(
        familiar_id="fam-1",
        channel_id=channel_id,
        channel_kind=channel_kind,
        turn_id="turn-1",
        history=cast("AsyncHistoryStore", None),
        bus=cast("EventBus", None),
        focus_manager=focus_manager,
        store=store,
    )


def _make_focus_manager() -> MagicMock:
    fm = MagicMock()
    fm.defer_shift = MagicMock()
    fm.get_focus = MagicMock(return_value=99)
    return fm


def _make_history_turn(id: int, role: str, content: str, channel_id: int = 99):  # noqa: A002
    """Build a minimal HistoryTurn."""
    from familiar_connect.history.store import HistoryTurn  # noqa: PLC0415

    return HistoryTurn(
        id=id,
        timestamp=datetime(2024, 1, 1, 0, 0, id, tzinfo=UTC),
        role=role,
        author=None,
        content=content,
        channel_id=channel_id,
    )


# ---------------------------------------------------------------------------
# ToolContext: new fields
# ---------------------------------------------------------------------------


class TestToolContextNewFields:
    def test_focus_manager_defaults_none(self) -> None:
        ctx = ToolContext(
            familiar_id="fam",
            channel_id=1,
            channel_kind="text",
            turn_id="t",
            history=cast("AsyncHistoryStore", None),
            bus=cast("EventBus", None),
        )
        assert ctx.focus_manager is None

    def test_store_defaults_none(self) -> None:
        ctx = ToolContext(
            familiar_id="fam",
            channel_id=1,
            channel_kind="text",
            turn_id="t",
            history=cast("AsyncHistoryStore", None),
            bus=cast("EventBus", None),
        )
        assert ctx.store is None


# ---------------------------------------------------------------------------
# shift_focus tool
# ---------------------------------------------------------------------------


class TestShiftFocusTool:
    @pytest.mark.asyncio
    async def test_calls_defer_shift(self) -> None:
        fm = _make_focus_manager()
        ctx = _make_ctx(focus_manager=fm)
        await _shift_focus_handler({"channel_id": 55}, ctx)
        fm.defer_shift.assert_called_once_with(55)

    @pytest.mark.asyncio
    async def test_returns_ok_json(self) -> None:
        fm = _make_focus_manager()
        ctx = _make_ctx(focus_manager=fm)
        result = await _shift_focus_handler({"channel_id": 55}, ctx)
        parsed = json.loads(result)
        assert parsed == {"ok": True, "channel_id": 55}

    @pytest.mark.asyncio
    async def test_returns_error_when_no_focus_manager(self) -> None:
        ctx = _make_ctx(focus_manager=None)
        result = await _shift_focus_handler({"channel_id": 55}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_error_when_missing_channel_id(self) -> None:
        fm = _make_focus_manager()
        ctx = _make_ctx(focus_manager=fm)
        result = await _shift_focus_handler({}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_build_shift_focus_tool_name(self) -> None:
        tool = _build_shift_focus_tool_direct()
        assert tool.name == "shift_focus"

    def test_build_shift_focus_tool_schema(self) -> None:
        tool = _build_shift_focus_tool_direct()
        props = tool.parameters["properties"]
        assert "channel_id" in props
        assert props["channel_id"]["type"] == "integer"
        assert tool.parameters["required"] == ["channel_id"]


# ---------------------------------------------------------------------------
# silent tool
# ---------------------------------------------------------------------------


class TestSilentTool:
    @pytest.mark.asyncio
    async def test_returns_silent_result_sentinel(self) -> None:
        ctx = _make_ctx()
        result = await _silent_handler({"reasoning": "not relevant"}, ctx)
        assert result == SILENT_RESULT

    @pytest.mark.asyncio
    async def test_returns_sentinel_even_with_empty_reasoning(self) -> None:
        ctx = _make_ctx()
        result = await _silent_handler({"reasoning": ""}, ctx)
        assert result == SILENT_RESULT

    def test_silent_result_is_string_constant(self) -> None:
        assert isinstance(SILENT_RESULT, str)

    def test_build_silent_tool_name(self) -> None:
        tool = _build_silent_tool_direct()
        assert tool.name == "silent"

    def test_build_silent_tool_schema(self) -> None:
        tool = _build_silent_tool_direct()
        props = tool.parameters["properties"]
        assert "reasoning" in props
        assert props["reasoning"]["type"] == "string"
        assert tool.parameters["required"] == ["reasoning"]


# ---------------------------------------------------------------------------
# Agentic loop: silent tool detection
# ---------------------------------------------------------------------------


@dataclass
class _Script:
    deltas: list[LLMDelta] = field(default_factory=list)


class _ScriptedStreamLLM(LLMClient):
    """Minimal scripted LLM for loop tests."""

    def __init__(self, scripts: list[_Script]) -> None:
        super().__init__(api_key="k", model="m")
        self._scripts = scripts

    async def stream_completion(
        self,
        messages: list[Message],  # noqa: ARG002
        *,
        tools: list[dict] | None = None,  # noqa: ARG002
    ) -> AsyncIterator[LLMDelta]:
        if not self._scripts:
            return
        script = self._scripts.pop(0)
        for delta in script.deltas:
            yield delta


def _delta_tool_call(call_id: str, name: str, args: dict) -> LLMDelta:
    return LLMDelta(
        tool_calls=[
            {
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    )


def _delta_finish(reason: str) -> LLMDelta:
    return LLMDelta(finish_reason=reason)


class TestAgenticLoopSilentDetection:
    @pytest.mark.asyncio
    async def test_silent_tool_sets_is_silent_true(self) -> None:
        registry = ToolRegistry()
        registry.register(_build_silent_tool_direct())

        scripts = [
            _Script(
                deltas=[
                    _delta_tool_call("c1", "silent", {"reasoning": "not relevant now"}),
                    _delta_finish("tool_calls"),
                ]
            ),
        ]
        llm = _ScriptedStreamLLM(scripts)
        ctx = _make_ctx()
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="hi")],
            registry=registry,
            ctx=ctx,
        )
        assert result.is_silent is True

    @pytest.mark.asyncio
    async def test_normal_tool_does_not_set_is_silent(self) -> None:
        async def _noop(  # noqa: RUF029
            args: dict,  # noqa: ARG001
            ctx: ToolContext,  # noqa: ARG001
        ) -> str:
            return "ok"

        registry = ToolRegistry()
        registry.register(
            Tool(name="noop", description="", parameters={}, handler=_noop)
        )

        scripts = [
            _Script(
                deltas=[
                    _delta_tool_call("c1", "noop", {}),
                    _delta_finish("tool_calls"),
                ]
            ),
            _Script(deltas=[LLMDelta(content="done"), _delta_finish("stop")]),
        ]
        llm = _ScriptedStreamLLM(scripts)
        ctx = _make_ctx()
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="hi")],
            registry=registry,
            ctx=ctx,
        )
        assert result.is_silent is False

    @pytest.mark.asyncio
    async def test_no_tools_loop_is_not_silent(self) -> None:
        scripts = [_Script(deltas=[LLMDelta(content="hello"), _delta_finish("stop")])]
        llm = _ScriptedStreamLLM(scripts)
        ctx = _make_ctx()
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="hi")],
            registry=ToolRegistry(),
            ctx=ctx,
        )
        assert result.is_silent is False


# ---------------------------------------------------------------------------
# read_channel tool
# ---------------------------------------------------------------------------


class TestReadChannelTool:
    @pytest.mark.asyncio
    async def test_returns_recent_turns_as_json(self) -> None:
        turns = [
            _make_history_turn(1, "user", "hello", 99),
            _make_history_turn(2, "assistant", "hi there", 99),
        ]
        store = MagicMock()
        store.recent = AsyncMock(return_value=turns)
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        result = await _read_channel_handler({"limit": 20}, ctx)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["role"] == "user"
        assert parsed[0]["content"] == "hello"
        assert parsed[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_passes_limit_to_store(self) -> None:
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        await _read_channel_handler({"limit": 10}, ctx)
        store.recent.assert_awaited_once()
        call_kwargs = store.recent.call_args
        assert call_kwargs.kwargs.get("limit") == 10

    @pytest.mark.asyncio
    async def test_clamps_limit_to_50(self) -> None:
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        await _read_channel_handler({"limit": 100}, ctx)
        call_kwargs = store.recent.call_args
        assert call_kwargs.kwargs.get("limit") == 50

    @pytest.mark.asyncio
    async def test_returns_error_when_no_store(self) -> None:
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=None)
        result = await _read_channel_handler({}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_error_when_no_focus_manager(self) -> None:
        store = MagicMock()
        ctx = _make_ctx(focus_manager=None, store=store)
        result = await _read_channel_handler({}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_error_when_no_text_focus(self) -> None:
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=None)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        result = await _read_channel_handler({}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_build_read_channel_tool_name(self) -> None:
        tool = _build_read_channel_tool_direct()
        assert tool.name == "read_channel"

    def test_build_read_channel_tool_schema_has_limit(self) -> None:
        tool = _build_read_channel_tool_direct()
        props = tool.parameters["properties"]
        assert "limit" in props
        assert props["limit"]["maximum"] == 50


# ---------------------------------------------------------------------------
# builtins.py factories
# ---------------------------------------------------------------------------


class TestBuiltinsFactories:
    def test_build_shift_focus_tool_importable_from_builtins(self) -> None:
        tool = build_shift_focus_tool()
        assert tool.name == "shift_focus"

    def test_build_silent_tool_importable_from_builtins(self) -> None:
        tool = build_silent_tool()
        assert tool.name == "silent"

    def test_build_read_channel_tool_importable_from_builtins(self) -> None:
        tool = build_read_channel_tool()
        assert tool.name == "read_channel"

    def test_build_voice_registry_includes_silent(self) -> None:
        scheduler = MagicMock()
        reg = build_voice_registry(scheduler)
        names = {t.name for t in reg.tools()}
        assert "silent" in names

    def test_build_text_registry_includes_silent(self) -> None:
        scheduler = MagicMock()
        reg = build_text_registry(scheduler)
        names = {t.name for t in reg.tools()}
        assert "silent" in names

    def test_build_voice_registry_includes_shift_focus_when_fm_provided(self) -> None:
        scheduler = MagicMock()
        fm = _make_focus_manager()
        reg = build_voice_registry(scheduler, focus_manager=fm)
        names = {t.name for t in reg.tools()}
        assert "shift_focus" in names

    def test_build_text_registry_includes_shift_focus_when_fm_provided(self) -> None:
        scheduler = MagicMock()
        fm = _make_focus_manager()
        reg = build_text_registry(scheduler, focus_manager=fm)
        names = {t.name for t in reg.tools()}
        assert "shift_focus" in names

    def test_build_text_registry_includes_read_channel_when_fm_provided(self) -> None:
        scheduler = MagicMock()
        fm = _make_focus_manager()
        reg = build_text_registry(scheduler, focus_manager=fm)
        names = {t.name for t in reg.tools()}
        assert "read_channel" in names

    def test_build_voice_registry_no_shift_focus_without_fm(self) -> None:
        scheduler = MagicMock()
        reg = build_voice_registry(scheduler)
        names = {t.name for t in reg.tools()}
        assert "shift_focus" not in names

    def test_build_text_registry_no_shift_focus_without_fm(self) -> None:
        scheduler = MagicMock()
        reg = build_text_registry(scheduler)
        names = {t.name for t in reg.tools()}
        assert "shift_focus" not in names
