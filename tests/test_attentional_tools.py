"""Tests for attentional tools: shift_focus, silent, read_channel, start_activity.

TDD red-first: covers all specified behaviors before implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_connect.activities.config import ActivityType
from familiar_connect.llm import LLMClient, LLMDelta, Message
from familiar_connect.tools.builtins import (
    build_read_channel_tool,
    build_shift_focus_tool,
    build_silent_tool,
    build_start_activity_tool,
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
    fm.shift_now = AsyncMock()
    fm.get_focus = MagicMock(return_value=99)
    fm.is_subscribed = MagicMock(return_value=True)
    fm.subscribed_channels = MagicMock(return_value=[55, 99])
    fm.channel_label = MagicMock(side_effect=lambda c: f"#{c}")
    fm.catch_up_limit = 20
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
    async def test_calls_shift_now(self) -> None:
        fm = _make_focus_manager()
        ctx = _make_ctx(focus_manager=fm)
        await _shift_focus_handler({"channel_id": 55}, ctx)
        fm.shift_now.assert_awaited_once_with(55)

    @pytest.mark.asyncio
    async def test_returns_ok_json(self) -> None:
        # no store wired → degrades to bare ack (back-compat)
        fm = _make_focus_manager()
        ctx = _make_ctx(focus_manager=fm)
        result = await _shift_focus_handler({"channel_id": 55}, ctx)
        parsed = json.loads(result)
        assert parsed == {"ok": True, "channel_id": 55}

    @pytest.mark.asyncio
    async def test_returns_messages_from_target_channel(self) -> None:
        # content-bearing shift: returns target channel's recent turns so
        # model sees the channel in-turn rather than narrating blind
        turns = [
            _make_history_turn(1, "user", "first", 55),
            _make_history_turn(2, "user", "second", 55),
        ]
        store = MagicMock()
        store.recent = AsyncMock(return_value=turns)
        fm = _make_focus_manager()
        ctx = _make_ctx(focus_manager=fm, store=store)
        result = await _shift_focus_handler({"channel_id": 55}, ctx)
        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert parsed["channel_id"] == 55
        assert [m["content"] for m in parsed["messages"]] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_fetches_target_channel_not_current_focus(self) -> None:
        # unlike read_channel (current focus), shift reads the *target*
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)  # current focus differs
        ctx = _make_ctx(focus_manager=fm, store=store)
        await _shift_focus_handler({"channel_id": 55}, ctx)
        assert store.recent.call_args.kwargs.get("channel_id") == 55

    @pytest.mark.asyncio
    async def test_preview_limit_matches_catch_up_window(self) -> None:
        # perception == consumption: the preview she sees is exactly the
        # catch-up window shift_now promotes (rest is missed)
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.catch_up_limit = 7
        ctx = _make_ctx(focus_manager=fm, store=store)
        await _shift_focus_handler({"channel_id": 55}, ctx)
        assert store.recent.call_args.kwargs.get("limit") == 7

    @pytest.mark.asyncio
    async def test_empty_channel_returns_empty_messages(self) -> None:
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        ctx = _make_ctx(focus_manager=fm, store=store)
        result = await _shift_focus_handler({"channel_id": 55}, ctx)
        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert parsed["messages"] == []

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

    @pytest.mark.asyncio
    async def test_rejects_unsubscribed_channel(self) -> None:
        # guard: cannot focus a channel the familiar isn't subscribed to
        fm = _make_focus_manager()
        fm.is_subscribed = MagicMock(return_value=False)
        ctx = _make_ctx(focus_manager=fm)
        result = await _shift_focus_handler({"channel_id": 12345}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed
        fm.shift_now.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unsubscribed_error_lists_available_channels(self) -> None:
        # error surfaces valid targets so the model can recover, not stall
        fm = _make_focus_manager()
        fm.is_subscribed = MagicMock(return_value=False)
        fm.subscribed_channels = MagicMock(return_value=[55, 99])
        ctx = _make_ctx(focus_manager=fm)
        result = await _shift_focus_handler({"channel_id": 12345}, ctx)
        parsed = json.loads(result)
        ids = [c["channel_id"] for c in parsed["available_channels"]]
        assert ids == [55, 99]

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

    @pytest.mark.asyncio
    async def test_silent_tool_does_not_call_on_iteration_end(self) -> None:
        """on_iteration_end must not fire for silent iterations.

        Silent call + its reasoning would be persisted to history,
        re-seeding the model's rationale for silence on the next turn.
        """
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

        iter_end_calls: list[tuple[Message, list[Message]]] = []

        async def _on_end(  # noqa: RUF029
            assistant: Message, tool_msgs: list[Message]
        ) -> None:
            iter_end_calls.append((assistant, tool_msgs))

        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="hi")],
            registry=registry,
            ctx=ctx,
            on_iteration_end=_on_end,
        )
        assert result.is_silent is True
        assert iter_end_calls == []


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

    @pytest.mark.asyncio
    async def test_passes_before_id_to_store(self) -> None:
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        await _read_channel_handler({"limit": 10, "before_id": 7}, ctx)
        call_kwargs = store.recent.call_args
        assert call_kwargs.kwargs.get("before_id") == 7

    @pytest.mark.asyncio
    async def test_before_id_defaults_none(self) -> None:
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        await _read_channel_handler({}, ctx)
        call_kwargs = store.recent.call_args
        assert call_kwargs.kwargs.get("before_id") is None

    @pytest.mark.asyncio
    async def test_around_id_calls_turns_around(self) -> None:
        turns = [
            _make_history_turn(9, "user", "before", 99),
            _make_history_turn(10, "assistant", "anchor", 99),
            _make_history_turn(11, "user", "after", 99),
        ]
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        store.turns_around = AsyncMock(return_value=turns)
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        result = await _read_channel_handler({"around_id": 10, "limit": 20}, ctx)
        store.turns_around.assert_awaited_once_with(
            familiar_id="fam-1",
            channel_id=99,
            turn_id=10,
            before=10,
            after=10,
        )
        store.recent.assert_not_awaited()
        parsed = json.loads(result)
        assert [t["id"] for t in parsed] == [9, 10, 11]
        assert parsed[1]["content"] == "anchor"

    @pytest.mark.asyncio
    async def test_around_id_split_respects_max_limit(self) -> None:
        store = MagicMock()
        store.turns_around = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        await _read_channel_handler({"around_id": 10, "limit": 100}, ctx)
        call_kwargs = store.turns_around.call_args
        assert call_kwargs.kwargs.get("before") == 25
        assert call_kwargs.kwargs.get("after") == 25

    @pytest.mark.asyncio
    async def test_around_id_small_limit_keeps_anchor_window(self) -> None:
        store = MagicMock()
        store.turns_around = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        await _read_channel_handler({"around_id": 10, "limit": 1}, ctx)
        call_kwargs = store.turns_around.call_args
        assert call_kwargs.kwargs["before"] >= 1
        assert call_kwargs.kwargs["after"] >= 1

    @pytest.mark.asyncio
    async def test_before_id_and_around_id_mutually_exclusive(self) -> None:
        store = MagicMock()
        store.recent = AsyncMock(return_value=[])
        store.turns_around = AsyncMock(return_value=[])
        fm = _make_focus_manager()
        fm.get_focus = MagicMock(return_value=99)
        ctx = _make_ctx(focus_manager=fm, store=store, channel_kind="text")

        result = await _read_channel_handler({"before_id": 5, "around_id": 10}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed
        store.recent.assert_not_awaited()
        store.turns_around.assert_not_awaited()

    def test_build_read_channel_tool_name(self) -> None:
        tool = _build_read_channel_tool_direct()
        assert tool.name == "read_channel"

    def test_build_read_channel_tool_schema_has_limit(self) -> None:
        tool = _build_read_channel_tool_direct()
        props = tool.parameters["properties"]
        assert "limit" in props
        assert props["limit"]["maximum"] == 50

    def test_build_read_channel_tool_schema_has_paging_params(self) -> None:
        tool = _build_read_channel_tool_direct()
        props = tool.parameters["properties"]
        assert "before_id" in props
        assert "around_id" in props
        assert props["around_id"]["type"] == "integer"


# ---------------------------------------------------------------------------
# start_activity tool
# ---------------------------------------------------------------------------


class _FakeActivityEngine:
    """Canned catalog + defer_start recorder; satisfies StartActivityEngine."""

    def __init__(self, result: dict | None = None) -> None:
        self.catalog: tuple[ActivityType, ...] = (
            ActivityType(
                id="creek_walk",
                label="a creek walk",
                duration_minutes=(20, 40),
                seed="x",
            ),
            ActivityType(
                id="hatbox",
                label="tending the hatbox",
                duration_minutes=(10, 25),
                seed="y",
            ),
        )
        self.calls: list[tuple[str, str | None]] = []
        self.active: object | None = None
        self._result = result or {
            "ack": "ok",
            "label": "a creek walk",
            "duration_minutes": 30,
        }

    def defer_start(self, type_id: str, note: str | None = None) -> dict[str, Any]:
        self.calls.append((type_id, note))
        return self._result


class TestStartActivityTool:
    def test_tool_name(self) -> None:
        tool = build_start_activity_tool(_FakeActivityEngine())
        assert tool.name == "start_activity"

    def test_activity_enum_built_from_catalog(self) -> None:
        tool = build_start_activity_tool(_FakeActivityEngine())
        props = tool.parameters["properties"]
        assert props["activity"]["type"] == "string"
        assert props["activity"]["enum"] == ["creek_walk", "hatbox"]

    def test_activity_description_includes_labels(self) -> None:
        tool = build_start_activity_tool(_FakeActivityEngine())
        desc = tool.parameters["properties"]["activity"]["description"]
        assert "a creek walk" in desc
        assert "tending the hatbox" in desc

    def test_note_optional_string(self) -> None:
        tool = build_start_activity_tool(_FakeActivityEngine())
        props = tool.parameters["properties"]
        assert props["note"]["type"] == "string"
        assert tool.parameters["required"] == ["activity"]

    def test_description_within_budget(self) -> None:
        tool = build_start_activity_tool(_FakeActivityEngine())
        assert len(tool.description) <= 450

    def test_description_carries_when_to_go_policy(self) -> None:
        # zero character-card growth: policy lives entirely here
        desc = build_start_activity_tool(_FakeActivityEngine()).description.lower()
        assert "quiet" in desc
        assert "goodbye" in desc
        assert "miss" in desc

    @pytest.mark.asyncio
    async def test_handler_calls_defer_start_with_note(self) -> None:
        engine = _FakeActivityEngine()
        tool = build_start_activity_tool(engine)
        result = await tool.handler(
            {"activity": "creek_walk", "note": "want to see the herons"},
            _make_ctx(),
        )
        assert engine.calls == [("creek_walk", "want to see the herons")]
        assert isinstance(result, str)
        assert json.loads(result) == {
            "ack": "ok",
            "label": "a creek walk",
            "duration_minutes": 30,
        }

    @pytest.mark.asyncio
    async def test_handler_note_defaults_none(self) -> None:
        engine = _FakeActivityEngine()
        tool = build_start_activity_tool(engine)
        await tool.handler({"activity": "hatbox"}, _make_ctx())
        assert engine.calls == [("hatbox", None)]

    @pytest.mark.asyncio
    async def test_handler_passes_engine_error_through(self) -> None:
        engine = _FakeActivityEngine(result={"error": "already out"})
        tool = build_start_activity_tool(engine)
        result = await tool.handler({"activity": "hatbox"}, _make_ctx())
        assert isinstance(result, str)
        assert json.loads(result) == {"error": "already out"}

    @pytest.mark.asyncio
    async def test_handler_missing_activity_returns_error(self) -> None:
        engine = _FakeActivityEngine()
        tool = build_start_activity_tool(engine)
        result = await tool.handler({}, _make_ctx())
        assert isinstance(result, str)
        assert "error" in json.loads(result)
        assert engine.calls == []

    @pytest.mark.asyncio
    async def test_handler_non_string_note_returns_error(self) -> None:
        engine = _FakeActivityEngine()
        tool = build_start_activity_tool(engine)
        result = await tool.handler({"activity": "hatbox", "note": 5}, _make_ctx())
        assert isinstance(result, str)
        assert "error" in json.loads(result)
        assert engine.calls == []

    @pytest.mark.asyncio
    async def test_called_while_already_out_is_silent_equivalent(self) -> None:
        # eval finding: stay-out intent misroutes to start_activity with
        # meta narration that would post live. belt: while out, the call
        # signals stay-out — return the silent sentinel, no defer_start
        engine = _FakeActivityEngine()
        engine.active = object()  # any truthy activity record
        tool = build_start_activity_tool(engine)
        result = await tool.handler({"activity": "creek_walk"}, _make_ctx())
        assert result == SILENT_RESULT
        assert engine.calls == []

    def test_description_demands_in_character_goodbye(self) -> None:
        # eval finding: goodbye voice-thinness — description must ask
        # for an in-character goodbye
        tool = build_start_activity_tool(_FakeActivityEngine())
        assert "in-character goodbye" in tool.description

    def test_scheduled_entry_appends_availability_window(self) -> None:
        # capability hint: a scheduled entry advertises WHEN it's choosable;
        # an unscheduled entry stays clean
        engine = _FakeActivityEngine()
        engine.catalog = (
            ActivityType(
                id="weekday_rounds",
                label="weekday rounds",
                duration_minutes=(20, 40),
                seed="x",
                active_days=frozenset({0, 1, 2, 3, 4}),
                active_hours=(time(9, 0), time(17, 0)),
            ),
            ActivityType(
                id="creek_walk",
                label="a creek walk",
                duration_minutes=(20, 40),
                seed="y",
            ),
        )
        tool = build_start_activity_tool(engine)
        desc = tool.parameters["properties"]["activity"]["description"]
        segments = desc.split("; ")
        weekday_seg = next(s for s in segments if "weekday_rounds" in s)
        creek_seg = next(s for s in segments if "creek_walk" in s)

        # 1. scheduled entry carries its window: Mon=0-indexed days + hours.
        # Pin order (not just membership) so a transposition in _WEEKDAY_ABBR
        # or a dropped sorted() is caught.
        assert "Mon Tue Wed Thu Fri" in weekday_seg
        assert "Sun" not in weekday_seg  # Mon-Fri set guards Mon=0 rendering
        assert "09:00-17:00" in weekday_seg

        # 2. unscheduled entry carries no schedule clause
        assert "[" not in creek_seg
        assert ":" not in creek_seg

        # 3. regression: enum lists every id, base 'id' = label text intact
        assert tool.parameters["properties"]["activity"]["enum"] == [
            "weekday_rounds",
            "creek_walk",
        ]
        assert "'weekday_rounds' = weekday rounds" in desc
        assert "'creek_walk' = a creek walk" in desc

    def test_active_days_only_renders_days_without_hours(self) -> None:
        # all-day-but-weekdays config: days clause, no time clause.
        # Guards against gating the whole clause on BOTH fields being set.
        engine = _FakeActivityEngine()
        engine.catalog = (
            ActivityType(
                id="market_day",
                label="the saturday market",
                duration_minutes=(20, 40),
                seed="x",
                active_days=frozenset({5}),
            ),
        )
        desc = build_start_activity_tool(engine).parameters["properties"]["activity"][
            "description"
        ]
        # closing bracket right after the day means no hours were appended
        assert "[Sat]" in desc

    def test_active_hours_only_renders_hours_without_days(self) -> None:
        # every-day-but-certain-hours config: time clause, no day tokens.
        # Guards against gating the whole clause on BOTH fields being set.
        engine = _FakeActivityEngine()
        engine.catalog = (
            ActivityType(
                id="quiet_hours",
                label="winding down",
                duration_minutes=(20, 40),
                seed="x",
                active_hours=(time(9, 0), time(17, 0)),
            ),
        )
        desc = build_start_activity_tool(engine).parameters["properties"]["activity"][
            "description"
        ]
        # opening bracket right before the time means no day prefix
        assert "[09:00-17:00]" in desc
        for abbr in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
            assert abbr not in desc


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

    def test_build_text_registry_includes_start_activity_with_engine(self) -> None:
        scheduler = MagicMock()
        reg = build_text_registry(scheduler, activity_engine=_FakeActivityEngine())
        names = {t.name for t in reg.tools()}
        assert "start_activity" in names

    def test_build_text_registry_no_start_activity_without_engine(self) -> None:
        scheduler = MagicMock()
        reg = build_text_registry(scheduler)
        names = {t.name for t in reg.tools()}
        assert "start_activity" not in names

    def test_build_text_registry_binds_describe_constraints(self) -> None:
        # run.py threads per-familiar constraints through the builder
        scheduler = MagicMock()
        reg = build_text_registry(
            scheduler, image_tools=True, describe_constraints="be brief"
        )
        names = {t.name for t in reg.tools()}
        assert "view_image" in names

    def test_build_voice_registry_never_has_start_activity(self) -> None:
        scheduler = MagicMock()
        fm = _make_focus_manager()
        reg = build_voice_registry(scheduler, focus_manager=fm)
        names = {t.name for t in reg.tools()}
        assert "start_activity" not in names
