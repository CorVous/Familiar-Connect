"""Tests for the agentic loop helper.

Drives a fake :class:`LLMClient` through one or more iterations of
``stream_completion``, with a :class:`ToolRegistry` providing handlers.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from familiar_connect.llm import LLMClient, LLMDelta, Message
from familiar_connect.tools.loop import AgenticResult, agentic_loop
from familiar_connect.tools.registry import Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.history.async_store import AsyncHistoryStore


# ---------------------------------------------------------------------------
# Scripted streaming LLM
# ---------------------------------------------------------------------------


@dataclass
class _Script:
    """One ``stream_completion`` invocation's worth of deltas."""

    deltas: list[LLMDelta] = field(default_factory=list)


class _ScriptedStreamLLM(LLMClient):
    """LLM stand-in that yields pre-scripted deltas per call."""

    def __init__(self, scripts: list[_Script]) -> None:
        super().__init__(api_key="k", model="m")
        self._scripts = scripts
        self.calls: list[list[Message]] = []
        self.tool_payloads: list[list[dict] | None] = []

    async def stream_completion(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMDelta]:
        self.calls.append([Message(**m.__dict__) for m in messages])
        self.tool_payloads.append(tools)
        if not self._scripts:
            return
        script = self._scripts.pop(0)
        for delta in script.deltas:
            yield delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> ToolContext:
    # handlers in these tests never touch ``history`` or ``bus`` — pass
    # casted ``None`` so the unit suite doesn't need real subsystems.
    return ToolContext(
        familiar_id="fam-1",
        channel_id=42,
        channel_kind="text",
        turn_id="turn-1",
        history=cast("AsyncHistoryStore", None),
        bus=cast("EventBus", None),
    )


async def _ok_handler(_args: dict, _ctx: ToolContext) -> str:
    await asyncio.sleep(0)
    return "ok"


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _delta_text(text: str) -> LLMDelta:
    return LLMDelta(content=text)


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgenticLoopTermination:
    @pytest.mark.asyncio
    async def test_terminates_when_no_tool_calls(self) -> None:
        llm = _ScriptedStreamLLM([
            _Script(deltas=[_delta_text("Hello there"), _delta_finish("stop")])
        ])
        registry = ToolRegistry()
        result: AgenticResult = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="hi")],
            registry=registry,
            ctx=_make_ctx(),
        )
        assert result.final_content == "Hello there"
        assert result.iterations == 1
        assert result.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_empty_registry_passes_no_tools(self) -> None:
        llm = _ScriptedStreamLLM([_Script(deltas=[_delta_text("hi")])])
        await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="hi")],
            registry=ToolRegistry(),
            ctx=_make_ctx(),
        )
        assert llm.tool_payloads == [None]

    @pytest.mark.asyncio
    async def test_non_empty_registry_passes_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(
            Tool(name="noop", description="", parameters={}, handler=_ok_handler)
        )
        llm = _ScriptedStreamLLM([_Script(deltas=[_delta_text("hi")])])
        await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="hi")],
            registry=registry,
            ctx=_make_ctx(),
        )
        assert llm.tool_payloads[0] is not None
        assert llm.tool_payloads[0][0]["function"]["name"] == "noop"


class TestAgenticLoopToolExecution:
    @pytest.mark.asyncio
    async def test_executes_tool_then_re_calls_llm(self) -> None:
        handler_seen: list[dict] = []

        async def _h(args: dict, _ctx: ToolContext) -> str:
            await asyncio.sleep(0)
            handler_seen.append(args)
            return json.dumps({"ok": True})

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="set_alarm",
                description="",
                parameters={"type": "object"},
                handler=_h,
            )
        )

        scripts = [
            # iteration 1: model emits a tool call
            _Script(
                deltas=[
                    _delta_tool_call("c1", "set_alarm", {"reason": "wake"}),
                    _delta_finish("tool_calls"),
                ]
            ),
            # iteration 2: model emits final text
            _Script(deltas=[_delta_text("Done."), _delta_finish("stop")]),
        ]
        llm = _ScriptedStreamLLM(scripts)
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="set alarm")],
            registry=registry,
            ctx=_make_ctx(),
        )
        assert handler_seen == [{"reason": "wake"}]
        assert result.final_content == "Done."
        assert result.iterations == 2
        assert result.tool_calls_made == 1

        # Second call to LLM should include the tool-result message
        second_call_msgs = llm.calls[1]
        tool_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "c1"
        assert json.loads(tool_msgs[0].content_str) == {"ok": True}

    @pytest.mark.asyncio
    async def test_handler_exception_surfaced_as_tool_error(self) -> None:
        async def _broken(_args: dict, _ctx: ToolContext) -> str:
            await asyncio.sleep(0)
            msg = "boom"
            raise RuntimeError(msg)

        registry = ToolRegistry()
        registry.register(
            Tool(name="broken", description="", parameters={}, handler=_broken)
        )
        scripts = [
            _Script(
                deltas=[
                    _delta_tool_call("c1", "broken", {}),
                    _delta_finish("tool_calls"),
                ]
            ),
            _Script(deltas=[_delta_text("Tool failed, sorry."), _delta_finish("stop")]),
        ]
        llm = _ScriptedStreamLLM(scripts)
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="x")],
            registry=registry,
            ctx=_make_ctx(),
        )
        assert result.final_content == "Tool failed, sorry."
        tool_msgs = [m for m in llm.calls[1] if m.role == "tool"]
        body = json.loads(tool_msgs[0].content_str)
        assert "error" in body
        assert "boom" in body["error"]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self) -> None:
        registry = ToolRegistry()
        scripts = [
            _Script(
                deltas=[
                    _delta_tool_call("c1", "ghost", {}),
                    _delta_finish("tool_calls"),
                ]
            ),
            _Script(deltas=[_delta_text("oh well"), _delta_finish("stop")]),
        ]
        llm = _ScriptedStreamLLM(scripts)
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="x")],
            registry=registry,
            ctx=_make_ctx(),
        )
        assert result.iterations == 2
        tool_msgs = [m for m in llm.calls[1] if m.role == "tool"]
        body = json.loads(tool_msgs[0].content_str)
        assert "error" in body
        assert "ghost" in body["error"]

    @pytest.mark.asyncio
    async def test_invalid_args_json_returns_error_result(self) -> None:
        registry = ToolRegistry()
        registry.register(
            Tool(name="t", description="", parameters={}, handler=_ok_handler)
        )
        scripts = [
            _Script(
                deltas=[
                    LLMDelta(
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "t",
                                    "arguments": "{not valid json",
                                },
                            }
                        ]
                    ),
                    _delta_finish("tool_calls"),
                ]
            ),
            _Script(deltas=[_delta_text("recovered"), _delta_finish("stop")]),
        ]
        llm = _ScriptedStreamLLM(scripts)
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="x")],
            registry=registry,
            ctx=_make_ctx(),
        )
        assert result.iterations == 2
        tool_msgs = [m for m in llm.calls[1] if m.role == "tool"]
        body = json.loads(tool_msgs[0].content_str)
        assert "error" in body


class TestAgenticLoopGuards:
    @pytest.mark.asyncio
    async def test_caps_at_max_iterations(self) -> None:
        registry = ToolRegistry()
        registry.register(
            Tool(name="t", description="", parameters={}, handler=_ok_handler)
        )
        # 10 iterations, each emitting another tool call
        scripts = [
            _Script(
                deltas=[
                    _delta_tool_call(f"c{i}", "t", {}),
                    _delta_finish("tool_calls"),
                ]
            )
            for i in range(10)
        ]
        llm = _ScriptedStreamLLM(scripts)
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="loop")],
            registry=registry,
            ctx=_make_ctx(),
            max_iterations=3,
        )
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_handler_timeout_returns_error(self) -> None:
        async def _slow(_args: dict, _ctx: ToolContext) -> str:
            await asyncio.sleep(5)
            return "never"

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="slow",
                description="",
                parameters={},
                handler=_slow,
                timeout_s=0.05,
            )
        )
        scripts = [
            _Script(
                deltas=[
                    _delta_tool_call("c1", "slow", {}),
                    _delta_finish("tool_calls"),
                ]
            ),
            _Script(deltas=[_delta_text("ok"), _delta_finish("stop")]),
        ]
        llm = _ScriptedStreamLLM(scripts)
        result = await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="x")],
            registry=registry,
            ctx=_make_ctx(),
        )
        tool_msgs = [m for m in llm.calls[1] if m.role == "tool"]
        body = json.loads(tool_msgs[0].content_str)
        assert "error" in body
        assert "timeout" in body["error"].lower()
        assert result.iterations == 2


class TestAgenticLoopCallbacks:
    @pytest.mark.asyncio
    async def test_on_delta_called_per_chunk(self) -> None:
        seen: list[LLMDelta] = []

        async def _on_delta(d: LLMDelta) -> None:
            await asyncio.sleep(0)
            seen.append(d)

        llm = _ScriptedStreamLLM([
            _Script(
                deltas=[
                    _delta_text("a"),
                    _delta_text("b"),
                    _delta_finish("stop"),
                ]
            )
        ])
        await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="x")],
            registry=ToolRegistry(),
            ctx=_make_ctx(),
            on_delta=_on_delta,
        )
        contents = [d.content for d in seen if d.content]
        assert contents == ["a", "b"]

    @pytest.mark.asyncio
    async def test_on_iteration_end_receives_assistant_and_tool_messages(self) -> None:
        async def _h(_args: dict, _ctx: ToolContext) -> str:
            await asyncio.sleep(0)
            return "result"

        registry = ToolRegistry()
        registry.register(Tool(name="t", description="", parameters={}, handler=_h))
        scripts = [
            _Script(
                deltas=[
                    _delta_text("Working..."),
                    _delta_tool_call("c1", "t", {}),
                    _delta_finish("tool_calls"),
                ]
            ),
            _Script(deltas=[_delta_text("Done"), _delta_finish("stop")]),
        ]
        events: list[tuple[Message, list[Message]]] = []

        async def _on_iter(asst: Message, tools: list[Message]) -> None:
            await asyncio.sleep(0)
            events.append((asst, tools))

        llm = _ScriptedStreamLLM(scripts)
        await agentic_loop(
            llm=llm,
            messages=[Message(role="user", content="x")],
            registry=registry,
            ctx=_make_ctx(),
            on_iteration_end=_on_iter,
        )
        assert len(events) == 2
        # Iteration 1: assistant text + tool_calls, plus one tool result message
        asst1, tools1 = events[0]
        assert asst1.content == "Working..."
        assert asst1.tool_calls is not None
        assert len(tools1) == 1
        assert tools1[0].role == "tool"
        # Iteration 2: assistant final text, no tools
        asst2, tools2 = events[1]
        assert asst2.content == "Done"
        assert not tools2
