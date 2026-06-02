"""Tests for the tool registry."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from familiar_connect.tools.registry import ImageResult, Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.history.async_store import AsyncHistoryStore


async def _noop_handler(_args: dict, _ctx: ToolContext) -> str:
    await asyncio.sleep(0)
    return "ok"


def _make_tool(name: str = "echo") -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
        handler=_noop_handler,
    )


def test_register_and_get_round_trip() -> None:
    registry = ToolRegistry()
    tool = _make_tool()
    registry.register(tool)
    assert registry.get("echo") is tool


def test_get_unknown_name_raises_key_error() -> None:
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get("nope")


def test_duplicate_name_register_raises_value_error() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool("echo"))
    with pytest.raises(ValueError, match="echo"):
        registry.register(_make_tool("echo"))


def test_as_openai_tools_shape_matches_function_calling_schema() -> None:
    registry = ToolRegistry()
    params = {
        "type": "object",
        "properties": {
            "when": {"type": "string", "format": "date-time"},
            "reason": {"type": "string", "maxLength": 200},
        },
        "required": ["reason"],
    }
    registry.register(
        Tool(
            name="set_alarm",
            description="Schedule a wake.",
            parameters=params,
            handler=_noop_handler,
        )
    )
    assert registry.as_openai_tools() == [
        {
            "type": "function",
            "function": {
                "name": "set_alarm",
                "description": "Schedule a wake.",
                "parameters": params,
            },
        }
    ]


def test_as_openai_tools_empty_registry_returns_empty_list() -> None:
    registry = ToolRegistry()
    assert registry.as_openai_tools() == []


def test_tools_iteration_returns_all_registered() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool("a"))
    registry.register(_make_tool("b"))
    assert sorted(t.name for t in registry.tools()) == ["a", "b"]


def test_default_timeout_is_set() -> None:
    tool = _make_tool()
    assert tool.timeout_s > 0


def _make_ctx() -> ToolContext:
    return ToolContext(
        familiar_id="fam-1",
        channel_id=42,
        channel_kind="text",
        turn_id="turn-1",
        history=cast("AsyncHistoryStore", None),
        bus=cast("EventBus", None),
    )


def test_image_result_carries_both() -> None:
    result = ImageResult(description="a cat", jpeg_base64="abc123")
    assert result.description == "a cat"
    assert result.jpeg_base64 == "abc123"
    assert result.media_type == "image/jpeg"


def test_tool_context_images_defaults_empty() -> None:
    ctx = _make_ctx()
    assert ctx.images == {}


def test_tool_context_description_llm_defaults_none() -> None:
    ctx = _make_ctx()
    assert ctx.description_llm is None
