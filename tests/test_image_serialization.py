"""Tests for ImageResult serialisation in loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from familiar_connect.llm import LLMClient, LLMDelta, Message
from familiar_connect.tools.loop import (
    agentic_loop,
    serialize_image_result,
    tool_content_as_text,
)
from familiar_connect.tools.registry import ImageResult, Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.history.async_store import AsyncHistoryStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> ToolContext:
    return ToolContext(
        familiar_id="fam-1",
        channel_id=42,
        channel_kind="text",
        turn_id="turn-1",
        history=cast("AsyncHistoryStore", None),
        bus=cast("EventBus", None),
    )


@dataclass
class _Script:
    deltas: list[LLMDelta] = field(default_factory=list)


class _ScriptedLLM(LLMClient):
    def __init__(self, scripts: list[_Script], *, multimodal: bool = False) -> None:
        super().__init__(api_key="k", model="m", multimodal=multimodal)
        self._scripts = scripts

    async def stream_completion(  # type: ignore[override]
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


def _make_tool_call_delta(name: str, args: str = "{}") -> LLMDelta:
    return LLMDelta(
        tool_calls=[
            {
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ]
    )


async def _image_handler(_args: dict, _ctx: ToolContext) -> ImageResult:  # noqa: RUF029
    return ImageResult(description="a cat", jpeg_base64="abc123")


def _make_image_tool() -> Tool:
    return Tool(
        name="view_image",
        description="view an image",
        parameters={"type": "object", "properties": {}},
        handler=_image_handler,
    )


# ---------------------------------------------------------------------------
# serialize_image_result unit tests
# ---------------------------------------------------------------------------


def test_serialize_image_result_text_only() -> None:
    res = ImageResult(description="a cat", jpeg_base64="abc123")
    result = serialize_image_result(res, multimodal=False)
    assert result == "a cat"


def test_serialize_image_result_multimodal() -> None:
    res = ImageResult(description="a cat", jpeg_base64="abc123")
    result = serialize_image_result(res, multimodal=True)
    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": "a cat"}
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"


def test_tool_content_as_text_passthrough_str() -> None:
    assert tool_content_as_text("hello") == "hello"


def test_tool_content_as_text_extracts_text_blocks() -> None:
    content = [
        {"type": "text", "text": "a cat"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]
    assert tool_content_as_text(content) == "a cat"


# ---------------------------------------------------------------------------
# loop integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_serialises_image_result_textonly() -> None:
    registry = ToolRegistry()
    registry.register(_make_image_tool())

    scripts = [
        _Script(deltas=[_make_tool_call_delta("view_image")]),
        _Script(deltas=[LLMDelta(content="done")]),
    ]
    llm = _ScriptedLLM(scripts, multimodal=False)

    messages: list[Message] = [Message(role="system", content="sys")]
    result = await agentic_loop(
        llm=llm,
        messages=messages,
        registry=registry,
        ctx=_make_ctx(),
    )
    # find the tool message
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "a cat"
    assert result.final_content == "done"


@pytest.mark.asyncio
async def test_loop_serialises_image_result_multimodal() -> None:
    registry = ToolRegistry()
    registry.register(_make_image_tool())

    scripts = [
        _Script(deltas=[_make_tool_call_delta("view_image")]),
        _Script(deltas=[LLMDelta(content="done")]),
    ]
    llm = _ScriptedLLM(scripts, multimodal=True)

    messages: list[Message] = [Message(role="system", content="sys")]
    await agentic_loop(
        llm=llm,
        messages=messages,
        registry=registry,
        ctx=_make_ctx(),
    )
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert isinstance(tool_msgs[0].content, list)
    assert tool_msgs[0].content[0]["type"] == "text"
    assert tool_msgs[0].content[1]["type"] == "image_url"


def test_message_to_dict_passes_list_content() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {}},
    ]
    msg = Message(role="tool", content=content, tool_call_id="c1")
    d = msg.to_dict()
    assert d["content"] is content
