"""Tests for the view_image tool."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from familiar_connect.llm import LLMClient, Message
from familiar_connect.tools.image import build_view_image_tool
from familiar_connect.tools.registry import ImageResult, Tool, ToolContext

if TYPE_CHECKING:
    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.history.async_store import AsyncHistoryStore


def _make_tiny_jpeg() -> bytes:
    """Return minimal valid JPEG bytes (solid red 1x1)."""
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_tiny_png() -> bytes:
    """Return minimal valid PNG bytes."""
    img = Image.new("RGB", (10, 10), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _DescribeLLM(LLMClient):
    def __init__(self, reply: str = "a tiny dot") -> None:
        super().__init__(api_key="k", model="vision")
        self._reply = reply
        self.captured: list[Message] = []

    async def chat(self, messages: list[Message]) -> Message:  # type: ignore[override]
        self.captured.extend(messages)
        return Message(role="assistant", content=self._reply)


def _make_ctx(
    images: dict[str, str] | None = None,
    description_llm: LLMClient | None = None,
) -> ToolContext:
    return ToolContext(
        familiar_id="fam-1",
        channel_id=42,
        channel_kind="text",
        turn_id="turn-1",
        history=cast("AsyncHistoryStore", None),
        bus=cast("EventBus", None),
        images=images or {},
        description_llm=description_llm,
    )


# ---------------------------------------------------------------------------
# build_view_image_tool
# ---------------------------------------------------------------------------


def test_build_view_image_tool_returns_tool() -> None:
    tool = build_view_image_tool()
    assert isinstance(tool, Tool)
    assert tool.name == "view_image"


def test_view_image_tool_has_image_id_param() -> None:
    tool = build_view_image_tool()
    props = tool.parameters.get("properties", {})
    assert "image_id" in props
    assert "image_id" in tool.parameters.get("required", [])


# ---------------------------------------------------------------------------
# unknown id → JSON error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_view_image_unknown_id_returns_error() -> None:
    tool = build_view_image_tool()
    ctx = _make_ctx(images={})
    result = await tool.handler({"image_id": "img_99"}, ctx)
    assert isinstance(result, str)
    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# happy path — fetch + describe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_view_image_returns_image_result() -> None:
    png_bytes = _make_tiny_png()
    ctx = _make_ctx(
        images={"img_0": "http://cdn.example.com/cat.png"},
        description_llm=_DescribeLLM("a cat"),
    )
    tool = build_view_image_tool()
    with patch(
        "familiar_connect.tools.image._fetch_image_bytes",
        new=AsyncMock(return_value=png_bytes),
    ):
        result = await tool.handler({"image_id": "img_0"}, ctx)

    assert isinstance(result, ImageResult)
    assert result.description == "a cat"
    assert len(result.jpeg_base64) > 0


# ---------------------------------------------------------------------------
# no description LLM → placeholder description, still has bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_view_image_constraints_flow_into_description() -> None:
    """Constraints bound at tool construction reach the describe prompt."""
    png_bytes = _make_tiny_png()
    llm = _DescribeLLM("a cat")
    ctx = _make_ctx(
        images={"img_0": "http://cdn.example.com/cat.png"},
        description_llm=llm,
    )
    tool = build_view_image_tool(describe_constraints="Do not name characters.")
    with patch(
        "familiar_connect.tools.image._fetch_image_bytes",
        new=AsyncMock(return_value=png_bytes),
    ):
        await tool.handler({"image_id": "img_0"}, ctx)

    text_blocks = [
        b
        for m in llm.captured
        if isinstance(m.content, list)
        for b in m.content
        if b.get("type") == "text"
    ]
    assert any("Do not name characters." in b["text"] for b in text_blocks)


@pytest.mark.asyncio
async def test_view_image_no_description_llm_degrades() -> None:
    png_bytes = _make_tiny_png()
    ctx = _make_ctx(
        images={"img_0": "http://cdn.example.com/img.png"},
        description_llm=None,
    )
    tool = build_view_image_tool()
    with patch(
        "familiar_connect.tools.image._fetch_image_bytes",
        new=AsyncMock(return_value=png_bytes),
    ):
        result = await tool.handler({"image_id": "img_0"}, ctx)

    assert isinstance(result, ImageResult)
    # description is a non-empty placeholder string
    assert isinstance(result.description, str)
    assert result.description  # non-empty placeholder
    assert len(result.jpeg_base64) > 0
