"""Tests for image_describe module."""

from __future__ import annotations

import pytest

from familiar_connect.llm import LLMClient, Message
from familiar_connect.tools.image_describe import describe_image


class _CaptureLLM(LLMClient):
    """LLM stub that records the chat message and returns a fixed reply."""

    def __init__(self, reply: str = "a cat") -> None:
        super().__init__(api_key="k", model="test-vision")
        self.captured_messages: list[Message] = []
        self._reply = reply

    async def chat(self, messages: list[Message]) -> Message:  # type: ignore[override]
        self.captured_messages.extend(messages)
        return Message(role="assistant", content=self._reply)


@pytest.mark.asyncio
async def test_describe_calls_llm_with_vision_block() -> None:
    llm = _CaptureLLM(reply="a fluffy cat")
    result = await describe_image(llm=llm, jpeg_base64="abc123")

    assert result == "a fluffy cat"
    assert len(llm.captured_messages) >= 1
    user_msg = llm.captured_messages[-1]
    assert user_msg.role == "user"
    assert isinstance(user_msg.content, list)

    # check the image_url block is present
    image_blocks = [b for b in user_msg.content if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    url = image_blocks[0]["image_url"]["url"]
    assert url == "data:image/jpeg;base64,abc123"


@pytest.mark.asyncio
async def test_describe_uses_custom_media_type() -> None:
    llm = _CaptureLLM()
    await describe_image(llm=llm, jpeg_base64="abc123", media_type="image/png")

    user_msg = llm.captured_messages[-1]
    assert isinstance(user_msg.content, list)
    image_blocks = [b for b in user_msg.content if b.get("type") == "image_url"]
    assert isinstance(image_blocks[0], dict)
    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
