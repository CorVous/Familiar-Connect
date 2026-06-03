"""Vision-based image description using an LLMClient."""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.llm import LLMClient

_DESCRIBE_PROMPT = (
    "Describe this image concisely for a chat assistant. "
    "Focus on the main subject and any notable details. "
    "Describe what you actually see — colors, shapes, composition, text. "
    "Do not name specific characters, people, franchises, brands, or other "
    "proper nouns; describe appearance instead."
)


async def describe_image(
    *,
    llm: LLMClient,
    jpeg_base64: str,
    media_type: str = "image/jpeg",
) -> str:
    """Call vision model; return text description of image.

    Sends a single user message with text + image_url content blocks.
    """
    content = [
        {"type": "text", "text": _DESCRIBE_PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{jpeg_base64}"},
        },
    ]
    messages = [Message(role="user", content=content)]
    reply = await llm.chat(messages)
    return reply.content if isinstance(reply.content, str) else ""
