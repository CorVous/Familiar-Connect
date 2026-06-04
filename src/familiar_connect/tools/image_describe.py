"""Vision-based image description using an LLMClient."""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.llm import LLMClient

# neutral base — no character/persona constraints. per-familiar
# `constraints` (from [prompt].image_description_constraints) append
# here; base stays stable so authors add one sentence, not re-author
# the whole prompt.
_DESCRIBE_PROMPT = (
    "Describe this image concisely for a chat assistant. "
    "Focus on the main subject and any notable details."
)


async def describe_image(
    *,
    llm: LLMClient,
    jpeg_base64: str,
    media_type: str = "image/jpeg",
    constraints: str = "",
) -> str:
    """Call vision model; return text description of image.

    Sends single user message with text + image_url blocks. ``constraints``
    append to base prompt (per-familiar persona tuning); blank → base only.
    """
    extra = constraints.strip()
    prompt = f"{_DESCRIBE_PROMPT} {extra}" if extra else _DESCRIBE_PROMPT
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{jpeg_base64}"},
        },
    ]
    messages = [Message(role="user", content=content)]
    reply = await llm.chat(messages)
    return reply.content if isinstance(reply.content, str) else ""
