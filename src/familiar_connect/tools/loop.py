"""Agentic tool-execution loop.

Drives :meth:`LLMClient.stream_completion` through one or more
iterations: stream deltas, accumulate ``content`` + ``tool_calls``,
execute tools, append results, re-call. Terminates when model returns
no tool calls or ``max_iterations`` is reached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from familiar_connect import log_style as ls
from familiar_connect.llm import LLMDelta, Message
from familiar_connect.tools.registry import ImageResult

# Lazy import to avoid circular import; resolved at call time
_SILENT_RESULT: str | None = None


def _get_silent_result() -> str:
    global _SILENT_RESULT  # noqa: PLW0603
    if _SILENT_RESULT is None:
        try:
            from familiar_connect.tools.silent import SILENT_RESULT  # noqa: PLC0415

            _SILENT_RESULT = SILENT_RESULT
        except ImportError:
            _SILENT_RESULT = "__SILENT__"
    return _SILENT_RESULT


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.llm import LLMClient
    from familiar_connect.tools.registry import Tool, ToolContext, ToolRegistry

_logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 5

OnDelta = "Callable[[LLMDelta], Awaitable[None]]"
OnIterationEnd = "Callable[[Message, list[Message]], Awaitable[None]]"


@dataclass
class AgenticResult:
    """Outcome of an :func:`agentic_loop` run."""

    final_content: str
    iterations: int
    tool_calls_made: int
    transcript: list[Message] = field(default_factory=list)
    is_silent: bool = False  # Set when silent tool was called


def _accumulate_tool_calls(
    pending: dict[int, dict[str, Any]],
    fragments: list[dict[str, Any]],
) -> None:
    """Merge streaming tool-call fragments into ``pending`` by index."""
    for frag in fragments:
        idx = frag.get("index", 0)
        if not isinstance(idx, int):
            continue
        bucket = pending.setdefault(
            idx,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if isinstance(frag.get("id"), str):
            bucket["id"] = frag["id"]
        if isinstance(frag.get("type"), str):
            bucket["type"] = frag["type"]
        fn = frag.get("function") or {}
        if isinstance(fn, dict):
            if isinstance(fn.get("name"), str) and fn["name"]:
                bucket["function"]["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                bucket["function"]["arguments"] += fn["arguments"]


def _finalize_tool_calls(pending: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Order pending tool calls by index; strip empty ones."""
    return [pending[i] for i in sorted(pending) if pending[i].get("id")]


# Leaked tool-call: model writes the invocation as text instead of using
# the tool-calling channel. matches bare + namespaced ``<invoke …>`` tags.
_LEADING_INVOKE_RE = re.compile(r"\s*<(?:\w+:)?invoke\b")
_INVOKE_BLOCK_RE = re.compile(r"<(?:\w+:)?invoke\b.*?</(?:\w+:)?invoke>", re.DOTALL)
_INVOKE_NAME_RE = re.compile(r'<(?:\w+:)?invoke\b[^>]*\bname="([^"]+)"')
# Qwen3 thinking-mode artifact: model writes Python-style calls as plain text
# e.g. silent(reasoning="…") or read_channel() instead of using the tools API.
_PYTHON_SILENT_RE = re.compile(r"^\s*silent\s*\(", re.IGNORECASE)
_PYTHON_TOOL_RE = re.compile(r"^\s*(read_channel|shift_focus)\s*\(", re.IGNORECASE)


def _strip_leaked_tool_calls(content: str) -> tuple[str, bool]:
    """Strip leaked tool invocations emitted as plain text.

    Handles two formats:
    - XML: ``<invoke name="silent">…</invoke>``
    - Python: ``silent(reasoning="…")`` (Qwen3 thinking-mode artifact)

    Returns ``(cleaned, silent_leak)``. ``silent_leak`` True when a
    stripped invocation named the ``silent`` tool — caller treats turn as
    silent. Only fires when content *leads* with an invocation, so a
    stray mention mid-prose stays content.
    """
    if _LEADING_INVOKE_RE.match(content):
        silent_leak = "silent" in _INVOKE_NAME_RE.findall(content)
        cleaned = _INVOKE_BLOCK_RE.sub("", content).strip()
        return cleaned, silent_leak
    if _PYTHON_SILENT_RE.match(content):
        return "", True
    if _PYTHON_TOOL_RE.match(content):
        return "", False
    return content, False


def serialize_image_result(
    res: ImageResult,
    *,
    multimodal: bool,
) -> str | list[dict[str, Any]]:
    """Serialise ``ImageResult`` per slot's multimodal capability.

    ``multimodal=False`` → description string only (text-safe).
    ``multimodal=True`` → list with text + image_url content blocks.
    """
    if not multimodal:
        return res.description
    return [
        {"type": "text", "text": res.description},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{res.media_type};base64,{res.jpeg_base64}",
            },
        },
    ]


def tool_content_as_text(content: str | list[dict[str, Any]]) -> str:
    """Project tool-message content to plain text for history persistence.

    Extracts text blocks from multimodal list; returns str unchanged.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


async def _execute_tool(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
) -> str | ImageResult:
    """Run ``tool.handler`` with timeout; convert exceptions to error JSON."""
    try:
        return await asyncio.wait_for(tool.handler(args, ctx), timeout=tool.timeout_s)
    except TimeoutError:
        return json.dumps({"error": f"timeout after {tool.timeout_s}s"})
    except Exception as exc:  # noqa: BLE001 — surface anything as tool error
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


async def _run_tool_call(
    tc: dict[str, Any],
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    multimodal: bool = False,
) -> Message:
    """Resolve + execute one tool call; return ``role=tool`` message."""
    call_id = tc.get("id") or ""
    fn = tc.get("function") or {}
    name = fn.get("name") or ""
    raw_args = fn.get("arguments") or "{}"

    try:
        decoded = json.loads(raw_args) if raw_args.strip() else {}
    except (ValueError, json.JSONDecodeError) as exc:
        content = json.dumps({"error": f"invalid arguments JSON: {exc}"})
        return Message(role="tool", content=content, tool_call_id=call_id)
    if not isinstance(decoded, dict):
        content = json.dumps({"error": "invalid arguments JSON: not a JSON object"})
        return Message(role="tool", content=content, tool_call_id=call_id)
    args = decoded

    try:
        tool = registry.get(name)
    except KeyError:
        content = json.dumps({"error": f"unknown tool: {name}"})
        return Message(role="tool", content=content, tool_call_id=call_id)

    result = await _execute_tool(tool, args, ctx)
    if isinstance(result, ImageResult):
        serialised = serialize_image_result(result, multimodal=multimodal)
        return Message(role="tool", content=serialised, tool_call_id=call_id)
    return Message(role="tool", content=result, tool_call_id=call_id)


async def agentic_loop(
    *,
    llm: LLMClient,
    messages: list[Message],
    registry: ToolRegistry,
    ctx: ToolContext,
    on_delta: Callable[[LLMDelta], Awaitable[None]] | None = None,
    on_before_tools: Callable[[Message], Awaitable[None]] | None = None,
    on_iteration_end: Callable[[Message, list[Message]], Awaitable[None]] | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> AgenticResult:
    """Run streaming + tool execution until model stops calling tools.

    Args:
        llm: client whose ``stream_completion`` drives each iteration.
        messages: starting transcript; mutated in place with assistant
            + tool turns.
        registry: tools the model may call. Empty → ``tools=None`` on
            every call (no agentic loop; first iteration terminal).
        ctx: per-turn context handed to handlers.
        on_delta: awaited per ``LLMDelta`` so callers stream content to
            TTS/Discord/etc.
        on_before_tools: awaited *after* assistant message built but
            *before* tool handlers run. Voice mode uses this to inject
            a filler phrase when content empty and tools about to
            consume time.
        on_iteration_end: awaited once per iteration with the assistant
            message produced and tool-result messages just appended.
            Used by responders to persist intermediate turns.
        max_iterations: hard cap; protects against runaway tool loops.

    Returns:
        :class:`AgenticResult` with final assistant text + counters.

    """
    tools_payload = registry.as_openai_tools() if any(registry.tools()) else None
    last_content = ""
    iterations = 0
    tool_calls_made = 0

    while iterations < max_iterations:
        iterations += 1
        content_buf: list[str] = []
        pending_tool_calls: dict[int, dict[str, Any]] = {}

        async for delta in llm.stream_completion(messages, tools=tools_payload):
            if delta.content:
                content_buf.append(delta.content)
            if delta.tool_calls:
                _accumulate_tool_calls(pending_tool_calls, delta.tool_calls)
            if on_delta is not None:
                await on_delta(delta)

        content = "".join(content_buf)
        tool_calls = _finalize_tool_calls(pending_tool_calls)
        assistant_msg = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls or None,
        )
        messages.append(assistant_msg)

        if tool_calls and on_before_tools is not None:
            await on_before_tools(assistant_msg)

        tool_msgs: list[Message] = []
        multimodal = getattr(llm, "multimodal", False)
        for tc in tool_calls:
            tool_calls_made += 1
            tool_msg = await _run_tool_call(tc, registry, ctx, multimodal=multimodal)
            messages.append(tool_msg)
            tool_msgs.append(tool_msg)

        # Detect silent tool BEFORE on_iteration_end so the call +
        # its reasoning aren't persisted to history (which would
        # re-seed the model's rationale for silence next turn)
        silent_sentinel = _get_silent_result()
        if any(
            isinstance(m.content, str) and m.content == silent_sentinel
            for m in tool_msgs
        ):
            return AgenticResult(
                final_content="",
                iterations=iterations,
                tool_calls_made=tool_calls_made,
                transcript=messages,
                is_silent=True,
            )

        if on_iteration_end is not None:
            await on_iteration_end(assistant_msg, tool_msgs)

        last_content = content
        if not tool_calls:
            break
        if iterations >= max_iterations:
            _logger.warning(
                f"{ls.tag('Tools', ls.LY)} "
                f"{ls.kv('hit_max_iterations', str(max_iterations), vc=ls.LY)}"
            )
            break

    # Guard: model occasionally emits a tool-call as text rather than
    # invoking it. never ship/store that — it leaks and seeds a mimicry
    # cascade. a leaked ``silent`` call is honoured as silence.
    cleaned, silent_leak = _strip_leaked_tool_calls(last_content)
    if cleaned != last_content:
        _logger.warning(
            f"{ls.tag('Tools', ls.LY)} "
            f"{ls.kv('leaked_tool_call_stripped', 'true', vc=ls.LY)} "
            f"{ls.kv('silent', str(silent_leak), vc=ls.LY)}"
        )
    return AgenticResult(
        final_content=cleaned,
        iterations=iterations,
        tool_calls_made=tool_calls_made,
        transcript=messages,
        is_silent=silent_leak and not cleaned,
    )
