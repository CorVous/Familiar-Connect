"""In-process tool registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from familiar_connect.bus.protocols import EventBus
    from familiar_connect.focus import FocusManager
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.llm import LLMClient


ToolHandler = "Callable[[dict[str, Any], ToolContext], Awaitable[str | ImageResult]]"


@dataclass
class ImageResult:
    """Tool result carrying a JPEG (base64) + text description.

    Always carries both. Loop serialises per slot's ``multimodal`` flag.
    """

    description: str
    jpeg_base64: str
    media_type: str = "image/jpeg"


@dataclass
class ToolContext:
    """Per-call context handed to tool handlers.

    Handlers reach the rest of the system through this — no globals.
    ``scheduler`` optional so registries without the alarm tool can
    omit it.
    """

    familiar_id: str
    channel_id: int
    channel_kind: str  # "text" | "voice"
    turn_id: str
    history: AsyncHistoryStore
    bus: EventBus
    scheduler: Any | None = None  # AlarmScheduler — avoids import cycle
    images: dict[str, str] = field(default_factory=dict)  # img_id → URL
    description_llm: LLMClient | None = None  # vision model client
    focus_manager: FocusManager | None = None  # attentional focus controller
    store: AsyncHistoryStore | None = None  # explicit store ref for read_channel


@dataclass
class Tool:
    """One callable tool exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[str | ImageResult]]
    timeout_s: float = 10.0


class ToolRegistry:
    """Name-indexed bag of :class:`Tool`."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            msg = f"tool already registered: {tool.name}"
            raise ValueError(msg)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            msg = f"unknown tool: {name}"
            raise KeyError(msg)
        return self._tools[name]

    def tools(self) -> Iterable[Tool]:
        return self._tools.values()

    def as_openai_tools(self) -> list[dict[str, Any]]:
        """Serialize to OpenAI ``tools`` array shape.

        Each entry: ``{"type": "function", "function": {name,
        description, parameters}}``. Empty registry returns ``[]``.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]
