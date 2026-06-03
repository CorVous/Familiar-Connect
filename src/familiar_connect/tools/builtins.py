"""Registry-builder helpers; extracted so tests can call without run.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.tools.alarm import build_alarm_tool, build_cancel_alarm_tool
from familiar_connect.tools.image import build_view_image_tool
from familiar_connect.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from familiar_connect.tools.scheduler import AlarmScheduler


def build_voice_registry(scheduler: AlarmScheduler) -> ToolRegistry:
    """Voice-tier registry: alarm + cancel only. view_image excluded."""
    registry = ToolRegistry()
    registry.register(build_alarm_tool(scheduler))
    registry.register(build_cancel_alarm_tool(scheduler))
    return registry


def build_text_registry(
    scheduler: AlarmScheduler,
    *,
    image_tools: bool = False,
) -> ToolRegistry:
    """Text-tier registry: alarm + cancel; view_image when ``image_tools=True``."""
    registry = ToolRegistry()
    registry.register(build_alarm_tool(scheduler))
    registry.register(build_cancel_alarm_tool(scheduler))
    if image_tools:
        registry.register(build_view_image_tool())
    return registry
