"""Registry-builder helpers; extracted so tests can call without run.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.tools.alarm import build_alarm_tool, build_cancel_alarm_tool
from familiar_connect.tools.image import build_view_image_tool
from familiar_connect.tools.read_channel import build_read_channel_tool
from familiar_connect.tools.registry import ToolRegistry
from familiar_connect.tools.shift_focus import build_shift_focus_tool
from familiar_connect.tools.silent import build_silent_tool

if TYPE_CHECKING:
    from familiar_connect.focus import FocusManager
    from familiar_connect.tools.scheduler import AlarmScheduler


def build_voice_registry(
    scheduler: AlarmScheduler,
    *,
    focus_manager: FocusManager | None = None,
) -> ToolRegistry:
    """Voice-tier registry: alarm + cancel + silent; shift_focus when fm provided."""
    registry = ToolRegistry()
    registry.register(build_alarm_tool(scheduler))
    registry.register(build_cancel_alarm_tool(scheduler))
    registry.register(build_silent_tool())
    if focus_manager is not None:
        registry.register(build_shift_focus_tool())
    return registry


def build_text_registry(
    scheduler: AlarmScheduler,
    *,
    image_tools: bool = False,
    focus_manager: FocusManager | None = None,
) -> ToolRegistry:
    """Text-tier registry: alarm + cancel + silent; optional image/focus tools."""
    registry = ToolRegistry()
    registry.register(build_alarm_tool(scheduler))
    registry.register(build_cancel_alarm_tool(scheduler))
    registry.register(build_silent_tool())
    if image_tools:
        registry.register(build_view_image_tool())
    if focus_manager is not None:
        registry.register(build_shift_focus_tool())
        registry.register(build_read_channel_tool())
    return registry
