"""Tool calling subsystem.

In-process tool registry + agentic loop. Lets the familiar invoke
functions during a turn and resume with results.
"""

from __future__ import annotations

from familiar_connect.tools.registry import Tool, ToolContext, ToolRegistry

__all__ = ["Tool", "ToolContext", "ToolRegistry"]
