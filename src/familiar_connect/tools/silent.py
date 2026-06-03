"""``silent`` tool.

Suppresses familiar reply for this turn. Agentic loop detects the
sentinel value and returns ``AgenticResult(is_silent=True)`` without
re-prompting the model.

Reasoning logged by caller; the tool itself just returns the sentinel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from familiar_connect import log_style as ls
from familiar_connect.tools.registry import Tool

if TYPE_CHECKING:
    from familiar_connect.tools.registry import ToolContext

_logger = logging.getLogger(__name__)

SILENT_RESULT = "__SILENT__"


async def _silent_handler(args: dict[str, Any], ctx: ToolContext) -> str:  # noqa: RUF029
    reasoning = args.get("reasoning", "")
    _logger.info(
        f"{ls.tag('💤 silent', ls.B)} "
        f"{ls.kv('reason', reasoning, vc=ls.LB)}"
    )
    return SILENT_RESULT


def build_silent_tool() -> Tool:
    """Build the ``silent`` tool."""
    return Tool(
        name="silent",
        description=(
            "Stay silent this turn — do not send a reply. "
            "Use when observation or context-gathering is appropriate "
            "but a response would be intrusive or premature. "
            "Provide a brief reasoning."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why the familiar is staying silent",
                },
            },
            "required": ["reasoning"],
        },
        handler=_silent_handler,
    )
