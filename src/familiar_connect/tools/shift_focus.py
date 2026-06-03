"""``shift_focus`` tool.

Deferred focus shift: registers intent with FocusManager at handler
time; applied atomically at end_turn. Modality inferred from
SubscriptionRegistry inside FocusManager.defer_shift.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from familiar_connect import log_style as ls
from familiar_connect.tools.registry import Tool

if TYPE_CHECKING:
    from familiar_connect.tools.registry import ToolContext

_logger = logging.getLogger(__name__)


async def _shift_focus_handler(args: dict[str, Any], ctx: ToolContext) -> str:  # noqa: RUF029
    fm = ctx.focus_manager
    if fm is None:
        return json.dumps({"error": "focus_manager not wired into context"})

    channel_id = args.get("channel_id")
    if not isinstance(channel_id, int):
        return json.dumps({"error": "missing or invalid 'channel_id' (integer)"})

    fm.defer_shift(channel_id)
    _logger.info(
        f"{ls.tag('🔀 shift_focus', ls.LC)} "
        f"{ls.kv('channel', fm._ch(channel_id), vc=ls.LW)}"
    )
    return json.dumps({"ok": True, "channel_id": channel_id})


def build_shift_focus_tool() -> Tool:
    """Build the ``shift_focus`` tool."""
    return Tool(
        name="shift_focus",
        description=(
            "Shift attentional focus to a different channel. "
            "The change is deferred and applied at end of turn. "
            "Modality (text/voice) inferred from channel subscription."
        ),
        parameters={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "integer",
                    "description": "Discord channel id to focus on",
                },
            },
            "required": ["channel_id"],
        },
        handler=_shift_focus_handler,
    )
