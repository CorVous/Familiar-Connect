"""``read_channel`` tool.

Read-only peek into focused text channel history. Returns recent turns
without updating consumed_at. Voice not supported.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from familiar_connect import log_style as ls
from familiar_connect.tools.channel_view import serialize_turns
from familiar_connect.tools.registry import Tool

if TYPE_CHECKING:
    from familiar_connect.tools.registry import ToolContext

_logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50


async def _read_channel_handler(args: dict[str, Any], ctx: ToolContext) -> str:
    fm = ctx.focus_manager
    if fm is None:
        return json.dumps({"error": "focus_manager not wired into context"})

    store = ctx.store
    if store is None:
        return json.dumps({"error": "store not wired into context"})

    channel_id = fm.get_focus("text")
    if channel_id is None:
        return json.dumps({"error": "no text focus active"})

    raw_limit = args.get("limit", _DEFAULT_LIMIT)
    limit = min(raw_limit, _MAX_LIMIT) if isinstance(raw_limit, int) else _DEFAULT_LIMIT

    before_id = args.get("before_id")
    around_id = args.get("around_id")
    if before_id is not None and around_id is not None:
        return json.dumps({"error": "before_id and around_id are mutually exclusive"})

    # deliberately unfiltered by archive watermark: fresh eyes may
    # scroll archived past (filter applies only to the prompt window)
    if around_id is not None:
        half = max(1, limit // 2)
        turns = await store.turns_around(
            familiar_id=ctx.familiar_id,
            channel_id=channel_id,
            turn_id=around_id,
            before=half,
            after=half,
        )
    else:
        turns = await store.recent(
            familiar_id=ctx.familiar_id,
            channel_id=channel_id,
            limit=limit,
            before_id=before_id,
        )

    result = serialize_turns(turns)
    _logger.info(
        f"{ls.tag('📖 read_channel', ls.LM)} "
        f"{ls.kv('channel', fm.channel_label(channel_id), vc=ls.LW)} "
        f"{ls.kv('turns', str(len(result)), vc=ls.LW)}"
    )
    return json.dumps(result)


def build_read_channel_tool() -> Tool:
    """Build the ``read_channel`` tool."""
    return Tool(
        name="read_channel",
        description=(
            "Read recent turns from the currently focused text channel. "
            "Page back with before_id, or jump to a turn with around_id. "
            "Read-only; does not consume or acknowledge messages. "
            "Voice focus not supported."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_LIMIT,
                    "default": _DEFAULT_LIMIT,
                    "description": "Number of turns to return (max 50).",
                },
                "before_id": {
                    "type": "integer",
                    "description": "Return turns with id < before_id (for paging).",
                },
                "around_id": {
                    "type": "integer",
                    "description": (
                        "Jump to this turn id; returns surrounding turns. "
                        "Cannot combine with before_id."
                    ),
                },
            },
            "required": [],
        },
        handler=_read_channel_handler,
    )
