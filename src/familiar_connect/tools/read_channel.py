"""``read_channel`` tool.

Read-only peek into focused text channel history. Returns recent turns
without updating consumed_at. Voice not supported.
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

    turns = await store.recent(
        familiar_id=ctx.familiar_id,
        channel_id=channel_id,
        limit=limit,
    )

    result = []
    for t in turns:
        author_name: str | None = None
        if t.author is not None:
            author_name = t.author.display_name or t.author.username
        result.append({
            "id": t.id,
            "role": t.role,
            "author": author_name,
            "content": t.content,
            "timestamp": t.timestamp.isoformat(),
        })
    _logger.info(
        f"{ls.tag('📖 read_channel', ls.LM)} "
        f"{ls.kv('channel', fm._ch(channel_id), vc=ls.LW)} "
        f"{ls.kv('turns', str(len(result)), vc=ls.LW)}"
    )
    return json.dumps(result)


def build_read_channel_tool() -> Tool:
    """Build the ``read_channel`` tool."""
    return Tool(
        name="read_channel",
        description=(
            "Read recent turns from the currently focused text channel. "
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
            },
            "required": [],
        },
        handler=_read_channel_handler,
    )
