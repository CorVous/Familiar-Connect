"""``shift_focus`` tool.

Deferred focus shift: registers intent with FocusManager at handler
time; applied atomically at end_turn. Modality inferred from
SubscriptionRegistry inside FocusManager.defer_shift.

Content-bearing: handler eagerly fetches target channel's recent turns
and returns them so the model sees the channel in-turn (agentic loop
feeds tool result back) rather than narrating a channel it can't see.
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

_PREVIEW_LIMIT = 20


async def _shift_focus_handler(args: dict[str, Any], ctx: ToolContext) -> str:
    fm = ctx.focus_manager
    if fm is None:
        return json.dumps({"error": "focus_manager not wired into context"})

    channel_id = args.get("channel_id")
    if not isinstance(channel_id, int):
        return json.dumps({"error": "missing or invalid 'channel_id' (integer)"})

    fm.defer_shift(channel_id)

    payload: dict[str, Any] = {"ok": True, "channel_id": channel_id}
    # Eager content fetch — voice/empty channels yield [] (model learns
    # it's empty). store may be absent in fm-only contexts → bare ack
    if ctx.store is not None:
        turns = await ctx.store.recent(
            familiar_id=ctx.familiar_id,
            channel_id=channel_id,
            limit=_PREVIEW_LIMIT,
        )
        payload["messages"] = serialize_turns(turns)

    _logger.info(
        f"{ls.tag('🔀 shift_focus', ls.LC)} "
        f"{ls.kv('channel', fm.channel_label(channel_id), vc=ls.LW)} "
        f"{ls.kv('preview', str(len(payload.get('messages', []))), vc=ls.LW)}"
    )
    return json.dumps(payload)


def build_shift_focus_tool() -> Tool:
    """Build the ``shift_focus`` tool."""
    return Tool(
        name="shift_focus",
        description=(
            "Shift attentional focus to a different channel. Returns that "
            "channel's recent messages so you can see it before responding "
            "(empty list = nothing there yet). Your reply will post to the "
            "new channel. Modality (text/voice) inferred from channel "
            "subscription."
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
