"""``shift_focus`` tool.

Immediate focus shift: applies at handler time via
``FocusManager.shift_now`` (modality inferred from SubscriptionRegistry).
No deferral — focus moves the moment the tool is called, so a silent
turn still leaves her where she went and nothing leaks.

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

    # guard: only subscribed channels are valid targets — focusing a
    # dead channel would strand attention. list valid targets so the
    # model can recover in-turn rather than narrating a channel it
    # can't reach.
    if not fm.is_subscribed(channel_id):
        available = [
            {"channel_id": cid, "label": fm.channel_label(cid)}
            for cid in fm.subscribed_channels()
        ]
        _logger.info(
            f"{ls.tag('🔀 shift_focus', ls.LC)} rejected "
            f"{ls.kv('channel', str(channel_id), vc=ls.LW)} (not subscribed)"
        )
        return json.dumps({
            "error": (
                f"channel {channel_id} is not subscribed — cannot focus "
                "there. Pick one of available_channels."
            ),
            "available_channels": available,
        })

    await fm.shift_now(channel_id)

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
            "Move your attention to a different channel — for real, right "
            "now. You stop following your current channel until you shift "
            "back, and any reply this turn posts to the new channel. Returns "
            "the target's recent messages (empty list = nothing there yet). "
            "Use it to actually go somewhere, not to glance. Modality "
            "(text/voice) inferred from channel subscription."
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
