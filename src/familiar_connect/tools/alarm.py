"""``set_alarm`` / ``cancel_alarm`` tools.

The tools schedule a future wake by inserting an alarm row and
registering an in-process ``asyncio`` sleep task. When the timer
fires, the scheduler publishes :data:`TOPIC_ALARM_FIRED`; the
:class:`AlarmWaker` processor turns that into a synthetic system turn
in the originating channel.

Both tools route the wake to the channel the user spoke in
(``ctx.channel_id`` / ``ctx.channel_kind``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from familiar_connect.tools.registry import Tool

if TYPE_CHECKING:
    from familiar_connect.tools.registry import ToolContext
    from familiar_connect.tools.scheduler import AlarmScheduler


_MAX_REASON_LEN = 200
_MIN_DELAY_S = 1
_MAX_DELAY_S = 60 * 60 * 24 * 365  # one year cap — defensive
# allow tiny past-skew so an immediate ``now`` doesn't bounce.
_PAST_SKEW_S = 5


def _resolve_when(args: dict[str, Any]) -> datetime | str:
    """Return target ``datetime`` or an error string."""
    when_str = args.get("when")
    delay = args.get("delay_seconds")

    if isinstance(when_str, str) and when_str:
        try:
            target = datetime.fromisoformat(when_str)
        except ValueError as exc:
            return f"invalid 'when' (must be ISO-8601): {exc}"
        if target.tzinfo is None:
            return "invalid 'when' (must include timezone, e.g. '...+00:00')"
        skew = (datetime.now(tz=UTC) - target).total_seconds()
        if skew > _PAST_SKEW_S:
            return f"'when' is {int(skew)}s in the past"
        return target

    if isinstance(delay, int) and not isinstance(delay, bool):
        if delay < _MIN_DELAY_S:
            return f"'delay_seconds' must be ≥ {_MIN_DELAY_S}"
        if delay > _MAX_DELAY_S:
            return f"'delay_seconds' must be ≤ {_MAX_DELAY_S}"
        return datetime.now(tz=UTC) + _delta(delay)

    return "missing 'when' (ISO-8601 timestamp) or 'delay_seconds' (int)"


def _delta(seconds: int):  # noqa: ANN202 — local helper
    from datetime import timedelta  # noqa: PLC0415 — defer import until called

    return timedelta(seconds=seconds)


async def _set_alarm_handler(
    args: dict[str, Any],
    ctx: ToolContext,
) -> str:
    scheduler = ctx.scheduler
    if scheduler is None:
        return json.dumps({"error": "alarm scheduler not wired into context"})

    reason = args.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return json.dumps({"error": "missing or empty 'reason' (string)"})
    if len(reason) > _MAX_REASON_LEN:
        return json.dumps({"error": f"'reason' too long (>{_MAX_REASON_LEN} chars)"})

    resolved = _resolve_when(args)
    if isinstance(resolved, str):
        return json.dumps({"error": resolved})

    alarm_id = await scheduler.add(
        channel_id=ctx.channel_id,
        channel_kind=ctx.channel_kind,
        scheduled_at=resolved,
        reason=reason,
        originating_turn_id=ctx.turn_id,
    )
    return json.dumps({
        "alarm_id": alarm_id,
        "scheduled_at": resolved.isoformat(),
        "ack": "ok",
    })


async def _cancel_alarm_handler(
    args: dict[str, Any],
    ctx: ToolContext,
) -> str:
    scheduler = ctx.scheduler
    if scheduler is None:
        return json.dumps({"error": "alarm scheduler not wired into context"})

    alarm_id = args.get("alarm_id")
    if not isinstance(alarm_id, str) or not alarm_id:
        return json.dumps({"error": "missing or empty 'alarm_id'"})

    ok = await scheduler.cancel(alarm_id=alarm_id)
    if not ok:
        return json.dumps({"error": f"no pending alarm with id {alarm_id}"})
    return json.dumps({"alarm_id": alarm_id, "ack": "ok"})


def build_alarm_tool(scheduler: AlarmScheduler) -> Tool:  # noqa: ARG001 — accepted for symmetry
    """Build the ``set_alarm`` tool.

    ``scheduler`` is reserved for future binding; the live scheduler
    is reached via ``ctx.scheduler`` to keep the registry decoupled
    from concrete instances.
    """
    return Tool(
        name="set_alarm",
        description=(
            "Schedule a future wake. The familiar will be re-prompted "
            "in the channel where this tool was called when the time "
            "arrives. Provide one of 'when' (ISO-8601 UTC) or "
            "'delay_seconds' (positive integer)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "when": {
                    "type": "string",
                    "description": (
                        "Absolute target time as ISO-8601 with timezone "
                        "(e.g. '2030-01-01T12:00:00+00:00')."
                    ),
                },
                "delay_seconds": {
                    "type": "integer",
                    "minimum": _MIN_DELAY_S,
                    "maximum": _MAX_DELAY_S,
                    "description": "Wait this many seconds before waking.",
                },
                "reason": {
                    "type": "string",
                    "maxLength": _MAX_REASON_LEN,
                    "description": ("Short note shown back to the familiar on wake."),
                },
            },
            "required": ["reason"],
        },
        handler=_set_alarm_handler,
    )


def build_cancel_alarm_tool(scheduler: AlarmScheduler) -> Tool:  # noqa: ARG001
    """Build the ``cancel_alarm`` tool."""
    return Tool(
        name="cancel_alarm",
        description=(
            "Cancel a previously scheduled alarm. The id is the "
            "'alarm_id' returned from set_alarm."
        ),
        parameters={
            "type": "object",
            "properties": {
                "alarm_id": {
                    "type": "string",
                    "description": "Id of the alarm to cancel.",
                },
            },
            "required": ["alarm_id"],
        },
        handler=_cancel_alarm_handler,
    )
