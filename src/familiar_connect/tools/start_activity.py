"""``start_activity`` tool.

Stages a global absence via :meth:`ActivityEngine.defer_start`; actual
departure applied by ``engine.end_turn()`` after the reply ships
(shift_focus deferral precedent). Activity enum built from the
engine's catalog at registry-build time, so each familiar's sidecar
``activities.toml`` shapes the schema.

Description carries the ENTIRE when-to-go policy — design decision:
zero character-card growth. Budget-sensitive; keep tight.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from familiar_connect import log_style as ls
from familiar_connect.tools.registry import Tool
from familiar_connect.tools.silent import SILENT_RESULT

if TYPE_CHECKING:
    from familiar_connect.activities.config import ActivityType
    from familiar_connect.tools.registry import ToolContext

_logger = logging.getLogger(__name__)


class StartActivityEngine(Protocol):
    """Structural slice of ActivityEngine the tool needs."""

    @property
    def catalog(self) -> tuple[ActivityType, ...]: ...

    @property
    def active(self) -> object | None: ...

    def defer_start(self, type_id: str, note: str | None = None) -> dict[str, Any]: ...


_DESCRIPTION = (
    "Head out and do something away from the screen for a while. Use "
    "when the current scene has wrapped up or the channel has gone "
    "quiet. You'll be away and may "
    "miss messages while out. You leave when this reply sends: with "
    "people around, say your in-character goodbye in this same message; "
    "from a quiet channel, call silent() too and slip away unannounced. "
    "Don't start one in the middle of a conversation you have a stake in."
)


def build_start_activity_tool(engine: StartActivityEngine) -> Tool:
    """Build the ``start_activity`` tool bound to *engine*."""
    catalog = engine.catalog

    async def _start_activity_handler(  # noqa: RUF029 — handler signature is async
        args: dict[str, Any],
        ctx: ToolContext,  # noqa: ARG001 — engine bound at build time
    ) -> str:
        if engine.active is not None:
            # belt for the stay-out misroute (eval finding): calling
            # start_activity while already out signals stay-out intent —
            # silent sentinel keeps the meta narration off the channel
            _logger.info(
                f"{ls.tag('🚶 start_activity', ls.G)} "
                f"{ls.kv('outcome', 'already-out → silent', vc=ls.LW)}"
            )
            return SILENT_RESULT
        activity = args.get("activity")
        if not isinstance(activity, str) or not activity:
            return json.dumps({"error": "missing or empty 'activity' (string)"})
        note = args.get("note")
        if note is not None and not isinstance(note, str):
            return json.dumps({"error": "'note' must be a string"})

        result = engine.defer_start(activity, note=note)
        outcome = "error" if "error" in result else "staged"
        _logger.info(
            f"{ls.tag('🚶 start_activity', ls.G)} "
            f"{ls.kv('activity', activity, vc=ls.G)} "
            f"{ls.kv('outcome', outcome, vc=ls.LW)}"
        )
        return json.dumps(result)

    return Tool(
        name="start_activity",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "activity": {
                    "type": "string",
                    "enum": [t.id for t in catalog],
                    "description": (
                        "What to go do: "
                        + "; ".join(f"'{t.id}' = {t.label}" for t in catalog)
                        + "."
                    ),
                },
                "note": {
                    "type": "string",
                    "description": (
                        "Optional intent — what you have in mind for it; "
                        "seeds the experience."
                    ),
                },
            },
            "required": ["activity"],
        },
        handler=_start_activity_handler,
    )
