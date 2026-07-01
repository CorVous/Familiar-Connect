"""Shared channel-history serialization for focus tools.

Used by ``read_channel`` and ``shift_focus`` to render recent turns as
JSON-friendly dicts. Keeps both tools on one format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryTurn

# Tool-result turns persist a serialized JSON dump of channel messages as
# their ``content``. Re-embedding that dump in a later preview's recent-turns
# window compounds across previews until one turn overflows the LLM context,
# so we render a fixed placeholder instead of the payload.
_TOOL_RESULT_PLACEHOLDER = "[tool result omitted]"


def serialize_turns(turns: list[HistoryTurn]) -> list[dict[str, Any]]:
    """Render turns as ``{id, role, author, content, timestamp}`` dicts.

    ``role="tool"`` turns carry tool-result payloads that must not be
    re-embedded (see :data:`_TOOL_RESULT_PLACEHOLDER`); their content is
    replaced with a placeholder. All other roles pass through verbatim.
    """
    result: list[dict[str, Any]] = []
    for t in turns:
        author_name: str | None = None
        if t.author is not None:
            author_name = t.author.display_name or t.author.username
        content = _TOOL_RESULT_PLACEHOLDER if t.role == "tool" else t.content
        result.append({
            "id": t.id,
            "role": t.role,
            "author": author_name,
            "content": content,
            "timestamp": t.timestamp.isoformat(),
        })
    return result
