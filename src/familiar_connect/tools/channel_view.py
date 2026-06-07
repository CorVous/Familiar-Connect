"""Shared channel-history serialization for focus tools.

Used by ``read_channel`` and ``shift_focus`` to render recent turns as
JSON-friendly dicts. Keeps both tools on one format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryTurn


def serialize_turns(turns: list[HistoryTurn]) -> list[dict[str, Any]]:
    """Render turns as ``{id, role, author, content, timestamp}`` dicts."""
    result: list[dict[str, Any]] = []
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
    return result
