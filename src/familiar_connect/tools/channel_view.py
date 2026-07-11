"""Shared channel-history serialization for focus tools.

Used by ``read_channel`` and ``shift_focus`` to render recent turns as
JSON-friendly dicts. Keeps both tools on one format.

Previews surface *conversation* only. ``role="tool"`` turns persist a
serialized JSON dump of channel messages as their ``content`` (the
familiar's own bookkeeping), and empty assistant turns are tool-call
scaffolding husks — neither is a message a person sent. Both are
excluded, which also closes a recursion: re-embedding a prior preview's
tool dump in a later preview's recent-turns window would compound across
previews until one turn overflows the LLM context. Dropping tool turns
entirely closes that vector by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryTurn


def serialize_turns(turns: list[HistoryTurn]) -> list[dict[str, Any]]:
    """Render conversation turns as ``{id, role, author, content, timestamp}``.

    Excludes ``role="tool"`` turns (bookkeeping payloads) and empty-content
    turns (tool-call scaffolding husks) so previews carry only real
    user/assistant messages. Surviving turns pass through verbatim.
    """
    result: list[dict[str, Any]] = []
    for t in turns:
        if t.role == "tool" or not t.content.strip():
            continue
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
