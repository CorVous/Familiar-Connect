"""Tests for shared channel-history serialization (``serialize_turns``).

Guards the root-cause fix for context overflow: ``role="tool"`` turns
carry serialized JSON dumps of channel messages as ``content``. When a
later preview's recent-turns window re-embeds such a turn verbatim, the
dump compounds across previews until one turn overflows the LLM context.
``serialize_turns`` must replace any tool-result payload with a fixed
placeholder so the recursion vector stays closed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from familiar_connect.history.store import HistoryTurn
from familiar_connect.identity import Author
from familiar_connect.tools.channel_view import serialize_turns

TOOL_PLACEHOLDER = "[tool result omitted]"


def _turn(
    *,
    id: int,  # noqa: A002
    role: str,
    content: str,
    author: Author | None = None,
) -> HistoryTurn:
    """Build a minimal HistoryTurn for serialization tests."""
    return HistoryTurn(
        id=id,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=id),
        role=role,
        author=author,
        content=content,
    )


def test_tool_turn_content_replaced_with_placeholder() -> None:
    payload = json.dumps([{"id": n, "content": f"msg {n}"} for n in range(50)])
    turn = _turn(id=1, role="tool", content=payload)

    [entry] = serialize_turns([turn])

    assert entry["content"] == TOOL_PLACEHOLDER
    assert payload not in json.dumps(serialize_turns([turn]))


def test_user_and_assistant_content_pass_through_verbatim() -> None:
    user = _turn(id=1, role="user", content="hello there")
    assistant = _turn(id=2, role="assistant", content="general kenobi")

    user_entry, assistant_entry = serialize_turns([user, assistant])

    assert user_entry["content"] == "hello there"
    assert assistant_entry["content"] == "general kenobi"


def test_author_resolution_prefers_display_name() -> None:
    author = Author(
        platform="discord",
        user_id="42",
        username="cor_handle",
        display_name="Cor",
    )
    [entry] = serialize_turns([_turn(id=1, role="user", content="hi", author=author)])

    assert entry["author"] == "Cor"


def test_author_resolution_falls_back_to_username() -> None:
    author = Author(
        platform="discord",
        user_id="42",
        username="cor_handle",
        display_name="",
    )
    [entry] = serialize_turns([_turn(id=1, role="user", content="hi", author=author)])

    assert entry["author"] == "cor_handle"


def test_authorless_turn_resolves_to_none() -> None:
    [entry] = serialize_turns([_turn(id=1, role="user", content="hi")])

    assert entry["author"] is None


def test_huge_tool_payload_does_not_inflate_output() -> None:
    huge = "x" * 500_000
    turn = _turn(id=1, role="tool", content=huge)

    serialized = json.dumps(serialize_turns([turn]))

    assert huge not in serialized
    assert len(serialized) < 1_000


def test_dict_shape_is_stable() -> None:
    [entry] = serialize_turns([_turn(id=7, role="user", content="hi")])

    assert list(entry.keys()) == ["id", "role", "author", "content", "timestamp"]


def test_recursion_closed_when_serialized_dump_re_ingested() -> None:
    # ``read_channel`` persists ``json.dumps(serialize_turns(...))`` as a
    # ``role="tool"`` turn. Re-ingesting that dump must not re-embed it.
    prior = [_turn(id=n, role="user", content=f"msg {n}") for n in range(50)]
    dump = json.dumps(serialize_turns(prior))
    tool_turn = _turn(id=99, role="tool", content=dump)

    [entry] = serialize_turns([tool_turn])

    assert entry["content"] == TOOL_PLACEHOLDER
    assert len(json.dumps([entry])) < 1_000


def test_serialize_output_is_idempotent() -> None:
    turns = [
        _turn(id=1, role="user", content="hi"),
        _turn(id=2, role="tool", content="big dump"),
        _turn(id=3, role="assistant", content="ok"),
    ]

    assert serialize_turns(turns) == serialize_turns(turns)
