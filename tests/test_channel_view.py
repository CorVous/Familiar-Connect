"""Tests for shared channel-history serialization (``serialize_turns``).

Previews surface *conversation* only. ``role="tool"`` turns carry the
familiar's own bookkeeping (serialized channel dumps as ``content``) and
empty assistant turns are tool-call scaffolding husks — neither is a
message a person sent. ``serialize_turns`` excludes both, so a preview
never re-embeds a prior preview's tool payload: the
preview -> persist -> preview recursion is closed by construction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from familiar_connect.history.store import HistoryTurn
from familiar_connect.identity import Author
from familiar_connect.tools.channel_view import serialize_turns


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


def test_tool_and_empty_turns_excluded_only_real_messages() -> None:
    payload = json.dumps([{"id": n, "content": f"msg {n}"} for n in range(50)])
    window = [
        _turn(id=1, role="user", content="hey"),
        _turn(id=2, role="assistant", content=""),  # tool-call husk
        _turn(id=3, role="tool", content=payload),  # tool result bookkeeping
        _turn(id=4, role="assistant", content="hello!"),
        _turn(id=5, role="user", content="   "),  # whitespace-only husk
    ]

    out = serialize_turns(window)

    assert all(e["role"] != "tool" for e in out)
    assert all(e["content"].strip() for e in out)
    # Only the real messages survive, in order, byte-identical content.
    assert [(e["role"], e["content"]) for e in out] == [
        ("user", "hey"),
        ("assistant", "hello!"),
    ]


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
    # ``role="tool"`` turn. Re-ingesting that dump must not re-embed it —
    # the tool turn is absent from the output entirely.
    prior = [_turn(id=n, role="user", content=f"msg {n}") for n in range(50)]
    dump = json.dumps(serialize_turns(prior))
    tool_turn = _turn(id=99, role="tool", content=dump)

    assert serialize_turns([tool_turn]) == []


def test_serialize_output_is_idempotent() -> None:
    turns = [
        _turn(id=1, role="user", content="hi"),
        _turn(id=2, role="tool", content="big dump"),
        _turn(id=3, role="assistant", content="ok"),
    ]

    assert serialize_turns(turns) == serialize_turns(turns)
