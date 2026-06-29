"""Tests for the structured-output PROVISIONING interface (issue #167).

Covers the two halves of :mod:`familiar_connect.structured_request`:

  * :func:`render_contract` — a declarative :class:`Schema` becomes the
    reply-shape contract text (skeleton + bullets + trailing notes), so no
    feature module hand-types a ``Reply JSON only: {...}`` string.
  * :func:`request_structured` — the call/parse/re-ask loop: it returns
    the parsed root on a good reply, re-asks with a clear correction on a
    wrong shape up to ``max_retries``, degrades to ``ok=False`` rather
    than raising, and lets transport errors propagate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.llm import LLMClient, Message
from familiar_connect.structured_request import (
    DEFAULT_MAX_RETRIES,
    Field,
    Schema,
    StructuredReply,
    render_contract,
    request_structured,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeLLM(LLMClient):
    """Minimal LLM stub: pops canned replies, records each prompt sent."""

    def __init__(self, replies: Sequence[str]) -> None:
        super().__init__(api_key="k", model="m", slot="test")
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        content = self._replies.pop(0) if self._replies else "{}"
        return Message(role="assistant", content=content)


class _BoomLLM(LLMClient):
    """Stub whose ``chat`` raises — stands in for a transport failure."""

    def __init__(self) -> None:
        super().__init__(api_key="k", model="m", slot="boom")

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
        msg = "network down"
        raise RuntimeError(msg)


_ARRAY_SCHEMA = Schema(
    fields=(
        Field("text", '"<one sentence>"', desc="the reflection"),
        Field("ids", "[<id>...]", desc="turn ids it draws from", required=False),
    ),
    root="array",
    empty_note="Reply [] when nothing stands out.",
)
_CONTAINER_SCHEMA = Schema(
    fields=(
        Field("text", '"<stance>"'),
        Field("turn_ids", "[<id>...]"),
    ),
    root="object",
    container="candidates",
    empty_note="Empty list when nothing stands out.",
)
_FLAT_SCHEMA = Schema(
    fields=(Field("superseded_ids", "[<id>...]"),),
    root="object",
    constraints=("Only use ids from the list below.",),
)


class TestRenderContract:
    """``render_contract`` — declarative schema → prompt contract text."""

    def test_array_root_renders_skeleton_bullets_and_empty_note(self) -> None:
        text = render_contract(_ARRAY_SCHEMA)
        assert '[{"text": "<one sentence>", "ids": [<id>...]}, ...]' in text
        assert "Each item's fields:" in text
        assert "- `text`: the reflection" in text
        # required=False renders the optional marker
        assert "- `ids` (optional): turn ids it draws from" in text
        assert "Reply [] when nothing stands out." in text

    def test_object_container_wraps_item_list(self) -> None:
        text = render_contract(_CONTAINER_SCHEMA)
        assert '{"candidates": [{"text": "<stance>", "turn_ids": [<id>...]}, ...]}' in (
            text
        )
        assert "Empty list when nothing stands out." in text

    def test_flat_object_renders_fields_and_constraints(self) -> None:
        text = render_contract(_FLAT_SCHEMA)
        assert '{"superseded_ids": [<id>...]}' in text
        # no field descs ⇒ no bullet block
        assert "Fields:" not in text
        assert "Only use ids from the list below." in text

    def test_container_on_array_root_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="container is only valid"):
            Schema(fields=(Field("x", "1"),), root="array", container="oops")


def _prompt() -> list[Message]:
    return [
        Message(role="system", content="do the thing"),
        Message(role="user", content="data"),
    ]


class TestRequestStructured:
    """``request_structured`` — call, parse to the root, re-ask, degrade."""

    @pytest.mark.asyncio
    async def test_object_success_first_try(self) -> None:
        llm = _FakeLLM(['{"candidates": [{"text": "x", "turn_ids": [1]}]}'])
        result = await request_structured(
            llm, messages=_prompt(), schema=_CONTAINER_SCHEMA
        )
        assert result.ok is True
        assert result.attempts == 1
        assert result.value == {"candidates": [{"text": "x", "turn_ids": [1]}]}
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_array_success_first_try(self) -> None:
        llm = _FakeLLM(['[{"text": "a"}]'])
        result = await request_structured(llm, messages=_prompt(), schema=_ARRAY_SCHEMA)
        assert result.ok is True
        assert result.value == [{"text": "a"}]

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_with_correction(self) -> None:
        # First reply is garbage, second is valid: one retry recovers it.
        llm = _FakeLLM(["not json at all", '{"superseded_ids": [7]}'])
        result = await request_structured(
            llm, messages=_prompt(), schema=_FLAT_SCHEMA, max_retries=1
        )
        assert result.ok is True
        assert result.attempts == 2
        assert result.value == {"superseded_ids": [7]}
        # The second call carried the bad reply + a corrective user turn
        # that restates the contract.
        second_convo = llm.calls[1]
        assert second_convo[-2].role == "assistant"
        assert second_convo[-2].content == "not json at all"
        assert second_convo[-1].role == "user"
        assert "could not be used" in second_convo[-1].content_str
        assert "superseded_ids" in second_convo[-1].content_str

    @pytest.mark.asyncio
    async def test_degrades_after_exhausting_retries(self) -> None:
        llm = _FakeLLM(["nope", "still nope"])
        result = await request_structured(
            llm, messages=_prompt(), schema=_FLAT_SCHEMA, max_retries=1
        )
        assert result == StructuredReply(value=None, ok=False, attempts=2)
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_wrong_root_type_is_a_shape_failure(self) -> None:
        # An object site that gets a bare array must re-ask, not accept it.
        llm = _FakeLLM(['["a", "b"]', '{"superseded_ids": []}'])
        result = await request_structured(
            llm, messages=_prompt(), schema=_FLAT_SCHEMA, max_retries=1
        )
        assert result.ok is True
        assert result.value == {"superseded_ids": []}
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_zero_retries_takes_one_attempt_only(self) -> None:
        llm = _FakeLLM(["garbage"])
        result = await request_structured(
            llm, messages=_prompt(), schema=_FLAT_SCHEMA, max_retries=0
        )
        assert result.ok is False
        assert result.attempts == 1
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_does_not_mutate_caller_messages(self) -> None:
        llm = _FakeLLM(["bad", '{"superseded_ids": []}'])
        messages = _prompt()
        await request_structured(
            llm, messages=messages, schema=_FLAT_SCHEMA, max_retries=1
        )
        # Caller's list is untouched even though a retry happened.
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_fenced_json_is_accepted(self) -> None:
        # The parser strips ``` fences; a fenced-but-valid reply succeeds
        # on the first try (no wasted retry).
        llm = _FakeLLM(['```json\n{"superseded_ids": [1]}\n```'])
        result = await request_structured(
            llm, messages=_prompt(), schema=_FLAT_SCHEMA, max_retries=1
        )
        assert result.ok is True
        assert result.attempts == 1
        assert result.value == {"superseded_ids": [1]}

    @pytest.mark.asyncio
    async def test_transport_errors_propagate(self) -> None:
        with pytest.raises(RuntimeError, match="network down"):
            await request_structured(
                _BoomLLM(), messages=_prompt(), schema=_FLAT_SCHEMA
            )

    def test_default_max_retries_is_one(self) -> None:
        # The engineerable knob (#167) — pinned so a change is deliberate.
        assert DEFAULT_MAX_RETRIES == 1
