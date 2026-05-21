"""Tool-calling integration tests for :class:`TextResponder`.

Verifies that when ``llm.tool_calling_enabled`` is True and a registry
is provided, the responder runs the agentic loop end-to-end:

* iteration 1: model emits a tool_call → tool handler runs
* iteration 2: model emits final text → posted via ``send_text``
* history has user + assistant(tool_calls) + tool + assistant(text)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from familiar_connect.bus import InProcessEventBus, TurnRouter
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.context import (
    Assembler,
    CharacterCardLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, LLMDelta, Message
from familiar_connect.processors.text_responder import TextResponder
from familiar_connect.tools.registry import Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Scripted LLM that supports streaming tool calls
# ---------------------------------------------------------------------------


class _ScriptedToolLLM(LLMClient):
    """LLM stand-in that yields LLMDeltas from a per-call script."""

    def __init__(self, scripts: list[list[LLMDelta]]) -> None:
        super().__init__(api_key="k", model="m", tool_calling=True)
        self._scripts = list(scripts)
        self.calls: list[list[Message]] = []
        self.tool_payloads: list[list[dict] | None] = []

    async def stream_completion(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMDelta]:
        self.calls.append(list(messages))
        self.tool_payloads.append(tools)
        script = self._scripts.pop(0) if self._scripts else []
        for d in script:
            yield d


class _CapturingSend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str | None, tuple[int, ...]]] = []

    async def __call__(
        self,
        channel_id: int,
        content: str,
        reply_to_message_id: str | None = None,
        mention_user_ids: tuple[int, ...] = (),
    ) -> str | None:
        self.calls.append((channel_id, content, reply_to_message_id, mention_user_ids))
        return "msg-1"


def _discord_text_event(
    *,
    channel_id: int = 42,
    content: str = "set an alarm",
) -> Event:
    return Event(
        event_id="e-1",
        turn_id="e-1",
        session_id=f"discord:{channel_id}",
        parent_event_ids=(),
        topic=TOPIC_DISCORD_TEXT,
        timestamp=datetime.now(tz=UTC),
        sequence_number=1,
        payload={
            "familiar_id": "fam",
            "channel_id": channel_id,
            "guild_id": 99,
            "author": Author(
                platform="discord",
                user_id="1",
                username="alice",
                display_name="Alice",
            ),
            "content": content,
        },
    )


def _tc_delta(call_id: str, name: str, args: dict) -> LLMDelta:
    return LLMDelta(
        tool_calls=[
            {
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    )


def _make_responder(
    *,
    llm: LLMClient,
    send: _CapturingSend,
    tmp_path: Path,
    registry: ToolRegistry,
    tool_context_factory,  # noqa: ANN001 — closure type fine for tests
) -> tuple[TextResponder, TurnRouter, HistoryStore]:
    card = tmp_path / "character.md"
    card.write_text("You are a familiar.\n")
    store = HistoryStore(":memory:")
    assembler = Assembler(
        layers=[
            CharacterCardLayer(card_path=card),
            RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20),
        ]
    )
    router = TurnRouter()
    responder = TextResponder(
        assembler=assembler,
        llm_client=llm,
        send_text=send,
        history_store=AsyncHistoryStore(store),
        router=router,
        familiar_id="fam",
        tool_registry=registry,
        tool_context_factory=tool_context_factory,
    )
    return responder, router, store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_responder_runs_agentic_loop(tmp_path: Path) -> None:
    handler_seen: list[dict] = []

    async def _h(args: dict, _ctx: ToolContext) -> str:
        await asyncio.sleep(0)
        handler_seen.append(args)
        return json.dumps({"ok": True})

    registry = ToolRegistry()
    registry.register(Tool(name="set_alarm", description="", parameters={}, handler=_h))

    scripts = [
        # iteration 1: tool call (no spoken content)
        [
            _tc_delta("c1", "set_alarm", {"reason": "ping", "delay_seconds": 30}),
            LLMDelta(finish_reason="tool_calls"),
        ],
        # iteration 2: final text
        [LLMDelta(content="Alarm set."), LLMDelta(finish_reason="stop")],
    ]
    llm = _ScriptedToolLLM(scripts)
    send = _CapturingSend()

    def _ctx_factory(channel_id: int, turn_id: str) -> ToolContext:
        return ToolContext(
            familiar_id="fam",
            channel_id=channel_id,
            channel_kind="text",
            turn_id=turn_id,
            history=AsyncHistoryStore(HistoryStore(":memory:")),
            bus=InProcessEventBus(),
        )

    responder, _, store = _make_responder(
        llm=llm,
        send=send,
        tmp_path=tmp_path,
        registry=registry,
        tool_context_factory=_ctx_factory,
    )
    bus = InProcessEventBus()
    await bus.start()
    try:
        await responder.handle(_discord_text_event(), bus)
    finally:
        await bus.shutdown()

    # Tool ran with the right args
    assert handler_seen == [{"reason": "ping", "delay_seconds": 30}]
    # Final text was posted
    assert len(send.calls) == 1
    assert send.calls[0][0] == 42
    assert send.calls[0][1] == "Alarm set."

    # expected history sequence: user, assistant(tool_calls), tool, assistant(text)
    rows = store._conn.execute(
        "SELECT role, content, tool_calls_json, tool_call_id "
        "FROM turns WHERE familiar_id = 'fam' ORDER BY id"
    ).fetchall()
    roles = [r["role"] for r in rows]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # middle assistant carries tool_calls
    assert rows[1]["tool_calls_json"]
    tc = json.loads(rows[1]["tool_calls_json"])
    assert tc[0]["function"]["name"] == "set_alarm"
    # tool message references the call
    assert rows[2]["tool_call_id"] == "c1"
    assert json.loads(rows[2]["content"]) == {"ok": True}
    # final assistant content
    assert rows[3]["content"] == "Alarm set."
    assert rows[3]["tool_calls_json"] is None
