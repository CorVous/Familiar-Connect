"""Tool-calling integration tests for :class:`VoiceResponder`.

Two scenarios:

1. **Content + tool_call**: model emits text first, then a tool_call.
   The TTS player must speak the text *before* the tool handler runs.
2. **Empty content + tool_call**: the filler backstop kicks in — a
   short stock filler phrase is spoken before the tool handler so
   the user never hears silence mid-turn.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from familiar_connect.bus import InProcessEventBus, TurnRouter
from familiar_connect.bus.envelope import Event
from familiar_connect.bus.topics import (
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)
from familiar_connect.context import (
    Assembler,
    CharacterCardLayer,
    RecentHistoryLayer,
)
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.llm import LLMClient, LLMDelta, Message
from familiar_connect.processors.voice_responder import VoiceResponder
from familiar_connect.tools.registry import Tool, ToolContext, ToolRegistry
from familiar_connect.tts_player import MockTTSPlayer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from familiar_connect.bus.envelope import TurnScope


class _ScriptedToolLLM(LLMClient):
    """Yield scripted LLMDeltas per stream_completion call."""

    def __init__(self, scripts: list[list[LLMDelta]]) -> None:
        super().__init__(api_key="k", model="m", tool_calling=True)
        self._scripts = list(scripts)
        self.calls: list[list[Message]] = []

    async def stream_completion(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,  # noqa: ARG002
    ) -> AsyncIterator[LLMDelta]:
        self.calls.append(list(messages))
        script = self._scripts.pop(0) if self._scripts else []
        for d in script:
            yield d


def _activity_start(turn_id: str = "t-1") -> Event:
    return Event(
        event_id=f"act-{turn_id}",
        turn_id=turn_id,
        session_id="voice:1",
        parent_event_ids=(),
        topic=TOPIC_VOICE_ACTIVITY_START,
        timestamp=datetime.now(tz=UTC),
        sequence_number=1,
        payload=None,
    )


def _final(text: str, turn_id: str = "t-1") -> Event:
    return Event(
        event_id=f"final-{turn_id}",
        turn_id=turn_id,
        session_id="voice:1",
        parent_event_ids=(),
        topic=TOPIC_VOICE_TRANSCRIPT_FINAL,
        timestamp=datetime.now(tz=UTC),
        sequence_number=2,
        payload={
            "text": text,
            "confidence": 0.9,
            "start": 0.0,
            "end": 1.0,
            "speaker": None,
            "user_id": None,
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
    player: MockTTSPlayer,
    tmp_path: Path,
    registry: ToolRegistry,
    tool_context_factory,  # noqa: ANN001
    tool_filler_phrases: tuple[str, ...] = ("one sec...",),
) -> tuple[VoiceResponder, TurnRouter, HistoryStore]:
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
    responder = VoiceResponder(
        assembler=assembler,
        llm_client=llm,
        tts_player=player,
        history_store=AsyncHistoryStore(store),
        router=router,
        familiar_id="fam",
        tool_registry=registry,
        tool_context_factory=tool_context_factory,
        tool_filler_phrases=tool_filler_phrases,
    )
    return responder, router, store


@pytest.mark.asyncio
async def test_speak_completes_before_tool_runs(tmp_path: Path) -> None:
    """Voice ordering: spoken content arrives at TTS before the handler runs."""
    tool_started_at: list[float] = []

    async def _h(_args: dict, _ctx: ToolContext) -> str:
        await asyncio.sleep(0)
        tool_started_at.append(time.monotonic())
        return json.dumps({"ok": True})

    registry = ToolRegistry()
    registry.register(Tool(name="set_alarm", description="", parameters={}, handler=_h))

    scripts = [
        # iteration 1: text + tool_call in the SAME iteration; text comes first
        [
            LLMDelta(content="Sure, one moment."),
            _tc_delta("c1", "set_alarm", {"reason": "x", "delay_seconds": 30}),
            LLMDelta(finish_reason="tool_calls"),
        ],
        # iteration 2: short final text
        [LLMDelta(content="Done."), LLMDelta(finish_reason="stop")],
    ]
    llm = _ScriptedToolLLM(scripts)
    player = MockTTSPlayer(ms_per_word=2)

    def _ctx_factory(channel_id: int, turn_id: str) -> ToolContext:
        return ToolContext(
            familiar_id="fam",
            channel_id=channel_id,
            channel_kind="voice",
            turn_id=turn_id,
            history=AsyncHistoryStore(HistoryStore(":memory:")),
            bus=InProcessEventBus(),
        )

    responder, _, _ = _make_responder(
        llm=llm,
        player=player,
        tmp_path=tmp_path,
        registry=registry,
        tool_context_factory=_ctx_factory,
    )
    bus = InProcessEventBus()
    await bus.start()
    try:
        await responder.handle(_activity_start("t-1"), bus)
        await responder.handle(_final("set an alarm", "t-1"), bus)
        await responder.wait_until_idle()
    finally:
        await bus.shutdown()

    spoken_texts = [c[0] for c in player.calls]
    # iteration 1 text spoken before tool, then iteration 2 text spoken after
    assert spoken_texts, "expected TTS to have been called"
    assert "Sure" in spoken_texts[0] or "Sure" in "".join(spoken_texts)
    # tool ran exactly once
    assert len(tool_started_at) == 1


@pytest.mark.asyncio
async def test_filler_spoken_when_tool_call_has_empty_content(tmp_path: Path) -> None:
    """Empty-content tool_call → filler phrase spoken before handler runs."""
    tool_ran_at: list[float] = []
    spoken_at_tool_run: list[tuple[str, ...]] = []
    spoken_running: list[str] = []

    async def _h(_args: dict, _ctx: ToolContext) -> str:
        await asyncio.sleep(0)
        # capture what's been spoken by the time the tool starts
        spoken_at_tool_run.append(tuple(spoken_running))
        tool_ran_at.append(time.monotonic())
        return json.dumps({"ok": True})

    registry = ToolRegistry()
    registry.register(Tool(name="set_alarm", description="", parameters={}, handler=_h))

    scripts = [
        # iteration 1: ONLY a tool_call, no content — filler should fire
        [
            _tc_delta("c1", "set_alarm", {"reason": "x", "delay_seconds": 30}),
            LLMDelta(finish_reason="tool_calls"),
        ],
        # iteration 2: final text
        [LLMDelta(content="Done."), LLMDelta(finish_reason="stop")],
    ]
    llm = _ScriptedToolLLM(scripts)

    class _RecordingPlayer(MockTTSPlayer):
        async def speak(self, text: str, *, scope: TurnScope) -> None:
            spoken_running.append(text)
            await super().speak(text, scope=scope)

    player = _RecordingPlayer(ms_per_word=2)

    def _ctx_factory(channel_id: int, turn_id: str) -> ToolContext:
        return ToolContext(
            familiar_id="fam",
            channel_id=channel_id,
            channel_kind="voice",
            turn_id=turn_id,
            history=AsyncHistoryStore(HistoryStore(":memory:")),
            bus=InProcessEventBus(),
        )

    responder, _, _ = _make_responder(
        llm=llm,
        player=player,
        tmp_path=tmp_path,
        registry=registry,
        tool_context_factory=_ctx_factory,
        tool_filler_phrases=("hang on...",),
    )
    bus = InProcessEventBus()
    await bus.start()
    try:
        await responder.handle(_activity_start("t-2"), bus)
        await responder.handle(_final("set alarm", "t-2"), bus)
        await responder.wait_until_idle()
    finally:
        await bus.shutdown()

    # filler must have been spoken before the tool ran
    assert tool_ran_at, "tool never ran"
    assert spoken_at_tool_run, "speak() never recorded"
    pre_tool_spoken = spoken_at_tool_run[0]
    assert any("hang on" in s for s in pre_tool_spoken), (
        f"filler not spoken before tool: {pre_tool_spoken}"
    )
