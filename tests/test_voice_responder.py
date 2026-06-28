"""Tests for :class:`familiar_connect.processors.voice_responder.VoiceResponder`.

End-to-end voice turn: activity.start → cancel prior → FINAL →
assemble prompt → stream LLM → speak via TTS. Barge-in bites here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

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
from familiar_connect.diagnostics.collector import (
    get_span_collector,
    reset_span_collector,
)
from familiar_connect.diagnostics.voice_budget import reset_voice_budget_recorder
from familiar_connect.focus import FocusManager
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message
from familiar_connect.processors.voice_responder import VoiceResponder
from familiar_connect.tts_player import MockTTSPlayer


class _ScriptedLLM(LLMClient):
    """LLM stand-in with configurable per-delta delay."""

    def __init__(
        self,
        *,
        deltas: list[str],
        delay_ms: int = 0,
    ) -> None:
        super().__init__(api_key="k", model="m")
        self._deltas = list(deltas)
        self._delay_ms = delay_ms

    async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
        return Message(role="assistant", content="".join(self._deltas))

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[Message],  # noqa: ARG002
    ) -> AsyncIterator[str]:
        for d in self._deltas:
            if self._delay_ms:
                await asyncio.sleep(self._delay_ms / 1000.0)
            yield d


def _mk_activity_start(
    session_id: str = "voice:1",
    turn_id: str = "t-1",
    user_id: int | None = None,
) -> Event:
    return Event(
        event_id=f"act-{turn_id}",
        turn_id=turn_id,
        session_id=session_id,
        parent_event_ids=(),
        topic=TOPIC_VOICE_ACTIVITY_START,
        timestamp=datetime.now(tz=UTC),
        sequence_number=1,
        payload={"user_id": user_id} if user_id is not None else None,
    )


def _mk_final(
    text: str,
    session_id: str = "voice:1",
    turn_id: str = "t-1",
    seq: int = 2,
    user_id: int | None = None,
) -> Event:
    return Event(
        event_id=f"final-{turn_id}",
        turn_id=turn_id,
        session_id=session_id,
        parent_event_ids=(),
        topic=TOPIC_VOICE_TRANSCRIPT_FINAL,
        timestamp=datetime.now(tz=UTC),
        sequence_number=seq,
        payload={
            "text": text,
            "confidence": 0.9,
            "start": 0.0,
            "end": 1.0,
            "speaker": None,
            "user_id": user_id,
        },
    )


def _make_responder(
    *,
    llm: LLMClient,
    player: MockTTSPlayer,
    tmp_path: Path,
    router: TurnRouter | None = None,
    store: HistoryStore | None = None,
    member_resolver: Callable[[int, int], Author | None] | None = None,
    post_history_instructions: str = "",
) -> tuple[VoiceResponder, TurnRouter, HistoryStore]:
    card = tmp_path / "character.md"
    card.write_text("You are a familiar.\n")
    store = store or HistoryStore(":memory:")
    assembler = Assembler(
        layers=[
            CharacterCardLayer(card_path=card),
            RecentHistoryLayer(store=AsyncHistoryStore(store), window_size=20),
        ]
    )
    router = router or TurnRouter()
    responder = VoiceResponder(
        assembler=assembler,
        llm_client=llm,
        tts_player=player,
        history_store=AsyncHistoryStore(store),
        router=router,
        familiar_id="fam",
        member_resolver=member_resolver,
        post_history_instructions=post_history_instructions,
    )
    return responder, router, store


def _focus_manager(
    *, channel_id: int, channel_name: str, guild_name: str | None
) -> FocusManager:
    """FocusManager with the voice channel's name + server pre-populated.

    The voice reminder/logs key off the live ``channel_id``, not a focus
    pointer, so only ``channel_names``/``guild_names`` need seeding.
    """
    fm = FocusManager(familiar_id="fam", store=MagicMock(), subscriptions=MagicMock())
    fm.channel_names[channel_id] = channel_name
    if guild_name is not None:
        fm.guild_names[channel_id] = guild_name
    return fm


class TestActivityStart:
    @pytest.mark.asyncio
    async def test_activity_start_begins_turn(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["hi"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-new"), bus)
        scope = router.active_scope("voice:1")
        assert scope is not None
        assert scope.turn_id == "t-new"
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_second_activity_start_cancels_first(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["hi"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        first = router.active_scope("voice:1")
        assert first is not None

        await responder.handle(_mk_activity_start(turn_id="t-2"), bus)
        second = router.active_scope("voice:1")
        assert second is not None
        assert second.turn_id == "t-2"
        assert first.is_cancelled() is True
        await bus.shutdown()


class TestFinalReply:
    @pytest.mark.asyncio
    async def test_full_reply_on_final(self, tmp_path: Path) -> None:
        llm = _ScriptedLLM(deltas=["Hello", ", ", "world", "."])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _router, store = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert player.calls
        spoken, cancelled = player.calls[0]
        assert spoken == "Hello, world."
        assert cancelled is False
        # user turn recorded
        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        contents = [t.content for t in turns]
        assert any("hi there" in c for c in contents)
        assert any("Hello, world." in c for c in contents)

    @pytest.mark.asyncio
    async def test_trailing_reminder_carries_post_history_instructions(
        self, tmp_path: Path
    ) -> None:
        captured: list[list[Message]] = []

        class _CapturingLLM(LLMClient):
            def __init__(self) -> None:
                super().__init__(api_key="k", model="m")

            async def chat(self, messages: list[Message]) -> Message:
                captured.append(list(messages))
                return Message(role="assistant", content="ok")

            async def chat_stream(  # type: ignore[override]
                self,
                messages: list[Message],
            ) -> AsyncIterator[str]:
                captured.append(list(messages))
                yield "ok"

        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(
            llm=_CapturingLLM(),
            player=player,
            tmp_path=tmp_path,
            post_history_instructions="# Etiquette\n\nPrefer <silent>.",
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert captured, "LLM was never invoked"
        trailing = captured[0][-1]
        assert trailing.role == "system"
        assert "# Etiquette" in trailing.content_str
        assert trailing.content_str.index("You are speaking aloud") < (
            trailing.content_str.index("# Etiquette")
        )

    @pytest.mark.asyncio
    async def test_respond_decision_logged_on_full_reply(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Affirmative branch logs ``decision=respond`` once per turn.

        Mirrors the silent-branch log so every turn produces exactly
        one decision line.
        """
        llm = _ScriptedLLM(deltas=["Hello", ", ", "world", "."])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.processors.voice_responder"
        ):
            await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
            await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
            await responder.wait_until_idle()
        await bus.shutdown()

        # ``ls.kv`` interleaves ANSI codes between key and value, so match
        # ``decision`` and ``respond`` independently rather than as one
        # contiguous substring.
        decisions = [
            r
            for r in caplog.records
            if "decision" in r.getMessage() and "respond" in r.getMessage()
        ]
        assert len(decisions) == 1
        assert "t-1" in decisions[0].getMessage()

    @pytest.mark.asyncio
    async def test_silent_sentinel_skips_tts_and_assistant_turn(
        self, tmp_path: Path
    ) -> None:
        """``<silent>`` reply gates the response.

        TTS not invoked, no assistant turn recorded. User turn still
        recorded.
        """
        llm = _ScriptedLLM(deltas=["<silent>"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, store = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi nobody", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert player.calls == []
        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        assert all(t.role != "assistant" for t in turns)
        assert any(t.role == "user" and "hi nobody" in t.content for t in turns)

    @pytest.mark.asyncio
    async def test_empty_reply_skips_tts_and_assistant_turn(
        self, tmp_path: Path
    ) -> None:
        """Whitespace-only LLM reply must not reach TTS.

        Cartesia rejects empty ``transcript`` with HTTP 400. Mirrors the
        ``TextResponder`` guard so voice and text behave consistently.
        """
        llm = _ScriptedLLM(deltas=["   ", "\n", "\t"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, store = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert player.calls == []
        # No assistant turn recorded; user turn still recorded.
        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        assert all(t.role != "assistant" for t in turns)
        assert any(t.role == "user" and "hi there" in t.content for t in turns)

    @pytest.mark.asyncio
    async def test_stale_final_ignored(self, tmp_path: Path) -> None:
        """A FINAL whose turn_id mismatches the active scope is dropped."""
        llm = _ScriptedLLM(deltas=["ignored"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-new"), bus)
        # final from an older utterance
        await responder.handle(_mk_final("old", turn_id="t-OLD"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()
        assert player.calls == []


class TestVoiceBudgetSpans:
    """Per-turn voice latency phase spans land in the SpanCollector.

    See ``familiar_connect.diagnostics.voice_budget``. Mock TTS player
    doesn't reach DiscordVoicePlayer so ``playback_start`` /
    ``voice.total`` aren't asserted here — only the LLM/TTS handoff.
    """

    @pytest.mark.asyncio
    async def test_ttft_to_tts_gap_recorded(self, tmp_path: Path) -> None:
        reset_span_collector()
        reset_voice_budget_recorder()

        llm = _ScriptedLLM(deltas=["Hello there. ", "How are you?"], delay_ms=20)
        player = MockTTSPlayer(ms_per_word=1)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        names = [r.name for r in get_span_collector().all()]
        # ttft_to_tts is the gap from first LLM delta to first TTS
        # speak() call. A meaningful sentence-streaming win shows up
        # here once it's compared against the LLM stream length.
        assert "voice.ttft_to_tts" in names

    @pytest.mark.asyncio
    async def test_no_budget_spans_on_silent_reply(self, tmp_path: Path) -> None:
        """Silent sentinel suppresses TTS — no ``tts_first_audio`` to record."""
        reset_span_collector()
        reset_voice_budget_recorder()

        llm = _ScriptedLLM(deltas=["<silent>"])
        player = MockTTSPlayer(ms_per_word=1)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        names = [r.name for r in get_span_collector().all()]
        assert "voice.ttft_to_tts" not in names


class TestSentenceStreaming:
    """Sentence-level TTS streaming — feeds TTS as sentences finalise.

    Drops time-to-first-audio from "after the LLM finishes" to "after
    the first sentence". See [Voice pipeline — sentence
    streaming](../docs/architecture/voice-pipeline.md#sentence-streaming).
    """

    @pytest.mark.asyncio
    async def test_multi_sentence_reply_speaks_each_sentence(
        self, tmp_path: Path
    ) -> None:
        """Three-sentence reply → three ordered TTS calls."""
        deltas = ["Hello there. ", "How are you? ", "Nice to meet you."]
        llm = _ScriptedLLM(deltas=deltas)
        player = MockTTSPlayer(ms_per_word=1)
        responder, _, store = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        spoken = [text for text, _ in player.calls]
        assert spoken == ["Hello there.", "How are you?", "Nice to meet you."]
        # Assistant turn records the full concatenated reply.
        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        assistants = [t.content for t in turns if t.role == "assistant"]
        assert assistants == ["Hello there. How are you? Nice to meet you."]

    @pytest.mark.asyncio
    async def test_first_sentence_reaches_tts_before_stream_ends(
        self, tmp_path: Path
    ) -> None:
        """Latency win: first ``speak`` fires before later deltas land."""
        # Three sentences with a long gap before the last delta. If TTS
        # only fires after the LLM completes, the first ``speak`` would
        # be delayed by the full stream duration.
        llm = _ScriptedLLM(
            deltas=["First sentence. ", "Second sentence. ", "Third."],
            delay_ms=50,
        )
        player = MockTTSPlayer(ms_per_word=1)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()

        loop = asyncio.get_running_loop()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        t_start = loop.time()
        await responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert player.calls
        # MockTTSPlayer doesn't expose per-call timestamps; instead
        # assert at least 2 sentences were emitted, proving aggregator
        # didn't buffer the whole reply.
        assert len(player.calls) >= 2
        # Stream took ~150 ms total; if streaming worked, the wall
        # clock between activity_start and the first speak attempt is
        # bounded by the first delta's delivery time, not the whole
        # stream. Sanity-check that total duration is bounded.
        total = loop.time() - t_start
        assert total < 1.0  # would be much higher if pipeline regressed

    @pytest.mark.asyncio
    async def test_silent_sentinel_still_gates_with_streamer(
        self, tmp_path: Path
    ) -> None:
        """``<silent>`` from a streaming reply still suppresses TTS entirely."""
        # Sentinel split across deltas — exercises mid-buffer detection.
        llm = _ScriptedLLM(deltas=["<sil", "ent>"])
        player = MockTTSPlayer(ms_per_word=1)
        responder, _, store = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert player.calls == []
        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        assert all(t.role != "assistant" for t in turns)

    @pytest.mark.asyncio
    async def test_partial_buffer_flushed_when_stream_ends(
        self, tmp_path: Path
    ) -> None:
        """Reply without trailing punctuation still speaks via flush."""
        llm = _ScriptedLLM(deltas=["just a fragment"])
        player = MockTTSPlayer(ms_per_word=1)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert [t for t, _ in player.calls] == ["just a fragment"]


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_during_llm_stream_prevents_speech(
        self, tmp_path: Path
    ) -> None:
        """A new activity.start mid-LLM-stream cancels speech entirely."""
        # 10 deltas x 50ms each = 500ms LLM stream
        llm = _ScriptedLLM(deltas=[f"d{i} " for i in range(10)], delay_ms=50)
        player = MockTTSPlayer(ms_per_word=5)
        responder, _router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)

        async def bargein() -> None:
            await asyncio.sleep(0.08)  # 80ms into stream
            await responder.handle(_mk_activity_start(turn_id="t-2"), bus)

        # Run responder and barge-in concurrently
        task_reply = asyncio.create_task(
            responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        )
        task_barge = asyncio.create_task(bargein())
        await asyncio.gather(task_reply, task_barge)
        await responder.wait_until_idle()

        # Player should either have not been called, or been called with a
        # cancelled flag. The key assertion: we didn't speak the full text.
        full_text = "".join(f"d{i} " for i in range(10)).strip()
        if player.calls:
            spoken, cancelled = player.calls[0]
            assert cancelled is True or spoken != full_text

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_barge_in_logs_preempted_decision(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Cancel-before-decision emits ``decision=preempted``.

        Regression: turns superseded by a barge-in produced
        ``[LLM call] status=cancelled`` but **no** ``decision=…`` line,
        so a continuously-speaking user generated a chain of cancelled
        LLM calls with no way to correlate them to which transcript
        was preempted.

        Scenario reproduces the production case: first delta keeps
        ``SilentDetector`` undecided (``<sil`` matches ``<silent>``
        prefix), the barge-in lands before the second delta, and the
        cancel-check at the top of the stream loop fires before any
        decision is logged.
        """
        llm = _ScriptedLLM(deltas=["<sil", "ent>"], delay_ms=100)
        player = MockTTSPlayer(ms_per_word=5)
        responder, _router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        async def bargein() -> None:
            # Land between first and second delta so the cancel check
            # at the loop top wins before the silent gate latches.
            await asyncio.sleep(0.15)
            await responder.handle(_mk_activity_start(turn_id="t-2"), bus)

        with caplog.at_level(
            logging.INFO, logger="familiar_connect.processors.voice_responder"
        ):
            await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
            task_reply = asyncio.create_task(
                responder.handle(_mk_final("hi", turn_id="t-1"), bus)
            )
            task_barge = asyncio.create_task(bargein())
            await asyncio.gather(task_reply, task_barge)
            await responder.wait_until_idle()
        await bus.shutdown()

        # Match ``decision`` and ``preempted`` independently — ``ls.kv``
        # interleaves ANSI codes between key and value.
        preempted = [
            r
            for r in caplog.records
            if "decision" in r.getMessage()
            and "preempted" in r.getMessage()
            and "t-1" in r.getMessage()
        ]
        # No silent/respond should fire for t-1 — gate never latched.
        other = [
            r
            for r in caplog.records
            if "decision" in r.getMessage()
            and ("silent" in r.getMessage() or "respond" in r.getMessage())
            and "t-1" in r.getMessage()
        ]
        assert other == [], (
            "unexpected silent/respond decision for t-1: "
            f"{[r.getMessage() for r in other]}"
        )
        assert len(preempted) == 1, (
            "expected one preempted decision for t-1; "
            f"got {len(preempted)} from records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_barge_in_during_speech_cuts_playback_fast(
        self, tmp_path: Path
    ) -> None:
        """Sub-200ms interruption of in-progress TTS playback.

        This is the plan's architectural proof: speak a long utterance,
        fire a new activity.start mid-playback, assert TTS stops
        well under 200 ms.
        """
        llm = _ScriptedLLM(deltas=["hello world"], delay_ms=0)
        # 50 words x 20ms each = 1000ms of playback
        player = MockTTSPlayer(ms_per_word=20, poll_ms=5)
        # Make LLM return a very long text so MockTTSPlayer has lots to play
        llm = _ScriptedLLM(
            deltas=[" ".join(f"w{i}" for i in range(50))],
            delay_ms=0,
        )
        responder, _router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)

        loop = asyncio.get_running_loop()
        task_reply = asyncio.create_task(
            responder.handle(_mk_final("go ahead", turn_id="t-1"), bus)
        )
        # barge in only once playback is genuinely in progress — avoids the
        # race where a loaded runner preempts t-1 before speak() is ever called
        await asyncio.wait_for(player.speak_started.wait(), timeout=5)
        cancel_time = loop.time()
        await responder.handle(_mk_activity_start(turn_id="t-2"), bus)
        await task_reply
        # handle() spawns _on_final as a task — wait for the in-flight speak
        # to observe cancellation before timing.
        await responder.wait_until_idle()
        t_reply_done = loop.time()

        cut_latency_ms = int((t_reply_done - cancel_time) * 1000)
        # The plan's architectural proof: sub-200ms cancel
        assert cut_latency_ms < 200, (
            f"barge-in took {cut_latency_ms}ms; should be under 200"
        )
        assert player.calls
        assert player.calls[0][1] is True  # was cancelled
        await bus.shutdown()


class TestPerUserBargeIn:
    """Per-user scope: cross-user activity.start must NOT cancel a reply.

    Discord delivers per-SSRC audio so every speaker fires their own
    activity.start. Channel-level scope keys cause every user's start
    to barge every other user's in-flight reply, so the bot rarely
    gets to finish speaking. Scope must be (channel, user_id).
    """

    @pytest.mark.asyncio
    async def test_cross_user_start_does_not_cancel(self, tmp_path: Path) -> None:
        """Alice starts → Bob starts → Alice's scope still active."""
        llm = _ScriptedLLM(deltas=["x"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-alice", user_id=101), bus)
        await responder.handle(_mk_activity_start(turn_id="t-bob", user_id=202), bus)

        alice = router.active_scope("voice:1:user:101")
        bob = router.active_scope("voice:1:user:202")
        assert alice is not None
        assert bob is not None
        assert alice.is_cancelled() is False
        assert alice.turn_id == "t-alice"
        assert bob.turn_id == "t-bob"
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_same_user_start_still_cancels(self, tmp_path: Path) -> None:
        """Same speaker barging themselves still wins — preserves self-barge."""
        llm = _ScriptedLLM(deltas=["x"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-1", user_id=101), bus)
        first = router.active_scope("voice:1:user:101")
        assert first is not None

        await responder.handle(_mk_activity_start(turn_id="t-2", user_id=101), bus)
        second = router.active_scope("voice:1:user:101")
        assert second is not None
        assert second.turn_id == "t-2"
        assert first.is_cancelled() is True
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_cross_user_does_not_cancel_in_flight_reply(
        self, tmp_path: Path
    ) -> None:
        """Reply to Alice keeps streaming when Bob starts speaking."""
        llm = _ScriptedLLM(deltas=[f"d{i} " for i in range(5)], delay_ms=30)
        player = MockTTSPlayer(ms_per_word=5)
        responder, _router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-alice", user_id=101), bus)

        async def bob_speaks() -> None:
            await asyncio.sleep(0.05)
            await responder.handle(
                _mk_activity_start(turn_id="t-bob", user_id=202), bus
            )

        task_reply = asyncio.create_task(
            responder.handle(_mk_final("hi", turn_id="t-alice", user_id=101), bus)
        )
        task_bob = asyncio.create_task(bob_speaks())
        await asyncio.gather(task_reply, task_bob)
        await responder.wait_until_idle()

        # Alice's reply should have been spoken in full despite Bob talking.
        full = "".join(f"d{i} " for i in range(5))
        assert player.calls
        spoken, cancelled = player.calls[0]
        assert cancelled is False
        assert spoken == full
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_other_users_continuous_starts_do_not_cut_playback(
        self, tmp_path: Path
    ) -> None:
        """Bob's repeated activity.starts mid-playback must not cut Alice off.

        Regression: production logs showed Cassidy's reply being cut by
        Cor's continuous transcription. Cor's first activity.start
        registered a scope; every subsequent one had ``prior is not None``
        and triggered a global ``tts.stop()``, halting the discord voice
        client mid-Cassidy-reply.
        """
        llm = _ScriptedLLM(deltas=["hello "], delay_ms=0)
        # 200ms playback so Bob's barrage lands while speak() is in flight.
        player = MockTTSPlayer(ms_per_word=200, poll_ms=5)
        responder, _router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        await responder.handle(_mk_activity_start(turn_id="t-alice", user_id=101), bus)

        async def bob_keeps_starting() -> None:
            # let Alice's speak() begin first
            await asyncio.sleep(0.04)
            # first start: no prior scope for Bob, no stop fired
            await responder.handle(
                _mk_activity_start(turn_id="t-bob-1", user_id=202), bus
            )
            # subsequent starts: each has a prior scope -> would fire tts.stop()
            for i in range(2, 7):
                await asyncio.sleep(0.02)
                await responder.handle(
                    _mk_activity_start(turn_id=f"t-bob-{i}", user_id=202), bus
                )

        task_reply = asyncio.create_task(
            responder.handle(_mk_final("hi", turn_id="t-alice", user_id=101), bus)
        )
        task_bob = asyncio.create_task(bob_keeps_starting())
        await asyncio.gather(task_reply, task_bob)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert player.calls
        spoken, cancelled = player.calls[0]
        assert cancelled is False, (
            "Bob's repeated activity.starts cut Alice's playback short"
        )
        assert spoken == "hello "


class TestCrossUserReplyGate:
    """Per-channel reply gate serializes cross-user replies.

    Two speakers talking in one window each spawn an independent
    reply pipeline. Without serialization both assemble before
    either commits an assistant turn, so both answer the same
    moment — the production double-response. A per-channel lock
    orders generation behind the already-serial TTS playback: the
    second pipeline assembles only after the first commits, sees
    the reply in context, and can resolve ``<silent>``.
    """

    @pytest.mark.asyncio
    async def test_concurrent_cross_user_finals_serialize_to_one_reply(
        self, tmp_path: Path
    ) -> None:
        """Two speakers, one window → streams never overlap, one reply.

        The LLM stand-in both detects overlap (max concurrent streams)
        and mimics the model: it answers when no assistant turn is in
        context yet, and emits ``<silent>`` once it sees the bot has
        already replied. Without the gate both pipelines assemble
        before either commits, so both stream concurrently and both
        answer. The generous in-stream delay makes that interleaving
        deterministic rather than racy.
        """

        class _BarrierLLM(LLMClient):
            """Forces overlap if the pipeline allows it.

            First stream to enter parks on a barrier waiting for a
            second; a second entrant releases it. Without the gate the
            second pipeline enters and ``max_active`` reaches 2. With
            the gate it can't enter until the first exits, so the
            barrier times out and ``max_active`` stays 1 — deterministic
            either way, no reliance on scheduler luck.
            """

            def __init__(self) -> None:
                super().__init__(api_key="k", model="m")
                self.active = 0
                self.max_active = 0
                self._entered = 0
                self._second = asyncio.Event()

            async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
                return Message(role="assistant", content="x")

            async def chat_stream(  # type: ignore[override]
                self,
                messages: list[Message],
            ) -> AsyncIterator[str]:
                already = any(m.role == "assistant" for m in messages)
                self.active += 1
                self._entered += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    if self._entered >= 2:
                        self._second.set()
                    else:
                        # give a concurrent pipeline a chance to join;
                        # the gate prevents it, so this times out
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(self._second.wait(), timeout=0.3)
                    yield "<silent>" if already else "Sure thing."
                finally:
                    self.active -= 1

        llm = _BarrierLLM()
        player = MockTTSPlayer(ms_per_word=1)
        responder, _router, store = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        # Alice and Bob both speak in the same window.
        await responder.handle(_mk_activity_start(turn_id="t-alice", user_id=101), bus)
        await responder.handle(_mk_activity_start(turn_id="t-bob", user_id=202), bus)
        await responder.handle(
            _mk_final("did you watch it", turn_id="t-alice", user_id=101), bus
        )
        await responder.handle(
            _mk_final("what did you think", turn_id="t-bob", user_id=202), bus
        )
        await responder.wait_until_idle()
        await bus.shutdown()

        # Core contract: reply generation never overlaps per channel.
        assert llm.max_active == 1, (
            f"reply streams overlapped (max_active={llm.max_active}); "
            "per-channel gate did not serialize cross-user replies"
        )
        # Behavioral outcome: exactly one reply; the loser went silent.
        assert len(player.calls) == 1
        assert player.calls[0][0] == "Sure thing."
        turns = store.recent(familiar_id="fam", channel_id=1, limit=20)
        assistants = [t for t in turns if t.role == "assistant"]
        assert len(assistants) == 1, (
            f"expected one assistant turn, got {len(assistants)}: "
            f"{[t.content for t in assistants]}"
        )
        # Both speakers' user turns still recorded — observation isn't gated.
        users = [t.content for t in turns if t.role == "user"]
        assert any("did you watch it" in c for c in users)
        assert any("what did you think" in c for c in users)


class TestVoiceAuthorResolution:
    """Voice user turns must resolve user_id → Author for prompt attribution.

    Without this, voice turns store author=None and prompts show plain
    ``[user]`` instead of ``[HH:MM Cor Vous]`` — the model can't tell
    speakers apart.
    """

    @pytest.mark.asyncio
    async def test_member_resolver_threads_author_into_history(
        self, tmp_path: Path
    ) -> None:
        llm = _ScriptedLLM(deltas=["ack"])
        player = MockTTSPlayer(ms_per_word=5)

        def resolver(channel_id: int, user_id: int) -> Author | None:
            assert channel_id == 1
            assert user_id == 101
            return Author(
                platform="discord",
                user_id="101",
                username="cassidy",
                display_name="Cor Vous",
            )

        responder, _router, store = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path, member_resolver=resolver
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1", user_id=101), bus)
        await responder.handle(
            _mk_final("hello there", turn_id="t-1", user_id=101), bus
        )
        await responder.wait_until_idle()
        await bus.shutdown()

        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        user_turns = [t for t in turns if t.role == "user"]
        assert user_turns, "user voice turn was not recorded"
        author = user_turns[-1].author
        assert author is not None
        assert author.display_name == "Cor Vous"
        assert author.user_id == "101"

    @pytest.mark.asyncio
    async def test_resolver_miss_falls_back_to_anon_user_turn(
        self, tmp_path: Path
    ) -> None:
        """Resolver returning None must not crash; user turn still recorded."""
        llm = _ScriptedLLM(deltas=["ack"])
        player = MockTTSPlayer(ms_per_word=5)

        def resolver(channel_id: int, user_id: int) -> Author | None:  # noqa: ARG001
            return None

        responder, _router, store = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path, member_resolver=resolver
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1", user_id=101), bus)
        await responder.handle(
            _mk_final("hello there", turn_id="t-1", user_id=101), bus
        )
        await responder.wait_until_idle()
        await bus.shutdown()

        turns = store.recent(familiar_id="fam", channel_id=1, limit=10)
        assert any("hello there" in t.content for t in turns)


class TestTrailingReminder:
    """The LLM should see a trailing ``system`` message after recent history.

    Re-emits the time + sentinels block plus the per-mode operating
    directive ("You are speaking aloud…") at the tail of the context
    so recency-biased models honor the format gate.
    """

    @pytest.mark.asyncio
    async def test_trailing_system_carries_voice_directive(
        self, tmp_path: Path
    ) -> None:
        captured: list[list[Message]] = []

        class _CapturingLLM(LLMClient):
            def __init__(self) -> None:
                super().__init__(api_key="k", model="m")

            async def chat(self, messages: list[Message]) -> Message:
                captured.append(list(messages))
                return Message(role="assistant", content="ok")

            async def chat_stream(  # type: ignore[override]
                self,
                messages: list[Message],
            ) -> AsyncIterator[str]:
                captured.append(list(messages))
                yield "ok"

        llm = _CapturingLLM()
        player = MockTTSPlayer(ms_per_word=1)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert captured, "LLM was never invoked"
        msgs = captured[0]
        assert msgs[-1].role == "system"
        assert "You are speaking aloud" in msgs[-1].content
        # The tail block also restates the time.
        assert "It is now" in msgs[-1].content


class TestDispatchLoop:
    """Production dispatch pattern (`async for event in bus.subscribe(...)`).

    Regression: previously the dispatcher awaited ``responder.handle()``
    sequentially, so a slow ``_on_final`` blocked subsequent
    ``voice.activity.start`` events from ever reaching ``prior.cancel()``
    in time. The voice reply for an old utterance therefore played in
    full even after the user spoke again.
    """

    @pytest.mark.asyncio
    async def test_dispatcher_unblocked_during_in_flight_final(
        self, tmp_path: Path
    ) -> None:
        """Barge-in must take effect while the prior LLM stream is parked."""

        class _BlockingLLM(LLMClient):
            def __init__(self) -> None:
                super().__init__(api_key="k", model="m")
                self.started = asyncio.Event()
                self.unblock = asyncio.Event()

            async def chat(self, messages: list[Message]) -> Message:  # noqa: ARG002
                return Message(role="assistant", content="x")

            async def chat_stream(  # type: ignore[override]
                self,
                messages: list[Message],  # noqa: ARG002
            ) -> AsyncIterator[str]:
                self.started.set()
                await self.unblock.wait()
                yield "hello"

        llm = _BlockingLLM()
        player = MockTTSPlayer(ms_per_word=5)
        responder, router, _ = _make_responder(
            llm=llm, player=player, tmp_path=tmp_path
        )
        bus = InProcessEventBus()
        await bus.start()

        async def dispatcher() -> None:
            async for event in bus.subscribe(responder.topics):
                await responder.handle(event, bus)

        dispatch_task = asyncio.create_task(dispatcher(), name="dispatcher")
        try:
            # Let the dispatcher start its async-for loop and register
            # the subscription before any publish.
            await asyncio.sleep(0)
            await bus.publish(_mk_activity_start(turn_id="t-1"))
            await bus.publish(_mk_final("hi", turn_id="t-1"))
            # Wait until the responder is genuinely parked inside the LLM
            # stream — past this point any further events queue in the bus.
            await asyncio.wait_for(llm.started.wait(), timeout=1.0)
            # Barge-in: the dispatcher must keep pulling events. Under the
            # bug, it's stuck inside handle() awaiting the blocked LLM.
            await bus.publish(_mk_activity_start(turn_id="t-2"))
            for _ in range(50):
                await asyncio.sleep(0.01)
                scope = router.active_scope("voice:1")
                if scope is not None and scope.turn_id == "t-2":
                    break
            scope = router.active_scope("voice:1")
            assert scope is not None
            assert scope.turn_id == "t-2", (
                f"dispatcher blocked on in-flight final; active turn is "
                f"{scope.turn_id!r}, expected 't-2'"
            )
        finally:
            llm.unblock.set()
            await responder.wait_until_idle()
            await bus.shutdown()
            dispatch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await dispatch_task


class _CapturingLLM(LLMClient):
    """Records every message list handed to the LLM; replies ``ok``."""

    def __init__(self, captured: list[list[Message]]) -> None:
        super().__init__(api_key="k", model="m")
        self._captured = captured

    async def chat(self, messages: list[Message]) -> Message:
        self._captured.append(list(messages))
        return Message(role="assistant", content="ok")

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[Message],
    ) -> AsyncIterator[str]:
        self._captured.append(list(messages))
        yield "ok"


class TestTrailingReminderServerName:
    """Voice trailing reminder names the live channel's Discord server.

    Mirrors the text path: the server clause attaches to the trailing
    focus line only (the head stays byte-stable for its cache prefix),
    and is wired only when a focus manager is present.
    """

    @pytest.mark.asyncio
    async def test_trailing_names_server_and_channel(self, tmp_path: Path) -> None:
        captured: list[list[Message]] = []
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(
            llm=_CapturingLLM(captured), player=player, tmp_path=tmp_path
        )
        responder._focus_manager = _focus_manager(
            channel_id=1, channel_name="voice-chan", guild_name="My Server"
        )
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert captured, "LLM was never invoked"
        trailing = captured[0][-1].content_str
        assert "#voice-chan" in trailing
        assert '"My Server" server' in trailing

    @pytest.mark.asyncio
    async def test_trailing_omits_focus_without_focus_manager(
        self, tmp_path: Path
    ) -> None:
        """Backward-compat: no focus manager → no focus/server line at all."""
        captured: list[list[Message]] = []
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(
            llm=_CapturingLLM(captured), player=player, tmp_path=tmp_path
        )
        assert responder._focus_manager is None
        bus = InProcessEventBus()
        await bus.start()
        await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
        await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
        await responder.wait_until_idle()
        await bus.shutdown()

        assert captured, "LLM was never invoked"
        trailing = captured[0][-1].content_str
        assert "Your attention is currently on" not in trailing


class TestPerTurnOriginLogging:
    """Per-turn voice decision logs name the turn's server + channel.

    ``ch`` renders ``#name(id)`` and ``srv`` names the Discord server —
    both resolved from the live voice ``channel_id``, omitted gracefully
    when there's no focus manager or the guild is unknown.
    """

    @staticmethod
    def _decision_record(
        caplog: pytest.LogCaptureFixture, decision: str
    ) -> logging.LogRecord:
        records = [
            r
            for r in caplog.records
            if "decision" in r.getMessage() and decision in r.getMessage()
        ]
        assert len(records) == 1, [r.getMessage() for r in caplog.records]
        return records[0]

    @pytest.mark.asyncio
    async def test_respond_log_names_server_and_channel(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _ScriptedLLM(deltas=["Hello", ", ", "world", "."])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        responder._focus_manager = _focus_manager(
            channel_id=1, channel_name="voice-chan", guild_name="My Server"
        )
        bus = InProcessEventBus()
        await bus.start()
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.processors.voice_responder"
        ):
            await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
            await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
            await responder.wait_until_idle()
        await bus.shutdown()

        msg = self._decision_record(caplog, "respond").getMessage()
        # ``ls.kv`` interleaves ANSI between key and value; match the
        # channel label and server name as independent substrings.
        assert "#voice-chan" in msg
        assert "My Server" in msg

    @pytest.mark.asyncio
    async def test_respond_log_omits_server_without_focus_manager(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _ScriptedLLM(deltas=["Hello", ", ", "world", "."])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        assert responder._focus_manager is None
        bus = InProcessEventBus()
        await bus.start()
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.processors.voice_responder"
        ):
            await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
            await responder.handle(_mk_final("hi there", turn_id="t-1"), bus)
            await responder.wait_until_idle()
        await bus.shutdown()

        msg = self._decision_record(caplog, "respond").getMessage()
        assert "#1" in msg  # graceful fallback to raw channel id
        assert "srv=" not in msg  # never log srv=None

    @pytest.mark.asyncio
    async def test_silent_log_names_server_and_channel(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _ScriptedLLM(deltas=["<silent>"])
        player = MockTTSPlayer(ms_per_word=5)
        responder, _, _ = _make_responder(llm=llm, player=player, tmp_path=tmp_path)
        responder._focus_manager = _focus_manager(
            channel_id=1, channel_name="voice-chan", guild_name="My Server"
        )
        bus = InProcessEventBus()
        await bus.start()
        with caplog.at_level(
            logging.INFO, logger="familiar_connect.processors.voice_responder"
        ):
            await responder.handle(_mk_activity_start(turn_id="t-1"), bus)
            await responder.handle(_mk_final("hi nobody", turn_id="t-1"), bus)
            await responder.wait_until_idle()
        await bus.shutdown()

        msg = self._decision_record(caplog, "silent").getMessage()
        assert "#voice-chan" in msg
        assert "My Server" in msg
