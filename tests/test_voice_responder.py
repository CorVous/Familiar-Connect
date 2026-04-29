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
    CoreInstructionsLayer,
    RecentHistoryLayer,
)
from familiar_connect.diagnostics.collector import (
    get_span_collector,
    reset_span_collector,
)
from familiar_connect.diagnostics.voice_budget import reset_voice_budget_recorder
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
) -> tuple[VoiceResponder, TurnRouter, HistoryStore]:
    core = tmp_path / "core.md"
    core.write_text("You are a familiar.\n")
    store = store or HistoryStore(":memory:")
    assembler = Assembler(
        layers=[
            CoreInstructionsLayer(path=core),
            RecentHistoryLayer(store=store, window_size=20),
        ]
    )
    router = router or TurnRouter()
    responder = VoiceResponder(
        assembler=assembler,
        llm_client=llm,
        tts_player=player,
        history_store=store,
        router=router,
        familiar_id="fam",
        member_resolver=member_resolver,
    )
    return responder, router, store


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

        async def bargein_mid_speech() -> float:
            # LLM stream is instant; playback starts immediately. Cancel 100ms in.
            await asyncio.sleep(0.1)
            loop = asyncio.get_running_loop()
            cancel_time = loop.time()
            await responder.handle(_mk_activity_start(turn_id="t-2"), bus)
            return cancel_time

        loop = asyncio.get_running_loop()
        task_reply = asyncio.create_task(
            responder.handle(_mk_final("go ahead", turn_id="t-1"), bus)
        )
        task_barge = asyncio.create_task(bargein_mid_speech())
        cancel_time = await task_barge
        await task_reply
        # handle() now spawns _on_final as a task — wait for the in-flight
        # speak to actually finish (i.e. observe cancellation) before timing.
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
