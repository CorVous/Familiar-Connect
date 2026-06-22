"""Tests for :class:`ActivityEngine`.

State machine idle → active → returning → idle, persisted via the
activities table. Deferred start applied at ``end_turn`` (FocusManager
precedent), return timer task (AlarmScheduler precedent), gate
decisions for TextResponder, cut-short via ``notify_reply_sent``,
idle-nudge eligibility. Fake clock throughout — no real sleeps.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

import familiar_connect.sleep.maintenance as maintenance_mod
from familiar_connect.activities.config import ActivitiesConfig, ActivityType
from familiar_connect.activities.engine import (
    ACTIVITY_RETURN_MODE,
    SLEEP_RETURN_MODE,
    ActivityEngine,
    GateAction,
    GateDecision,
)
from familiar_connect.bus.bus import InProcessEventBus
from familiar_connect.bus.protocols import BackpressurePolicy
from familiar_connect.bus.topics import TOPIC_DISCORD_TEXT
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author, is_self_key
from familiar_connect.sleep.maintenance import SleepPromptText
from familiar_connect.sleep.opinion_formation import OpinionFact, OpinionPlan

from .conftest import FakeLLMClient, build_fake_llm_clients
from .test_text_responder import (
    _CapturingSend,
    _discord_text_event,
    _make_responder,
    _ScriptedLLM,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from familiar_connect.llm import Message

_FAMILIAR = "aria"
_BOT_ID = 99
_CHANNEL = 42
_NOON = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)


class FakeClock:
    """Mutable now() for injection."""

    def __init__(self, start: datetime = _NOON) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


class FakeFocus:
    """Minimal focus stand-in: fixed text focus."""

    def __init__(self, channel_id: int | None = _CHANNEL) -> None:
        self.channel_id = channel_id

    def get_focus(self, modality: str) -> int | None:
        return self.channel_id if modality == "text" else None


class PresenceRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, status: str, text: str | None) -> None:
        self.calls.append((status, text))


def _config(**overrides: Any) -> ActivitiesConfig:  # noqa: ANN401
    walk = ActivityType(
        id="walk",
        label="creek walk",
        duration_minutes=(20, 40),
        reachable=True,
        seed="A walk along the creek behind the house.",
    )
    hatbox = ActivityType(
        id="hatbox",
        label="hatbox tending",
        duration_minutes=(10, 20),
        reachable=False,
        seed="Tending the hatbox.",
    )
    defaults: dict[str, Any] = {
        "catalog": (walk, hatbox),
        "archive_after_minutes": 45,
        "idle_nudge_minutes": 20,
        "min_gap_minutes": 90,
        "active_hours": (time(10, 0), time(23, 0)),
    }
    defaults.update(overrides)
    return ActivitiesConfig(**defaults)


def _author() -> Author:
    return Author(platform="discord", user_id="1", username="cor", display_name="Cor")


def _payload(content: str, *, channel_id: int = _CHANNEL) -> dict[str, Any]:
    return {
        "familiar_id": _FAMILIAR,
        "channel_id": channel_id,
        "guild_id": None,
        "author": _author(),
        "content": content,
        "message_id": "1001",
        "reply_to_message_id": None,
        "mentions": (),
        "images": {},
    }


@pytest.fixture
def store(tmp_path: Path) -> AsyncHistoryStore:
    return AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _engine(
    store: AsyncHistoryStore,
    clock: FakeClock,
    *,
    bus: InProcessEventBus | None = None,
    config: ActivitiesConfig | None = None,
    presence: Callable[[str, str | None], Awaitable[None] | None] | None = None,
    focused: int | None = _CHANNEL,
    experience: str = "I walked along the creek and watched dragonflies.",
    voice_active: bool = False,
    bot_user_id: Callable[[], int | None] = lambda: _BOT_ID,
    nudge_tick: float = 60.0,
    familiar_id: str = _FAMILIAR,
    sleep_passes_enabled: bool = False,
    seed_dream_path: Path | None = None,
    sleep_prompts: SleepPromptText | None = None,
    sleep_window: tuple[time, time] | None = (time(0, 0), time(8, 0)),
    sleep_grace_minutes: int = 30,
) -> ActivityEngine:
    kwargs: dict[str, Any] = {}
    if sleep_prompts is not None:
        kwargs["sleep_prompts"] = sleep_prompts
    return ActivityEngine(
        store=store,
        config=config or _config(),
        llm_clients=build_fake_llm_clients(
            per_slot_replies={"background": [experience]}
        ),
        bus=bus or InProcessEventBus(),
        focus_manager=FakeFocus(focused),
        presence_cb=presence or PresenceRecorder(),
        familiar_id=familiar_id,
        display_tz="UTC",
        sleep_window=sleep_window,
        sleep_grace_minutes=sleep_grace_minutes,
        bot_user_id=bot_user_id,
        voice_active_fn=lambda: voice_active,
        now_fn=clock,
        rng=random.Random(7),  # noqa: S311 — deterministic test roll
        nudge_tick_seconds=nudge_tick,
        sleep_passes_enabled=sleep_passes_enabled,
        seed_dream_path=seed_dream_path,
        **kwargs,
    )


async def _start_activity(
    engine: ActivityEngine, type_id: str = "walk", note: str | None = None
) -> dict[str, Any]:
    ack = engine.defer_start(type_id, note)
    await engine.end_turn()
    return ack


class TestDeferStart:
    @pytest.mark.asyncio
    async def test_unknown_type_returns_error(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        result = engine.defer_start("scuba", None)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_ack_carries_label_and_rolled_duration(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        ack = engine.defer_start("walk", "want fresh air")
        assert ack.get("ack") == "ok"
        assert ack["label"] == "creek walk"
        assert 20 <= ack["duration_minutes"] <= 40

    @pytest.mark.asyncio
    async def test_does_not_start_until_end_turn(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        engine.defer_start("walk", None)
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
        assert presence.calls == []

    @pytest.mark.asyncio
    async def test_refuses_while_staged_or_active(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        assert engine.defer_start("walk", None).get("ack") == "ok"
        assert "error" in engine.defer_start("hatbox", None)
        await engine.end_turn()
        assert "error" in engine.defer_start("hatbox", None)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_refuses_while_voice_subscribed(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock, voice_active=True)
        result = engine.defer_start("walk", None)
        assert "error" in result
        assert "voice" in result["error"]


class TestEndTurnAppliesStart:
    @pytest.mark.asyncio
    async def test_creates_row_sets_presence_arms_timer(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        ack = await _start_activity(engine, "walk", "fresh air")
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.type_id == "walk"
        assert row.note == "fresh air"
        expected_return = clock.now + timedelta(minutes=ack["duration_minutes"])
        assert row.planned_return_at == expected_return
        assert presence.calls == [("idle", "creek walk")]
        assert engine.return_timer_armed
        await engine.stop()

    @pytest.mark.asyncio
    async def test_unreachable_departure_sets_dnd_presence(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        await _start_activity(engine, "hatbox")
        assert presence.calls == [("dnd", "hatbox tending")]
        await engine.stop()

    @pytest.mark.asyncio
    async def test_end_turn_without_staged_is_noop(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        await engine.end_turn()
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
        assert presence.calls == []


class TestGate:
    @pytest.mark.asyncio
    async def test_normal_when_idle(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        decision = engine.gate(_payload("hello"))
        assert isinstance(decision, GateDecision)
        assert decision.action is GateAction.NORMAL

    @pytest.mark.asyncio
    async def test_suppress_non_ping_while_active(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        assert engine.gate(_payload("anyone around?")).action is GateAction.SUPPRESS
        await engine.stop()

    @pytest.mark.asyncio
    async def test_judgment_on_ping_when_reachable(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=20)
        decision = engine.gate(_payload(f"hey <@{_BOT_ID}> you there?"))
        assert decision.action is GateAction.JUDGMENT
        assert decision.state_line is not None
        assert "creek walk" in decision.state_line
        assert "20 min" in decision.state_line
        assert "Cor" in decision.state_line
        assert "silent()" in decision.state_line
        # eval finding: stay-out intent misroutes to start_activity —
        # state line must steer it back to silent()
        assert "do not call start_activity" in decision.state_line
        await engine.stop()

    @pytest.mark.asyncio
    async def test_nickname_mention_form_counts(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        decision = engine.gate(_payload(f"<@!{_BOT_ID}> ping"))
        assert decision.action is GateAction.JUDGMENT
        await engine.stop()

    @pytest.mark.asyncio
    async def test_bare_name_does_not_count(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        assert engine.gate(_payload("aria where are you")).action is (
            GateAction.SUPPRESS
        )
        await engine.stop()

    @pytest.mark.asyncio
    async def test_suppress_ping_when_unreachable(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine, "hatbox")
        decision = engine.gate(_payload(f"<@{_BOT_ID}> hello?"))
        assert decision.action is GateAction.SUPPRESS
        await engine.stop()

    @pytest.mark.asyncio
    async def test_pings_bot_flag_true_counts_without_mention_string(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Reply-ping case: no ``<@id>`` in content, flag carries the ping."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        payload = _payload("you still there?")
        payload["pings_bot"] = True
        decision = engine.gate(payload)
        assert decision.action is GateAction.JUDGMENT
        assert decision.state_line is not None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_pings_bot_flag_false_overrides_content(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Flag authoritative when present — content scan is fallback only."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        payload = _payload(f"<@{_BOT_ID}> hello?")
        payload["pings_bot"] = False
        assert engine.gate(payload).action is GateAction.SUPPRESS
        await engine.stop()

    @pytest.mark.asyncio
    async def test_missing_flag_falls_back_to_content_scan(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Synthetic events / tests omit the flag — raw ``<@id>`` scan applies."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        payload = _payload(f"<@{_BOT_ID}> hello?")
        assert "pings_bot" not in payload
        assert engine.gate(payload).action is GateAction.JUDGMENT
        await engine.stop()

    @pytest.mark.asyncio
    async def test_alarm_payload_normal_while_out_reachable(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Her own alarm pierces the absence — never suppressed or judged."""
        engine = _engine(store, clock)
        await _start_activity(engine, "walk")
        payload = _payload("[alarm fired: check the tea]")
        payload["author"] = None
        payload["alarm"] = True
        assert engine.gate(payload).action is GateAction.NORMAL
        await engine.stop()

    @pytest.mark.asyncio
    async def test_alarm_payload_normal_while_out_unreachable(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine, "hatbox")
        payload = _payload("[alarm fired: check the tea]")
        payload["author"] = None
        payload["alarm"] = True
        assert engine.gate(payload).action is GateAction.NORMAL
        await engine.stop()

    @pytest.mark.asyncio
    async def test_unfocused_ping_suppressed(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Real ping off the focused channel ⇒ SUPPRESS, never JUDGMENT.

        Responder's suppressed path notes it for the return wake.
        """
        engine = _engine(store, clock)
        await _start_activity(engine)
        decision = engine.gate(_payload(f"<@{_BOT_ID}> hey", channel_id=_CHANNEL + 1))
        assert decision.action is GateAction.SUPPRESS
        await engine.stop()

    @pytest.mark.asyncio
    async def test_ping_with_no_focus_suppressed(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock, focused=None)
        await _start_activity(engine)
        decision = engine.gate(_payload(f"<@{_BOT_ID}> hey"))
        assert decision.action is GateAction.SUPPRESS
        await engine.stop()

    @pytest.mark.asyncio
    async def test_second_ping_same_author_suppressed(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """S2 latch: one judgment per author per absence."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        ping = _payload(f"<@{_BOT_ID}> you there?")
        assert engine.gate(ping).action is GateAction.JUDGMENT
        assert engine.gate(ping).action is GateAction.SUPPRESS
        await engine.stop()

    @pytest.mark.asyncio
    async def test_second_ping_different_author_judged(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        assert engine.gate(_payload(f"<@{_BOT_ID}> hi")).action is (GateAction.JUDGMENT)
        other = _payload(f"<@{_BOT_ID}> hi")
        other["author"] = Author(
            platform="discord", user_id="2", username="mia", display_name="Mia"
        )
        assert engine.gate(other).action is GateAction.JUDGMENT
        await engine.stop()

    @pytest.mark.asyncio
    async def test_judgment_latch_clears_at_next_activity_start(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        ping = _payload(f"<@{_BOT_ID}> you there?")
        assert engine.gate(ping).action is GateAction.JUDGMENT
        clock.advance(minutes=5)
        await engine.notify_reply_sent()
        await _start_activity(engine)
        assert engine.gate(ping).action is GateAction.JUDGMENT
        await engine.stop()

    @pytest.mark.asyncio
    async def test_state_line_truncates_long_author_name(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """S4: pathological display names must not balloon the state line."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        long_name = "N" * 80
        ping = _payload(f"<@{_BOT_ID}> hey")
        ping["author"] = Author(
            platform="discord", user_id="3", username="n", display_name=long_name
        )
        decision = engine.gate(ping)
        assert decision.action is GateAction.JUDGMENT
        assert decision.state_line is not None
        assert long_name not in decision.state_line
        assert "N" * 40 in decision.state_line  # truncated head survives
        await engine.stop()

    @pytest.mark.asyncio
    async def test_normal_while_returning(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """C1: mid-return corpse state must not suppress arriving events."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        engine._returning = True  # mid-return snapshot
        assert engine.gate(_payload("hello")).action is GateAction.NORMAL
        engine._returning = False
        await engine.stop()


class TestReturnFlow:
    @pytest.mark.asyncio
    async def test_cut_short_finishes_row_and_restores_presence(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        await _start_activity(engine)
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
        assert presence.calls[-1] == ("online", None)
        assert not engine.return_timer_armed
        await engine.stop()

    @pytest.mark.asyncio
    async def test_presence_failure_never_breaks_turn_flow(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Presence cb failures must be swallowed.

        Live incident 2026-06-12: change_presence hit a closing Discord
        websocket mid-reconnect; the error escaped end_turn and killed
        the run TaskGroup.
        """

        def _boom(status: str, text: str | None) -> None:  # noqa: ARG001
            raise ConnectionResetError

        engine = _engine(store, clock, presence=_boom)
        await _start_activity(engine)  # departure presence flip raises
        assert engine.active is not None  # departure still completed
        assert engine.return_timer_armed  # timer armed despite cb failure
        clock.advance(minutes=10)
        await engine.notify_reply_sent()  # return presence flip raises too
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_cut_short_keeps_nudge_loop_armed(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """First cut-short must not kill idle nudges for the process lifetime."""
        engine = _engine(store, clock)
        await engine.start()
        await _start_activity(engine)
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        assert engine.nudge_loop_armed
        await engine.stop()

    @pytest.mark.asyncio
    async def test_experience_turn_marked_and_persisted(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock, experience="The creek was high after the rain.")
        await _start_activity(engine)
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert len(marked) == 1
        # her own narration — assistant role, not system authority
        assert marked[0].role == "assistant"
        assert marked[0].content.startswith("[returned from creek walk]")
        assert "The creek was high after the rain." in marked[0].content
        await engine.stop()

    @pytest.mark.asyncio
    async def test_return_turn_carries_activity_return_mode(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Extractor skip keys on ``turns.mode``, not the display prefix."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert len(marked) == 1
        assert marked[0].mode == ACTIVITY_RETURN_MODE
        assert marked[0].mode == "activity_return"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_event_fact_written_mechanically(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        facts = store.sync.recent_facts(familiar_id=_FAMILIAR, limit=5)
        assert len(facts) == 1
        assert "creek walk" in facts[0].text
        # spent-<when>-<label> phrasing composes with phrase labels
        assert "spent" in facts[0].text
        assert "went on a" not in facts[0].text
        # date + daypart in display_tz (UTC noon → afternoon)
        assert "Jun 12" in facts[0].text
        assert "afternoon" in facts[0].text
        await engine.stop()

    @pytest.mark.asyncio
    async def test_long_absence_sets_archive_watermark_at_departure(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        departure_turn = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="see you",
            author=_author(),
        )
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=60)
        await engine.notify_reply_sent()
        mark = store.sync.get_archive_watermark(
            familiar_id=_FAMILIAR, channel_id=_CHANNEL
        )
        assert mark == departure_turn.id
        await engine.stop()

    @pytest.mark.asyncio
    async def test_long_absence_archives_all_channels(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Archive break applies to every channel, not just focused.

        Absence is global — she leaves the whole screen.
        """
        other_channel = _CHANNEL + 1
        other_turn = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=other_channel,
            role="user",
            content="art link chatter",
            author=_author(),
        )
        departure_turn = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="see you",
            author=_author(),
        )
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=90)
        await engine.notify_reply_sent()
        # other channel's pre-departure turn behind the break too
        assert (
            store.sync.get_archive_watermark(
                familiar_id=_FAMILIAR, channel_id=other_channel
            )
            == departure_turn.id
        )
        window = store.sync.recent_cross_channel(
            familiar_id=_FAMILIAR, limit=50, respect_archive=True
        )
        ids = {t.id for t in window}
        assert other_turn.id not in ids
        assert departure_turn.id not in ids
        # the return-experience turn is post-departure — visible
        assert any(t.mode == "activity_return" for t in window)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_short_absence_leaves_no_watermark(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="see you",
            author=_author(),
        )
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        mark = store.sync.get_archive_watermark(
            familiar_id=_FAMILIAR, channel_id=_CHANNEL
        )
        assert mark is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_missed_ping_publishes_wake_event(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus)
        await _start_activity(engine)
        clock.advance(minutes=10)
        store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content=f"<@{_BOT_ID}> where did you go?",
            author=_author(),
        )
        store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="probably out again",
            author=_author(),
        )
        clock.advance(minutes=10)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.notify_reply_sent()
        try:
            event = await asyncio.wait_for(anext(sub), timeout=1.0)
            assert event.topic == TOPIC_DISCORD_TEXT
            payload = event.payload
            assert payload["channel_id"] == _CHANNEL
            assert payload["author"] is None
            assert "creek walk" in payload["content"]
            assert "where did you go?" in payload["content"]
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_no_pings_no_wake_event(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus)
        await _start_activity(engine)
        store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="just chatting",
            author=_author(),
        )
        clock.advance(minutes=10)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.notify_reply_sent()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(anext(sub), timeout=0.2)
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_past_due_reload_arms_floor_timer_not_inline_return(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """C3: past-due reload must NOT return inline at boot.

        Bus consumers + Discord login don't exist yet — the return
        timer is armed at now+floor so the wake lands on a live stack.
        """
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        armed: list[datetime] = []
        original = engine._arm_return_timer

        def _spy(planned: datetime) -> None:
            armed.append(planned)
            original(planned)

        monkeypatch.setattr(engine, "_arm_return_timer", _spy)
        store.sync.create_activity(
            familiar_id=_FAMILIAR,
            type_id="walk",
            label="creek walk",
            started_at=clock.now - timedelta(minutes=30),
            planned_return_at=clock.now - timedelta(minutes=5),
            note=None,
        )
        await engine.start()
        # no inline return — row still active, timer armed at floor
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is not None
        assert engine.return_timer_armed
        assert armed == [clock.now + timedelta(seconds=20)]
        # away presence re-established, but no inline return flip to online
        assert presence.calls == [("idle", "creek walk")]
        # still absent until the floor timer fires
        assert engine.gate(_payload("hello")).action is GateAction.SUPPRESS
        await engine.stop()


class TestRestart:
    @pytest.mark.asyncio
    async def test_start_rearms_future_return(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        store.sync.create_activity(
            familiar_id=_FAMILIAR,
            type_id="walk",
            label="creek walk",
            started_at=clock.now - timedelta(minutes=5),
            planned_return_at=clock.now + timedelta(minutes=25),
            note=None,
        )
        engine = _engine(store, clock)
        await engine.start()
        assert engine.return_timer_armed
        # gate still suppresses while reloaded activity is active
        assert engine.gate(_payload("hello")).action is GateAction.SUPPRESS
        await engine.stop()

    @pytest.mark.asyncio
    async def test_start_restores_idle_presence_for_reachable_row(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        store.sync.create_activity(
            familiar_id=_FAMILIAR,
            type_id="walk",
            label="creek walk",
            started_at=clock.now - timedelta(minutes=5),
            planned_return_at=clock.now + timedelta(minutes=25),
            note=None,
        )
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        await engine.start()
        assert presence.calls == [("idle", "creek walk")]
        await engine.stop()

    @pytest.mark.asyncio
    async def test_start_restores_dnd_presence_for_unreachable_row(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        store.sync.create_activity(
            familiar_id=_FAMILIAR,
            type_id="hatbox",
            label="hatbox tending",
            started_at=clock.now - timedelta(minutes=5),
            planned_return_at=clock.now + timedelta(minutes=15),
            note=None,
        )
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        await engine.start()
        assert presence.calls == [("dnd", "hatbox tending")]
        await engine.stop()

    @pytest.mark.asyncio
    async def test_resync_presence_reissues_away_for_active_row(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Post-ready resync — boot-reload away call was dropped pre-ready."""
        store.sync.create_activity(
            familiar_id=_FAMILIAR,
            type_id="hatbox",
            label="hatbox tending",
            started_at=clock.now - timedelta(minutes=5),
            planned_return_at=clock.now + timedelta(minutes=15),
            note=None,
        )
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        await engine.start()
        presence.calls.clear()  # isolate the resync re-issue
        await engine.resync_presence()
        assert presence.calls == [("dnd", "hatbox tending")]
        await engine.stop()

    @pytest.mark.asyncio
    async def test_resync_presence_noop_when_idle(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        presence = PresenceRecorder()
        engine = _engine(store, clock, presence=presence)
        await engine.start()
        await engine.resync_presence()
        assert presence.calls == []
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_timer(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        assert engine.return_timer_armed
        await engine.stop()
        assert not engine.return_timer_armed
        # row remains active in DB — restart-safe
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is not None


class TestShouldNudge:
    @pytest.mark.asyncio
    async def test_eligible_when_quiet_and_inside_hours(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        clock.advance(minutes=30)
        assert engine.should_nudge(clock.now) is True

    @pytest.mark.asyncio
    async def test_not_eligible_while_active(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=30)
        assert engine.should_nudge(clock.now) is False
        await engine.stop()

    @pytest.mark.asyncio
    async def test_not_eligible_when_channel_recently_active(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        clock.advance(minutes=30)
        engine.note_traffic()
        clock.advance(minutes=5)
        assert engine.should_nudge(clock.now) is False

    @pytest.mark.asyncio
    async def test_min_gap_since_last_activity(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        clock.advance(minutes=30)  # quiet ≥ idle_nudge, but gap < 90
        assert engine.should_nudge(clock.now) is False
        clock.advance(minutes=70)  # gap now 100 ≥ 90
        assert engine.should_nudge(clock.now) is True
        await engine.stop()

    @pytest.mark.asyncio
    async def test_outside_active_hours(self, store: AsyncHistoryStore) -> None:
        night = FakeClock(datetime(2026, 6, 12, 3, 0, tzinfo=UTC))
        engine = _engine(store, night)
        night.advance(minutes=30)
        assert engine.should_nudge(night.now) is False

    @pytest.mark.asyncio
    async def test_wrapped_active_hours(self, store: AsyncHistoryStore) -> None:
        config = _config(active_hours=(time(22, 0), time(2, 0)))
        late = FakeClock(datetime(2026, 6, 12, 23, 30, tzinfo=UTC))
        engine = _engine(store, late, config=config)
        late.advance(minutes=30)
        assert engine.should_nudge(late.now) is True

    @pytest.mark.asyncio
    async def test_debounce_after_mark_nudge_pending(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        clock.advance(minutes=30)
        assert engine.should_nudge(clock.now) is True
        engine.mark_nudge_pending()
        assert engine.should_nudge(clock.now) is False


class TestNoteMissedPing:
    """Live-gated ping capture — merged with content scan at return."""

    @pytest.mark.asyncio
    async def test_reply_ping_without_mention_string_reaches_wake(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Content scan can't see reply-pings; live note carries them."""
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus)
        await _start_activity(engine)
        clock.advance(minutes=10)
        turn = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="you still there?",  # no <@id> — reply-ping shape
            author=_author(),
        )
        engine.note_missed_ping(turn.id)
        clock.advance(minutes=10)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.notify_reply_sent()
        try:
            event = await asyncio.wait_for(anext(sub), timeout=1.0)
            assert "you still there?" in event.payload["content"]
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_live_note_dedupes_with_content_scan(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus)
        await _start_activity(engine)
        clock.advance(minutes=10)
        turn = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content=f"<@{_BOT_ID}> where did you go?",
            author=_author(),
        )
        engine.note_missed_ping(turn.id)
        engine.note_missed_ping(turn.id)  # double note — idempotent
        clock.advance(minutes=10)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.notify_reply_sent()
        try:
            event = await asyncio.wait_for(anext(sub), timeout=1.0)
            assert event.payload["content"].count("where did you go?") == 1
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_cross_channel_ping_reaches_wake(
        self, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Unfocused real ping, end to end through a real TextResponder.

        Gate suppresses it (not focused ⇒ no judgment), responder
        stages the turn and notes the ping, return wake surfaces it
        at the focused channel.
        """
        sync_store = HistoryStore(tmp_path / "history.db")
        astore = AsyncHistoryStore(sync_store)
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(astore, clock, bus=bus, familiar_id="fam")
        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=_ScriptedLLM(deltas=["should never stream"]),
            send=send,
            tmp_path=tmp_path,
            store=sync_store,
            activity_engine=engine,
        )
        await _start_activity(engine)
        clock.advance(minutes=10)
        await responder.handle(
            _discord_text_event(
                channel_id=77, content="come look at this", pings_bot=True
            ),
            bus,
        )
        # unfocused ping while out: suppressed — no judgment reply
        assert send.calls == []
        clock.advance(minutes=10)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.notify_reply_sent()
        try:
            event = await asyncio.wait_for(anext(sub), timeout=1.0)
            # wake lands at the focused channel, content carries the ping
            assert event.payload["channel_id"] == _CHANNEL
            assert "come look at this" in event.payload["content"]
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_cleared_at_activity_start(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        """Stale ids from before departure never leak into the wake."""
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus)
        turn = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content="old ping",
            author=_author(),
        )
        engine.note_missed_ping(turn.id)
        await _start_activity(engine)
        clock.advance(minutes=10)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.notify_reply_sent()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(anext(sub), timeout=0.2)
        finally:
            await engine.stop()
            await bus.shutdown()


class TestBotUserIdProvider:
    """Late-bound bot user id — run.py wires it before Discord login."""

    @pytest.mark.asyncio
    async def test_callable_resolves_at_gate_time(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        holder: dict[str, int | None] = {"id": None}
        engine = _engine(store, clock, bot_user_id=lambda: holder["id"])
        await _start_activity(engine)
        ping = _payload(f"<@{_BOT_ID}> hey")
        # id unknown yet (pre-ready) — content scan can't match
        assert engine.gate(ping).action is GateAction.SUPPRESS
        holder["id"] = _BOT_ID
        assert engine.gate(ping).action is GateAction.JUDGMENT
        await engine.stop()


class TestNudgeLoop:
    """Engine-owned periodic nudge — publishes wake when channel quiet."""

    @pytest.mark.asyncio
    async def test_publishes_wake_into_focused_channel_when_quiet(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus, nudge_tick=0.01)
        clock.advance(minutes=30)  # quiet ≥ idle_nudge_minutes
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.start()
        try:
            event = await asyncio.wait_for(anext(sub), timeout=2.0)
            payload = event.payload
            assert payload["channel_id"] == _CHANNEL
            assert payload["familiar_id"] == _FAMILIAR
            assert payload["author"] is None
            assert "quiet" in payload["content"]
            assert "start_activity" in payload["content"]
            # debounced — next tick must not fire again
            assert engine.should_nudge(clock.now) is False
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_no_nudge_while_traffic_fresh(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus, nudge_tick=0.01)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.start()  # quiet clock fresh — ineligible
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(anext(sub), timeout=0.2)
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_start_arms_and_stop_cancels_loop(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        engine = _engine(store, clock)
        assert engine.nudge_loop_armed is False
        await engine.start()
        assert engine.nudge_loop_armed is True
        await engine.stop()
        assert engine.nudge_loop_armed is False


async def _raise_runtime_error(*args: object, **kwargs: object) -> None:  # noqa: RUF029 — async to stand in for store/bus coroutine methods
    del args, kwargs
    raise RuntimeError


class TestEndTurnHardening:
    """F1: ``end_turn`` must never raise into a responder turn."""

    @pytest.mark.asyncio
    async def test_end_turn_never_raises_when_create_fails(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _engine(store, clock)
        monkeypatch.setattr(store, "create_activity", _raise_runtime_error)
        engine.defer_start("walk", None)
        await engine.end_turn()  # must not raise
        # coherent state: she never left, staged dropped, next defer clean
        assert engine.active is None
        assert not engine.return_timer_armed
        assert engine.defer_start("walk", None).get("ack") == "ok"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_end_turn_partial_failure_still_arms_return_timer(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Row committed ⇒ she left; timer must still bring her back."""
        engine = _engine(store, clock)
        monkeypatch.setattr(store, "latest_id", _raise_runtime_error)
        engine.defer_start("walk", None)
        await engine.end_turn()  # must not raise
        assert engine.active is not None
        assert engine.return_timer_armed
        await engine.stop()


class TestReturnHardening:
    """F1: return flow commits first; later steps are best-effort."""

    @pytest.mark.asyncio
    async def test_return_commits_even_when_return_turn_write_fails(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=10)
        monkeypatch.setattr(store, "append_turn", _raise_runtime_error)
        await engine.notify_reply_sent()  # must not raise
        # finish_activity ran before the failing step — she is back
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
        assert engine.active is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_event_fact_failure_does_not_kill_remaining_steps(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus)
        await _start_activity(engine)
        clock.advance(minutes=10)
        store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content=f"<@{_BOT_ID}> where did you go?",
            author=_author(),
        )
        monkeypatch.setattr(store, "append_fact", _raise_runtime_error)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        await engine.notify_reply_sent()  # must not raise
        try:
            assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
            # return turn still written
            turns = store.sync.recent(
                familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10
            )
            assert any(t.content.startswith("[returned from") for t in turns)
            # wake still published
            event = await asyncio.wait_for(anext(sub), timeout=1.0)
            assert "where did you go?" in event.payload["content"]
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_finish_activity_failure_still_leaves_her_back(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Commit failing must not strand her absent in memory."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        clock.advance(minutes=10)
        monkeypatch.setattr(store, "finish_activity", _raise_runtime_error)
        await engine.notify_reply_sent()  # must not raise
        assert engine.active is None
        assert engine.gate(_payload("hello")).action is GateAction.NORMAL
        # DB row stays active — restart replays the return
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is not None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_wake_publish_failure_still_restores_presence(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bus = InProcessEventBus()
        await bus.start()
        presence = PresenceRecorder()
        engine = _engine(store, clock, bus=bus, presence=presence)
        await _start_activity(engine)
        clock.advance(minutes=10)
        store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=_CHANNEL,
            role="user",
            content=f"<@{_BOT_ID}> hello?",
            author=_author(),
        )
        monkeypatch.setattr(bus, "publish", _raise_runtime_error)
        await engine.notify_reply_sent()  # must not raise
        assert presence.calls[-1] == ("online", None)
        assert engine.active is None
        await engine.stop()
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_sleep_then_return_swallows_exceptions(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No orphan task exceptions from the return timer."""
        engine = _engine(store, clock)
        await _start_activity(engine)
        await engine._cancel_return_timer()  # drive return manually
        monkeypatch.setattr(engine, "_run_return", _raise_runtime_error)
        # past-due ⇒ no sleep; failure must be logged, not raised
        await engine._sleep_then_return(clock.now)
        await engine.stop()


class TestStagedPromotionAtReturn:
    """F3 call site: she reads the screen when she gets back."""

    @pytest.mark.asyncio
    async def test_absence_chatter_promoted_at_return(
        self, store: AsyncHistoryStore, clock: FakeClock
    ) -> None:
        other = _CHANNEL + 1
        engine = _engine(store, clock)
        # pre-absence staged turn keeps its attentional semantics
        before = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=other,
            role="user",
            content="old unread",
            author=_author(),
            consumed=False,
        )
        await _start_activity(engine)
        clock.advance(minutes=10)
        during = store.sync.append_turn(
            familiar_id=_FAMILIAR,
            channel_id=other,
            role="user",
            content="chatter while she was out",
            author=_author(),
            consumed=False,
        )
        clock.advance(minutes=10)
        await engine.notify_reply_sent()
        window = store.sync.recent_cross_channel(familiar_id=_FAMILIAR, limit=50)
        ids = {t.id for t in window}
        assert during.id in ids  # promoted — visible at return
        assert before.id not in ids  # pre-absence stays staged
        await engine.stop()


class TestResponderInLoop:
    """Real TextResponder consuming the engine's bus traffic.

    Pins C1: the return wake must gate NORMAL even when the responder
    handles it while the engine is still inside ``_run_return``.
    """

    @pytest.mark.asyncio
    async def test_return_wake_gets_normal_reply_mid_return(
        self, clock: FakeClock, tmp_path: Path
    ) -> None:
        sync_store = HistoryStore(tmp_path / "history.db")
        astore = AsyncHistoryStore(sync_store)
        bus = InProcessEventBus()
        await bus.start()
        handled = asyncio.Event()
        prompts: list[list[Message]] = []

        class _CapturingScriptedLLM(_ScriptedLLM):
            """Record prompts — corpse state shows up as a judgment line."""

            async def chat_stream(  # type: ignore[override]
                self, messages: list[Message]
            ) -> AsyncIterator[str]:
                prompts.append(list(messages))
                async for delta in super().chat_stream(messages):
                    yield delta

        async def _presence(status: str, text: str | None) -> None:
            del text
            if status == "online":
                # hold the engine inside _run_return until the
                # responder has consumed the wake — the "responder
                # sees corpse absence state" race is deterministic
                await asyncio.wait_for(handled.wait(), timeout=2.0)

        engine = _engine(astore, clock, bus=bus, presence=_presence, familiar_id="fam")
        send = _CapturingSend()
        responder, _, _ = _make_responder(
            llm=_CapturingScriptedLLM(deltas=["good to be back!"]),
            send=send,
            tmp_path=tmp_path,
            store=sync_store,
            activity_engine=engine,
        )
        await _start_activity(engine)
        clock.advance(minutes=10)
        sync_store.append_turn(
            familiar_id="fam",
            channel_id=_CHANNEL,
            role="user",
            content=f"<@{_BOT_ID}> where did you go?",
            author=_author(),
        )
        clock.advance(minutes=10)
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)

        async def _consume_wake() -> None:
            event = await anext(sub)
            await responder.handle(event, bus)
            handled.set()

        consumer = asyncio.create_task(_consume_wake())
        try:
            await asyncio.wait_for(engine.notify_reply_sent(), timeout=5.0)
            await asyncio.wait_for(consumer, timeout=2.0)
            # wake produced a real reply — not a suppressed/staged drop
            assert send.calls, "return wake suppressed by corpse absence state"
            turns = sync_store.recent(familiar_id="fam", channel_id=_CHANNEL, limit=20)
            assert any(
                t.role == "assistant" and t.content == "good to be back!" for t in turns
            )
            # NORMAL turn, not a judgment off the corpse state: no
            # engine state line may leak into the prompt
            assert prompts
            assert all(
                "Replying means heading back" not in m.content_str for m in prompts[0]
            )
            # wake turn consumed, not staged
            wake_turns = [
                t
                for t in turns
                if t.role == "user" and "missed pings while away" in t.content
            ]
            assert wake_turns
            assert all(t.consumed_at is not None for t in wake_turns)
        finally:
            consumer.cancel()
            await engine.stop()
            await bus.shutdown()


def _sleep_type() -> ActivityType:
    return ActivityType(
        id="sleep",
        label="asleep",
        duration_minutes=None,
        reachable=False,
        seed="The night's dream, retold on waking.",
    )


def _sleep_config() -> ActivitiesConfig:
    base = _config()
    return _config(catalog=(*base.catalog, _sleep_type()))


def _night_clock(hour: int, minute: int = 0, day: int = 13) -> FakeClock:
    return FakeClock(datetime(2026, 6, day, hour, minute, tzinfo=UTC))


class TestSleepSchedule:
    """Behavioral sleep window — bedtime nudge, grace backstop, fixed wake."""

    @pytest.mark.asyncio
    async def test_defer_start_sleep_returns_at_window_end(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(0, 10)
        engine = _engine(store, clock, config=_sleep_config())
        ack = engine.defer_start("sleep", None)
        assert ack.get("ack") == "ok"
        assert ack["duration_minutes"] == 470  # 00:10 → 08:00
        await engine.end_turn()
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.planned_return_at == datetime(2026, 6, 13, 8, 0, tzinfo=UTC)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_arms_from_character_config_window(
        self, store: AsyncHistoryStore
    ) -> None:
        """Window sourced from ctor (character config), not the catalog entry.

        Sleep entry stays in the catalog for identification; the schedule
        force-sleeps past grace using the ctor-supplied window/grace.
        """
        clock = _night_clock(0, 31)  # past 00:00 + 30 grace
        engine = _engine(
            store,
            clock,
            config=_sleep_config(),
            sleep_window=(time(0, 0), time(8, 0)),
            sleep_grace_minutes=30,
        )
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.type_id == "sleep"
        assert row.planned_return_at == datetime(2026, 6, 13, 8, 0, tzinfo=UTC)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_no_window_disarms_schedule(
        self, store: AsyncHistoryStore
    ) -> None:
        """``sleep_window=None`` ⇒ schedule never fires even with a sleep entry."""
        clock = _night_clock(0, 31)
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_window=None
        )
        await engine._sleep_schedule_tick(clock.now)
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_bedtime_nudge_published_once_per_occurrence(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(0, 5)
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus, config=_sleep_config())
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        try:
            await engine._sleep_schedule_tick(clock.now)
            event = await asyncio.wait_for(anext(sub), timeout=1.0)
            assert "sleep" in event.payload["content"]
            assert "start_activity" in event.payload["content"]
            # nudge only — nothing force-started pre-grace
            assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
            # debounced — same occurrence never nudges twice
            clock.advance(minutes=5)
            await engine._sleep_schedule_tick(clock.now)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(anext(sub), timeout=0.2)
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_force_sleep_after_grace(self, store: AsyncHistoryStore) -> None:
        clock = _night_clock(0, 31)
        presence = PresenceRecorder()
        engine = _engine(store, clock, config=_sleep_config(), presence=presence)
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.type_id == "sleep"
        assert row.planned_return_at == datetime(2026, 6, 13, 8, 0, tzinfo=UTC)
        assert presence.calls == [("dnd", "asleep")]
        assert engine.return_timer_armed
        await engine.stop()

    @pytest.mark.asyncio
    async def test_fixed_wake_regardless_of_start_time(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(3, 0)
        engine = _engine(store, clock, config=_sleep_config())
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.planned_return_at == datetime(2026, 6, 13, 8, 0, tzinfo=UTC)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_nothing_happens_outside_window(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(12, 0)
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus, config=_sleep_config())
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        try:
            await engine._sleep_schedule_tick(clock.now)
            assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(anext(sub), timeout=0.2)
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_already_slept_this_window_no_refire(
        self, store: AsyncHistoryStore
    ) -> None:
        """Cut-short or completed sleep STARTED this window blocks re-entry."""
        clock = _night_clock(1, 0)
        slept_id = store.sync.create_activity(
            familiar_id=_FAMILIAR,
            type_id="sleep",
            label="asleep",
            started_at=datetime(2026, 6, 13, 0, 5, tzinfo=UTC),
            planned_return_at=datetime(2026, 6, 13, 8, 0, tzinfo=UTC),
            note=None,
        )
        store.sync.finish_activity(
            activity_id=slept_id,
            status="completed",
            actual_return_at=datetime(2026, 6, 13, 0, 50, tzinfo=UTC),
            experience_text=None,
        )
        bus = InProcessEventBus()
        await bus.start()
        engine = _engine(store, clock, bus=bus, config=_sleep_config())
        sub = bus.subscribe((TOPIC_DISCORD_TEXT,), policy=BackpressurePolicy.UNBOUNDED)
        try:
            await engine._sleep_schedule_tick(clock.now)
            assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(anext(sub), timeout=0.2)
        finally:
            await engine.stop()
            await bus.shutdown()

    @pytest.mark.asyncio
    async def test_prior_night_sleep_does_not_block_tonight(
        self, store: AsyncHistoryStore
    ) -> None:
        old_id = store.sync.create_activity(
            familiar_id=_FAMILIAR,
            type_id="sleep",
            label="asleep",
            started_at=datetime(2026, 6, 12, 0, 10, tzinfo=UTC),
            planned_return_at=datetime(2026, 6, 12, 8, 0, tzinfo=UTC),
            note=None,
        )
        store.sync.finish_activity(
            activity_id=old_id,
            status="completed",
            actual_return_at=datetime(2026, 6, 12, 8, 0, tzinfo=UTC),
            experience_text=None,
        )
        clock = _night_clock(0, 45)
        engine = _engine(store, clock, config=_sleep_config())
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.type_id == "sleep"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_backstop_waits_while_out_on_other_activity(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(0, 0)
        engine = _engine(store, clock, config=_sleep_config())
        await _start_activity(engine, "walk")
        clock.advance(minutes=45)  # past grace, but she's out
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.type_id == "walk"
        await engine.notify_reply_sent()  # back from the walk
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.type_id == "sleep"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_wrapped_window_evening_side(self, store: AsyncHistoryStore) -> None:
        clock = _night_clock(23, 45, day=12)
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_window=(time(23, 0), time(7, 0))
        )
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.planned_return_at == datetime(2026, 6, 13, 7, 0, tzinfo=UTC)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_wrapped_window_morning_side(self, store: AsyncHistoryStore) -> None:
        """Occurrence started yesterday 23:00; 01:00 is past grace."""
        clock = _night_clock(1, 0, day=13)
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_window=(time(23, 0), time(7, 0))
        )
        await engine._sleep_schedule_tick(clock.now)
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.planned_return_at == datetime(2026, 6, 13, 7, 0, tzinfo=UTC)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_boot_mid_window_first_tick_force_sleeps(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(1, 0)
        engine = _engine(store, clock, config=_sleep_config(), nudge_tick=0.01)
        await engine.start()
        try:
            for _ in range(200):
                if store.sync.active_activity(familiar_id=_FAMILIAR) is not None:
                    break
                await asyncio.sleep(0.01)
            row = store.sync.active_activity(familiar_id=_FAMILIAR)
            assert row is not None
            assert row.type_id == "sleep"
        finally:
            await engine.stop()


def _opinion_plan(*texts: str) -> OpinionPlan:
    opinions = tuple(
        OpinionFact(
            text=t,
            source_turn_ids=(1,),
            valid_from_date="2026-06-12",
            self_grounded=True,
            importance=5,
        )
        for t in texts
    )
    return OpinionPlan(
        familiar_id=_FAMILIAR,
        opinions=opinions,
        rejected=(),
        flags=(),
        new_last_turn_id=5,
    )


class TestSleepLifecyclePasses:
    """Sleep departure fires consolidation then opinions (apply=True)."""

    @pytest.mark.asyncio
    async def test_consolidation_then_opinion_applied_on_departure(
        self,
        store: AsyncHistoryStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clock = _night_clock(0, 45)
        calls: list[tuple[str, dict[str, Any]]] = []
        cplan = SimpleNamespace(retire=[], rewrite=[], mutated_count=2)
        dplan = _opinion_plan("Rain is best heard from indoors.")

        async def fake_consolidation(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            calls.append(("consolidation", kw))
            return cplan

        async def fake_opinion(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            calls.append(("opinion", kw))
            return dplan

        monkeypatch.setattr(
            maintenance_mod, "execute_consolidation", fake_consolidation
        )
        monkeypatch.setattr(maintenance_mod, "execute_opinion_formation", fake_opinion)
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_passes_enabled=True
        )
        await engine._sleep_schedule_tick(clock.now)
        assert engine._sleep_passes_task is not None
        await engine._sleep_passes_task
        assert [c[0] for c in calls] == ["consolidation", "opinion"]
        assert calls[0][1]["apply"] is True
        assert calls[1][1]["apply"] is True
        assert calls[1][1]["display_tz"] == "UTC"
        # opinion plan now flows straight into prose at pass completion —
        # formed opinion reaches the prose-gen prompt, prose persisted
        llm = cast("FakeLLMClient", engine._llm_clients["background"])
        assert any(
            "Rain is best heard from indoors." in c[-1].content_str for c in llm.calls
        )
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.experience_text is not None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_consolidation_denylist_threaded_into_opinion(
        self,
        store: AsyncHistoryStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clock = _night_clock(0, 45)
        fact = store.sync.append_fact(
            familiar_id=_FAMILIAR,
            channel_id=None,
            text="a known bit, retired tonight",
            source_turn_ids=(),
        )
        cplan = SimpleNamespace(
            retire=[SimpleNamespace(fact_ids=[fact.id])], rewrite=[], mutated_count=1
        )
        seen: dict[str, Any] = {}

        async def fake_consolidation(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            return cplan

        async def fake_opinion(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            seen.update(kw)
            return _opinion_plan()

        monkeypatch.setattr(
            maintenance_mod, "execute_consolidation", fake_consolidation
        )
        monkeypatch.setattr(maintenance_mod, "execute_opinion_formation", fake_opinion)
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_passes_enabled=True
        )
        await engine._sleep_schedule_tick(clock.now)
        assert engine._sleep_passes_task is not None
        await engine._sleep_passes_task
        assert seen["denylist"] == ("a known bit, retired tonight",)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_configured_sleep_prompts_thread_into_passes(
        self,
        store: AsyncHistoryStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Engine-held config prompt text reaches the execute kwargs."""
        clock = _night_clock(0, 45)
        seen_c: dict[str, Any] = {}
        seen_o: dict[str, Any] = {}

        async def fake_consolidation(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            seen_c.update(kw)
            return SimpleNamespace(retire=[], rewrite=[], mutated_count=0)

        async def fake_opinion(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            seen_o.update(kw)
            return _opinion_plan()

        monkeypatch.setattr(
            maintenance_mod, "execute_consolidation", fake_consolidation
        )
        monkeypatch.setattr(maintenance_mod, "execute_opinion_formation", fake_opinion)
        prompts = SleepPromptText(
            consolidation_system="CFG consolidation",
            stance_system="CFG stance {self_name}",
            synthesis_system="CFG synthesis {self_name}",
        )
        engine = _engine(
            store,
            clock,
            config=_sleep_config(),
            sleep_passes_enabled=True,
            sleep_prompts=prompts,
        )
        await engine._sleep_schedule_tick(clock.now)
        assert engine._sleep_passes_task is not None
        await engine._sleep_passes_task
        assert seen_c["system"] == "CFG consolidation"
        assert seen_o["stance_system"] == "CFG stance {self_name}"
        assert seen_o["synthesis_system"] == "CFG synthesis {self_name}"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_not_fired_for_non_sleep_activity(
        self,
        store: AsyncHistoryStore,
        clock: FakeClock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fail(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            msg = "passes must not run for a walk"
            raise AssertionError(msg)

        monkeypatch.setattr(maintenance_mod, "execute_consolidation", fail)
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_passes_enabled=True
        )
        await _start_activity(engine, "walk")
        assert engine._sleep_passes_task is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_skipped_when_sleep_passes_disabled(
        self,
        store: AsyncHistoryStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clock = _night_clock(0, 45)
        calls: list[str] = []

        async def fake_consolidation(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            calls.append("consolidation")

        monkeypatch.setattr(
            maintenance_mod, "execute_consolidation", fake_consolidation
        )
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_passes_enabled=False
        )
        await engine._sleep_schedule_tick(clock.now)
        assert engine._sleep_passes_task is not None
        await engine._sleep_passes_task
        assert calls == []
        await engine.stop()

    @pytest.mark.asyncio
    async def test_failure_logged_keeps_none_never_blocks_return(
        self,
        store: AsyncHistoryStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clock = _night_clock(0, 45)

        async def boom(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            msg = "llm down"
            raise RuntimeError(msg)

        monkeypatch.setattr(maintenance_mod, "execute_consolidation", boom)
        engine = _engine(
            store, clock, config=_sleep_config(), sleep_passes_enabled=True
        )
        await engine._sleep_schedule_tick(clock.now)
        assert engine._sleep_passes_task is not None
        await engine._sleep_passes_task  # must not raise
        assert engine._last_opinion_plan is None
        # return flow still completes
        await engine._cancel_return_timer()
        clock.now = datetime(2026, 6, 13, 8, 0, tzinfo=UTC)
        await engine._run_return(status="completed")
        assert store.sync.active_activity(familiar_id=_FAMILIAR) is None
        await engine.stop()

    @pytest.mark.asyncio
    async def test_passes_persist_dream_prose_and_journal(
        self,
        store: AsyncHistoryStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dream prose produced + persisted right after passes finish."""
        clock = _night_clock(0, 45)
        cplan = SimpleNamespace(retire=[], rewrite=[], mutated_count=0)
        dplan = _opinion_plan("Rain is best heard from indoors.")

        async def fake_consolidation(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            return cplan

        async def fake_opinion(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            return dplan

        monkeypatch.setattr(
            maintenance_mod, "execute_consolidation", fake_consolidation
        )
        monkeypatch.setattr(maintenance_mod, "execute_opinion_formation", fake_opinion)
        # spy the journal write rather than reading it back: a just-committed
        # facts INSERT is intermittently invisible to a same-test read through
        # pyturso 0.5.1 (no intervening fact write to refresh the view), which
        # flaked this test ~5%. Verifying the engine ISSUES the dream-framed
        # self: append is deterministic and is the engine's actual contract.
        appended: list[dict[str, Any]] = []
        orig_append = store.sync.append_fact

        def spy_append(**kw: Any) -> Any:  # noqa: ANN401
            appended.append(kw)
            return orig_append(**kw)

        monkeypatch.setattr(store.sync, "append_fact", spy_append)
        engine = _engine(
            store,
            clock,
            config=_sleep_config(),
            sleep_passes_enabled=True,
            experience="A dream of rain on glass.",
        )
        await engine._sleep_schedule_tick(clock.now)
        assert engine.active is not None
        assert engine._sleep_passes_task is not None
        await engine._sleep_passes_task
        # persisted on the activity row
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.experience_text == "A dream of rain on glass."
        # in-memory record updated too (no-restart path)
        assert engine.active is not None
        assert engine.active.experience_text == "A dream of rain on glass."
        # dream-journal fact minted at pass completion: dream-framed, carries
        # the prose, self-subject only
        dream_appends = [c for c in appended if "dreamed" in c["text"]]
        assert len(dream_appends) == 1
        assert "A dream of rain on glass." in dream_appends[0]["text"]
        assert all(is_self_key(s.canonical_key) for s in dream_appends[0]["subjects"])
        await engine.stop()

    @pytest.mark.asyncio
    async def test_persist_no_ops_when_return_beat_passes(
        self,
        store: AsyncHistoryStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return finished the sleep row mid-passes — late persist no-ops.

        Simulate return-beats-passes: dream pass clears ``_active`` (as a
        finishing return would). The persist block must not write
        experience_text to the already-finished row nor mint a 2nd journal.
        """
        clock = _night_clock(0, 45)
        cplan = SimpleNamespace(retire=[], rewrite=[], mutated_count=0)
        dplan = _opinion_plan("Rain is best heard from indoors.")

        async def fake_consolidation(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            return cplan

        engine = _engine(
            store,
            clock,
            config=_sleep_config(),
            sleep_passes_enabled=True,
            experience="A dream of rain on glass.",
        )

        async def fake_opinion(**kw: Any) -> Any:  # noqa: ANN401, RUF029
            del kw
            # side effect: a return finished/cleared the row mid-passes
            engine._active = None
            return dplan

        monkeypatch.setattr(
            maintenance_mod, "execute_consolidation", fake_consolidation
        )
        monkeypatch.setattr(maintenance_mod, "execute_opinion_formation", fake_opinion)
        await engine._sleep_schedule_tick(clock.now)
        assert engine._sleep_passes_task is not None
        await engine._sleep_passes_task
        # sleep row must stay unwritten by the late passes
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.experience_text is None
        # no dream-journal fact minted by the late path
        facts = store.sync.recent_facts(familiar_id=_FAMILIAR, limit=10)
        journal = [f for f in facts if "dreamed" in f.text]
        assert journal == []
        await engine.stop()


async def _sleep_and_wake(engine: ActivityEngine, clock: FakeClock) -> None:
    """Force sleep at clock.now, then drive the fixed wake."""
    await engine._sleep_schedule_tick(clock.now)
    assert engine.active is not None
    await engine._cancel_return_timer()
    clock.now = engine.active.planned_return_at
    await engine._run_return(status="completed")


class TestDreamReturn:
    """Sleep return narrates the dream; mode + journal fact differ."""

    @pytest.mark.asyncio
    async def test_return_turn_uses_sleep_return_mode(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(0, 45)
        engine = _engine(
            store, clock, config=_sleep_config(), experience="I dreamed of spools."
        )
        await _sleep_and_wake(engine, clock)
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert len(marked) == 1
        assert marked[0].content == "[returned from asleep] I dreamed of spools."
        assert marked[0].mode == SLEEP_RETURN_MODE
        assert marked[0].mode == "sleep_return"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_prompt_carries_seed_and_minted_opinions(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(0, 45)
        engine = _engine(store, clock, config=_sleep_config())
        await engine._sleep_schedule_tick(clock.now)
        assert engine.active is not None
        # passes task lands the plan after departure — simulate that
        engine._last_opinion_plan = _opinion_plan("Rain is best heard from indoors.")
        await engine._cancel_return_timer()
        clock.now = engine.active.planned_return_at
        await engine._run_return(status="completed")
        llm = cast("FakeLLMClient", engine._llm_clients["background"])
        assert llm.calls, "dream prose generation never reached the LLM"
        system, user = llm.calls[-1][0], llm.calls[-1][1]
        assert "dream" in system.content_str.lower()
        assert "The night's dream, retold on waking." in user.content_str
        assert "Rain is best heard from indoors." in user.content_str
        await engine.stop()

    @pytest.mark.asyncio
    async def test_prose_seed_only_when_no_plan(self, store: AsyncHistoryStore) -> None:
        clock = _night_clock(0, 45)
        engine = _engine(store, clock, config=_sleep_config())
        assert engine._last_opinion_plan is None
        await _sleep_and_wake(engine, clock)
        llm = cast("FakeLLMClient", engine._llm_clients["background"])
        assert llm.calls
        assert "The night's dream, retold on waking." in llm.calls[-1][1].content_str
        await engine.stop()

    @pytest.mark.asyncio
    async def test_dream_journal_self_fact_minted(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(0, 45)
        engine = _engine(
            store, clock, config=_sleep_config(), experience="I dreamed of spools."
        )
        await _sleep_and_wake(engine, clock)
        facts = store.sync.recent_facts(familiar_id=_FAMILIAR, limit=10)
        journal = [f for f in facts if "dreamed" in f.text]
        assert len(journal) == 1
        assert "I dreamed of spools." in journal[0].text
        assert any(is_self_key(s.canonical_key) for s in journal[0].subjects)
        # mechanical event-fact still written alongside
        assert any("spent" in f.text and "asleep" in f.text for f in facts)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_stock_line(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(0, 45)
        engine = _engine(store, clock, config=_sleep_config(), experience="")
        await _sleep_and_wake(engine, clock)
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from asleep]")]
        assert len(marked) == 1
        assert len(marked[0].content) > len("[returned from asleep] ")
        await engine.stop()

    @pytest.mark.asyncio
    async def test_return_reuses_persisted_prose_no_regen(
        self, store: AsyncHistoryStore
    ) -> None:
        """Prose already on the row: reuse verbatim, no LLM, no 2nd journal."""
        clock = _night_clock(0, 45)
        engine = _engine(store, clock, config=_sleep_config())
        await engine._sleep_schedule_tick(clock.now)
        active = engine.active
        assert active is not None
        # simulate passes having persisted the dream (row + in-memory)
        engine._active = replace(active, experience_text="Persisted dream.")
        store.sync.set_activity_experience(
            activity_id=active.id, experience_text="Persisted dream."
        )
        await engine._cancel_return_timer()
        clock.now = active.planned_return_at
        await engine._run_return(status="completed")
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert marked[0].content == "[returned from asleep] Persisted dream."
        # no prose LLM call at wake
        llm = cast("FakeLLMClient", engine._llm_clients["background"])
        assert llm.calls == []
        # journal not re-appended (it was minted at pass completion)
        facts = store.sync.recent_facts(familiar_id=_FAMILIAR, limit=10)
        assert [f for f in facts if "dreamed" in f.text] == []
        await engine.stop()

    @pytest.mark.asyncio
    async def test_restart_reload_reuses_persisted_prose(
        self, store: AsyncHistoryStore
    ) -> None:
        """Mid-sleep restart: reloaded row carries prose → reuse, no regen."""
        clock = _night_clock(0, 45)
        engine = _engine(store, clock, config=_sleep_config())
        await engine._sleep_schedule_tick(clock.now)
        active = engine.active
        assert active is not None
        store.sync.set_activity_experience(
            activity_id=active.id, experience_text="Survived the restart."
        )
        await engine.stop()
        # boot a fresh engine — reloads the active row WITH prose
        engine2 = _engine(store, clock, config=_sleep_config())
        await engine2.start()
        reloaded = engine2.active
        assert reloaded is not None
        assert reloaded.experience_text == "Survived the restart."
        await engine2._cancel_return_timer()
        clock.now = reloaded.planned_return_at
        await engine2._run_return(status="completed")
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert marked[0].content == "[returned from asleep] Survived the restart."
        assert cast("FakeLLMClient", engine2._llm_clients["background"]).calls == []
        await engine2.stop()

    @pytest.mark.asyncio
    async def test_fallback_generates_and_journals_once_when_unpersisted(
        self, store: AsyncHistoryStore
    ) -> None:
        """Passes never persisted: wake generates seed-only + journals once."""
        clock = _night_clock(0, 45)
        engine = _engine(
            store, clock, config=_sleep_config(), experience="Fallback dream."
        )
        await engine._sleep_schedule_tick(clock.now)
        active = engine.active
        assert active is not None
        assert active.experience_text is None
        await engine._cancel_return_timer()
        clock.now = active.planned_return_at
        await engine._run_return(status="completed")
        llm = cast("FakeLLMClient", engine._llm_clients["background"])
        assert llm.calls, "fallback should reach the LLM"
        facts = store.sync.recent_facts(familiar_id=_FAMILIAR, limit=10)
        journal = [f for f in facts if "dreamed" in f.text]
        assert len(journal) == 1
        assert "Fallback dream." in journal[0].text
        await engine.stop()


class TestSeedDreamConsumable:
    """Authored first dream — verbatim once, then generation."""

    @pytest.mark.asyncio
    async def test_used_verbatim_then_renamed(
        self, store: AsyncHistoryStore, tmp_path: Path
    ) -> None:
        seed_path = tmp_path / "seed_dream.md"
        seed_path.write_text("The Spools, hand-authored.")
        clock = _night_clock(0, 45)
        engine = _engine(
            store, clock, config=_sleep_config(), seed_dream_path=seed_path
        )
        await _sleep_and_wake(engine, clock)
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert marked[0].content == "[returned from asleep] The Spools, hand-authored."
        # generation skipped entirely
        assert cast("FakeLLMClient", engine._llm_clients["background"]).calls == []
        # consumed — idempotent on the next night
        assert not seed_path.exists()
        consumed = tmp_path / "seed_dream.consumed.md"
        assert consumed.exists()
        assert consumed.read_text() == "The Spools, hand-authored."
        await engine.stop()

    @pytest.mark.asyncio
    async def test_second_night_generates(
        self, store: AsyncHistoryStore, tmp_path: Path
    ) -> None:
        seed_path = tmp_path / "seed_dream.md"
        seed_path.write_text("The Spools.")
        clock = _night_clock(0, 45)
        engine = _engine(
            store,
            clock,
            config=_sleep_config(),
            seed_dream_path=seed_path,
            experience="A generated second dream.",
        )
        await _sleep_and_wake(engine, clock)
        clock.now = datetime(2026, 6, 14, 0, 45, tzinfo=UTC)
        await _sleep_and_wake(engine, clock)
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert len(marked) == 2
        assert "A generated second dream." in marked[-1].content
        await engine.stop()

    @pytest.mark.asyncio
    async def test_missing_file_generates(
        self, store: AsyncHistoryStore, tmp_path: Path
    ) -> None:
        clock = _night_clock(0, 45)
        engine = _engine(
            store,
            clock,
            config=_sleep_config(),
            seed_dream_path=tmp_path / "seed_dream.md",
            experience="A generated dream.",
        )
        await _sleep_and_wake(engine, clock)
        turns = store.sync.recent(familiar_id=_FAMILIAR, channel_id=_CHANNEL, limit=10)
        marked = [t for t in turns if t.content.startswith("[returned from")]
        assert "A generated dream." in marked[0].content
        await engine.stop()


class TestEarlyBedGuard:
    """start_activity('sleep') far from the window is refused.

    The fixed wake at window END would otherwise turn a midday tool
    call into a ~20h absence. Within an hour of bedtime is fine.
    """

    @pytest.mark.asyncio
    async def test_midday_sleep_refused(self, store: AsyncHistoryStore) -> None:
        clock = _night_clock(12, 0)
        engine = _engine(store, clock, config=_sleep_config())
        result = engine.defer_start("sleep", None)
        assert "error" in result
        assert "window" in result["error"]

    @pytest.mark.asyncio
    async def test_within_hour_before_window_allowed(
        self, store: AsyncHistoryStore
    ) -> None:
        clock = _night_clock(23, 30, day=12)  # window 00:00-08:00 next day
        engine = _engine(store, clock, config=_sleep_config())
        ack = engine.defer_start("sleep", None)
        assert ack.get("ack") == "ok"
        await engine.end_turn()
        row = store.sync.active_activity(familiar_id=_FAMILIAR)
        assert row is not None
        assert row.planned_return_at == datetime(2026, 6, 13, 8, 0, tzinfo=UTC)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_inside_window_allowed(self, store: AsyncHistoryStore) -> None:
        clock = _night_clock(2, 0)
        engine = _engine(store, clock, config=_sleep_config())
        assert engine.defer_start("sleep", None).get("ack") == "ok"
        await engine.end_turn()
        await engine.stop()
