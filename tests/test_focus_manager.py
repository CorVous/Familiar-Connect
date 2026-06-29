"""Tests for FocusManager and SubscriptionRegistry.kind_for.

TDD red-first: covers all specified behaviors before implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_connect.focus import FocusManager
from familiar_connect.history.store import FocusPointers, Promotion
from familiar_connect.subscriptions import (
    SubscriptionKind,
    SubscriptionRegistry,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# SubscriptionRegistry.kind_for
# ---------------------------------------------------------------------------


class TestKindFor:
    def test_returns_none_when_not_subscribed(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        assert reg.kind_for(channel_id=99) is None

    def test_returns_text_kind(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=None)
        assert reg.kind_for(channel_id=42) is SubscriptionKind.text

    def test_returns_voice_kind(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=7, kind=SubscriptionKind.voice, guild_id=1)
        assert reg.kind_for(channel_id=7) is SubscriptionKind.voice

    def test_prefers_first_kind_when_both_present(self, tmp_path: Path) -> None:
        """Channel with both text and voice: returns a valid SubscriptionKind."""
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=5, kind=SubscriptionKind.text, guild_id=None)
        reg.add(channel_id=5, kind=SubscriptionKind.voice, guild_id=1)
        result = reg.kind_for(channel_id=5)
        assert result in {SubscriptionKind.text, SubscriptionKind.voice}

    def test_unrelated_channel_unaffected(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=10, kind=SubscriptionKind.text, guild_id=None)
        assert reg.kind_for(channel_id=20) is None


# ---------------------------------------------------------------------------
# FocusManager helpers
# ---------------------------------------------------------------------------


def _make_store(
    *,
    text_channel_id: int | None = None,
    voice_channel_id: int | None = None,
    promote_count: int = 0,
) -> MagicMock:
    """Mock AsyncHistoryStore with configurable return values."""
    store = MagicMock()
    if text_channel_id is not None or voice_channel_id is not None:
        fp = FocusPointers(
            text_channel_id=text_channel_id,
            voice_channel_id=voice_channel_id,
            updated_at=datetime.now(UTC),
        )
    else:
        fp = None
    store.get_focus_pointers = AsyncMock(return_value=fp)
    store.set_focus_pointers = AsyncMock(return_value=None)
    store.promote_staged_turns = AsyncMock(
        return_value=Promotion(consumed=promote_count, missed=0)
    )
    return store


def _make_registry(
    tmp_path: Path,
    *,
    channel_kind: dict[int, SubscriptionKind] | None = None,
) -> SubscriptionRegistry:
    reg = SubscriptionRegistry(tmp_path / "subs.toml")
    if channel_kind:
        for ch, kind in channel_kind.items():
            guild = 1 if kind is SubscriptionKind.voice else None
            reg.add(channel_id=ch, kind=kind, guild_id=guild)
    return reg


# ---------------------------------------------------------------------------
# FocusManager.initialize
# ---------------------------------------------------------------------------


class TestFocusManagerInitialize:
    @pytest.mark.asyncio
    async def test_initialize_loads_focus_pointers(self, tmp_path: Path) -> None:

        store = _make_store(text_channel_id=10, voice_channel_id=20)
        reg = _make_registry(
            tmp_path,
            channel_kind={10: SubscriptionKind.text, 20: SubscriptionKind.voice},
        )
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.get_focus("text") == 10
        assert fm.get_focus("voice") == 20

    @pytest.mark.asyncio
    async def test_initialize_with_no_db_entry_stays_none(self, tmp_path: Path) -> None:

        store = _make_store()  # get_focus_pointers returns None
        reg = _make_registry(tmp_path)
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.get_focus("text") is None
        assert fm.get_focus("voice") is None

    @pytest.mark.asyncio
    async def test_initialize_drops_unsubscribed_text_focus(
        self, tmp_path: Path
    ) -> None:
        # persisted pointer to a channel no longer subscribed → cleared,
        # not stranded (startup fallback then re-seeds a live channel)
        store = _make_store(text_channel_id=10)
        reg = _make_registry(tmp_path)  # 10 not subscribed
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.get_focus("text") is None

    @pytest.mark.asyncio
    async def test_initialize_drops_unsubscribed_voice_focus(
        self, tmp_path: Path
    ) -> None:
        store = _make_store(voice_channel_id=20)
        reg = _make_registry(tmp_path)  # 20 not subscribed
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.get_focus("voice") is None

    @pytest.mark.asyncio
    async def test_initialize_keeps_subscribed_focus(self, tmp_path: Path) -> None:
        store = _make_store(text_channel_id=10, voice_channel_id=20)
        reg = _make_registry(
            tmp_path,
            channel_kind={10: SubscriptionKind.text, 20: SubscriptionKind.voice},
        )
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.get_focus("text") == 10
        assert fm.get_focus("voice") == 20


# ---------------------------------------------------------------------------
# FocusManager.is_subscribed / subscribed_channels
# ---------------------------------------------------------------------------


class TestFocusManagerSubscriptionHelpers:
    def test_is_subscribed_true(self, tmp_path: Path) -> None:
        store = _make_store()
        reg = _make_registry(tmp_path, channel_kind={5: SubscriptionKind.text})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        assert fm.is_subscribed(5) is True

    def test_is_subscribed_false(self, tmp_path: Path) -> None:
        store = _make_store()
        reg = _make_registry(tmp_path, channel_kind={5: SubscriptionKind.text})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        assert fm.is_subscribed(99) is False

    def test_subscribed_channels_lists_text_and_voice(self, tmp_path: Path) -> None:
        store = _make_store()
        reg = _make_registry(
            tmp_path,
            channel_kind={5: SubscriptionKind.text, 8: SubscriptionKind.voice},
        )
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        assert fm.subscribed_channels() == [5, 8]

    def test_subscribed_channels_dedups_dual_kind(self, tmp_path: Path) -> None:
        # channel hosting both text + voice appears once
        store = _make_store()
        reg = _make_registry(tmp_path)
        reg.add(channel_id=5, kind=SubscriptionKind.text, guild_id=None)
        reg.add(channel_id=5, kind=SubscriptionKind.voice, guild_id=1)
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        assert fm.subscribed_channels() == [5]


# ---------------------------------------------------------------------------
# FocusManager.is_focused
# ---------------------------------------------------------------------------


class TestFocusManagerIsFocused:
    @pytest.mark.asyncio
    async def test_is_focused_true_for_text_channel(self, tmp_path: Path) -> None:

        store = _make_store(text_channel_id=42)
        reg = _make_registry(tmp_path, channel_kind={42: SubscriptionKind.text})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.is_focused(42) is True

    @pytest.mark.asyncio
    async def test_is_focused_true_for_voice_channel(self, tmp_path: Path) -> None:

        store = _make_store(voice_channel_id=77)
        reg = _make_registry(tmp_path, channel_kind={77: SubscriptionKind.voice})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.is_focused(77) is True

    @pytest.mark.asyncio
    async def test_is_focused_false_for_unfocused_channel(self, tmp_path: Path) -> None:

        store = _make_store(text_channel_id=1)
        reg = _make_registry(tmp_path)
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.is_focused(99) is False

    @pytest.mark.asyncio
    async def test_is_focused_false_when_no_focus(self, tmp_path: Path) -> None:

        store = _make_store()
        reg = _make_registry(tmp_path)
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        assert fm.is_focused(1) is False


# ---------------------------------------------------------------------------
# FocusManager.shift_now (immediate) + end_turn (idle-clock only)
# ---------------------------------------------------------------------------


class TestFocusManagerShiftNow:
    @pytest.mark.asyncio
    async def test_shift_now_text_promotes_staged_turns(self, tmp_path: Path) -> None:

        store = _make_store(promote_count=3)
        reg = _make_registry(tmp_path, channel_kind={5: SubscriptionKind.text})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.shift_now(channel_id=5)
        store.promote_staged_turns.assert_awaited_once_with(
            familiar_id="fam", channel_id=5, catch_up_limit=20
        )

    @pytest.mark.asyncio
    async def test_shift_now_text_updates_text_focus(self, tmp_path: Path) -> None:

        store = _make_store(promote_count=0)
        reg = _make_registry(tmp_path, channel_kind={5: SubscriptionKind.text})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.shift_now(channel_id=5)
        assert fm.get_focus("text") == 5

    @pytest.mark.asyncio
    async def test_shift_now_voice_updates_voice_focus(self, tmp_path: Path) -> None:

        store = _make_store()
        reg = _make_registry(tmp_path, channel_kind={8: SubscriptionKind.voice})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.shift_now(channel_id=8)
        assert fm.get_focus("voice") == 8

    @pytest.mark.asyncio
    async def test_shift_now_voice_does_not_promote(self, tmp_path: Path) -> None:

        store = _make_store()
        reg = _make_registry(tmp_path, channel_kind={8: SubscriptionKind.voice})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.shift_now(channel_id=8)
        store.promote_staged_turns.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shift_now_persists_pointers(self, tmp_path: Path) -> None:

        store = _make_store()
        reg = _make_registry(tmp_path, channel_kind={5: SubscriptionKind.text})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.shift_now(channel_id=5)
        store.set_focus_pointers.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_end_turn_does_not_touch_focus(self, tmp_path: Path) -> None:
        # end_turn is idle-clock bookkeeping only; never moves/persists focus
        store = _make_store()
        reg = _make_registry(tmp_path)
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.end_turn()
        store.promote_staged_turns.assert_not_awaited()
        store.set_focus_pointers.assert_not_awaited()


# ---------------------------------------------------------------------------
# Idle-drift wake
# ---------------------------------------------------------------------------


class _FakeClock:
    """Mutable monotonic clock for deterministic idle tests."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestFocusManagerIdleWake:
    def _fm(
        self,
        tmp_path: Path,
        *,
        clock: _FakeClock,
        unread_nudge_enabled: bool = True,
        nudge_debounce_seconds: float = 10.0,
        text_focus: int = 1,
    ) -> FocusManager:
        store = _make_store(text_channel_id=text_focus)
        reg = _make_registry(tmp_path, channel_kind={text_focus: SubscriptionKind.text})
        return FocusManager(
            familiar_id="fam",
            store=store,
            subscriptions=reg,
            clock=clock,
            unread_nudge_enabled=unread_nudge_enabled,
            nudge_debounce_seconds=nudge_debounce_seconds,
        )

    @pytest.mark.asyncio
    async def test_should_wake_true_on_arrival_when_focused_channel_active(
        self, tmp_path: Path
    ) -> None:
        # arrival nudges immediately, even though the focused channel was
        # just active (clock not advanced) — debounce is the sole throttle
        clock = _FakeClock()
        fm = self._fm(tmp_path, clock=clock, text_focus=1)
        await fm.initialize()
        assert fm.should_wake(99) is True

    @pytest.mark.asyncio
    async def test_should_wake_false_for_focused_channel(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        fm = self._fm(tmp_path, clock=clock, text_focus=1)
        await fm.initialize()
        clock.advance(90.0)
        # focused channel never wakes itself
        assert fm.should_wake(1) is False

    @pytest.mark.asyncio
    async def test_should_wake_false_when_disabled(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        fm = self._fm(tmp_path, clock=clock, unread_nudge_enabled=False)
        await fm.initialize()
        clock.advance(10_000.0)
        assert fm.should_wake(99) is False

    @pytest.mark.asyncio
    async def test_should_wake_false_within_debounce(self, tmp_path: Path) -> None:
        # rapid arrivals grouped: nudge suppressed within debounce window
        clock = _FakeClock()
        fm = self._fm(tmp_path, clock=clock)  # debounce=10s
        await fm.initialize()
        clock.advance(90.0)
        assert fm.should_wake(99) is True
        fm.mark_nudge_pending()
        clock.advance(5.0)  # within 10s debounce
        assert fm.should_wake(99) is False

    @pytest.mark.asyncio
    async def test_should_wake_true_after_debounce_expires(
        self, tmp_path: Path
    ) -> None:
        # nudge recovers after debounce window — no end_turn needed
        clock = _FakeClock()
        fm = self._fm(tmp_path, clock=clock)  # debounce=10s
        await fm.initialize()
        clock.advance(90.0)
        fm.mark_nudge_pending()
        clock.advance(15.0)  # past debounce, focused channel still idle
        assert fm.should_wake(99) is True

    @pytest.mark.asyncio
    async def test_should_wake_repeats_after_each_debounce(
        self, tmp_path: Path
    ) -> None:
        # each unread cycle after debounce fires another nudge
        clock = _FakeClock()
        fm = self._fm(tmp_path, clock=clock)  # debounce=10s
        await fm.initialize()
        clock.advance(90.0)
        assert fm.should_wake(99) is True
        fm.mark_nudge_pending()
        clock.advance(15.0)
        assert fm.should_wake(99) is True  # second nudge fires
        fm.mark_nudge_pending()
        clock.advance(5.0)  # within second debounce window
        assert fm.should_wake(99) is False

    @pytest.mark.asyncio
    async def test_wake_does_not_move_focus(self, tmp_path: Path) -> None:
        # nudge must NOT auto-shift; focus only moves when the model decides
        clock = _FakeClock()
        fm = self._fm(tmp_path, clock=clock, text_focus=1)
        await fm.initialize()
        clock.advance(90.0)
        fm.mark_nudge_pending()
        assert fm.get_focus("text") == 1


# ---------------------------------------------------------------------------
# Two modalities stay independent
# ---------------------------------------------------------------------------


class TestModalitiesIndependent:
    @pytest.mark.asyncio
    async def test_text_shift_does_not_affect_voice(self, tmp_path: Path) -> None:

        store = _make_store(voice_channel_id=99)
        reg = _make_registry(
            tmp_path,
            channel_kind={5: SubscriptionKind.text, 99: SubscriptionKind.voice},
        )
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        await fm.shift_now(channel_id=5)
        # voice should be unchanged
        assert fm.get_focus("voice") == 99
        assert fm.get_focus("text") == 5

    @pytest.mark.asyncio
    async def test_voice_shift_does_not_affect_text(self, tmp_path: Path) -> None:

        store = _make_store(text_channel_id=11)
        reg = _make_registry(
            tmp_path,
            channel_kind={8: SubscriptionKind.voice, 11: SubscriptionKind.text},
        )
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        await fm.initialize()
        await fm.shift_now(channel_id=8)
        assert fm.get_focus("text") == 11
        assert fm.get_focus("voice") == 8


# ---------------------------------------------------------------------------
# set_focus_immediately
# ---------------------------------------------------------------------------


class TestSetFocusImmediately:
    def test_sets_text_focus(self, tmp_path: Path) -> None:

        store = _make_store()
        reg = _make_registry(tmp_path)
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        fm.set_focus_immediately(channel_id=3, modality="text")
        assert fm.get_focus("text") == 3

    def test_sets_voice_focus(self, tmp_path: Path) -> None:

        store = _make_store()
        reg = _make_registry(tmp_path)
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        fm.set_focus_immediately(channel_id=4, modality="voice")
        assert fm.get_focus("voice") == 4


# ---------------------------------------------------------------------------
# FocusManager.presence_text
# ---------------------------------------------------------------------------


class TestPresenceText:
    def _fm(self, tmp_path: Path) -> FocusManager:
        store = _make_store()
        reg = _make_registry(tmp_path)
        return FocusManager(familiar_id="fam", store=store, subscriptions=reg)

    def test_returns_none_with_no_focus(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        assert fm.presence_text() is None

    def test_returns_channel_name(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        fm.channel_names[42] = "general"
        fm.set_focus_immediately(42, "text")
        assert fm.presence_text() == "#general"

    def test_falls_back_to_channel_id_when_name_unknown(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        fm.set_focus_immediately(42, "text")
        assert fm.presence_text() == "#42"


class TestPresenceGuild:
    def _fm(self, tmp_path: Path) -> FocusManager:
        store = _make_store()
        reg = _make_registry(tmp_path)
        return FocusManager(familiar_id="fam", store=store, subscriptions=reg)

    def test_returns_none_with_no_focus(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        assert fm.presence_guild() is None

    def test_returns_none_when_guild_unknown(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        fm.set_focus_immediately(42, "text")
        assert fm.presence_guild() is None

    def test_returns_guild_name(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        fm.guild_names[42] = "Sapphire"
        fm.set_focus_immediately(42, "text")
        assert fm.presence_guild() == "Sapphire"


class TestGuildNameFor:
    """``guild_name_for`` — single lookup point for a channel's server name."""

    def _fm(self, tmp_path: Path) -> FocusManager:
        store = _make_store()
        reg = _make_registry(tmp_path)
        return FocusManager(familiar_id="fam", store=store, subscriptions=reg)

    def test_returns_name_for_known_channel(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        fm.guild_names[42] = "My Server"
        assert fm.guild_name_for(42) == "My Server"

    def test_returns_none_for_unknown_channel(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        fm.guild_names[42] = "My Server"
        assert fm.guild_name_for(99) is None

    def test_returns_none_for_none_input(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        fm.guild_names[42] = "My Server"
        assert fm.guild_name_for(None) is None


# ---------------------------------------------------------------------------
# FocusManager.on_shift
# ---------------------------------------------------------------------------


class TestOnShift:
    def _fm(self, tmp_path: Path) -> FocusManager:
        store = _make_store()
        reg = _make_registry(tmp_path)
        return FocusManager(familiar_id="fam", store=store, subscriptions=reg)

    def test_on_shift_is_none_by_default(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        assert fm.on_shift is None

    @pytest.mark.asyncio
    async def test_on_shift_called_after_shift_now(self, tmp_path: Path) -> None:
        store = _make_store()
        reg = _make_registry(tmp_path, channel_kind={5: SubscriptionKind.text})
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        callback = AsyncMock()
        fm.on_shift = callback
        await fm.shift_now(channel_id=5)
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_shift_fires_per_shift(self, tmp_path: Path) -> None:
        store = _make_store()
        reg = _make_registry(
            tmp_path,
            channel_kind={5: SubscriptionKind.text, 8: SubscriptionKind.voice},
        )
        fm = FocusManager(familiar_id="fam", store=store, subscriptions=reg)
        callback = AsyncMock()
        fm.on_shift = callback
        await fm.shift_now(channel_id=5)
        await fm.shift_now(channel_id=8)
        assert callback.await_count == 2

    @pytest.mark.asyncio
    async def test_on_shift_not_called_by_end_turn(self, tmp_path: Path) -> None:
        fm = self._fm(tmp_path)
        callback = AsyncMock()
        fm.on_shift = callback
        await fm.end_turn()
        callback.assert_not_awaited()
