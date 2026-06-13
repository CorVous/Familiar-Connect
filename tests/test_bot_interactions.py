"""Slash-command interaction guards.

Handlers race Discord's 3s ACK window; a missed ACK surfaces as
``NotFound (10062)``. :func:`_defer_interaction` claims the window and
:func:`_reply` delivers the confirmation — both must treat a dead
interaction as benign (action already ran).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from familiar_connect.activities.config import ActivitiesConfig, ActivityType
from familiar_connect.activities.engine import ActivityEngine
from familiar_connect.bot import (
    BotHandle,
    _defer_interaction,
    _register_events,
    _reply,
    build_activity_presence_cb,
    message_pings_bot,
)
from familiar_connect.bus.bus import InProcessEventBus
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore

from .conftest import build_fake_llm_clients

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    from familiar_connect.familiar import Familiar


def _not_found() -> discord.NotFound:
    resp = SimpleNamespace(status=404, reason="Not Found")
    return discord.NotFound(resp, "Unknown interaction")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def _ctx(*, defer: AsyncMock | None = None, followup: AsyncMock | None = None):
    return SimpleNamespace(
        defer=defer or AsyncMock(),
        followup=SimpleNamespace(send=followup or AsyncMock()),
        command=SimpleNamespace(name="subscribe-text"),
    )


@pytest.mark.asyncio
async def test_defer_returns_true_on_success() -> None:
    ctx = _ctx()
    assert await _defer_interaction(ctx) is True
    ctx.defer.assert_awaited_once_with(ephemeral=True)


@pytest.mark.asyncio
async def test_defer_returns_false_on_dead_interaction() -> None:
    ctx = _ctx(defer=AsyncMock(side_effect=_not_found()))
    # must not raise — a stale interaction is benign
    assert await _defer_interaction(ctx) is False


@pytest.mark.asyncio
async def test_reply_sends_followup() -> None:
    ctx = _ctx()
    await _reply(ctx, "ok")
    ctx.followup.send.assert_awaited_once_with("ok", ephemeral=True)


@pytest.mark.asyncio
async def test_reply_swallows_dead_interaction() -> None:
    ctx = _ctx(followup=AsyncMock(side_effect=_not_found()))
    # must not raise
    await _reply(ctx, "ok")


class TestMessagePingsBot:
    """Ping detection over ``message.mentions``.

    Carries both ``<@id>`` mentions and reply-ping targets (py-cord);
    roles/@everyone live elsewhere and never count.
    """

    def test_true_when_bot_in_mentions(self) -> None:
        msg = SimpleNamespace(mentions=[SimpleNamespace(id=7), SimpleNamespace(id=99)])
        assert message_pings_bot(cast("discord.Message", msg), 99) is True

    def test_false_when_bot_absent(self) -> None:
        msg = SimpleNamespace(mentions=[SimpleNamespace(id=7)])
        assert message_pings_bot(cast("discord.Message", msg), 99) is False

    def test_false_when_bot_user_id_unknown(self) -> None:
        msg = SimpleNamespace(mentions=[SimpleNamespace(id=99)])
        assert message_pings_bot(cast("discord.Message", msg), None) is False

    def test_false_on_empty_mentions(self) -> None:
        msg = SimpleNamespace(mentions=[])
        assert message_pings_bot(cast("discord.Message", msg), 99) is False


def _presence_handle(*, ready: bool = True) -> tuple[BotHandle, MagicMock]:
    """Handle + raw bot mock (assertions on the mock keep ty happy)."""
    bot = MagicMock(name="bot")
    bot.is_ready.return_value = ready
    bot.change_presence = AsyncMock()
    fm = MagicMock(name="focus_manager")
    fm.presence_guild.return_value = "Guild"
    fm.presence_text.return_value = "general"
    return BotHandle(bot=bot, send_text=AsyncMock(), focus_manager=fm), bot


class TestActivityPresenceCb:
    """ActivityEngine presence callback — idle while out, focus restore back."""

    @pytest.mark.asyncio
    async def test_idle_sets_idle_status_with_label(self) -> None:
        handle, bot = _presence_handle()
        cb = build_activity_presence_cb(handle)
        await cb("idle", "creek walk")
        kwargs = bot.change_presence.await_args.kwargs
        assert kwargs["status"] is discord.Status.idle
        assert kwargs["activity"].name == "creek walk"

    @pytest.mark.asyncio
    async def test_dnd_sets_dnd_status_with_label(self) -> None:
        handle, bot = _presence_handle()
        cb = build_activity_presence_cb(handle)
        await cb("dnd", "hatbox tending")
        kwargs = bot.change_presence.await_args.kwargs
        assert kwargs["status"] is discord.Status.dnd
        assert kwargs["activity"].name == "hatbox tending"

    @pytest.mark.asyncio
    async def test_online_restores_focus_presence(self) -> None:
        handle, bot = _presence_handle()
        cb = build_activity_presence_cb(handle)
        await cb("online", None)
        kwargs = bot.change_presence.await_args.kwargs
        assert kwargs["status"] is discord.Status.online
        # _sync_presence path — custom status carries focus label
        assert "general" in kwargs["activity"].state

    @pytest.mark.asyncio
    async def test_noop_when_bot_not_ready(self) -> None:
        handle, bot = _presence_handle(ready=False)
        cb = build_activity_presence_cb(handle)
        await cb("idle", "creek walk")
        bot.change_presence.assert_not_awaited()


class _NoFocus:
    """FocusLike stand-in — no focused channel."""

    def get_focus(self, modality: str) -> int | None:
        del modality
        return None


class TestOnReadyPresenceResync:
    """on_ready must end with away presence when mid-activity.

    Boot order: engine.start() runs pre-login, its away call is
    dropped by the cb's ready guard, and ``on_ready``'s focus sync
    sets online — the post-sync resync has to win.
    """

    @staticmethod
    def _engine(handle: BotHandle, tmp_path: Path) -> ActivityEngine:
        store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
        hatbox = ActivityType(
            id="hatbox",
            label="hatbox tending",
            duration_minutes=(10, 20),
            reachable=False,
            seed="Tending the hatbox.",
        )
        now = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)
        store.sync.create_activity(
            familiar_id="aria",
            type_id="hatbox",
            label="hatbox tending",
            started_at=now - timedelta(minutes=5),
            planned_return_at=now + timedelta(minutes=15),
            note=None,
        )
        return ActivityEngine(
            store=store,
            config=ActivitiesConfig(catalog=(hatbox,)),
            llm_clients=build_fake_llm_clients(),
            bus=InProcessEventBus(),
            focus_manager=_NoFocus(),
            presence_cb=build_activity_presence_cb(handle),
            familiar_id="aria",
            display_tz="UTC",
            bot_user_id=lambda: 99,
            now_fn=lambda: now,
        )

    @pytest.mark.asyncio
    async def test_ready_after_reload_ends_with_away_presence(
        self, tmp_path: Path
    ) -> None:
        events: dict[str, Callable[..., Coroutine[None, None, None]]] = {}
        handle, bot = _presence_handle(ready=False)
        bot.user = SimpleNamespace(id=99)
        bot.guilds = []
        bot.event.side_effect = lambda coro: events.setdefault(coro.__name__, coro)
        engine = self._engine(handle, tmp_path)
        handle.activity_engine = engine
        familiar = cast("Familiar", SimpleNamespace(bot_user_id=None))
        _register_events(bot, familiar, MagicMock(), handle)
        # boot order: engine reload fires pre-ready — cb drops the call
        await engine.start()
        bot.change_presence.assert_not_awaited()
        bot.is_ready.return_value = True
        await events["on_ready"]()
        calls = bot.change_presence.await_args_list
        # _sync_presence first (online), resync wins with away (dnd)
        assert calls[0].kwargs["status"] is discord.Status.online
        assert calls[-1].kwargs["status"] is discord.Status.dnd
        assert calls[-1].kwargs["activity"].name == "hatbox tending"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_ready_without_engine_skips_resync(self) -> None:
        events: dict[str, Callable[..., Coroutine[None, None, None]]] = {}
        handle, bot = _presence_handle(ready=True)
        bot.user = SimpleNamespace(id=99)
        bot.guilds = []
        bot.event.side_effect = lambda coro: events.setdefault(coro.__name__, coro)
        familiar = cast("Familiar", SimpleNamespace(bot_user_id=None))
        _register_events(bot, familiar, MagicMock(), handle)
        # activities disabled — clean no-op, only the focus sync runs
        await events["on_ready"]()
        calls = bot.change_presence.await_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["status"] is discord.Status.online
