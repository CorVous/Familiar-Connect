"""Slash-command interaction guards.

Handlers race Discord's 3s ACK window; a missed ACK surfaces as
``NotFound (10062)``. :func:`_defer_interaction` claims the window and
:func:`_reply` delivers the confirmation — both must treat a dead
interaction as benign (action already ran).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from familiar_connect.bot import (
    BotHandle,
    _defer_interaction,
    _reply,
    build_activity_presence_cb,
    message_pings_bot,
)


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
