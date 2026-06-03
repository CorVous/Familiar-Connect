"""Slash-command interaction guards.

Handlers race Discord's 3s ACK window; a missed ACK surfaces as
``NotFound (10062)``. :func:`_defer_interaction` claims the window and
:func:`_reply` delivers the confirmation — both must treat a dead
interaction as benign (action already ran).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from familiar_connect.bot import _defer_interaction, _reply


def _not_found() -> discord.NotFound:
    resp = SimpleNamespace(status=404, reason="Not Found")
    return discord.NotFound(resp, "Unknown interaction")  # type: ignore[arg-type]


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
