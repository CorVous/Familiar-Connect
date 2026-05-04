"""Tests for :mod:`familiar_connect.typing_interrupt`.

The handler is the policy seam between Discord ``on_typing`` events and
the :class:`TextResponder`. It owns:

* cancelling the in-flight turn when a real user starts typing
  (``respond_to_typing`` ON; default);
* exponential backoff when *another bot* is typing in the channel,
  protecting against pingpong with another familiar-connect instance.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from familiar_connect.bus import TurnRouter
from familiar_connect.config import DiscordTextConfig
from familiar_connect.typing_interrupt import TypingInterruptHandler


def _make_handler(
    *,
    config: DiscordTextConfig | None = None,
    bot_user_id: int | None = 999,
    subscribed: set[int] | None = None,
    router: TurnRouter | None = None,
) -> tuple[TypingInterruptHandler, TurnRouter]:
    config = config or DiscordTextConfig()
    router = router or TurnRouter()
    subscribed = subscribed if subscribed is not None else {42}
    handler = TypingInterruptHandler(
        config=config,
        router=router,
        is_subscribed=subscribed.__contains__,
        bot_user_id_provider=lambda: bot_user_id,
    )
    return handler, router


class TestUserTypingCancelsActiveTurn:
    def test_user_typing_cancels_scope(self) -> None:
        handler, router = _make_handler()
        scope = router.begin_turn(session_id="discord:42", turn_id="t-1")
        handler.notify_typing(channel_id=42, user_id=7, is_bot=False)
        assert scope.is_cancelled() is True

    def test_no_active_scope_is_noop(self) -> None:
        handler, _ = _make_handler()
        # nothing raised — handler tolerates idle channels
        handler.notify_typing(channel_id=42, user_id=7, is_bot=False)

    def test_bot_self_typing_ignored(self) -> None:
        """Our own typing-indicator events must not cancel our own turn."""
        handler, router = _make_handler(bot_user_id=999)
        scope = router.begin_turn(session_id="discord:42", turn_id="t-1")
        handler.notify_typing(channel_id=42, user_id=999, is_bot=True)
        assert scope.is_cancelled() is False

    def test_unsubscribed_channel_ignored(self) -> None:
        handler, router = _make_handler(subscribed=set())
        scope = router.begin_turn(session_id="discord:42", turn_id="t-1")
        handler.notify_typing(channel_id=42, user_id=7, is_bot=False)
        assert scope.is_cancelled() is False

    def test_disabled_via_config(self) -> None:
        handler, router = _make_handler(
            config=DiscordTextConfig(respond_to_typing=False),
        )
        scope = router.begin_turn(session_id="discord:42", turn_id="t-1")
        handler.notify_typing(channel_id=42, user_id=7, is_bot=False)
        assert scope.is_cancelled() is False


class TestBotTypingExponentialBackoff:
    def test_first_bot_typing_sets_initial_backoff(self) -> None:
        cfg = DiscordTextConfig(typing_backoff_initial_s=2.0, typing_backoff_max_s=8.0)
        handler, _ = _make_handler(config=cfg)
        before = time.monotonic()
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        deadline = handler.backoff_deadline(42)
        assert deadline is not None
        # deadline ~= now + 2s; allow generous tolerance for test scheduling
        assert before + 1.5 <= deadline <= before + 2.6

    def test_repeated_bot_typing_doubles_backoff(self) -> None:
        cfg = DiscordTextConfig(typing_backoff_initial_s=1.0, typing_backoff_max_s=16.0)
        handler, _ = _make_handler(config=cfg)
        # consecutive bot-typing events without intervening user message
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        first = handler.current_backoff_s(42)
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        second = handler.current_backoff_s(42)
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        third = handler.current_backoff_s(42)
        assert first == pytest.approx(1.0)
        assert second == pytest.approx(2.0)
        assert third == pytest.approx(4.0)

    def test_backoff_caps_at_max(self) -> None:
        cfg = DiscordTextConfig(typing_backoff_initial_s=4.0, typing_backoff_max_s=8.0)
        handler, _ = _make_handler(config=cfg)
        for _ in range(5):
            handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        assert handler.current_backoff_s(42) == pytest.approx(8.0)

    def test_user_message_resets_backoff(self) -> None:
        cfg = DiscordTextConfig(typing_backoff_initial_s=1.0, typing_backoff_max_s=16.0)
        handler, _ = _make_handler(config=cfg)
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        assert handler.current_backoff_s(42) == pytest.approx(2.0)
        # user actually says something — bot path is alive again
        handler.notify_user_message(channel_id=42)
        # next bot typing starts the ladder over
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        assert handler.current_backoff_s(42) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_wait_for_backoff_sleeps_until_deadline(self) -> None:
        cfg = DiscordTextConfig(typing_backoff_initial_s=0.05, typing_backoff_max_s=0.1)
        handler, _ = _make_handler(config=cfg)
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        start = time.monotonic()
        await handler.wait_for_backoff(42)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04

    @pytest.mark.asyncio
    async def test_wait_for_backoff_returns_immediately_when_idle(self) -> None:
        handler, _ = _make_handler()
        start = time.monotonic()
        await handler.wait_for_backoff(42)
        assert time.monotonic() - start < 0.05

    def test_disabled_via_config_skips_backoff(self) -> None:
        cfg = DiscordTextConfig(respond_to_typing=False)
        handler, _ = _make_handler(config=cfg)
        handler.notify_typing(channel_id=42, user_id=123, is_bot=True)
        assert handler.backoff_deadline(42) is None


def test_module_loads() -> None:
    """Smoke: handler constructible, sleep runs cleanly."""

    async def _smoke() -> None:
        handler, _ = _make_handler()
        await handler.wait_for_backoff(99)

    asyncio.run(_smoke())
