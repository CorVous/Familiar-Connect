"""Typing-event policy: interruption + bot-pingpong backoff.

Discord ``on_typing`` fires when any user (or bot) starts typing in a
channel. :class:`TypingInterruptHandler` translates those events into:

* immediate :class:`TurnRouter` cancellation when a real user is typing
  (``[discord.text].respond_to_typing`` ON; default), so the bot stops
  generating instead of speaking over the user;
* exponential backoff when *another bot* is typing — protects against
  pingpong with another familiar-connect instance whose typing
  indicator we'd otherwise mirror by replying.

Wired by :mod:`familiar_connect.bot` (``on_typing``) and queried by
:class:`TextResponder` before assembling a reply.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls

if TYPE_CHECKING:
    from collections.abc import Callable

    from familiar_connect.bus.router import TurnRouter
    from familiar_connect.config import DiscordTextConfig

_logger = logging.getLogger(__name__)


class TypingInterruptHandler:
    """Per-channel typing-event policy.

    Stateless across processes — backoff state lives in memory, reset
    on user activity. Safe to share across the asyncio event loop.
    """

    def __init__(
        self,
        *,
        config: DiscordTextConfig,
        router: TurnRouter,
        is_subscribed: Callable[[int], bool],
        bot_user_id_provider: Callable[[], int | None],
    ) -> None:
        self._config = config
        self._router = router
        self._is_subscribed = is_subscribed
        self._bot_user_id = bot_user_id_provider
        # per-channel backoff ladder. ``_next_backoff_s`` is the value
        # to apply on the *next* bot-typing event; doubles after each.
        # ``_deadline`` is the monotonic clock time the channel is
        # parked until.
        self._next_backoff_s: dict[int, float] = {}
        self._deadline: dict[int, float] = {}

    def notify_typing(self, *, channel_id: int, user_id: int, is_bot: bool) -> None:
        """Translate a Discord ``on_typing`` event into policy actions."""
        if not self._config.respond_to_typing:
            return
        if not self._is_subscribed(channel_id):
            return
        bot_user_id = self._bot_user_id()
        if bot_user_id is not None and user_id == bot_user_id:
            return
        if is_bot:
            self._apply_bot_backoff(channel_id)
            return
        self._cancel_active_turn(channel_id, user_id)

    def notify_user_message(self, *, channel_id: int) -> None:
        """Reset backoff ladder; a real user message means the lane is live."""
        self._next_backoff_s.pop(channel_id, None)
        self._deadline.pop(channel_id, None)

    def backoff_deadline(self, channel_id: int) -> float | None:
        """Monotonic deadline if currently parked; ``None`` when free."""
        deadline = self._deadline.get(channel_id)
        if deadline is None:
            return None
        if deadline <= time.monotonic():
            # expired — clean up so the next call sees a free lane
            self._deadline.pop(channel_id, None)
            return None
        return deadline

    def current_backoff_s(self, channel_id: int) -> float:
        """Backoff window most recently applied to *channel_id*.

        Test seam — production callers should use :meth:`wait_for_backoff`.
        """
        return self._current_applied_s.get(channel_id, 0.0)

    async def wait_for_backoff(self, channel_id: int) -> None:
        """Sleep until the channel's backoff deadline (no-op when idle)."""
        deadline = self.backoff_deadline(channel_id)
        if deadline is None:
            return
        delay = max(0.0, deadline - time.monotonic())
        if delay <= 0:
            return
        _logger.info(
            f"{ls.tag('💬 Text', ls.B)} "
            f"{ls.kv('typing_backoff', f'{delay:.2f}s', vc=ls.LB)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LC)}"
        )
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @property
    def _current_applied_s(self) -> dict[int, float]:
        # lazily-created sibling map; kept here so the public surface
        # stays small. last value applied per channel — survives until
        # the next user message resets the ladder.
        if not hasattr(self, "_applied"):
            self._applied: dict[int, float] = {}
        return self._applied

    def _apply_bot_backoff(self, channel_id: int) -> None:
        next_s = self._next_backoff_s.get(
            channel_id, self._config.typing_backoff_initial_s
        )
        applied = min(next_s, self._config.typing_backoff_max_s)
        self._current_applied_s[channel_id] = applied
        self._deadline[channel_id] = time.monotonic() + applied
        # double for the next event; cap on read so the ladder doesn't
        # silently overflow the float range.
        self._next_backoff_s[channel_id] = min(
            applied * 2.0, self._config.typing_backoff_max_s
        )
        _logger.info(
            f"{ls.tag('💬 Text', ls.Y)} "
            f"{ls.kv('bot_typing', f'{applied:.2f}s', vc=ls.LY)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LC)}"
        )

    def _cancel_active_turn(self, channel_id: int, user_id: int) -> None:
        session_id = f"discord:{channel_id}"
        scope = self._router.active_scope(session_id)
        if scope is None:
            return
        scope.cancel()
        _logger.info(
            f"{ls.tag('💬 Text', ls.Y)} "
            f"{ls.kv('typing_cancel', 'user', vc=ls.LY)} "
            f"{ls.kv('user', str(user_id), vc=ls.LC)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LC)}"
        )
