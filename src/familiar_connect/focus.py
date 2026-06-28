"""Attentional focus controller for a single familiar."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.subscriptions import SubscriptionKind

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.subscriptions import SubscriptionRegistry

_logger = logging.getLogger(__name__)

# Debounce window: rapid arrivals within this window share one nudge
_DEFAULT_NUDGE_DEBOUNCE_S = 30.0


class FocusManager:
    """Per-familiar attentional focus controller.

    Two independent focus pointers (text, voice). Focus shifts are
    model-decided (via shift_focus tool) and applied immediately at
    tool-call time under per-modality lock — no deferral. A silent turn
    therefore still leaves her where she went; there is no pending
    state to leak into a later turn.

    Unread nudge: when a non-focused channel gets traffic, ``should_wake``
    flags that the model deserves a turn (the responder fires a synthetic
    wake) — the arrival itself fires it, with no idle-silence requirement.
    ``unread_nudge_enabled`` gates the behavior on/off; the debounce window
    is the sole throttle. The nudge never moves focus — only the model's
    shift_focus does. Rapid arrivals within ``nudge_debounce_seconds`` are
    grouped into one nudge; the next unread after the window fires again.
    """

    def __init__(
        self,
        *,
        familiar_id: str,
        store: AsyncHistoryStore,
        subscriptions: SubscriptionRegistry,
        clock: Callable[[], float] = time.monotonic,
        unread_nudge_enabled: bool = True,
        nudge_debounce_seconds: float = _DEFAULT_NUDGE_DEBOUNCE_S,
    ) -> None:
        self._familiar_id = familiar_id
        self._store = store
        self._subscriptions = subscriptions
        self._text_focus: int | None = None
        self._voice_focus: int | None = None
        self._text_lock = asyncio.Lock()
        self._voice_lock = asyncio.Lock()
        # Unread-nudge state
        self._clock = clock
        self._unread_nudge_enabled = unread_nudge_enabled
        self._nudge_debounce_seconds = nudge_debounce_seconds
        self._last_nudge: float = float("-inf")  # Never nudged initially
        # Channel_id → display name; populated by bot on_ready
        self.channel_names: dict[int, str] = {}
        # Channel_id → guild name; populated by bot on_ready
        self.guild_names: dict[int, str] = {}
        # Called once after each applied shift; None disables
        self.on_shift: Callable[[], Awaitable[None]] | None = None

    async def initialize(self) -> None:
        """Load persisted focus pointers from DB.

        Pointers to channels no longer subscribed are dropped (avoid
        stranding focus on a dead channel); startup fallback re-seeds.
        """
        ptrs = await self._store.get_focus_pointers(self._familiar_id)
        if ptrs is not None:
            self._text_focus = self._keep_if_subscribed(ptrs.text_channel_id)
            self._voice_focus = self._keep_if_subscribed(ptrs.voice_channel_id)
            _logger.info(
                f"{ls.tag('Focus', ls.LC)} loaded "
                f"{ls.kv('text', self.channel_label(self._text_focus), vc=ls.LW)} "
                f"{ls.kv('voice', self.channel_label(self._voice_focus), vc=ls.LW)}"
            )

    def _keep_if_subscribed(self, channel_id: int | None) -> int | None:
        """Pass through channel_id when subscribed; else None (warn on drop)."""
        if channel_id is None or self.is_subscribed(channel_id):
            return channel_id
        _logger.warning(
            f"{ls.tag('Focus', ls.LC)} dropped stale pointer "
            f"{ls.kv('channel', self.channel_label(channel_id), vc=ls.LW)} "
            "(no longer subscribed)"
        )
        return None

    def get_focus(self, modality: str) -> int | None:
        """Return current focus channel_id for modality."""
        return self._text_focus if modality == "text" else self._voice_focus

    def is_subscribed(self, channel_id: int) -> bool:
        """Whether channel_id is a known text/voice subscription."""
        return self._subscriptions.kind_for(channel_id) is not None

    def subscribed_channels(self) -> list[int]:
        """Sorted, deduped channel_ids across text + voice subscriptions."""
        return sorted({sub.channel_id for sub in self._subscriptions.all()})

    def is_focused(self, channel_id: int) -> bool:
        """Check if channel_id is the active text or voice focus."""
        return channel_id in {self._text_focus, self._voice_focus}

    async def shift_now(self, channel_id: int) -> None:
        """Apply focus shift immediately (at tool-call time).

        Modality inferred from subscriptions. Text shift promotes the
        target's staged backlog to consumed, moves the pointer, persists,
        and fires ``on_shift``. No deferral — focus moves now, so a silent
        turn still leaves her where she went and nothing leaks.
        """
        kind = self._subscriptions.kind_for(channel_id)
        modality = "voice" if kind is SubscriptionKind.voice else "text"
        lock = self._voice_lock if modality == "voice" else self._text_lock
        async with lock:
            if modality == "text":
                count = await self._store.promote_staged_turns(
                    familiar_id=self._familiar_id, channel_id=channel_id
                )
                self._text_focus = channel_id
                _logger.info(
                    f"{ls.tag('🔀 Focus', ls.LC)} "
                    f"{ls.kv('text', self.channel_label(channel_id), vc=ls.LW)} "
                    f"{ls.kv('promoted', str(count), vc=ls.LG)}"
                )
            else:
                self._voice_focus = channel_id
                _logger.info(
                    f"{ls.tag('🔀 Focus', ls.LC)} "
                    f"{ls.kv('voice', self.channel_label(channel_id), vc=ls.LW)}"
                )
            await self._store.set_focus_pointers(
                self._familiar_id,
                text_channel_id=self._text_focus,
                voice_channel_id=self._voice_focus,
            )
        if self.on_shift is not None:
            await self.on_shift()

    def should_wake(self, channel_id: int) -> bool:
        """Whether a non-focused arrival warrants a nudge.

        True when nudges are enabled, the channel is unfocused, and we
        are outside the debounce window — the arrival itself fires the
        nudge, regardless of how recently the focused channel was active.
        Debounce is the sole throttle: rapid arrivals share one nudge.
        """
        if not self._unread_nudge_enabled:
            return False
        if self.is_focused(channel_id):
            return False
        now = self._clock()
        if (now - self._last_nudge) < self._nudge_debounce_seconds:
            return False
        return True

    def mark_nudge_pending(self) -> None:
        """Record nudge timestamp to start debounce window."""
        self._last_nudge = self._clock()

    async def end_turn(self) -> None:
        """Responder end-of-turn hook — intentionally a no-op.

        Focus shifts apply immediately (``shift_now``), so there is no
        per-turn focus state to reset. Retained because both responders
        call it as their end-of-turn signal; stays async so they can
        await it uniformly.
        """
        return

    def channel_label(self, channel_id: int | None) -> str:
        """Format channel_id as '#name(id)' or '#id' when name unknown."""
        if channel_id is None:
            return "none"
        name = self.channel_names.get(channel_id)
        return f"#{name}({channel_id})" if name else f"#{channel_id}"

    def guild_name_for(self, channel_id: int | None) -> str | None:
        """Server name for channel_id; None for None input or unknown channel."""
        return self.guild_names.get(channel_id) if channel_id is not None else None

    def presence_guild(self) -> str | None:
        """Guild name for current text focus channel; None when unset or unknown."""
        return self.guild_name_for(self._text_focus)

    def presence_text(self) -> str | None:
        """'#channel-name' for current text focus; None when unset."""
        channel_id = self._text_focus
        if channel_id is None:
            return None
        name = self.channel_names.get(channel_id, str(channel_id))
        return f"#{name}"

    def set_focus_immediately(self, channel_id: int, modality: str) -> None:
        """Set focus without deferral (used at startup)."""
        if modality == "text":
            self._text_focus = channel_id
        else:
            self._voice_focus = channel_id
        _logger.info(
            f"{ls.tag('Focus', ls.LC)} default "
            f"{ls.kv(modality, self.channel_label(channel_id), vc=ls.LW)}"
        )
