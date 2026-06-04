"""Attentional focus controller for a single familiar."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.subscriptions import SubscriptionKind

if TYPE_CHECKING:
    from collections.abc import Callable

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.subscriptions import SubscriptionRegistry

_logger = logging.getLogger(__name__)

# default focused-channel silence before a non-focused arrival nudges the
# model. 0 disables. module constant for now; promote to config if tuned.
_DEFAULT_IDLE_WAKE_S = 120.0


class FocusManager:
    """Per-familiar attentional focus controller.

    Two independent focus pointers (text, voice). Focus shifts are
    model-decided (via shift_focus tool), deferred to end_turn,
    then applied atomically under per-modality lock.

    Idle nudge: when the focused channel falls silent for
    ``idle_wake_seconds`` and a non-focused channel gets traffic,
    ``should_wake`` flags that the model deserves a turn (the responder
    fires a synthetic wake). The nudge never moves focus — only the
    model's shift_focus does. ``_nudge_pending`` dedupes arrival bursts.
    """

    def __init__(
        self,
        *,
        familiar_id: str,
        store: AsyncHistoryStore,
        subscriptions: SubscriptionRegistry,
        clock: Callable[[], float] = time.monotonic,
        idle_wake_seconds: float = _DEFAULT_IDLE_WAKE_S,
    ) -> None:
        self._familiar_id = familiar_id
        self._store = store
        self._subscriptions = subscriptions
        self._text_focus: int | None = None
        self._voice_focus: int | None = None
        self._pending_shift: dict[str, int] = {}  # modality -> channel_id
        self._text_lock = asyncio.Lock()
        self._voice_lock = asyncio.Lock()
        # idle-nudge state: last focused-turn time + dedupe flag
        self._clock = clock
        self._idle_wake_seconds = idle_wake_seconds
        self._last_active = clock()
        self._nudge_pending = False
        # channel_id → display name; populated by bot on_ready
        self.channel_names: dict[int, str] = {}

    async def initialize(self) -> None:
        """Load persisted focus pointers from DB."""
        ptrs = await self._store.get_focus_pointers(self._familiar_id)
        if ptrs is not None:
            self._text_focus = ptrs.text_channel_id
            self._voice_focus = ptrs.voice_channel_id
            _logger.info(
                f"{ls.tag('Focus', ls.LC)} loaded "
                f"{ls.kv('text', self.channel_label(self._text_focus), vc=ls.LW)} "
                f"{ls.kv('voice', self.channel_label(self._voice_focus), vc=ls.LW)}"
            )

    def get_focus(self, modality: str) -> int | None:
        """Return current focus channel_id for modality."""
        return self._text_focus if modality == "text" else self._voice_focus

    def is_focused(self, channel_id: int) -> bool:
        """Check if channel_id is the active text or voice focus."""
        return channel_id in {self._text_focus, self._voice_focus}

    def defer_shift(self, channel_id: int) -> None:
        """Register deferred focus shift; modality inferred from subscriptions."""
        kind = self._subscriptions.kind_for(channel_id)
        modality = "voice" if kind is SubscriptionKind.voice else "text"
        self._pending_shift[modality] = channel_id
        _logger.debug("defer shift channel=%d modality=%s", channel_id, modality)

    def should_wake(self, channel_id: int) -> bool:
        """Whether a non-focused arrival warrants an idle nudge.

        True when nudges enabled, channel unfocused, focused channel
        silent ≥ threshold, and no nudge already pending.
        """
        if self._idle_wake_seconds <= 0 or self._nudge_pending:
            return False
        if self.is_focused(channel_id):
            return False
        return (self._clock() - self._last_active) >= self._idle_wake_seconds

    def mark_nudge_pending(self) -> None:
        """Flag that a nudge turn is in flight; cleared at end_turn."""
        self._nudge_pending = True

    async def end_turn(self) -> None:
        """Apply pending focus shifts; called after each turn completes."""
        # focused turn happened → reset idle clock + release nudge dedupe
        self._last_active = self._clock()
        self._nudge_pending = False
        for modality, channel_id in list(self._pending_shift.items()):
            lock = self._voice_lock if modality == "voice" else self._text_lock
            async with lock:
                del self._pending_shift[modality]
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

    def channel_label(self, channel_id: int | None) -> str:
        """Format channel_id as '#name(id)' or '#id' when name unknown."""
        if channel_id is None:
            return "none"
        name = self.channel_names.get(channel_id)
        return f"#{name}({channel_id})" if name else f"#{channel_id}"

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
