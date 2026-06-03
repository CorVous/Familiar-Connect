"""Attentional focus controller for a single familiar."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.subscriptions import SubscriptionKind

if TYPE_CHECKING:
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.subscriptions import SubscriptionRegistry

_logger = logging.getLogger(__name__)


class FocusManager:
    """Per-familiar attentional focus controller.

    Two independent focus pointers (text, voice). Focus shifts are
    model-decided (via shift_focus tool), deferred to end_turn,
    then applied atomically under per-modality lock.
    """

    def __init__(
        self,
        *,
        familiar_id: str,
        store: AsyncHistoryStore,
        subscriptions: SubscriptionRegistry,
    ) -> None:
        self._familiar_id = familiar_id
        self._store = store
        self._subscriptions = subscriptions
        self._text_focus: int | None = None
        self._voice_focus: int | None = None
        self._pending_shift: dict[str, int] = {}  # modality -> channel_id
        self._text_lock = asyncio.Lock()
        self._voice_lock = asyncio.Lock()
        # idle gate: held while a turn is active; non-focused arrivals
        # contend for it to detect idle and trigger wake
        self._idle_gate = asyncio.Lock()
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
                f"{ls.kv('text', self._ch(self._text_focus), vc=ls.LW)} "
                f"{ls.kv('voice', self._ch(self._voice_focus), vc=ls.LW)}"
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

    async def end_turn(self) -> None:
        """Apply pending focus shifts; called after each turn completes."""
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
                        f"{ls.kv('text', self._ch(channel_id), vc=ls.LW)} "
                        f"{ls.kv('promoted', str(count), vc=ls.LG)}"
                    )
                else:
                    self._voice_focus = channel_id
                    _logger.info(
                        f"{ls.tag('🔀 Focus', ls.LC)} "
                        f"{ls.kv('voice', self._ch(channel_id), vc=ls.LW)}"
                    )
                await self._store.set_focus_pointers(
                    self._familiar_id,
                    text_channel_id=self._text_focus,
                    voice_channel_id=self._voice_focus,
                )

    def _ch(self, channel_id: int | None) -> str:
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
            f"{ls.kv(modality, self._ch(channel_id), vc=ls.LW)}"
        )
