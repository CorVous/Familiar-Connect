"""Chunked text-channel response delivery with mid-flight cancellation.

Used by ``_run_text_response`` when ``ChannelConfig.typing_simulation.enabled``
is True. Splits the LLM reply into paragraphs (with sentence fallback for
oversize paragraphs), then for each chunk shows ``typing…`` for a
length-proportional delay, sends it, pauses, repeats. A new user message
on the channel cancels the remaining chunks; already-sent chunks are
persisted as the assistant turn.

- :func:`split_reply_into_chunks` — pure; paragraph → sentence → hard-cap
- :func:`compute_typing_delay` — pure; chars/sec clamped to [min, max]
- :class:`TextDeliveryTracker` — per-channel in-flight task + sent scratch
- :class:`TextDeliveryRegistry` — lazy per-channel tracker lookup
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from familiar_connect.config import TypingSimulationConfig


_logger = logging.getLogger(__name__)

# Discord hard limit is 2000 chars; leave a bit of headroom for the
# typing-sim path which sends chunks one at a time.
_DISCORD_HARD_CAP = 1900

# Simple English sentence splitter — splits after . ! ? … when followed
# by whitespace. Punts on abbreviations (Dr., U.S.); acceptable for RP prose.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+")


def split_reply_into_chunks(
    text: str,
    *,
    sentence_split_threshold: int,
    hard_cap: int = _DISCORD_HARD_CAP,
) -> list[str]:
    """Split *text* into delivery chunks.

    Paragraphs (blank-line separated) are the primary unit. A paragraph
    whose length exceeds *sentence_split_threshold* is further split at
    sentence boundaries. Any chunk still over *hard_cap* is split at the
    last whitespace before the cap (rare; safety net for Discord).

    Empty/whitespace chunks are dropped. Order preserved.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= sentence_split_threshold:
            chunks.append(para)
            continue
        # oversize paragraph → split by sentence
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
        chunks.extend(sentences or [para])

    # final safety net: hard-cap any remaining oversize chunk
    capped: list[str] = []
    for chunk in chunks:
        remaining = chunk
        while len(remaining) > hard_cap:
            # prefer split at last whitespace before cap
            cut = remaining.rfind(" ", 0, hard_cap)
            if cut <= 0:
                cut = hard_cap
            capped.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            capped.append(remaining)
    return capped


def compute_typing_delay(chunk: str, cfg: TypingSimulationConfig) -> float:
    """Return typing-indicator duration for *chunk*, clamped to config range.

    ``len(chunk) / cfg.chars_per_second`` clamped to
    ``[cfg.min_delay_s, cfg.max_delay_s]``. Guards against
    ``chars_per_second <= 0`` by returning ``cfg.max_delay_s``.
    """
    if cfg.chars_per_second <= 0:
        return cfg.max_delay_s
    raw = len(chunk) / cfg.chars_per_second
    return max(cfg.min_delay_s, min(cfg.max_delay_s, raw))


@dataclass
class TextDeliveryTracker:
    """Per-channel in-flight delivery state.

    Pure state — behaviour lives in :mod:`familiar_connect.bot`. Pattern
    mirrors :class:`familiar_connect.voice.interruption.ResponseTracker`.

    :param channel_id: Discord channel this tracker serves.
    :param task: in-flight delivery task, or ``None`` when idle.
    :param sent_chunks: chunks already committed to Discord; flushed as
        the assistant turn on completion or cancellation.
    """

    channel_id: int
    task: asyncio.Task[Any] | None = None
    sent_chunks: list[str] = field(default_factory=list)

    def start(self, task: asyncio.Task[Any]) -> None:
        """Register *task* as the in-flight delivery; clears any prior scratch."""
        self.task = task
        self.sent_chunks = []

    def mark_sent(self, chunk: str) -> None:
        """Record *chunk* as successfully delivered to Discord."""
        self.sent_chunks.append(chunk)

    def is_active(self) -> bool:
        """Return ``True`` when a delivery task is registered and not done."""
        return self.task is not None and not self.task.done()

    async def cancel_and_wait(self) -> list[str]:
        """Cancel the in-flight delivery and await its completion.

        Returns the snapshot of already-sent chunks. Safe to call when
        idle (returns an empty list and no-ops).
        """
        if self.task is None:
            return []
        if not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        sent = list(self.sent_chunks)
        self.task = None
        return sent

    def clear(self) -> None:
        """Drop task reference and sent-chunks scratch."""
        self.task = None
        self.sent_chunks = []


class TextDeliveryRegistry:
    """Per-channel :class:`TextDeliveryTracker` lookup; lazy-creates on first use.

    Mirrors :class:`familiar_connect.voice.interruption.ResponseTrackerRegistry`.
    """

    def __init__(self) -> None:
        self._by_channel: dict[int, TextDeliveryTracker] = {}

    def get(self, channel_id: int) -> TextDeliveryTracker:
        """Return the tracker for *channel_id*, creating one on first use."""
        tracker = self._by_channel.get(channel_id)
        if tracker is None:
            tracker = TextDeliveryTracker(channel_id=channel_id)
            self._by_channel[channel_id] = tracker
        return tracker

    def drop(self, channel_id: int) -> None:
        """Remove *channel_id*'s tracker (called on unsubscribe)."""
        self._by_channel.pop(channel_id, None)

    def snapshot(self) -> dict[int, TextDeliveryTracker]:
        """Return a shallow copy of the channel→tracker map (for tests)."""
        return dict(self._by_channel)
