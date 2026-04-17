"""HistoryProvider — sliding window of recent turns + rolling summary.

Emits up to two contributions per request:

- ``recent_history`` — most recent N turns *per channel*, rendered as text.
- ``history_summary`` — rolling LLM-built summary of older turns *globally*,
  cached by ``last_summarised_id`` and rebuilt only when stale.

Summariser runs **off the critical path**: on cache miss or stale cache,
``contribute`` returns the current cache (empty or stale) immediately and
schedules a background ``asyncio.Task`` to rebuild. The fresh value lands
on the next turn. Soft deadline on the background task guards against
runaway stalls. Dedupe is per provider instance (one task per
``(kind, familiar_id, channel_id, target_max_id)`` key). Familiar rebuilds
the provider per turn, so cross-turn duplicate refreshes are possible but
bounded — SQLite serialises cache writes and the later write wins.

See docs/architecture/context-pipeline.md and
docs/architecture/configuration-model.md for the hybrid per-channel /
per-familiar design rationale.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from familiar_connect.config import ChannelMode
from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer
from familiar_connect.identity import format_turn_for_transcript
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from familiar_connect.context.types import ContextRequest
    from familiar_connect.history.store import (
        HistoryStore,
        HistoryTurn,
        OtherChannelInfo,
    )
    from familiar_connect.llm import LLMClient

RefreshKey = tuple[Any, ...]


_logger = logging.getLogger(__name__)


HISTORY_RECENT_PRIORITY = 80
"""Priority assigned to the recent-history Contribution.

Lower than CharacterProvider (100) so the persona always survives
budget pressure ahead of mid-conversation context, but higher than
the eventual content-search and author-note layers."""

HISTORY_SUMMARY_PRIORITY = 60
"""Priority assigned to the rolling-summary Contribution.

Lower than recent_history because the summary is a derivative
artefact of the same conversation; if we have to drop something, we
drop the lossy summary first."""

CROSS_CONTEXT_PRIORITY = 55
"""Priority assigned to cross-context "meanwhile elsewhere" contributions.

Below within-channel summary (60) so budget pressure drops cross-channel
summaries first."""

_MAX_CROSS_CHANNELS = 2
"""How many other channels to surface cross-context summaries for."""

DEFAULT_WINDOW_SIZE = 20
"""How many of the most recent turns to surface verbatim by default."""

DEFAULT_DEADLINE_S = 15.0
"""Hard cap the pipeline enforces on this provider's contribute() call."""

DEFAULT_SUMMARY_TIMEOUT_S = 12.0
"""Soft cap on the summariser sub-call. Strictly less than
DEFAULT_DEADLINE_S so the recent-history layer always returns even
when the model stalls."""

_MAX_SUMMARY_TOKENS_HINT = 256
"""Approximate target length advertised to the summariser in the prompt."""

_SUMMARY_PROMPT_TEMPLATE = (
    "Summarise the following conversation in at most {max_tokens} tokens. "
    "Focus on facts, decisions, and emotional beats that matter for "
    "continuing the conversation. Plain prose, no headings.\n\n"
    "----- conversation -----\n"
    "{transcript}\n"
    "----- end conversation -----\n\n"
    "Summary:"
)

_MODE_DESCRIPTIONS: dict[str, str] = {
    "full_rp": "live roleplay scene",
    "text_conversation_rp": "text-message / chatroom conversation",
    "imitate_voice": "voice call",
}

_CROSS_CONTEXT_PROMPT_TEMPLATE = (
    "You are summarising recent activity from another channel for context. "
    "The reader is currently in a {viewer_mode_desc}. The source channel "
    "is a {source_mode_desc}. Summarise what happened in 1-2 sentences, "
    'framed as "Meanwhile, in the {source_mode_desc}, …". '
    "Focus on facts and decisions relevant to ongoing interaction.\n\n"
    "----- conversation -----\n"
    "{transcript}\n"
    "----- end conversation -----\n\n"
    "Summary:"
)


class HistoryProvider:
    """ContextProvider that surfaces recent turns + a rolling summary.

    ``window_size`` controls per-channel recent turns. ``summary_lag``
    (defaults to ``window_size``) excludes the N most-recent-globally
    turns from the summary, so single-channel scenarios produce zero
    overlap between recent and summary layers.

    Constructed per turn with a fixed ``mode``; ``None`` disables mode
    filtering (backwards-compatible default).
    """

    id = "history"
    deadline_s = DEFAULT_DEADLINE_S

    def __init__(
        self,
        *,
        store: HistoryStore,
        llm_client: LLMClient,
        window_size: int = DEFAULT_WINDOW_SIZE,
        summary_lag: int | None = None,
        mode: ChannelMode | None = None,
        summary_timeout_s: float = DEFAULT_SUMMARY_TIMEOUT_S,
    ) -> None:
        if window_size <= 0:
            msg = f"window_size must be > 0, got {window_size}"
            raise ValueError(msg)
        self._store = store
        self._llm_client = llm_client
        self._window_size = window_size
        self._summary_lag = summary_lag if summary_lag is not None else window_size
        self._mode = mode
        self._summary_timeout_s = summary_timeout_s
        # instance-local dedupe map; per-turn rebuild means cross-turn
        # dedupe is best-effort (see module docstring).
        self._pending_refresh: dict[RefreshKey, asyncio.Task[None]] = {}

    async def contribute(
        self,
        request: ContextRequest,
    ) -> list[Contribution]:
        """Return up to two Contributions for *request*.

        - No global history → empty list (the budgeter handles it).
        - Global history fits in ``window_size`` → only recent.
        - Global history overflows → recent + summary (cached when
          possible, regenerated when stale, falls back to the stale
          cache or to nothing if the summariser fails).
        """
        recent = self._store.recent(
            familiar_id=request.familiar_id,
            channel_id=request.channel_id,
            limit=self._window_size,
            mode=self._mode,
        )

        latest = self._store.latest_id(
            familiar_id=request.familiar_id,
            channel_id=request.channel_id,
        )
        if latest is None:
            # no global history at all — nothing to surface
            return []

        contributions: list[Contribution] = []
        if recent:
            contributions.append(self._recent_contribution(recent))

        # only build a summary if at least one turn lives outside the
        # rolling-window region
        if latest > self._summary_lag:
            target_max_id = latest - self._summary_lag
            summary_text = await self._fetch_or_build_summary(request, target_max_id)
            if summary_text:
                contributions.append(self._summary_contribution(summary_text))

        # cross-context: summarise activity in other channels
        if self._mode is not None:
            cross = await self._build_cross_context_contributions(request)
            contributions.extend(cross)

        return contributions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch_or_build_summary(
        self,
        request: ContextRequest,
        target_max_id: int,
    ) -> str:
        """Return cache now; schedule background refresh on miss/stale."""
        cached = self._store.get_summary(
            familiar_id=request.familiar_id,
            channel_id=request.channel_id,
        )
        if cached is not None and cached.last_summarised_id >= target_max_id:
            return cached.summary_text

        # cache missing or stale — fire-and-forget refresh, return
        # whatever the cache has now (possibly empty) immediately
        key: RefreshKey = (
            "summary",
            request.familiar_id,
            request.channel_id,
            target_max_id,
        )
        self._schedule_refresh(key, self._refresh_summary(request, target_max_id))
        return cached.summary_text if cached is not None else ""

    async def _build_summary(
        self,
        request: ContextRequest,
        target_max_id: int,
    ) -> str:
        older = self._store.older_than(
            familiar_id=request.familiar_id,
            max_id=target_max_id,
            channel_id=request.channel_id,
        )
        if not older:
            return ""

        prompt = _SUMMARY_PROMPT_TEMPLATE.format(
            max_tokens=_MAX_SUMMARY_TOKENS_HINT,
            transcript=_render_turns(older),
        )
        reply = await self._llm_client.chat(
            [Message(role="user", content=prompt)],
        )
        summary = reply.content
        if not summary:
            return ""

        # update cache so future calls reuse this work
        self._store.put_summary(
            familiar_id=request.familiar_id,
            channel_id=request.channel_id,
            last_summarised_id=target_max_id,
            summary_text=summary,
        )
        return summary

    async def _build_cross_context_contributions(
        self,
        request: ContextRequest,
    ) -> list[Contribution]:
        """Build cross-context summaries for other channels' activity.

        For ``full_rp`` mode the summaries are built and cached (so the
        renderer can read them for mid-chat breadcrumbs) but NOT emitted
        as ``Layer.history_summary`` contributions — that would duplicate
        the information the renderer already places at the gap.
        """
        assert self._mode is not None  # caller guards  # noqa: S101
        others = self._store.distinct_other_channels(
            familiar_id=request.familiar_id,
            exclude_channel_id=request.channel_id,
        )
        if not others:
            return []

        contributions: list[Contribution] = []
        viewer_mode_desc = _MODE_DESCRIPTIONS.get(self._mode.value, self._mode.value)
        emit = self._mode is not ChannelMode.full_rp

        for info in others[:_MAX_CROSS_CHANNELS]:
            text = await self._fetch_or_build_cross_context(
                request, info, viewer_mode_desc
            )
            if text and emit:
                contributions.append(
                    Contribution(
                        layer=Layer.history_summary,
                        priority=CROSS_CONTEXT_PRIORITY,
                        text=text,
                        estimated_tokens=estimate_tokens(text),
                        source=f"history:cross_channel:{info.channel_id}",
                    )
                )
        return contributions

    async def _fetch_or_build_cross_context(
        self,
        request: ContextRequest,
        info: OtherChannelInfo,
        viewer_mode_desc: str,
    ) -> str:
        """Return cache now; schedule background refresh on miss/stale."""
        assert self._mode is not None  # noqa: S101
        cached = self._store.get_cross_context(
            familiar_id=request.familiar_id,
            viewer_mode=self._mode.value,
            source_channel_id=info.channel_id,
        )
        if cached is not None and cached.source_last_id >= info.latest_id:
            return cached.summary_text

        # cache miss or stale — fire-and-forget refresh
        key: RefreshKey = (
            "cross",
            request.familiar_id,
            self._mode.value,
            info.channel_id,
            info.latest_id,
        )
        self._schedule_refresh(
            key,
            self._refresh_cross_context(request, info, viewer_mode_desc),
        )
        return cached.summary_text if cached is not None else ""

    def _schedule_refresh(
        self,
        key: RefreshKey,
        coro: Coroutine[Any, Any, None],
    ) -> None:
        """Schedule *coro* as a background task unless one already in flight for *key*.

        Dedupe is per-instance; see module docstring for cross-turn caveat.
        """
        if key in self._pending_refresh:
            coro.close()  # dedupe; identical refresh already running
            return
        task = asyncio.create_task(coro)
        self._pending_refresh[key] = task
        task.add_done_callback(
            lambda _t, k=key: self._pending_refresh.pop(k, None),
        )

    async def _refresh_summary(
        self,
        request: ContextRequest,
        target_max_id: int,
    ) -> None:
        """Background summariser refresh under soft deadline.

        Errors logged, swallowed.
        """
        try:
            async with asyncio.timeout(self._summary_timeout_s):
                await self._build_summary(request, target_max_id)
        except TimeoutError:
            _logger.warning(
                "history summariser background refresh timed out after %.3fs "
                "for familiar=%s channel=%s",
                self._summary_timeout_s,
                request.familiar_id,
                request.channel_id,
            )
        except Exception as exc:  # noqa: BLE001 — isolation by design
            _logger.warning(
                "history summariser background refresh raised %s: %s",
                type(exc).__name__,
                exc,
            )

    async def _refresh_cross_context(
        self,
        request: ContextRequest,
        info: OtherChannelInfo,
        viewer_mode_desc: str,
    ) -> None:
        """Background cross-context refresh under soft deadline.

        Errors logged, swallowed.
        """
        try:
            async with asyncio.timeout(self._summary_timeout_s):
                await self._build_cross_context(request, info, viewer_mode_desc)
        except TimeoutError:
            _logger.warning(
                "cross-context summariser background refresh timed out for channel=%s",
                info.channel_id,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "cross-context summariser background refresh raised %s: %s",
                type(exc).__name__,
                exc,
            )

    async def _build_cross_context(
        self,
        request: ContextRequest,
        info: OtherChannelInfo,
        viewer_mode_desc: str,
    ) -> str:
        """Build a cross-context POV summary for one other channel."""
        assert self._mode is not None  # noqa: S101
        turns = self._store.recent(
            familiar_id=request.familiar_id,
            channel_id=info.channel_id,
            limit=self._window_size,
        )
        if not turns:
            return ""

        source_mode_desc = _MODE_DESCRIPTIONS.get(
            info.mode or "", info.mode or "unknown"
        )
        prompt = _CROSS_CONTEXT_PROMPT_TEMPLATE.format(
            viewer_mode_desc=viewer_mode_desc,
            source_mode_desc=source_mode_desc,
            transcript=_render_turns(turns),
        )
        reply = await self._llm_client.chat(
            [Message(role="user", content=prompt)],
        )
        summary = reply.content
        if not summary:
            return ""

        self._store.put_cross_context(
            familiar_id=request.familiar_id,
            viewer_mode=self._mode.value,
            source_channel_id=info.channel_id,
            source_last_id=info.latest_id,
            summary_text=summary,
        )
        return summary

    def _recent_contribution(self, turns: list[HistoryTurn]) -> Contribution:
        text = _render_turns(turns)
        return Contribution(
            layer=Layer.recent_history,
            priority=HISTORY_RECENT_PRIORITY,
            text=text,
            estimated_tokens=estimate_tokens(text),
            source="history:recent",
        )

    def _summary_contribution(self, summary_text: str) -> Contribution:
        return Contribution(
            layer=Layer.history_summary,
            priority=HISTORY_SUMMARY_PRIORITY,
            text=summary_text,
            estimated_tokens=estimate_tokens(summary_text),
            source="history:summary",
        )


def _render_turns(turns: list[HistoryTurn]) -> str:
    """Render a list of HistoryTurns as a single text block.

    Delegates line formatting to :func:`identity.format_turn_for_transcript`
    so the history-summary provider and memory writer stay in sync.
    """
    return "\n".join(
        format_turn_for_transcript(t.role, t.author, t.content) for t in turns
    )
