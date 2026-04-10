"""HistoryProvider — sliding window of recent turns + rolling summary.

Step 6 of future-features/context-management.md. Reads turns from the
:class:`HistoryStore` for the request's
``(familiar_id, channel_id)`` and emits up to two contributions:

- A ``recent_history`` Contribution containing the most recent N
  turns *in this channel* rendered as a single block of text. The
  recent window is partitioned per channel so two simultaneous
  conversations don't bleed into each other.
- A ``history_summary`` Contribution containing a rolling summary of
  every turn the familiar has heard *globally* (across all channels)
  except the most recent ``summary_lag`` turns, built by a cheap
  :class:`SideModel` and cached in the store under ``familiar_id``.
  The cache is keyed by ``last_summarised_id``; the summariser is
  only re-invoked when enough new turns have arrived globally that
  the cached watermark has fallen behind.

The provider has its own internal soft deadline for the summariser,
shorter than the pipeline's per-provider deadline, so a slow side-
model never costs us the recent-history layer. If the summariser
exceeds the soft deadline (or raises) and there is a stale cached
summary, the stale value is returned. If neither path produces text,
the provider returns just the recent layer and lets the pipeline
move on.

The split between per-channel recent window and per-familiar global
summary is the "hybrid" option from
``future-features/configuration-levels.md``, picked specifically
because it forces multi-channel scalability to be a first-class
concern from day one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer

if TYPE_CHECKING:
    from familiar_connect.config import ChannelMode
    from familiar_connect.context.side_model import SideModel
    from familiar_connect.context.types import ContextRequest
    from familiar_connect.history.store import HistoryStore, HistoryTurn


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

DEFAULT_WINDOW_SIZE = 20
"""How many of the most recent turns to surface verbatim by default."""

DEFAULT_DEADLINE_S = 15.0
"""Hard cap the pipeline enforces on this provider's contribute() call."""

DEFAULT_SUMMARY_TIMEOUT_S = 12.0
"""Soft cap on the summariser sub-call. Strictly less than
DEFAULT_DEADLINE_S so the recent-history layer always returns even
when the cheap model stalls."""

DEFAULT_MAX_SUMMARY_TOKENS = 256
"""Approximate target length the summariser is asked to produce."""

_SUMMARY_PROMPT_TEMPLATE = (
    "Summarise the following conversation in at most {max_tokens} tokens. "
    "Focus on facts, decisions, and emotional beats that matter for "
    "continuing the conversation. Plain prose, no headings.\n\n"
    "----- conversation -----\n"
    "{transcript}\n"
    "----- end conversation -----\n\n"
    "Summary:"
)


class HistoryProvider:
    """ContextProvider that surfaces recent turns + a rolling summary.

    Conforms to the ContextProvider Protocol structurally — no
    inheritance required.

    :param store: The :class:`HistoryStore` to read turns from and
        cache summaries in.
    :param side_model: The :class:`SideModel` used to build new
        summaries when the cache is stale.
    :param window_size: Number of recent turns *per channel* to
        surface verbatim. Defaults to ``DEFAULT_WINDOW_SIZE``.
    :param summary_lag: Number of most-recent-globally turns to
        exclude from the summary content. Defaults to ``window_size``,
        which means the summary covers everything older than the
        global rolling window. Setting it equal to ``window_size``
        also means single-channel scenarios produce zero overlap
        between the recent layer and the summary layer.
    :param mode: The :class:`ChannelMode` this provider instance is
        scoped to. One provider is constructed per turn inside
        :meth:`Familiar.build_pipeline`, so the mode is fixed for
        the lifetime of the provider. When ``None``, no mode
        filtering is applied (backwards-compatible default for
        existing callers and tests).
    :param summary_timeout_s: Soft cap on the summariser sub-call.
    :param max_summary_tokens: Approximate target length for the
        summariser's output.
    """

    id = "history"
    deadline_s = DEFAULT_DEADLINE_S

    def __init__(
        self,
        *,
        store: HistoryStore,
        side_model: SideModel,
        window_size: int = DEFAULT_WINDOW_SIZE,
        summary_lag: int | None = None,
        mode: ChannelMode | None = None,
        summary_timeout_s: float = DEFAULT_SUMMARY_TIMEOUT_S,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
    ) -> None:
        if window_size <= 0:
            msg = f"window_size must be > 0, got {window_size}"
            raise ValueError(msg)
        self._store = store
        self._side_model = side_model
        self._window_size = window_size
        self._summary_lag = summary_lag if summary_lag is not None else window_size
        self._mode = mode
        self._summary_timeout_s = summary_timeout_s
        self._max_summary_tokens = max_summary_tokens

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
            # No global history at all — nothing to surface, anywhere.
            return []

        contributions: list[Contribution] = []
        if recent:
            contributions.append(self._recent_contribution(recent))

        # Only build a summary if there's enough global history that
        # at least one turn lives outside the rolling-window region.
        if latest <= self._summary_lag:
            return contributions

        target_max_id = latest - self._summary_lag
        summary_text = await self._fetch_or_build_summary(request, target_max_id)
        if summary_text:
            contributions.append(self._summary_contribution(summary_text))

        return contributions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch_or_build_summary(
        self,
        request: ContextRequest,
        target_max_id: int,
    ) -> str:
        cached = self._store.get_summary(
            familiar_id=request.familiar_id,
            channel_id=request.channel_id,
        )
        if cached is not None and cached.last_summarised_id >= target_max_id:
            return cached.summary_text

        # Cache is missing or stale: try to (re)build it under the soft
        # deadline. On any failure, fall back to whatever the cache has —
        # better stale than nothing.
        try:
            async with asyncio.timeout(self._summary_timeout_s):
                return await self._build_summary(request, target_max_id)
        except TimeoutError:
            _logger.warning(
                "history summariser timed out after %.3fs for "
                "familiar=%s channel=%s; falling back to cache",
                self._summary_timeout_s,
                request.familiar_id,
                request.channel_id,
            )
        except Exception as exc:  # noqa: BLE001 — isolation by design
            _logger.warning(
                "history summariser raised %s: %s; falling back to cache",
                type(exc).__name__,
                exc,
            )

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
            max_tokens=self._max_summary_tokens,
            transcript=_render_turns(older),
        )
        summary = await self._side_model.complete(
            prompt,
            max_tokens=self._max_summary_tokens,
        )
        if not summary:
            return ""

        # Update the cache so future calls reuse this work.
        self._store.put_summary(
            familiar_id=request.familiar_id,
            channel_id=request.channel_id,
            last_summarised_id=target_max_id,
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

    Format is intentionally simple: one ``<role>[: speaker]: content``
    line per turn. The speaker is included for user turns so the
    familiar can address callers by name; assistant turns are labelled
    with their role only.
    """
    lines: list[str] = []
    for t in turns:
        if t.role == "user" and t.speaker:
            lines.append(f"{t.role} ({t.speaker}): {t.content}")
        else:
            lines.append(f"{t.role}: {t.content}")
    return "\n".join(lines)
