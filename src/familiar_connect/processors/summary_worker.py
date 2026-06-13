"""Watermark-driven summary worker.

Regenerates two kinds of summaries from raw ``turns`` table:

- **Focus-stream rolling summary** (stored in ``summaries`` under
  ``FOCUS_STREAM_CHANNEL_ID``): the consumed cross-channel stream —
  the conversation the familiar actually attended to. When
  ``turns_threshold`` new consumed turns accumulate past the composite
  ``(consumed_at, id)`` watermark, worker compounds prior summary plus
  new turns via LLM. Watermarking on ``consumed_at`` (not ``id``)
  catches late-promoted staged turns. First run bounded by
  ``backfill_cap``.
- **Cross-channel summary** (stored in ``cross_context_summaries``):
  per ``(viewer_channel, source_channel)`` pair listed in
  ``cross_channel_map``, summary produced whenever source channel
  gains ``cross_k`` new turns since last cached entry. Used by
  :class:`CrossChannelContextLayer`.

All LLM traffic is ``chat`` (not ``chat_stream``) — summary worker
runs off hot path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.history.store import FOCUS_STREAM_CHANNEL_ID
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.summary_worker")


class SummaryWorker:
    """Rebuilds side-index summaries off SQLite watermarks."""

    name: str = "summary-worker"

    def __init__(
        self,
        *,
        store: AsyncHistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        turns_threshold: int = 10,
        backfill_cap: int = 200,
        cross_channel_map: dict[int, list[int]] | None = None,
        cross_k: int = 5,
        tick_interval_s: float = 5.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._turns_threshold = max(1, turns_threshold)
        self._backfill_cap = max(1, backfill_cap)
        self._cross_map = cross_channel_map or {}
        self._cross_k = max(1, cross_k)
        self._tick_interval_s = tick_interval_s

    # ------------------------------------------------------------------
    # Public loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Forever loop; tick on interval. Cancel to stop."""
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — worker must not die
                _logger.warning(
                    f"{ls.tag('SummaryWorker', ls.R)} "
                    f"{ls.kv('tick_error', repr(exc), vc=ls.R)}"
                )
            await asyncio.sleep(self._tick_interval_s)

    @span("summary.tick")
    async def tick(self) -> None:
        """Refresh focus-stream + cross-channel summaries."""
        await self._refresh_focus_stream()

        for viewer_channel_id, sources in self._cross_map.items():
            for source_id in sources:
                await self._maybe_refresh_cross(viewer_channel_id, source_id)

    # ------------------------------------------------------------------
    # Focus-stream rolling summary
    # ------------------------------------------------------------------

    async def _refresh_focus_stream(self) -> None:
        """Compound the consumed cross-channel stream into one summary.

        Follows attention, not channels: watermark is the composite
        ``(consumed_at, id)`` cursor, so late-promoted staged turns aren't
        missed. Stored under ``FOCUS_STREAM_CHANNEL_ID``. First run is
        bounded by ``backfill_cap`` then compounds forward.
        """
        prior = await self._store.get_summary(
            familiar_id=self._familiar_id, channel_id=FOCUS_STREAM_CHANNEL_ID
        )
        if prior is not None:
            after_consumed_at = prior.last_consumed_at or ""
            after_id = prior.last_summarised_id
        else:
            after_consumed_at = ""
            after_id = 0

        new_turns = await self._store.consumed_turns_after(
            familiar_id=self._familiar_id,
            after_consumed_at=after_consumed_at,
            after_id=after_id,
            limit=max(self._turns_threshold, self._backfill_cap),
        )
        if len(new_turns) < self._turns_threshold:
            return

        prompt = _build_rolling_prompt(
            prior_summary=prior.summary_text if prior is not None else None,
            new_turns=new_turns,
        )
        reply = await self._llm.chat(prompt)
        text = reply.content_str.strip()
        if not text:
            return
        last = new_turns[-1]
        last_consumed = last.consumed_at.isoformat() if last.consumed_at else None
        await self._store.put_summary(
            familiar_id=self._familiar_id,
            channel_id=FOCUS_STREAM_CHANNEL_ID,
            last_summarised_id=last.id,
            summary_text=text,
            last_consumed_at=last_consumed,
        )
        _logger.info(
            f"{ls.tag('Summary', ls.LC)} "
            f"{ls.kv('focus_stream', str(len(new_turns)), vc=ls.LY)} "
            f"{ls.kv('watermark', str(last.id), vc=ls.LC)} "
            f"{ls.kv('chars', str(len(text)), vc=ls.LW)}"
        )

    # ------------------------------------------------------------------
    # Cross-channel summary
    # ------------------------------------------------------------------

    async def _maybe_refresh_cross(
        self, viewer_channel_id: int, source_channel_id: int
    ) -> None:
        source_latest = await self._store.latest_id(
            familiar_id=self._familiar_id, channel_id=source_channel_id
        )
        if source_latest is None or source_latest <= 0:
            return
        viewer_mode = f"voice:{viewer_channel_id}"
        prior = await self._store.get_cross_context(
            familiar_id=self._familiar_id,
            viewer_mode=viewer_mode,
            source_channel_id=source_channel_id,
        )
        prior_source_id = prior.source_last_id if prior is not None else 0
        gained = source_latest - prior_source_id
        if prior is not None and gained < self._cross_k:
            return
        if prior is None and source_latest < self._cross_k:
            return

        turns = await self._turns_in_range(
            channel_id=source_channel_id,
            min_id_exclusive=prior_source_id,
            max_id_inclusive=source_latest,
        )
        if not turns:
            return
        prompt = _build_cross_prompt(
            prior_summary=prior.summary_text if prior is not None else None,
            source_channel_id=source_channel_id,
            new_turns=turns,
        )
        reply = await self._llm.chat(prompt)
        text = reply.content_str.strip()
        if not text:
            return
        await self._store.put_cross_context(
            familiar_id=self._familiar_id,
            viewer_mode=viewer_mode,
            source_channel_id=source_channel_id,
            source_last_id=source_latest,
            summary_text=text,
        )
        _logger.info(
            f"{ls.tag('Cross', ls.LC)} "
            f"{ls.kv('viewer', viewer_mode, vc=ls.LC)} "
            f"{ls.kv('source', str(source_channel_id), vc=ls.LY)} "
            f"{ls.kv('watermark', str(source_latest), vc=ls.LC)}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _turns_in_range(
        self,
        *,
        channel_id: int,
        min_id_exclusive: int,
        max_id_inclusive: int,
    ) -> list[HistoryTurn]:
        return await self._store.turns_in_id_range(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            min_id_exclusive=min_id_exclusive,
            max_id_inclusive=max_id_inclusive,
        )


# ---------------------------------------------------------------------------
# Prompt builders (private — swap for a Jinja template if they get fancier)
# ---------------------------------------------------------------------------


def _build_rolling_prompt(
    *,
    prior_summary: str | None,
    new_turns: Iterable[HistoryTurn],
) -> list[Message]:
    header = (
        "You produce concise, retrieval-friendly summaries of a familiar's "
        "attended conversation across channels. 3-5 sentences. Preserve "
        "proper nouns, commitments, and open questions. Omit small talk."
    )
    body_lines: list[str] = []
    if prior_summary:
        body_lines.extend(["Previous summary:\n" + prior_summary, "\nNew turns:"])
    else:
        body_lines.append("Turns:")
    for t in new_turns:
        who = t.author.display_name if t.author is not None else t.role
        body_lines.append(f"- [#{t.channel_id} {who}] {t.content}")
    user = "\n".join(body_lines)
    return [
        Message(role="system", content=header),
        Message(role="user", content=user),
    ]


def _build_cross_prompt(
    *,
    prior_summary: str | None,
    source_channel_id: int,
    new_turns: Iterable[HistoryTurn],
) -> list[Message]:
    header = (
        "You are producing a short briefing about what's been "
        f"happening in channel #{source_channel_id}. 2-3 sentences. "
        "Strip interpersonal chatter; keep topics, decisions, names."
    )
    body_lines: list[str] = []
    if prior_summary:
        body_lines.extend([
            "Previous briefing:\n" + prior_summary,
            "\nNew turns since:",
        ])
    else:
        body_lines.append("Turns:")
    for t in new_turns:
        who = t.author.display_name if t.author is not None else t.role
        body_lines.append(f"- [{who}] {t.content}")
    return [
        Message(role="system", content=header),
        Message(role="user", content="\n".join(body_lines)),
    ]
