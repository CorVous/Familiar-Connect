"""Watermark-driven summary worker.

Regenerates two kinds of summaries from the raw ``turns`` table:

- **Per-channel rolling summary** (stored in ``summaries``): when the
  channel's ``latest_id - last_summarised_id`` exceeds
  ``turns_threshold``, the worker produces a new summary by feeding
  the prior summary plus the new turns to the LLM. Compounding
  strategy keeps cost bounded; a periodic full recompute is a
  future-phase addition (see plan § Design.4).
- **Cross-channel summary** (stored in ``cross_context_summaries``):
  per ``(viewer_channel, source_channel)`` pair listed in
  ``cross_channel_map``, a summary is produced whenever the source
  channel has gained ``cross_k`` new turns since the last cached
  entry. Used by :class:`CrossChannelContextLayer`.

All LLM traffic is ``chat`` (not ``chat_stream``) — the summary
worker runs off the hot path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.history.store import _TURN_COLS, _row_to_turn
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.history.store import HistoryStore, HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.summary_worker")


class SummaryWorker:
    """Rebuilds side-index summaries off SQLite watermarks."""

    name: str = "summary-worker"

    def __init__(
        self,
        *,
        store: HistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        turns_threshold: int = 10,
        cross_channel_map: dict[int, list[int]] | None = None,
        cross_k: int = 5,
        tick_interval_s: float = 5.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._turns_threshold = max(1, turns_threshold)
        self._cross_map = cross_channel_map or {}
        self._cross_k = max(1, cross_k)
        self._tick_interval_s = tick_interval_s

    # ------------------------------------------------------------------
    # Public loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Forever loop — tick on an interval. Cancel to stop."""
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
        """One pass: refresh any stale rolling + cross-channel summaries."""
        channels = self._channels_with_turns()
        for channel_id in channels:
            await self._maybe_refresh_rolling(channel_id)

        for viewer_channel_id, sources in self._cross_map.items():
            for source_id in sources:
                await self._maybe_refresh_cross(viewer_channel_id, source_id)

    # ------------------------------------------------------------------
    # Rolling summary
    # ------------------------------------------------------------------

    async def _maybe_refresh_rolling(self, channel_id: int) -> None:
        latest = self._store.latest_id(
            familiar_id=self._familiar_id, channel_id=channel_id
        )
        if latest is None or latest <= 0:
            return
        prior = self._store.get_summary(
            familiar_id=self._familiar_id, channel_id=channel_id
        )
        last_summarised = prior.last_summarised_id if prior is not None else 0
        if latest - last_summarised < self._turns_threshold:
            return

        new_turns = self._turns_in_range(
            channel_id=channel_id,
            min_id_exclusive=last_summarised,
            max_id_inclusive=latest,
        )
        if not new_turns:
            return

        prompt = _build_rolling_prompt(
            prior_summary=prior.summary_text if prior is not None else None,
            new_turns=new_turns,
        )
        reply = await self._llm.chat(prompt)
        text = reply.content.strip()
        if not text:
            return
        self._store.put_summary(
            familiar_id=self._familiar_id,
            channel_id=channel_id,
            last_summarised_id=latest,
            summary_text=text,
        )
        _logger.info(
            f"{ls.tag('Summary', ls.LC)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LY)} "
            f"{ls.kv('watermark', str(latest), vc=ls.LC)} "
            f"{ls.kv('chars', str(len(text)), vc=ls.LW)}"
        )

    # ------------------------------------------------------------------
    # Cross-channel summary
    # ------------------------------------------------------------------

    async def _maybe_refresh_cross(
        self, viewer_channel_id: int, source_channel_id: int
    ) -> None:
        source_latest = self._store.latest_id(
            familiar_id=self._familiar_id, channel_id=source_channel_id
        )
        if source_latest is None or source_latest <= 0:
            return
        viewer_mode = f"voice:{viewer_channel_id}"
        prior = self._store.get_cross_context(
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

        turns = self._turns_in_range(
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
        text = reply.content.strip()
        if not text:
            return
        self._store.put_cross_context(
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

    def _channels_with_turns(self) -> set[int]:
        """Return channels that have at least one turn for this familiar.

        Small N — a single scan of distinct channel ids is fine.
        """
        rows = self._store._conn.execute(  # noqa: SLF001 — tight coupling OK for the worker
            """
            SELECT DISTINCT channel_id
              FROM turns
             WHERE familiar_id = ?
            """,
            (self._familiar_id,),
        ).fetchall()
        return {int(r["channel_id"]) for r in rows}

    def _turns_in_range(
        self,
        *,
        channel_id: int,
        min_id_exclusive: int,
        max_id_inclusive: int,
    ) -> list[HistoryTurn]:
        rows = self._store._conn.execute(  # noqa: SLF001
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ?
               AND channel_id = ?
               AND id > ?
               AND id <= ?
             ORDER BY id ASC
            """,  # noqa: S608
            (self._familiar_id, channel_id, min_id_exclusive, max_id_inclusive),
        ).fetchall()
        return [_row_to_turn(r) for r in rows]


# ---------------------------------------------------------------------------
# Prompt builders (private — swap for a Jinja template if they get fancier)
# ---------------------------------------------------------------------------


def _build_rolling_prompt(
    *,
    prior_summary: str | None,
    new_turns: Iterable[HistoryTurn],
) -> list[Message]:
    header = (
        "You produce concise, retrieval-friendly summaries of chat "
        "history. 3-5 sentences. Preserve proper nouns, commitments, "
        "and open questions. Omit small talk."
    )
    body_lines: list[str] = []
    if prior_summary:
        body_lines.extend(["Previous summary:\n" + prior_summary, "\nNew turns:"])
    else:
        body_lines.append("Turns:")
    for t in new_turns:
        who = t.author.display_name if t.author is not None else t.role
        body_lines.append(f"- [{who}] {t.content}")
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
