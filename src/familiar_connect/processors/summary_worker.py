"""Watermark-driven summary worker.

Regenerates the focus-stream rolling summary from the raw ``turns``
table:

- **Focus-stream rolling summary** (stored in ``summaries`` under
  ``FOCUS_STREAM_CHANNEL_ID``): the consumed cross-channel stream —
  the conversation the familiar actually attended to. When
  ``turns_threshold`` new consumed turns accumulate past the composite
  ``(consumed_at, id)`` watermark, worker compounds prior summary plus
  new turns via LLM. Watermarking on ``consumed_at`` (not ``id``)
  catches late-promoted staged turns. First run bounded by
  ``backfill_cap``.

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
        tick_interval_s: float = 5.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._turns_threshold = max(1, turns_threshold)
        self._backfill_cap = max(1, backfill_cap)
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
        """Refresh focus-stream summary."""
        await self._refresh_focus_stream()

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
