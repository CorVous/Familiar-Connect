"""Watermark-driven people-dossier worker.

For each ``canonical_key`` that appears as a subject on at least one
current fact, maintains a compounded summary in ``people_dossiers``.
Refreshes when the subject's max ``facts.id`` exceeds the dossier's
``last_fact_id`` watermark — same compounding shape as
:class:`SummaryWorker` so dossier cost stays bounded.

Cadence is intentionally a quarter of :class:`SummaryWorker`'s tick
(20 s vs 5 s): people facts churn slower than turn-by-turn summaries,
and dossiers are read off SQLite by :class:`PeopleDossierLayer`
during prompt assembly — the read path doesn't wait on the worker.

All LLM traffic is ``chat`` (not ``chat_stream``) — the worker runs
off the hot path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.history.store import Fact, HistoryStore
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.people_dossier_worker")


class PeopleDossierWorker:
    """Rebuilds per-person dossiers off the facts watermark."""

    name: str = "people-dossier-worker"

    def __init__(
        self,
        *,
        store: HistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        tick_interval_s: float = 20.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
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
                    f"{ls.tag('PeopleDossier', ls.R)} "
                    f"{ls.kv('tick_error', repr(exc), vc=ls.R)}"
                )
            await asyncio.sleep(self._tick_interval_s)

    @span("people_dossier.tick")
    async def tick(self) -> None:
        """Refresh any dossier whose subject has new facts."""
        latest_per_subject = self._store.subjects_with_facts(
            familiar_id=self._familiar_id
        )
        for canonical_key, latest_fact_id in latest_per_subject.items():
            await self._maybe_refresh(canonical_key, latest_fact_id)

    # ------------------------------------------------------------------
    # Per-subject refresh
    # ------------------------------------------------------------------

    async def _maybe_refresh(self, canonical_key: str, latest_fact_id: int) -> None:
        prior = self._store.get_people_dossier(
            familiar_id=self._familiar_id, canonical_key=canonical_key
        )
        prior_wm = prior.last_fact_id if prior is not None else 0
        if latest_fact_id <= prior_wm:
            return

        new_facts = self._store.facts_for_subject(
            familiar_id=self._familiar_id,
            canonical_key=canonical_key,
            min_id_exclusive=prior_wm,
        )
        if not new_facts:
            return

        display = self._store.resolve_label(
            canonical_key=canonical_key,
            guild_id=None,
            familiar_id=self._familiar_id,
        )
        prompt = _build_dossier_prompt(
            display_name=display,
            prior_dossier=prior.dossier_text if prior is not None else None,
            new_facts=new_facts,
        )
        reply = await self._llm.chat(prompt)
        text = reply.content.strip()
        if not text:
            # Don't overwrite a real dossier with an empty reply.
            return
        self._store.put_people_dossier(
            familiar_id=self._familiar_id,
            canonical_key=canonical_key,
            last_fact_id=latest_fact_id,
            dossier_text=text,
        )
        _logger.info(
            f"{ls.tag('PeopleDossier', ls.LC)} "
            f"{ls.kv('subject', canonical_key, vc=ls.LY)} "
            f"{ls.kv('display', display, vc=ls.LW)} "
            f"{ls.kv('watermark', str(latest_fact_id), vc=ls.LC)} "
            f"{ls.kv('chars', str(len(text)), vc=ls.LW)}"
        )


# ---------------------------------------------------------------------------
# Prompt builder (private — swap for a Jinja template if it grows)
# ---------------------------------------------------------------------------


def _build_dossier_prompt(
    *,
    display_name: str,
    prior_dossier: str | None,
    new_facts: Iterable[Fact],
) -> list[Message]:
    header = (
        "You maintain a short, retrieval-friendly dossier about one "
        f"person ({display_name}) — 3-5 sentences. Preserve concrete "
        "details, names, places, commitments. Drop transient feelings "
        "and conversational filler. Reconcile contradictions in favour "
        "of newer evidence. Reply with the updated dossier text only."
    )
    body_lines: list[str] = []
    if prior_dossier:
        body_lines.extend([
            "Previous dossier:\n" + prior_dossier,
            "\nNew facts:",
        ])
    else:
        body_lines.append("Facts:")
    body_lines.extend(f"- {f.text}" for f in new_facts)
    return [
        Message(role="system", content=header),
        Message(role="user", content="\n".join(body_lines)),
    ]
