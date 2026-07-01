"""Watermark-driven people-dossier worker.

For each ``canonical_key`` appearing as subject on at least one
current fact, maintains a compounded summary in ``people_dossiers``.
Refreshes when subject's max ``facts.id`` exceeds dossier's
``last_fact_id`` watermark — same compounding shape as
:class:`SummaryWorker` so dossier cost stays bounded.

Cadence intentionally a quarter of :class:`SummaryWorker`'s tick
(20 s vs 5 s): people facts churn slower than turn-by-turn summaries,
and dossiers read off SQLite by :class:`PeopleDossierLayer` during
prompt assembly — read path doesn't wait on worker.

All LLM traffic is ``chat`` (not ``chat_stream``) — worker runs off
hot path.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.identity import is_ego_key
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import Fact
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.people_dossier_worker")

# self-dossier is always-injected; the dream pass writes momentary "texture"
# opinions at low importance that should stay out of it (they remain in the
# DB, RAG-recallable). durable stances land 7-9, texture 2-3. NULL-importance
# (legacy / extractor) facts are kept — only explicitly-low ones are filtered.
SELF_DOSSIER_MIN_IMPORTANCE = 5

# self prompt annotates facts "(importance N)"; strip any that the writer
# echoes back so metadata never lands in the always-injected self-record
_IMPORTANCE_TAG = re.compile(r"\(importance \d+\)\s*")


def _dossier_facts(facts: list[Fact], *, is_self: bool) -> list[Fact]:
    """Facts that should shape a dossier.

    Self-dossier drops low-importance texture, then orders kept facts
    importance-desc (NULL ranks at keep-threshold; stable sort
    preserves recency within tier) so writer sees most central stances
    first. Other subjects and NULL-importance facts pass through
    unchanged.
    """
    if not is_self:
        return facts
    kept = [
        f
        for f in facts
        if f.importance is None or f.importance >= SELF_DOSSIER_MIN_IMPORTANCE
    ]
    # stable sort, importance-desc; NULL ranks at the keep-threshold band
    kept.sort(
        key=lambda f: (
            f.importance if f.importance is not None else SELF_DOSSIER_MIN_IMPORTANCE
        ),
        reverse=True,
    )
    return kept


class PeopleDossierWorker:
    """Rebuilds per-person dossiers off facts watermark."""

    name: str = "people-dossier-worker"

    def __init__(
        self,
        *,
        store: AsyncHistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        familiar_display_name: str | None = None,
        tick_interval_s: float = 20.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        # label for the reserved self-subject; title-cased id when absent
        self._familiar_display_name = familiar_display_name or familiar_id.title()
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
                    f"{ls.tag('PeopleDossier', ls.R)} "
                    f"{ls.kv('tick_error', repr(exc), vc=ls.R)}"
                )
            await asyncio.sleep(self._tick_interval_s)

    @span("people_dossier.tick")
    async def tick(self) -> None:
        """Refresh any dossier whose subject has new facts."""
        latest_per_subject = await self._store.subjects_with_facts(
            familiar_id=self._familiar_id
        )
        for canonical_key, latest_fact_id in latest_per_subject.items():
            await self._maybe_refresh(canonical_key, latest_fact_id)

    # ------------------------------------------------------------------
    # Per-subject refresh
    # ------------------------------------------------------------------

    async def _maybe_refresh(self, canonical_key: str, latest_fact_id: int) -> None:
        prior = await self._store.get_people_dossier(
            familiar_id=self._familiar_id, canonical_key=canonical_key
        )
        prior_wm = prior.last_fact_id if prior is not None else 0
        if latest_fact_id <= prior_wm:
            return

        new_facts = await self._store.facts_for_subject(
            familiar_id=self._familiar_id,
            canonical_key=canonical_key,
            min_id_exclusive=prior_wm,
        )
        if not new_facts:
            return

        is_self = is_ego_key(canonical_key)
        dossier_facts = _dossier_facts(new_facts, is_self=is_self)
        if not dossier_facts:
            # window held only low-importance texture — advance the watermark
            # past it (keeping prior text) so we don't re-filter every tick.
            if prior is not None:
                await self._store.put_people_dossier(
                    familiar_id=self._familiar_id,
                    canonical_key=canonical_key,
                    last_fact_id=latest_fact_id,
                    dossier_text=prior.dossier_text,
                )
            return

        # self-subject resolves to the familiar's display name — store
        # has no account row for ``self:<id>``, so resolve_label would
        # fall through to the raw id portion.
        if is_self:
            display = self._familiar_display_name
        else:
            display = await self._store.resolve_label(
                canonical_key=canonical_key,
                guild_id=None,
                familiar_id=self._familiar_id,
            )
        prompt = _build_dossier_prompt(
            display_name=display,
            prior_dossier=prior.dossier_text if prior is not None else None,
            new_facts=dossier_facts,
            is_self=is_self,
        )
        reply = await self._llm.chat(prompt)
        text = reply.content_str.strip()
        if is_self:
            # defensive: writer may echo the "(importance N)" annotations
            text = _IMPORTANCE_TAG.sub("", text).strip()
        if not text:
            # Don't overwrite real dossier with empty reply
            return
        await self._store.put_people_dossier(
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
    is_self: bool = False,
) -> list[Message]:
    if is_self:
        # self-record is the substrate for consistently-forming opinions
        # (feeds the sleep cycle) — keep settled feelings/stances, shed
        # only momentary reactions. do NOT blanket-drop feelings.
        header = (
            f"You maintain {display_name}'s evolving self-record — who she "
            "is becoming — in 3-5 sentences. Preserve her settled opinions, "
            "stances, and feelings about people and things (the views she "
            "holds consistently), plus concrete choices and commitments. "
            "Drop only momentary, in-the-moment reactions and filler. "
            "Reconcile contradictions in favour of newer evidence. Facts "
            "carry an importance score (higher = more central/durable to "
            "who she is); weight higher-importance stances more heavily, "
            "and since the record is only 3-5 sentences, when space is "
            "tight favour durable high-importance stances over lower ones "
            "(never invent). Reply with the updated self-record text only."
        )
    else:
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
    if is_self:
        # annotate scored facts so the writer can weight them; NULL untagged
        body_lines.extend(
            f"- (importance {f.importance}) {f.text}"
            if f.importance is not None
            else f"- {f.text}"
            for f in new_facts
        )
    else:
        body_lines.extend(f"- {f.text}" for f in new_facts)
    return [
        Message(role="system", content=header),
        Message(role="user", content="\n".join(body_lines)),
    ]
