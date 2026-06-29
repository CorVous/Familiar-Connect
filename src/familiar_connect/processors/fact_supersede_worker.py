"""Watermark-driven fact supersession worker.

For each newly-appended fact, asks background LLM whether it
retires any prior current facts about the same subject — if so,
calls :meth:`HistoryStore.supersede` (existing-id form) to repoint
them at the new fact.

System-time bookkeeping half of the fact lifecycle. ``valid_to``
(world-time) is left to extractor and to speaker who anchors a
real-world end. Supersession runs slower than extractor (60 s
default) — fact churn bounded by conversation pace, missing a tick
of LLM-assisted retirement is harmless.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.llm import Message
from familiar_connect.structured_output import coerce_positive_int_list
from familiar_connect.structured_request import (
    Field,
    Schema,
    render_contract,
    request_structured,
)

if TYPE_CHECKING:
    from typing import Any

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import Fact
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.fact_supersede_worker")

# Reply-shape contract for "which priors does this new fact retire" —
# declared, rendered + parsed via :mod:`familiar_connect.structured_request`
# (#167) instead of a hand-typed JSON string in the prompt builder.
_SUPERSEDE_SCHEMA = Schema(
    fields=(Field("superseded_ids", "[<id>...]"),),
    root="object",
    empty_note="Empty list when nothing is retired.",
    constraints=(
        "Only include ids from the Prior facts list below — do not invent ids.",
    ),
)


class FactSupersedeWorker:
    """Retires prior facts replaced by newer ones about the same subject.

    :param batch_size: max new facts per tick. Each gets one LLM
        call, batched per subject.
    :param tick_interval_s: idle interval. 60 s default — slower
        than 15 s extractor since supersession isn't latency-critical
        and adds an LLM call per new fact.
    :param priors_max: cap on prior facts shown to LLM per subject.
        Prevents prompt bloat for long-running familiars with many
        priors per person.
    """

    name: str = "fact-supersede-worker"

    def __init__(
        self,
        *,
        store: AsyncHistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        batch_size: int = 5,
        tick_interval_s: float = 60.0,
        priors_max: int = 20,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._batch_size = max(1, batch_size)
        self._tick_interval_s = tick_interval_s
        self._priors_max = max(1, priors_max)
        self._last_seen_fact_id: int = 0

    def prime_watermark(self) -> None:
        """Skip historical backlog; start from latest current id.

        Called by projector factory at process start so fresh deploy
        doesn't burn LLM calls re-evaluating every old fact on first
        tick. Tests may call directly to assert no-new-facts path.
        """
        self._last_seen_fact_id = self._store.sync.latest_fact_id(
            familiar_id=self._familiar_id
        )

    async def run(self) -> None:
        """Forever loop; tick on interval. Cancel to stop."""
        self.prime_watermark()
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — worker must not die
                _logger.warning(
                    f"{ls.tag('Supersede', ls.R)} "
                    f"{ls.kv('tick_error', repr(exc), vc=ls.R)}"
                )
            await asyncio.sleep(self._tick_interval_s)

    @span("fact_supersede.tick")
    async def tick(self) -> int:
        """Evaluate up to ``batch_size`` new current facts; return supersedes."""
        candidates = await self._store.recent_facts(
            familiar_id=self._familiar_id,
            limit=self._batch_size,
            include_superseded=False,
        )
        new = [f for f in candidates if f.id > self._last_seen_fact_id]
        if not new:
            return 0

        # Oldest-new first so cascading retirements settle deterministically
        new.sort(key=lambda f: f.id)
        retired = 0
        for f_new in new:
            retired += await self._evaluate(f_new)

        # Advance watermark to highest id seen this tick (even on bad
        # llm output) — prevents loops on a fact whose prompt model
        # consistently fails to parse
        if candidates:
            self._last_seen_fact_id = max(f.id for f in candidates)

        if retired:
            _logger.info(
                f"{ls.tag('Supersede', ls.LM)} "
                f"{ls.kv('evaluated', str(len(new)), vc=ls.LW)} "
                f"{ls.kv('retired', str(retired), vc=ls.LM)} "
                f"{ls.kv('watermark', str(self._last_seen_fact_id), vc=ls.LW)}"
            )
        return retired

    async def _evaluate(self, f_new: Fact) -> int:
        """Ask LLM which priors `f_new` retires; supersede them; return count."""
        if not f_new.subjects:
            return 0
        retired = 0
        seen_priors: set[int] = set()
        for subject in f_new.subjects:
            priors = await self._store.facts_for_subject(
                familiar_id=self._familiar_id,
                canonical_key=subject.canonical_key,
                include_superseded=False,
            )
            # Exclude f_new itself, dedupe across subjects, cap to priors_max
            unique = [
                p for p in priors if p.id != f_new.id and p.id not in seen_priors
            ][-self._priors_max :]
            if not unique:
                continue
            seen_priors.update(p.id for p in unique)

            prompt = _build_supersede_prompt(f_new=f_new, priors=unique)
            result = await request_structured(
                self._llm, messages=prompt, schema=_SUPERSEDE_SCHEMA
            )
            ids = _superseded_ids(result.value, valid={p.id for p in unique})
            if not ids:
                continue
            # Existing-id form: repoint each old row at f_new (mints
            # nothing). Per-id skip-and-record — a prior already retired
            # by an earlier subject lands in `skipped`, not an exception.
            result = await self._store.supersede(
                familiar_id=self._familiar_id,
                obsolete_facts=ids,
                new_fact=f_new,
            )
            retired += len(result.superseded)
        return retired


def _build_supersede_prompt(*, f_new: Fact, priors: list[Fact]) -> list[Message]:
    """LLM prompt: which priors does ``f_new`` replace."""
    persona = (
        "You decide whether a new fact retires earlier facts about the "
        "same person. A fact is *retired* when the new one contradicts "
        "or directly replaces it (e.g., 'Alice loves hiking' is retired "
        'by "Alice now hates hiking"). A fact is NOT retired just '
        "because it's older or differently worded — facts about "
        "independent topics coexist."
    )
    header = f"{persona}\n\n{render_contract(_SUPERSEDE_SCHEMA)}"
    lines: list[str] = [
        f"New fact (id={f_new.id}): {f_new.text}",
        "",
        "Prior facts:",
    ]
    lines.extend(f"- id={p.id}: {p.text}" for p in priors)
    return [
        Message(role="system", content=header),
        Message(role="user", content="\n".join(lines)),
    ]


def _superseded_ids(value: Any, *, valid: set[int]) -> list[int]:  # noqa: ANN401 — parsed JSON
    """Distinct prior ids the model marked superseded, filtered to *valid*.

    *value* is the parsed reply object from
    :func:`familiar_connect.structured_request.request_structured`; a
    non-object, missing key, or non-list value all degrade to ``[]``.
    """
    if not isinstance(value, dict):
        return []
    return [
        i for i in coerce_positive_int_list(value.get("superseded_ids")) if i in valid
    ]
