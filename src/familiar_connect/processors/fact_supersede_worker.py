"""Watermark-driven fact supersession worker.

For each newly-appended fact, asks the background LLM whether it
retires any prior current facts about the same subject — and if so,
calls :meth:`HistoryStore.supersede_fact` to mark them.

This is the system-time bookkeeping half of the fact lifecycle.
``valid_to`` (world-time) is left to the extractor and to the speaker
who anchors a real-world end. Supersession runs slower than the
extractor (60 s default) — fact churn is bounded by conversation
pace, and missing a tick of LLM-assisted retirement is harmless.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import Fact
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.fact_supersede_worker")

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class FactSupersedeWorker:
    """Retires prior facts replaced by newer ones about the same subject.

    :param batch_size: max new facts evaluated per tick. Each gets one
        LLM call, batched per subject.
    :param tick_interval_s: idle interval. 60 s by default — much
        slower than the 15 s extractor since supersession isn't
        latency-critical and adds an LLM call per new fact.
    :param priors_max: cap on the number of prior facts shown to the
        LLM per subject. Prevents prompt bloat for long-running
        familiars with many priors per person.
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
        """Skip the historical backlog — start from the latest current id.

        Called by the projector factory at process start so a fresh
        deploy doesn't burn LLM calls re-evaluating every old fact in
        the store on the first tick. Tests may call this directly to
        assert the no-new-facts path.
        """
        self._last_seen_fact_id = self._store.sync.latest_fact_id(
            familiar_id=self._familiar_id
        )

    async def run(self) -> None:
        """Forever loop — tick on an interval. Cancel to stop."""
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
        """Evaluate up to ``batch_size`` new current facts. Return supersedes."""
        candidates = await self._store.recent_facts(
            familiar_id=self._familiar_id,
            limit=self._batch_size,
            include_superseded=False,
        )
        new = [f for f in candidates if f.id > self._last_seen_fact_id]
        if not new:
            return 0

        # oldest-new first so cascading retirements settle deterministically
        new.sort(key=lambda f: f.id)
        retired = 0
        for f_new in new:
            retired += await self._evaluate(f_new)

        # advance watermark to the highest id seen this tick (even on bad
        # llm output) — prevents loops on a fact whose prompt the model
        # consistently fails to parse.
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
        """Ask LLM which priors `f_new` retires; supersede them. Return count."""
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
            # exclude f_new itself, dedupe across subjects, cap to priors_max
            unique = [
                p for p in priors if p.id != f_new.id and p.id not in seen_priors
            ][-self._priors_max :]
            if not unique:
                continue
            seen_priors.update(p.id for p in unique)

            prompt = _build_supersede_prompt(f_new=f_new, priors=unique)
            reply = await self._llm.chat(prompt)
            ids = _parse_superseded_ids(
                reply.content,
                valid={p.id for p in unique},
            )
            for old_id in ids:
                try:
                    await self._store.supersede_fact(
                        familiar_id=self._familiar_id,
                        old_id=old_id,
                        new_id=f_new.id,
                    )
                except ValueError:
                    # already retired by an earlier subject in this fact
                    continue
                retired += 1
        return retired


def _build_supersede_prompt(*, f_new: Fact, priors: list[Fact]) -> list[Message]:
    """LLM prompt: which priors does ``f_new`` replace."""
    header = (
        "You decide whether a new fact retires earlier facts about the "
        "same person. A fact is *retired* when the new one contradicts "
        "or directly replaces it (e.g., 'Alice loves hiking' is retired "
        'by "Alice now hates hiking"). A fact is NOT retired just '
        "because it's older or differently worded — facts about "
        "independent topics coexist.\n\n"
        'Reply with JSON: ``{"superseded_ids": [<id>, ...]}``. '
        "Empty list when nothing is retired. Only include ids from the "
        "Prior facts list below — do not invent ids."
    )
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


def _parse_superseded_ids(reply: str, *, valid: set[int]) -> list[int]:
    """Extract ``superseded_ids`` from LLM reply; filter to *valid*.

    Permissive: bad JSON, missing key, non-list value, and non-int
    items all degrade to ``[]`` rather than raising.
    """
    if not reply or not reply.strip():
        return []
    cleaned = re.sub(r"```(?:json)?", "", reply, flags=re.IGNORECASE).strip()
    match = _JSON_OBJECT_RE.search(cleaned)
    blob = match.group(0) if match else cleaned
    try:
        parsed = json.loads(blob)
    except ValueError:
        return []
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("superseded_ids")
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            candidate = item
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            candidate = int(item.strip())
        else:
            continue
        if candidate in valid:
            out.append(candidate)
    return out
