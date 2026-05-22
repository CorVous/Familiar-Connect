"""Watermark-driven reflection worker (M3).

Compounds higher-order syntheses over recent turns + facts. Ticks
slower than :class:`PeopleDossierWorker` (default 60 s vs 20 s) —
goal is one reflection-write per ~20-30 new turns, not per fact.

Worker reads ``(last_turn_id, last_fact_id)`` from
``reflection_watermark``; when at least ``turns_threshold`` new turns
accumulate, asks LLM "what high-level questions do recent events
raise?" and persists each answer as one ``reflections`` row with
``cited_turn_ids`` / ``cited_fact_ids`` provenance. Rows whose only
cited ids LLM hallucinates are dropped silently; rows where some ids
are valid keep valid subset.

Two guardrails prevent runaway token spend:

* Per-tick window capped at ``max_turns_per_tick`` — even if
  watermark is far behind, only most recent N turns enter prompt.
  Reflections best-effort syntheses, not exhaustive.
* Watermark advances at end of every tick, even when LLM returns
  ``[]`` or every item dropped. Without this, no-op tick re-sends
  same growing window on next tick.

All LLM traffic is ``chat`` (not ``chat_stream``) — worker runs off
hot path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import Fact, HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.reflection_worker")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class ReflectionWorker:
    """Writes higher-order reflections off turns + facts watermark."""

    name: str = "reflection-worker"

    def __init__(
        self,
        *,
        store: AsyncHistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        turns_threshold: int = 20,
        max_reflections_per_tick: int = 3,
        max_turns_per_tick: int = 50,
        recent_facts_limit: int = 20,
        tick_interval_s: float = 60.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._turns_threshold = max(1, turns_threshold)
        self._max_per_tick = max(1, max_reflections_per_tick)
        self._max_turns_per_tick = max(1, max_turns_per_tick)
        self._recent_facts_limit = max(0, recent_facts_limit)
        self._tick_interval_s = tick_interval_s

    async def run(self) -> None:
        """Forever loop; tick on interval. Cancel to stop."""
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — worker must not die
                _logger.warning(
                    f"{ls.tag('Reflection', ls.R)} "
                    f"{ls.kv('tick_error', repr(exc), vc=ls.R)}"
                )
            await asyncio.sleep(self._tick_interval_s)

    @span("reflection.tick")
    async def tick(self) -> None:
        """One pass; write reflections if enough new turns accumulated."""
        latest_turn = await self._store.latest_id(familiar_id=self._familiar_id)
        if latest_turn is None or latest_turn <= 0:
            return
        prior_turn_wm, _prior_fact_wm = await self._store.latest_reflection_watermarks(
            familiar_id=self._familiar_id
        )
        if latest_turn - prior_turn_wm < self._turns_threshold:
            return

        latest_fact = await self._store.latest_fact_id(familiar_id=self._familiar_id)
        # always advance watermark to ``latest_turn`` regardless of
        # how tick lands — empty LLM reply, malformed JSON, all items
        # filtered are no-ops on reflections table but must not pin
        # worker to a growing window
        try:
            await self._do_tick(
                prior_turn_wm=prior_turn_wm,
                latest_turn=latest_turn,
                latest_fact=latest_fact,
            )
        finally:
            await self._store.set_reflection_watermark(
                familiar_id=self._familiar_id,
                last_turn_id=latest_turn,
                last_fact_id=latest_fact,
            )

    async def _do_tick(
        self,
        *,
        prior_turn_wm: int,
        latest_turn: int,
        latest_fact: int,
    ) -> None:
        new_turns = await self._turns_in_range(
            min_id_exclusive=prior_turn_wm,
            max_id_inclusive=latest_turn,
        )
        if not new_turns:
            return
        # cap window — bursty chat or first run on long-lived db
        # otherwise ships hundreds of turns per tick. keep tail (most
        # recent) since that's what reflection cares about; older
        # turns skipped, not deferred
        if len(new_turns) > self._max_turns_per_tick:
            new_turns = new_turns[-self._max_turns_per_tick :]

        recent_facts = await self._store.recent_facts(
            familiar_id=self._familiar_id, limit=self._recent_facts_limit
        )

        prompt = _build_reflection_prompt(
            new_turns=new_turns,
            recent_facts=recent_facts,
            max_reflections=self._max_per_tick,
        )
        reply = await self._llm.chat(prompt)
        items = _parse_reflections(reply.content)

        valid_turn_ids = {t.id for t in new_turns}
        valid_fact_ids = {f.id for f in recent_facts}
        # reflection may legitimately cite older facts surfaced via
        # dossier; widen valid set to all known facts so we don't
        # drop those unnecessarily
        all_known_fact_ids = (
            set(valid_fact_ids)
            if not valid_fact_ids
            else await self._store.all_fact_ids(familiar_id=self._familiar_id)
        )

        # per-channel scoping: pick most-frequent channel in batch,
        # fall back to None for cross-channel batches
        channel_id = _dominant_channel(new_turns)

        written = 0
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            raw_turns = item.get("cited_turn_ids", [])
            raw_facts = item.get("cited_fact_ids", [])
            cited_turns = (
                [
                    int(i)
                    for i in raw_turns
                    if isinstance(i, int) and i in valid_turn_ids
                ]
                if isinstance(raw_turns, list)
                else []
            )
            cited_facts = (
                [
                    int(i)
                    for i in raw_facts
                    if isinstance(i, int) and i in all_known_fact_ids
                ]
                if isinstance(raw_facts, list)
                else []
            )
            # require at least one valid citation; uncited reflection
            # is free-floating opinion, not synthesis
            if not cited_turns and not cited_facts:
                continue
            await self._store.append_reflection(
                familiar_id=self._familiar_id,
                channel_id=channel_id,
                text=text,
                cited_turn_ids=cited_turns,
                cited_fact_ids=cited_facts,
                last_turn_id=latest_turn,
                last_fact_id=latest_fact,
            )
            written += 1

        _logger.info(
            f"{ls.tag('Reflection', ls.LC)} "
            f"{ls.kv('new_turns', str(len(new_turns)), vc=ls.LY)} "
            f"{ls.kv('written', str(written), vc=ls.LC)} "
            f"{ls.kv('turn_wm', str(latest_turn), vc=ls.LW)} "
            f"{ls.kv('fact_wm', str(latest_fact), vc=ls.LW)}"
        )

    async def _turns_in_range(
        self,
        *,
        min_id_exclusive: int,
        max_id_inclusive: int,
    ) -> list[HistoryTurn]:
        return await self._store.turns_in_id_range(
            familiar_id=self._familiar_id,
            min_id_exclusive=min_id_exclusive,
            max_id_inclusive=max_id_inclusive,
        )


def _dominant_channel(turns: Iterable[HistoryTurn]) -> int | None:
    """Most-frequent channel id across *turns*; ``None`` for cross-channel."""
    counts: dict[int, int] = {}
    for t in turns:
        counts[t.channel_id] = counts.get(t.channel_id, 0) + 1
    if len(counts) == 1:
        return next(iter(counts))
    if not counts:
        return None
    # mixed batch — caller decides; default to most-frequent
    return max(counts.items(), key=operator.itemgetter(1))[0]


# ---------------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------------


def _build_reflection_prompt(
    *,
    new_turns: Iterable[HistoryTurn],
    recent_facts: Iterable[Fact],
    max_reflections: int,
) -> list[Message]:
    header = (
        "You write short, high-level reflections over recent chat "
        "history — patterns, recurring tensions, open questions, "
        "themes the participants keep circling back to. Skip blow-by-"
        "blow recaps; that's what summaries are for. Each reflection "
        "is one or two sentences.\n\n"
        f"Reply with a JSON array of at most {max_reflections} items. "
        "Each item has:\n"
        "- ``text`` (one or two sentences)\n"
        "- ``cited_turn_ids`` (list of turn ids the reflection draws "
        "from; pick the most representative, not all of them)\n"
        "- ``cited_fact_ids`` (list of fact ids if the reflection "
        "leans on stored facts; may be empty)\n\n"
        "Cite at least one turn id or fact id per reflection. If "
        "nothing of substance is happening, reply with []."
    )
    lines: list[str] = ["Recent turns (id prefixed):"]
    for t in new_turns:
        who = t.author.display_name if t.author is not None else t.role
        lines.append(f"- id={t.id} [{who}] {t.content}")
    facts = list(recent_facts)
    if facts:
        lines.extend(("", "Recent facts (id prefixed):"))
        lines.extend(f"- id={f.id} {f.text}" for f in facts)
    return [
        Message(role="system", content=header),
        Message(role="user", content="\n".join(lines)),
    ]


def _parse_reflections(reply: str) -> list[dict[str, object]]:
    """Permissive JSON-array parser; bad input → ``[]``."""
    if not reply or not reply.strip():
        return []
    cleaned = re.sub(r"```(?:json)?", "", reply, flags=re.IGNORECASE).strip()
    match = _JSON_ARRAY_RE.search(cleaned)
    blob = match.group(0) if match else cleaned
    try:
        parsed = json.loads(blob)
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, object]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        raw_turns = item.get("cited_turn_ids", [])
        raw_facts = item.get("cited_fact_ids", [])
        cited_turns: list[int] = []
        if isinstance(raw_turns, list):
            cited_turns = [int(x) for x in raw_turns if isinstance(x, int)]
        cited_facts: list[int] = []
        if isinstance(raw_facts, list):
            cited_facts = [int(x) for x in raw_facts if isinstance(x, int)]
        out.append({
            "text": text,
            "cited_turn_ids": cited_turns,
            "cited_fact_ids": cited_facts,
        })
    return out
