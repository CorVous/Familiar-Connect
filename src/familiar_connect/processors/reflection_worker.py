"""Watermark-driven reflection worker (M3).

Compounds higher-order syntheses over recent turns + facts. Ticks
slower than :class:`PeopleDossierWorker` (default 60 s vs 20 s) — the
goal is one reflection-write per ~20-30 new turns, not per fact.

The worker reads the latest reflection row's
``(last_turn_id, last_fact_id)`` as its watermark; when at least
``turns_threshold`` new turns have accumulated, it asks the LLM
"what high-level questions do recent events raise?" and persists each
answer as one ``reflections`` row with ``cited_turn_ids`` /
``cited_fact_ids`` provenance. Rows whose only cited ids the LLM
hallucinates are dropped silently; rows where some ids are valid keep
the valid subset.

All LLM traffic is ``chat`` (not ``chat_stream``) — the worker runs
off the hot path.
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
from familiar_connect.history.store import _TURN_COLS, _row_to_turn
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.history.store import Fact, HistoryStore, HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.reflection_worker")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class ReflectionWorker:
    """Writes higher-order reflections off the turns + facts watermark."""

    name: str = "reflection-worker"

    def __init__(
        self,
        *,
        store: HistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        turns_threshold: int = 20,
        max_reflections_per_tick: int = 3,
        recent_facts_limit: int = 20,
        tick_interval_s: float = 60.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._turns_threshold = max(1, turns_threshold)
        self._max_per_tick = max(1, max_reflections_per_tick)
        self._recent_facts_limit = max(0, recent_facts_limit)
        self._tick_interval_s = tick_interval_s

    async def run(self) -> None:
        """Forever loop — tick on an interval. Cancel to stop."""
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
        """One pass: write reflections if enough new turns accumulated."""
        latest_turn = self._store.latest_id(familiar_id=self._familiar_id)
        if latest_turn is None or latest_turn <= 0:
            return
        prior_turn_wm, _prior_fact_wm = self._store.latest_reflection_watermarks(
            familiar_id=self._familiar_id
        )
        if latest_turn - prior_turn_wm < self._turns_threshold:
            return

        new_turns = self._turns_in_range(
            min_id_exclusive=prior_turn_wm,
            max_id_inclusive=latest_turn,
        )
        if not new_turns:
            return

        recent_facts = self._store.recent_facts(
            familiar_id=self._familiar_id, limit=self._recent_facts_limit
        )
        latest_fact = self._store.latest_fact_id(familiar_id=self._familiar_id)

        prompt = _build_reflection_prompt(
            new_turns=new_turns,
            recent_facts=recent_facts,
            max_reflections=self._max_per_tick,
        )
        reply = await self._llm.chat(prompt)
        items = _parse_reflections(reply.content)

        valid_turn_ids = {t.id for t in new_turns}
        valid_fact_ids = {f.id for f in recent_facts}
        # Reflection may also legitimately cite older facts surfaced via
        # the dossier; widen the valid set to all known facts so we
        # don't drop those unnecessarily.
        all_known_fact_ids = (
            set(valid_fact_ids)
            if not valid_fact_ids
            else _all_fact_ids(self._store, familiar_id=self._familiar_id)
        )

        # Per-channel scoping: pick the most-frequent channel in the
        # batch, fall back to None for cross-channel batches.
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
            # Require at least one valid citation; an uncited reflection
            # is a free-floating opinion, not a synthesis.
            if not cited_turns and not cited_facts:
                continue
            self._store.append_reflection(
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

    def _turns_in_range(
        self,
        *,
        min_id_exclusive: int,
        max_id_inclusive: int,
    ) -> list[HistoryTurn]:
        rows = self._store._conn.execute(  # noqa: SLF001 — tight coupling for the worker
            f"""
            SELECT {_TURN_COLS}
              FROM turns
             WHERE familiar_id = ?
               AND id > ?
               AND id <= ?
             ORDER BY id ASC
            """,  # noqa: S608
            (self._familiar_id, min_id_exclusive, max_id_inclusive),
        ).fetchall()
        return [_row_to_turn(r) for r in rows]


def _all_fact_ids(store: HistoryStore, *, familiar_id: str) -> set[int]:
    """Return the set of all known fact ids for *familiar_id* (incl. superseded)."""
    rows = store._conn.execute(  # noqa: SLF001
        "SELECT id FROM facts WHERE familiar_id = ?",
        (familiar_id,),
    ).fetchall()
    return {int(r["id"]) for r in rows}


def _dominant_channel(turns: Iterable[HistoryTurn]) -> int | None:
    """Most-frequent channel id across *turns*; ``None`` for cross-channel."""
    counts: dict[int, int] = {}
    for t in turns:
        counts[t.channel_id] = counts.get(t.channel_id, 0) + 1
    if len(counts) == 1:
        return next(iter(counts))
    if not counts:
        return None
    # Mixed batch — caller decides; default to the most-frequent.
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
