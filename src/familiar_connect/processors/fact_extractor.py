"""Watermark-driven fact extractor.

Distils atomic facts from new turns using a cheap LLM pass, and
stores them in ``facts`` with ``source_turn_ids`` pointing back to
the originating rows in ``turns``. Advances the
``memory_writer_watermark`` once processed — even on malformed LLM
output — so the worker never loops forever on the same batch.

Phase-4 scope: the extraction prompt asks for a JSON array of
``{text, source_turn_ids}`` objects. A permissive JSON parser wraps
fences and extraneous prose common in chat-tuned models.
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
    from collections.abc import Iterable

    from familiar_connect.history.store import HistoryStore, HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.fact_extractor")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class FactExtractor:
    """Distils facts from new turns; forever loop with ``run()``."""

    name: str = "fact-extractor"

    def __init__(
        self,
        *,
        store: HistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        batch_size: int = 10,
        tick_interval_s: float = 15.0,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._batch_size = max(1, batch_size)
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
                    f"{ls.tag('FactExtractor', ls.R)} "
                    f"{ls.kv('tick_error', repr(exc), vc=ls.R)}"
                )
            await asyncio.sleep(self._tick_interval_s)

    @span("facts.tick")
    async def tick(self) -> None:
        """Process one batch of unprocessed turns, if enough accumulated."""
        new_turns = self._store.turns_since_watermark(
            familiar_id=self._familiar_id,
            limit=self._batch_size,
        )
        if len(new_turns) < self._batch_size:
            return

        prompt = _build_extract_prompt(new_turns)
        reply = await self._llm.chat(prompt)
        facts = _parse_facts(reply.content)
        valid_ids = {t.id for t in new_turns}
        channel_ids: dict[int, int] = {t.id: t.channel_id for t in new_turns}

        for fact in facts:
            raw_sources = fact.get("source_turn_ids", [])
            source_ids: list[int] = []
            if isinstance(raw_sources, list):
                source_ids = [
                    int(i)
                    for i in raw_sources
                    if isinstance(i, int) and i in valid_ids
                ]
            if not source_ids:
                # fall back to the whole batch rather than dropping the fact
                source_ids = list(valid_ids)
            # Prefer the channel of the first source turn for scoping.
            channel_id = channel_ids.get(source_ids[0])
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            self._store.append_fact(
                familiar_id=self._familiar_id,
                channel_id=channel_id,
                text=text,
                source_turn_ids=source_ids,
            )

        # Always advance watermark, even on empty/bad output, to prevent loops.
        last_id = new_turns[-1].id
        self._store.put_writer_watermark(
            familiar_id=self._familiar_id, last_written_id=last_id
        )
        _logger.info(
            f"{ls.tag('Facts', ls.LC)} "
            f"{ls.kv('batch_size', str(len(new_turns)), vc=ls.LY)} "
            f"{ls.kv('extracted', str(len(facts)), vc=ls.LC)} "
            f"{ls.kv('watermark', str(last_id), vc=ls.LW)}"
        )


# ---------------------------------------------------------------------------
# Prompt + parsing helpers (private)
# ---------------------------------------------------------------------------


def _build_extract_prompt(turns: Iterable[HistoryTurn]) -> list[Message]:
    header = (
        "Extract a short list of atomic facts from the chat turns "
        "below. Reply with a JSON array where each item has "
        "``text`` (one sentence) and ``source_turn_ids`` (a list of "
        "turn ids the fact was distilled from). Skip small talk and "
        "transient feelings. If nothing useful, reply with []."
    )
    lines: list[str] = ["Turns (id prefixed):"]
    for t in turns:
        who = t.author.display_name if t.author is not None else t.role
        lines.append(f"- id={t.id} [{who}] {t.content}")
    return [
        Message(role="system", content=header),
        Message(role="user", content="\n".join(lines)),
    ]


def _parse_facts(reply: str) -> list[dict[str, object]]:
    """Permissive JSON-array parser.

    Strips code fences, extracts the first balanced ``[...]`` blob,
    and coerces it via :func:`json.loads`. Malformed input returns
    ``[]`` rather than raising.
    """
    if not reply or not reply.strip():
        return []
    # Strip common code-fence prelude like ``` or ```json
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
        # Coerce source_turn_ids to ints
        raw_ids = item.get("source_turn_ids", [])
        ids: list[int] = []
        if isinstance(raw_ids, list):
            ids.extend(
                int(x)
                for x in raw_ids
                if isinstance(x, (int, str)) and str(x).lstrip("-").isdigit()
            )
        out.append({"text": item.get("text", ""), "source_turn_ids": ids})
    return out
