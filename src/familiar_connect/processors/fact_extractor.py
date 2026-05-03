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
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.spans import span
from familiar_connect.history.store import FactSubject
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.history.store import HistoryStore, HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.fact_extractor")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# Patterns that mark a "fact" as actually a self-capability statement
# (e.g., "I cannot remember names"). These belong in the system prompt
# or runtime config — not the facts store, where they'd silently expire
# the moment the underlying capability changes. Belt-and-braces post-
# filter; the extractor prompt also asks the model to skip them.
_SELF_CAPABILITY_RE = re.compile(
    r"""^\s*
        (?:
            i\s+(?:can(?:not|'t|\s+not)?|do(?:n't|\s+not)|am\s+(?:not|unable))
          | i'm\s+(?:not|unable)
          | i\s+have\s+no\b
          | as\s+(?:an?\s+)?(?:ai|assistant|language\s+model|llm)\b
          | the\s+(?:assistant|ai|familiar|model|bot)\b
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_self_capability(text: str) -> bool:
    """Heuristic prefix-match for self-capability "facts"."""
    return bool(_SELF_CAPABILITY_RE.match(text))


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
        participants_max: int = 30,
    ) -> None:
        self._store = store
        self._llm = llm_client
        self._familiar_id = familiar_id
        self._batch_size = max(1, batch_size)
        self._tick_interval_s = tick_interval_s
        self._participants_max = max(1, participants_max)

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

        participants = _build_participants(
            new_turns,
            store=self._store,
            familiar_id=self._familiar_id,
            max_total=self._participants_max,
        )
        prompt = _build_extract_prompt(new_turns, participants)
        reply = await self._llm.chat(prompt)
        facts = _parse_facts(reply.content)
        valid_ids = {t.id for t in new_turns}
        channel_ids: dict[int, int] = {t.id: t.channel_id for t in new_turns}
        ts_by_id: dict[int, datetime] = {t.id: t.timestamp for t in new_turns}

        dropped_self_cap = 0
        for fact in facts:
            raw_sources = fact.get("source_turn_ids", [])
            source_ids: list[int] = []
            if isinstance(raw_sources, list):
                source_ids = [
                    int(i) for i in raw_sources if isinstance(i, int) and i in valid_ids
                ]
            if not source_ids:
                # fall back to the whole batch rather than dropping the fact
                source_ids = list(valid_ids)
            # Prefer the channel of the first source turn for scoping.
            channel_id = channel_ids.get(source_ids[0])
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            if _is_self_capability(text):
                dropped_self_cap += 1
                _logger.debug(
                    f"{ls.tag('Facts', ls.Y)} "
                    f"{ls.kv('drop', 'self_capability', vc=ls.LY)} "
                    f"{ls.kv('text', ls.trunc(text, 120), vc=ls.LW)}"
                )
                continue
            subjects = _resolve_subjects(fact.get("subject_keys", []), participants)
            valid_from = _parse_iso_dt(fact.get("valid_from")) or ts_by_id.get(
                source_ids[0]
            )
            valid_to = _parse_iso_dt(fact.get("valid_to"))
            self._store.append_fact(
                familiar_id=self._familiar_id,
                channel_id=channel_id,
                text=text,
                source_turn_ids=source_ids,
                subjects=subjects,
                valid_from=valid_from,
                valid_to=valid_to,
            )
            # Mirror resolved subjects into ``turn_mentions``. Bridges
            # bare-text references like "what about Aria?" into the
            # same mention index that Discord ``@`` pings populate, so
            # ``PeopleDossierLayer`` can pick them up at the next
            # assemble — no separate read path, no fact-table join in
            # the hot path. Idempotent (PK-deduped on the store side).
            if subjects:
                keys = [s.canonical_key for s in subjects]
                for tid in source_ids:
                    self._store.record_mentions(turn_id=tid, canonical_keys=keys)

        # Always advance watermark, even on empty/bad output, to prevent loops.
        last_id = new_turns[-1].id
        self._store.put_writer_watermark(
            familiar_id=self._familiar_id, last_written_id=last_id
        )
        _logger.info(
            f"{ls.tag('Facts', ls.LC)} "
            f"{ls.kv('batch_size', str(len(new_turns)), vc=ls.LY)} "
            f"{ls.kv('extracted', str(len(facts)), vc=ls.LC)} "
            f"{ls.kv('dropped_self_cap', str(dropped_self_cap), vc=ls.LY)} "
            f"{ls.kv('watermark', str(last_id), vc=ls.LW)}"
        )


# ---------------------------------------------------------------------------
# Prompt + parsing helpers (private)
# ---------------------------------------------------------------------------


def _build_participants(
    turns: Iterable[HistoryTurn],
    *,
    store: HistoryStore,
    familiar_id: str,
    max_total: int = 30,
) -> dict[str, str]:
    """Map ``canonical_key`` → current display name for fact resolution.

    Two layers, batch-first:

    1. **Batch authors** — every turn whose author is set, keyed by
       canonical_key with per-turn ``guild_id`` for label resolution.
       Guarantees the active speakers are always represented even if
       the wider widen step would otherwise drop them.
    2. **Recent channel participants** — for each channel touched by
       the batch, ``recent_distinct_authors(limit=max_total)`` adds
       speakers from the recent past, capped at ``max_total`` total.
       Closes the bare-text reference gap: a batch where only Cass
       speaks can still link "Aria" in turn content to her canonical
       key when Aria spoke earlier in the channel.

    Resolution goes through :meth:`HistoryStore.resolve_label` so the
    manifest matches what the read path renders (per-guild nick wins
    over the snapshot label baked into each turn). Used both to
    inject the LLM-facing manifest and to validate ``subject_keys``
    coming back in the response — keys outside the manifest are
    dropped silently (the LLM may hallucinate).
    """
    out: dict[str, str] = {}
    # Batch authors first, with per-turn guild_id.
    channel_guilds: dict[int, int | None] = {}
    for t in turns:
        if t.channel_id is not None:
            # Last guild_id wins per channel; usually consistent within a batch.
            channel_guilds[t.channel_id] = t.guild_id
        if t.author is None:
            continue
        out[t.author.canonical_key] = store.resolve_label(
            canonical_key=t.author.canonical_key,
            guild_id=t.guild_id,
            familiar_id=familiar_id,
        )
    # Widen with recent channel participants, capped at max_total.
    for channel_id, guild_id in channel_guilds.items():
        if len(out) >= max_total:
            break
        for author in store.recent_distinct_authors(
            familiar_id=familiar_id,
            channel_id=channel_id,
            limit=max_total,
        ):
            if author.canonical_key in out:
                continue
            if len(out) >= max_total:
                break
            out[author.canonical_key] = store.resolve_label(
                canonical_key=author.canonical_key,
                guild_id=guild_id,
                familiar_id=familiar_id,
            )
    return out


def _resolve_subjects(
    raw: object,
    participants: dict[str, str],
) -> list[FactSubject]:
    """Validate LLM-emitted ``subject_keys`` against the manifest.

    Soft validation: unknown keys are dropped silently rather than
    invalidating the fact. The display_at_write is taken from the
    manifest (the LLM's view at extraction time), not from anything
    the LLM might have echoed.
    """
    if not isinstance(raw, list):
        return []
    out: list[FactSubject] = []
    seen: set[str] = set()
    for key in raw:
        if not isinstance(key, str) or key not in participants or key in seen:
            continue
        seen.add(key)
        out.append(FactSubject(canonical_key=key, display_at_write=participants[key]))
    return out


def _build_extract_prompt(
    turns: Iterable[HistoryTurn],
    participants: dict[str, str],
) -> list[Message]:
    header = (
        "Extract a short list of atomic facts about the people and "
        "events in the chat turns below — observations about the "
        "world, not about you.\n\n"
        "Reply with a JSON array. Each item has:\n"
        "- ``text`` (one sentence)\n"
        "- ``source_turn_ids`` (list of turn ids the fact was distilled from)\n"
        "- ``subject_keys`` (optional list of canonical keys from the "
        "Participants block, identifying which people the fact is "
        "about). Use this whenever the fact mentions someone by name "
        "and you can match that name to a participant. Leave it out "
        "or empty if you can't tell.\n"
        "- ``valid_from`` (optional ISO-8601 timestamp) — only set "
        "when the speaker explicitly anchors the fact to a different "
        "moment than 'now' (e.g., 'as of last June', 'back in 2019'). "
        "Otherwise omit; the source turn's timestamp is used.\n"
        "- ``valid_to`` (optional ISO-8601 timestamp) — only set when "
        "the fact is bounded to end at a known point.\n\n"
        "Skip small talk and transient feelings. If nothing useful, "
        "reply with []. Do NOT emit self-capability statements about "
        "yourself, the assistant, or your own limitations (e.g., 'I "
        "cannot remember names', 'the assistant has no internet "
        "access', 'as an AI, I…'). Those belong in the system "
        "prompt, not the facts store — they expire the moment a "
        "capability changes."
    )
    lines: list[str] = []
    if participants:
        lines.append("Participants (canonical_key — current display name):")
        for key, display in participants.items():
            lines.append(f"- {key} — {display}")
        lines.append("")
    lines.append("Turns (id prefixed):")
    for t in turns:
        who = t.author.label if t.author is not None else t.role
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
        subject_keys: list[str] = []
        raw_subjects = item.get("subject_keys", [])
        if isinstance(raw_subjects, list):
            subject_keys = [s for s in raw_subjects if isinstance(s, str)]
        out.append({
            "text": item.get("text", ""),
            "source_turn_ids": ids,
            "subject_keys": subject_keys,
            "valid_from": item.get("valid_from"),
            "valid_to": item.get("valid_to"),
        })
    return out


def _parse_iso_dt(raw: object) -> datetime | None:
    """Permissive ISO-8601 → ``datetime`` parse; ``None`` for bad input.

    The LLM may emit a date-only string (``2024-01-15``) or a full
    timestamp; ``datetime.fromisoformat`` accepts both. Anything else
    silently degrades to ``None`` so the caller falls back to the
    source turn's timestamp.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # date-only strings parse to naive datetimes; assume UTC for consistency.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
