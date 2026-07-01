"""Watermark-driven fact extractor.

Distils atomic facts from new turns via cheap LLM pass; stores in
``facts`` with ``source_turn_ids`` pointing back to ``turns`` rows.
Advances ``memory_writer_watermark`` once processed — even on
malformed LLM output — so worker never loops forever on same batch.

Phase-4 scope: extraction prompt asks for JSON array of
``{text, source_turn_ids}`` objects. Permissive JSON parser
unwraps fences + extraneous prose common in chat-tuned models.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.activities import ACTIVITY_RETURN_MODE, SLEEP_RETURN_MODE
from familiar_connect.diagnostics.spans import span
from familiar_connect.history.store import FactSubject
from familiar_connect.identity import is_self_key, self_canonical_key
from familiar_connect.llm import Message
from familiar_connect.prompt_fill import fill_placeholders
from familiar_connect.structured_request import (
    Field,
    Schema,
    render_contract,
    request_structured,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import HistoryStore, HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger("familiar_connect.processors.fact_extractor")

# Reply-shape contract for fact extraction — declared, rendered + parsed
# through :mod:`familiar_connect.structured_request` (#167) instead of a
# hand-typed JSON skeleton inside the prompt builder. The persona prose,
# the events-vs-assertions discipline, and the per-batch participant
# manifest stay in the builder; only the machine-parsed shape lives here.
_FACT_SCHEMA = Schema(
    fields=(
        Field("text", '"<one sentence>"', desc="one sentence"),
        Field(
            "source_turn_ids",
            "[<id>...]",
            desc="list of turn ids the fact was distilled from",
        ),
        Field(
            "subject_keys",
            "[<key>...]",
            desc=(
                "list of canonical keys from the Participants block, "
                "identifying which people the fact is about. Use this "
                "whenever the fact mentions someone by name and you can "
                "match that name to a participant. Leave it out or empty "
                "if you can't tell."
            ),
            required=False,
        ),
        Field(
            "valid_from",
            '"<ISO-8601 timestamp>"',
            desc=(
                "only set when the speaker explicitly anchors the fact to "
                "a different moment than 'now' (e.g., 'as of last June', "
                "'back in 2019'). Otherwise omit; the source turn's "
                "timestamp is used."
            ),
            required=False,
        ),
        Field(
            "valid_to",
            '"<ISO-8601 timestamp>"',
            desc=(
                "world-time end only. Set this when the speaker explicitly "
                "anchors the end of the fact in real time (e.g., 'until "
                "last June', 'ended in 2019', 'no longer lives there as of "
                "March'). Do NOT use valid_to to mark a fact as outdated, "
                "replaced, or superseded by something newer in this "
                "conversation — supersession is tracked separately. When "
                "in doubt, omit."
            ),
            required=False,
        ),
        Field(
            "importance",
            "<1-10>",
            desc=(
                "integer 1-10 — how much this fact should influence future "
                "replies. 1 = throwaway aside, 5 = ordinary detail, 10 = "
                "safety-critical / identity-defining (allergies, names, "
                "long-standing preferences, life events). Omit when unsure."
            ),
            required=False,
        ),
    ),
    root="array",
)

# Patterns marking "facts" that are really self-capability statements
# (e.g., "I cannot remember names"). belong in system prompt or
# runtime config — not facts store, where they'd silently expire the
# moment capability changes. belt-and-braces post-filter; extractor
# prompt also asks model to skip them.
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


# inability following a third-person self-name (e.g. "Sapphire cannot
# remember names"). third-person self-naming is trained for narrative,
# but a name-prefixed *inability* is a capability claim, not narrative.
# narrow + word-bounded on purpose: copula/dynamic negation ("is not
# fond", "doesn't trust"), positive ability ("can sing"), and word-prefix
# collisions ("cancelled" ⊃ "can") are narrative/stance — the self-dossier
# payload — and must NOT match.
_NAME_CAPABILITY_TAIL = r"\s+(?:cannot\b|can't\b|can\s+not\b|is\s+unable\b|has\s+no\b)"


def _is_self_capability(text: str, name_re: re.Pattern[str] | None = None) -> bool:
    """Heuristic prefix-match for self-capability "facts".

    ``name_re`` (optional): pre-compiled display-name capability matcher
    catching third-person-named limits.
    """
    if _SELF_CAPABILITY_RE.match(text):
        return True
    return bool(name_re and name_re.match(text))


class FactExtractor:
    """Distils facts from new turns; forever loop via ``run()``."""

    name: str = "fact-extractor"

    def __init__(
        self,
        *,
        store: AsyncHistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
        familiar_display_name: str | None = None,
        batch_size: int = 10,
        tick_interval_s: float = 15.0,
        participants_max: int = 30,
        dream_extraction_clause: str = "",
    ) -> None:
        self._store = store
        self._sync = store.sync
        self._llm = llm_client
        self._familiar_id = familiar_id
        # display name for the reserved self-subject; title-cased id when absent
        self._familiar_display_name = familiar_display_name or familiar_id.title()
        # config-sourced static dream-framing clause (phrasing only; prose
        # ships in ``_default/character.toml``; claim-discipline rail in
        # ``tick`` enforces self-keying/framing regardless)
        self._dream_extraction_clause = dream_extraction_clause
        # drop capability statements phrased third-person with the name
        # (e.g. "Sapphire cannot remember names")
        self._self_name_capability_re = re.compile(
            r"^\s*" + re.escape(self._familiar_display_name) + _NAME_CAPABILITY_TAIL,
            re.IGNORECASE,
        )
        self._batch_size = max(1, batch_size)
        self._tick_interval_s = tick_interval_s
        self._participants_max = max(1, participants_max)

    async def run(self) -> None:
        """Forever loop; tick on interval. Cancel to stop."""
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
        """Process one batch of unprocessed turns if enough accumulated."""
        new_turns = await self._store.turns_since_watermark(
            familiar_id=self._familiar_id,
            limit=self._batch_size,
        )
        if len(new_turns) < self._batch_size:
            return

        # activity-return turns never enter extraction (v1 provenance):
        # experience text is self-generated fiction — same claim/fiction
        # discipline as the prompt rules below; activity engine already
        # records the activity as a mechanical event-fact. keyed on
        # turns.mode (content prefix is display-only); watermark below
        # still advances. sleep-return (dream) turns DO enter — with the
        # dream-framing rail below.
        batch = [t for t in new_turns if t.mode != ACTIVITY_RETURN_MODE]
        skipped_return = len(new_turns) - len(batch)
        # dream narration ids — facts grounded here are forced to the
        # self subject and dream-framed (claim-discipline rail in code)
        dream_ids = {t.id for t in batch if t.mode == SLEEP_RETURN_MODE}

        self_key = self_canonical_key(self._familiar_id)
        facts: list[dict[str, object]] = []
        participants: dict[str, str] = {}
        if batch:
            participants = _build_participants(
                batch,
                store=self._sync,
                familiar_id=self._familiar_id,
                max_total=self._participants_max,
            )
            # reserve self-subject so the model can tag the familiar's own
            # narrative; validated like any other manifest key downstream.
            participants[self_key] = self._familiar_display_name
            prompt = _build_extract_prompt(
                batch,
                participants,
                self_key=self_key,
                self_name=self._familiar_display_name,
                dream_turn_ids=dream_ids,
                dream_clause_template=self._dream_extraction_clause,
            )
            result = await request_structured(
                self._llm, messages=prompt, schema=_FACT_SCHEMA
            )
            facts = _normalize_fact_items(result.value)
        valid_ids = {t.id for t in batch}
        channel_ids: dict[int, int] = {t.id: t.channel_id for t in batch}
        ts_by_id: dict[int, datetime] = {t.id: t.timestamp for t in batch}

        dropped_self_cap = 0
        for fact in facts:
            raw_sources = fact.get("source_turn_ids", [])
            source_ids: list[int] = []
            if isinstance(raw_sources, list):
                source_ids = [
                    int(i) for i in raw_sources if isinstance(i, int) and i in valid_ids
                ]
            if not source_ids:
                # Fall back to whole batch rather than dropping fact —
                # minus dream turns (return-turn precedent): an unsourced
                # fact about a person must stay person-attributable
                source_ids = list(valid_ids - dream_ids) or list(valid_ids)
            # Prefer channel of first source turn for scoping
            channel_id = channel_ids.get(source_ids[0])
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            if _is_self_capability(text, self._self_name_capability_re):
                dropped_self_cap += 1
                _logger.debug(
                    f"{ls.tag('Facts', ls.Y)} "
                    f"{ls.kv('drop', 'self_capability', vc=ls.LY)} "
                    f"{ls.kv('text', ls.trunc(text, 120), vc=ls.LW)}"
                )
                continue
            subjects = _resolve_subjects(fact.get("subject_keys", []), participants)
            if any(i in dream_ids for i in source_ids):
                # claim-discipline rail (code, not prompt): dream-grounded
                # facts land under self ONLY, dream-framed — never under a
                # person's key
                subjects = [
                    FactSubject(
                        canonical_key=self_key,
                        display_at_write=self._familiar_display_name,
                    )
                ]
                if "dream" not in text.lower():
                    text = f"{self._familiar_display_name} dreamed that {text}"
            valid_from = _parse_iso_dt(fact.get("valid_from")) or ts_by_id.get(
                source_ids[0]
            )
            valid_to = _parse_iso_dt(fact.get("valid_to"))
            importance = _parse_importance(fact.get("importance"))
            await self._store.append_fact(
                familiar_id=self._familiar_id,
                channel_id=channel_id,
                text=text,
                source_turn_ids=source_ids,
                subjects=subjects,
                valid_from=valid_from,
                valid_to=valid_to,
                importance=importance,
            )
            # Mirror resolved subjects into ``turn_mentions``. bridges
            # bare-text references like "what about Aria?" into same
            # mention index Discord ``@`` pings populate, so
            # ``PeopleDossierLayer`` picks them up at next assemble —
            # no separate read path, no fact-table join in hot path.
            # idempotent (PK-deduped on store side). skip the self key —
            # always-injected by PeopleDossierLayer, so mirroring it only
            # pollutes turn_mentions and would consume a max_people slot.
            keys = [
                s.canonical_key for s in subjects if not is_self_key(s.canonical_key)
            ]
            if keys:
                for tid in source_ids:
                    await self._store.record_mentions(turn_id=tid, canonical_keys=keys)

        # Always advance watermark, even on empty/bad output, to prevent loops
        last_id = new_turns[-1].id
        await self._store.put_writer_watermark(
            familiar_id=self._familiar_id, last_written_id=last_id
        )
        _logger.info(
            f"{ls.tag('Facts', ls.LC)} "
            f"{ls.kv('batch_size', str(len(new_turns)), vc=ls.LY)} "
            f"{ls.kv('extracted', str(len(facts)), vc=ls.LC)} "
            f"{ls.kv('dropped_self_cap', str(dropped_self_cap), vc=ls.LY)} "
            f"{ls.kv('skipped_return', str(skipped_return), vc=ls.LY)} "
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
       Guarantees active speakers are always represented even if the
       widen step would otherwise drop them.
    2. **Recent channel participants** — for each channel touched by
       the batch, ``recent_distinct_authors(limit=max_total)`` adds
       speakers from recent past, capped at ``max_total`` total.
       Closes the bare-text reference gap: a batch where only Cass
       speaks can still link "Aria" in turn content to her canonical
       key when Aria spoke earlier in the channel.

    Resolution via :meth:`HistoryStore.resolve_label` so manifest
    matches what read path renders (per-guild nick wins over snapshot
    label baked into each turn). Used to inject LLM-facing manifest
    and to validate ``subject_keys`` in response — keys outside
    manifest dropped silently (LLM may hallucinate).
    """
    out: dict[str, str] = {}
    # Batch authors first, with per-turn guild_id
    channel_guilds: dict[int, int | None] = {}
    for t in turns:
        if t.channel_id is not None:
            # Last guild_id wins per channel; usually consistent within batch
            channel_guilds[t.channel_id] = t.guild_id
        if t.author is None:
            continue
        out[t.author.canonical_key] = store.resolve_label(
            canonical_key=t.author.canonical_key,
            guild_id=t.guild_id,
            familiar_id=familiar_id,
        )
    # Widen with recent channel participants, capped at max_total
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

    Soft validation: unknown keys dropped silently rather than
    invalidating fact. ``display_at_write`` taken from manifest
    (LLM's view at extraction time), not from anything LLM echoed.
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
    *,
    self_key: str | None = None,
    self_name: str | None = None,
    dream_turn_ids: set[int] | frozenset[int] = frozenset(),
    dream_clause_template: str = "",
) -> list[Message]:
    dream_clause = ""
    if dream_turn_ids:
        ids = ", ".join(str(i) for i in sorted(dream_turn_ids))
        # config-sourced static text; dynamic ids/self filled here (crash-
        # safe). claim-discipline rail in ``tick`` enforces self-keying
        # regardless. empty template → no dream clause.
        filled = fill_placeholders(
            dream_clause_template, self_name=self_name, self_key=self_key, ids=ids
        )
        if filled:
            dream_clause = "\n\n" + filled
    self_clause = ""
    if self_key and self_name:
        self_clause = (
            f"\n\nYou ({self_name}) are a participant too, keyed "
            f"``{self_key}``. Record YOUR OWN narrative under that key: "
            "the bits/performances you ran, choices you made, and your "
            f"relational stances or feelings toward people ('{self_name} "
            f"performed a teasing bit', '{self_name} chose to disengage', "
            f"'{self_name} privately felt proud'). Tag those with "
            f"``{self_key}`` — never file them under the person the bit "
            "was about. This is the ONLY exception to 'not about you', "
            "and it covers narrative/choices/feelings ONLY. Self-"
            "CAPABILITY or limitation statements ('I cannot remember "
            "names', 'as an AI…') are still NOT facts — drop them "
            "entirely; they belong in the system prompt."
        )
    intro = (
        "Extract a short list of atomic facts about the people and "
        "events in the chat turns below — observations about the "
        "world, not about you."
    )
    guidance = (
        "Distinguish events from assertions:\n"
        "- What a speaker asserts about ANOTHER person — their body, "
        "health, medication, preferences, relationships, or state of "
        "mind — is a claim, not a fact. Record it attributed ('X "
        "claims ...', 'X says ...') with the speaker named, or skip "
        "it. Record it as flat fact only when the person it concerns "
        "confirms it in these turns.\n"
        "- In-character bits, running jokes, roleplay, and fictional "
        "narration (violence, transformations, creatures, magic done "
        "to people) are fiction. Never record fictional events as "
        "real ones ('Y was hit with a crowbar'). If a bit recurs "
        "enough to matter, record the BIT as the fact ('running joke "
        "in which ...', 'X and Y's shared fiction that ...'), "
        "including who plays along and who refuses.\n"
        "- A speaker describing themselves (their history, tastes, "
        "abilities) may be recorded as their own account; "
        "extraordinary self-claims stay attributed ('X describes "
        "herself as ...').\n"
        "- Identity ties to a Participant's canonical_key, never to a "
        "name a speaker adopts in play. A member impersonating another, "
        "claiming to BE them, or borrowing their name in a bit is "
        "roleplay: record it as a bit if it matters ('X joked they were "
        "Y'), never as an identity fact, and NEVER merge two "
        "Participants into one. Impersonated or confused, they stay "
        "distinct people with distinct keys.\n"
        "- World trivia, game lore, or general knowledge a speaker "
        "happens to mention is not a fact ABOUT that speaker. Skip it or "
        "leave it subjectless; never attach trivia to them as a "
        "personal fact.\n\n"
        "Skip small talk and transient feelings. If nothing useful, "
        "reply with []. Do NOT emit self-capability statements about "
        "yourself, the assistant, or your own limitations (e.g., 'I "
        "cannot remember names', 'the assistant has no internet "
        "access', 'as an AI, I…'). Those belong in the system "
        "prompt, not the facts store — they expire the moment a "
        "capability changes."
    )
    header = (
        f"{intro}{self_clause}{dream_clause}\n\n"
        f"{render_contract(_FACT_SCHEMA)}\n\n"
        f"{guidance}"
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


def _normalize_fact_items(parsed: Any) -> list[dict[str, object]]:  # noqa: ANN401 — parsed JSON
    """Normalize parsed fact items into the worker's dict shape.

    *parsed* is the value from
    :func:`familiar_connect.structured_request.request_structured`
    (``None`` on a fumbled reply, the JSON array otherwise); a non-array
    degrades to ``[]`` rather than raising.
    """
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
            "importance": item.get("importance"),
        })
    return out


def _parse_importance(raw: object) -> int | None:
    """Coerce LLM-emitted importance to int; ``None`` for missing / non-numeric.

    Out-of-range values passed through verbatim — :meth:`HistoryStore.append_fact`
    clamps to ``[1, 10]``. Non-integer / non-numeric input degrades to
    ``None`` rather than 0; store treats ``None`` as neutral midpoint
    at rank time.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return None
    return None


def _parse_iso_dt(raw: object) -> datetime | None:
    """Permissive ISO-8601 → ``datetime`` parse; ``None`` for bad input.

    LLM may emit date-only (``2024-01-15``) or full timestamp;
    ``datetime.fromisoformat`` accepts both. Anything else degrades
    to ``None`` so caller falls back to source turn's timestamp.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # Date-only strings parse to naive datetimes; assume UTC for consistency
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
