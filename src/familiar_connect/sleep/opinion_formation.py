"""Opinion-formation pass — form the familiar's stances.

Sleep function #2 (consolidation is #1). Reads the conversation log up
to the sleep watermark's turn axis, bucketed per calendar day in the
familiar's ``display_tz``, and forms its OPINIONS — stored as ordinary
facts routed to the ``self:`` subject, grounded in the turns that
demonstrate them via ``source_turn_ids`` (authored against its log).

Two-pass, mirroring consolidation's LLM-proposes/code-decides shape:
  1. per-day stance-moments — each carries that day's cited turn-ids
     (parallel-friendly, contamination stays local).
  2. ONE **synthesis** over all stance-moments — code enforces every
     formed opinion's grounding ⊆ the union of its input stance-moments'
     ids, so the synthesis can never invent grounding, and it sees the
     whole arc at once (the whole-window property a sequential carry
     would lose).

Opinions change only through this pass (grounded in experience), never
through consolidation (textual redundancy / contradiction) — see the
``self_subject`` rail in :mod:`familiar_connect.sleep.consolidation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from familiar_connect.history.store import FactSubject, _normalize_fact_text
from familiar_connect.identity import self_canonical_key
from familiar_connect.llm import Message
from familiar_connect.prompt_fill import fill_placeholders
from familiar_connect.structured_output import coerce_json, coerce_positive_int_list

if TYPE_CHECKING:
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import HistoryTurn, SleepWatermark
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)

DEFAULT_TURNS_MAX_PER_DAY = 600
DEFAULT_OPINION_CAP = 60

# Suffixes appended in code after the (formatted) static system text —
# JSON-shape contract + conditional blocks. Kept out of config: they
# describe the machine-parsed reply shape, not persona phrasing.
_STANCE_REPLY_SHAPE = (
    '\n\nReply JSON only: {"candidates": [{"text": "<stance>", '
    '"turn_ids": [<id>...]}]}. Empty list when nothing stands out.'
)
_SYNTHESIS_REPLY_SHAPE = (
    '\n\nReply JSON only: {"opinions": [{"text": "<their stance>", '
    '"source_turn_ids": [<id>...], "importance": <1-10>, '
    '"reason": "<why>"}]}.'
)


@dataclass(frozen=True)
class DayBatch:
    """One calendar day of turns (local to the familiar's ``display_tz``)."""

    date: str  # YYYY-MM-DD in display_tz
    turns: tuple[HistoryTurn, ...]

    @property
    def turn_ids(self) -> frozenset[int]:
        """Every turn id in this day — the grounding pool for the day."""
        return frozenset(t.id for t in self.turns)

    @property
    def self_turn_ids(self) -> frozenset[int]:
        """Turns the familiar authored (role=assistant) — its own acts."""
        return frozenset(t.id for t in self.turns if t.role == "assistant")


@dataclass(frozen=True)
class OpinionWindow:
    """The log slice one opinion pass reasons over, bucketed by local day."""

    familiar_id: str
    days: tuple[DayBatch, ...]
    prior_watermark: SleepWatermark | None
    max_turn_id: int


def bucket_by_day(
    turns: tuple[HistoryTurn, ...] | list[HistoryTurn],
    tz_name: str,
) -> list[DayBatch]:
    """Group turns into calendar-day batches in ``tz_name``, oldest first.

    Turn timestamps are UTC-aware; bucket key is the local date. Within a
    day turns stay id-ordered (chronological).
    """
    tz = ZoneInfo(tz_name)
    by_date: dict[str, list[HistoryTurn]] = {}
    for t in sorted(turns, key=lambda t: t.id):
        key = t.timestamp.astimezone(tz).strftime("%Y-%m-%d")
        by_date.setdefault(key, []).append(t)
    return [DayBatch(date=d, turns=tuple(ts)) for d, ts in sorted(by_date.items())]


async def gather_days(
    store: AsyncHistoryStore,
    *,
    familiar_id: str,
    display_tz: str,
) -> OpinionWindow:
    """Collect turns since the turn-axis watermark, bucketed by day."""
    prior = await store.get_sleep_watermark(familiar_id=familiar_id)
    min_turn = prior.last_turn_id if prior is not None else 0
    max_turn_id = await store.latest_fts_id(familiar_id=familiar_id)
    turns = await store.turns_in_id_range(
        familiar_id=familiar_id,
        min_id_exclusive=min_turn,
        max_id_inclusive=max_turn_id,
    )
    return OpinionWindow(
        familiar_id=familiar_id,
        days=tuple(bucket_by_day(turns, display_tz)),
        prior_watermark=prior,
        max_turn_id=max_turn_id,
    )


# ---------------------------------------------------------------------------
# stage 1 — per-day stance moments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StanceMoment:
    """A stance from one day, grounded in that day's turn-ids."""

    text: str
    date: str
    turn_ids: tuple[int, ...]


def _render_turn(t: HistoryTurn, self_name: str) -> str:
    """Render a turn for the prompt, marking the familiar's own turns.

    Only assistant-role turns are the familiar's; a user posting under
    its display name (impersonation — real in the log) renders with an
    explicit disambiguator so the model never reads it as the familiar's
    own act.
    """
    if t.role == "assistant":
        return f"[{t.id}] {self_name} (you): {t.content}"
    label = t.author.label if t.author is not None else t.role
    if label == self_name:
        label = f"{label} (a user, not you)"
    return f"[{t.id}] {label}: {t.content}"


def _coerce_importance(raw: Any) -> int:  # noqa: ANN401
    """1-10 importance from model output; clamp range, default 5.

    Never a rejection reason — bad/missing value degrades to neutral 5.
    """
    if isinstance(raw, bool):
        return 5
    if isinstance(raw, int):
        val = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        val = int(raw.strip())
    else:
        return 5
    return max(1, min(10, val))


def _extract_object(reply: str) -> dict[str, Any] | None:
    """Permissive: pull the first JSON object from an LLM reply, or None."""
    result = coerce_json(reply, expect="object")
    if not result.parsed_ok:
        return None
    # coerce_json only extracts; a non-object payload still degrades.
    return result.value if isinstance(result.value, dict) else None


def _build_stance_prompt(
    day: DayBatch,
    *,
    self_name: str,
    denylist: tuple[str, ...],
    system: str = "",
) -> list[Message]:
    """Build the per-day stance prompt.

    ``system`` is the config-sourced static instruction (``{self_name}``
    filled here, crash-safe; prose ships in ``_default/character.toml``);
    the known-bits deny block and the JSON reply shape are assembled in
    code and appended. Grounding rails enforce safety regardless of text.
    """
    deny = ""
    if denylist:
        deny = (
            "\n\nKNOWN BITS / NOISE (already judged not-real — "
            f"{self_name} may have a TAKE on these, e.g. finding a running "
            "joke tedious, but must never treat them as true events):\n"
            + "\n".join(f"- {d}" for d in denylist)
        )
    instruction = fill_placeholders(system, self_name=self_name)
    body = "\n".join(_render_turn(t, self_name) for t in day.turns)
    return [
        Message(role="system", content=instruction + deny + _STANCE_REPLY_SHAPE),
        Message(role="user", content=f"Day {day.date}:\n{body}"),
    ]


async def extract_stance_moments(
    llm: LLMClient,
    day: DayBatch,
    *,
    self_name: str,
    denylist: tuple[str, ...] = (),
    system: str = "",
) -> list[StanceMoment]:
    """Stage 1: pull stance-moments for one day, grounding ⊆ the day."""
    reply = await llm.chat(
        _build_stance_prompt(day, self_name=self_name, denylist=denylist, system=system)
    )
    obj = _extract_object(reply.content_str)
    if obj is None:
        return []
    raw = obj.get("candidates")
    if not isinstance(raw, list):
        return []
    out: list[StanceMoment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        # code-enforce: keep only ids that really belong to this day
        ids = tuple(
            i
            for i in coerce_positive_int_list(item.get("turn_ids"))
            if i in day.turn_ids
        )
        if not ids:
            continue
        out.append(StanceMoment(text=text, date=day.date, turn_ids=ids))
    return out


# ---------------------------------------------------------------------------
# stage 2 — synthesis across all stance moments
# ---------------------------------------------------------------------------


def _build_synthesis_prompt(
    stance_moments: list[StanceMoment],
    *,
    self_name: str,
    prior_self_dossier: str | None,
    system: str = "",
) -> list[Message]:
    """Build the synthesis prompt.

    ``system`` is the config-sourced static instruction (``{self_name}``
    filled here, crash-safe; prose ships in ``_default/character.toml``);
    the prior-dossier block and the JSON reply shape are assembled in
    code and appended. The grounding rail (source ids ⊆ stance-moment
    ids) enforces safety regardless.
    """
    prior = ""
    if prior_self_dossier:
        prior = (
            f"\n\n{self_name}'s existing self-understanding (refine/extend, "
            f"do not blindly repeat):\n{prior_self_dossier}"
        )
    instruction = fill_placeholders(system, self_name=self_name)
    lines = [f"- ({c.date}) ids={list(c.turn_ids)}: {c.text}" for c in stance_moments]
    body = "Stance-moments:\n" + "\n".join(lines)
    return [
        Message(role="system", content=instruction + prior + _SYNTHESIS_REPLY_SHAPE),
        Message(role="user", content=body),
    ]


async def synthesize(
    llm: LLMClient,
    stance_moments: list[StanceMoment],
    *,
    self_name: str,
    prior_self_dossier: str | None = None,
    system: str = "",
) -> list[dict[str, Any]]:
    """Stage 2: one call → raw opinion dicts (validated separately)."""
    if not stance_moments:
        return []
    reply = await llm.chat(
        _build_synthesis_prompt(
            stance_moments,
            self_name=self_name,
            prior_self_dossier=prior_self_dossier,
            system=system,
        )
    )
    obj = _extract_object(reply.content_str)
    if obj is None:
        return []
    raw = obj.get("opinions")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# validate — rails enforced in code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpinionFact:
    """An accepted opinion, ready to record as a ``self:`` fact."""

    text: str
    source_turn_ids: tuple[int, ...]
    valid_from_date: str  # YYYY-MM-DD, earliest grounding day
    self_grounded: bool  # ≥1 source turn the familiar authored
    importance: int  # LLM-rated 1-10 durability/centrality


@dataclass(frozen=True)
class RejectedOpinion:
    payload: dict[str, Any]
    rail: str
    detail: str


@dataclass(frozen=True)
class OpinionPlan:
    familiar_id: str
    opinions: tuple[OpinionFact, ...]
    rejected: tuple[RejectedOpinion, ...]
    flags: tuple[str, ...]
    new_last_turn_id: int
    days_considered: int = 0
    stance_moments_considered: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


def validate_opinions(
    raw: list[dict[str, Any]],
    *,
    stance_moments: list[StanceMoment],
    window: OpinionWindow,
    cap: int = DEFAULT_OPINION_CAP,
    notes: tuple[str, ...] = (),
) -> OpinionPlan:
    """Filter synthesized opinions through the grounding rails.

    Rails: source ids ⊆ the union of stance-moment ids (the synthesis can
    not invent grounding); ≥1 id (``ungrounded``); non-empty text
    (``empty_text``); normalized-text dedup across the plan
    (``duplicate``); total ≤ ``cap``. An opinion grounded in no turn the
    familiar authored is accepted but FLAGGED (it may be the room's
    stance, not the familiar's). ``valid_from`` = earliest grounding day
    (honest timeline).
    """
    grounding_union: set[int] = {i for c in stance_moments for i in c.turn_ids}
    turn_day: dict[int, str] = {t.id: d.date for d in window.days for t in d.turns}
    self_ids: set[int] = {
        t.id for d in window.days for t in d.turns if t.role == "assistant"
    }
    opinions: list[OpinionFact] = []
    rejected: list[RejectedOpinion] = []
    flags: list[str] = []
    seen_norm: set[str] = set()

    for payload in raw:
        text = str(payload.get("text", "")).strip()
        if not text:
            rejected.append(RejectedOpinion(payload, "empty_text", "blank text"))
            continue
        ids = tuple(coerce_positive_int_list(payload.get("source_turn_ids")))
        bad = [i for i in ids if i not in grounding_union]
        if not ids or bad:
            rejected.append(
                RejectedOpinion(payload, "ungrounded", f"ids={list(ids)} bad={bad}")
            )
            continue
        norm = _normalize_fact_text(text)
        if norm in seen_norm:
            rejected.append(RejectedOpinion(payload, "duplicate", text))
            continue
        if len(opinions) >= cap:
            rejected.append(RejectedOpinion(payload, "cap", f"cap={cap}"))
            continue
        seen_norm.add(norm)
        self_grounded = any(i in self_ids for i in ids)
        if not self_grounded:
            flags.append(f"no_self_authored: {text}")
        valid_from = min(turn_day[i] for i in ids if i in turn_day)
        opinions.append(
            OpinionFact(
                text=text,
                source_turn_ids=ids,
                valid_from_date=valid_from,
                self_grounded=self_grounded,
                importance=_coerce_importance(payload.get("importance")),
            )
        )

    return OpinionPlan(
        familiar_id=window.familiar_id,
        opinions=tuple(opinions),
        rejected=tuple(rejected),
        flags=tuple(flags),
        new_last_turn_id=window.max_turn_id,
        days_considered=len(window.days),
        stance_moments_considered=len(stance_moments),
        notes=notes,
    )


async def plan_opinions(
    store: AsyncHistoryStore,
    llm: LLMClient,
    *,
    familiar_id: str,
    display_tz: str,
    self_name: str,
    denylist: tuple[str, ...] = (),
    prior_self_dossier: str | None = None,
    cap: int = DEFAULT_OPINION_CAP,
    stance_system: str = "",
    synthesis_system: str = "",
) -> OpinionPlan:
    """Gather days → per-day stance-moments → one synthesis → validate.

    ``stance_system`` / ``synthesis_system`` are config-sourced static
    instructions (prose ships in ``_default/character.toml``); grounding
    rails enforce safety regardless of phrasing.
    """
    window = await gather_days(store, familiar_id=familiar_id, display_tz=display_tz)
    stance_moments: list[StanceMoment] = []
    for day in window.days:
        stance_moments.extend(
            await extract_stance_moments(
                llm, day, self_name=self_name, denylist=denylist, system=stance_system
            )
        )
    raw = await synthesize(
        llm,
        stance_moments,
        self_name=self_name,
        prior_self_dossier=prior_self_dossier,
        system=synthesis_system,
    )
    plan = validate_opinions(raw, stance_moments=stance_moments, window=window, cap=cap)
    _logger.info(
        "opinion plan familiar=%s days=%d stance_moments=%d opinions=%d "
        "rejected=%d flags=%d",
        familiar_id,
        plan.days_considered,
        plan.stance_moments_considered,
        len(plan.opinions),
        len(plan.rejected),
        len(plan.flags),
    )
    return plan


@dataclass(frozen=True)
class OpinionApplyReport:
    """Outcome of recording an opinion plan's opinions."""

    recorded: tuple[tuple[str, int], ...]  # (opinion text, fact id)
    watermark: int  # advanced last_turn_id


async def apply_opinions(
    store: AsyncHistoryStore,
    plan: OpinionPlan,
    *,
    familiar_display_name: str | None = None,
) -> OpinionApplyReport:
    """Record opinions as ``self:`` facts; advance the turn watermark."""
    fam = plan.familiar_id
    subj = FactSubject(
        canonical_key=self_canonical_key(fam),
        display_at_write=familiar_display_name or fam,
    )
    recorded: list[tuple[str, int]] = []
    for op in plan.opinions:
        # valid_from = midnight UTC of the source day — honest stance
        # timeline even though created_at is tonight.
        valid_from = datetime.fromisoformat(f"{op.valid_from_date}T00:00:00+00:00")
        fact = await store.append_fact(
            familiar_id=fam,
            channel_id=None,  # opinions are global stances, not channel-bound
            text=op.text,
            source_turn_ids=list(op.source_turn_ids),
            subjects=[subj],
            valid_from=valid_from,
            importance=op.importance,
        )
        recorded.append((op.text, fact.id))
    # opinion-formation owns the TURN axis only — leaves the consolidation
    # fact axis intact.
    await store.advance_sleep_watermark(
        familiar_id=fam, last_turn_id=plan.new_last_turn_id
    )
    _logger.info(
        "opinions applied familiar=%s recorded=%d turn_watermark=%d",
        fam,
        len(recorded),
        plan.new_last_turn_id,
    )
    return OpinionApplyReport(recorded=tuple(recorded), watermark=plan.new_last_turn_id)
