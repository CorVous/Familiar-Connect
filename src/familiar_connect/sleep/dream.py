"""Dream / opinion-formation pass — mint the familiar's stances.

Sleep function #2 (hygiene is #1). Reads the conversation log up to the
sleep watermark's turn axis, bucketed per calendar day in the familiar's
``display_tz``, and forms her OPINIONS — stored as ordinary facts routed
to the ``self:`` subject, grounded in the turns that demonstrate them via
``source_turn_ids`` ("authored against her log").

Two-pass, mirroring hygiene's LLM-proposes/code-decides shape:
  1. per-day **candidate** stance-moments — each carries that day's cited
     turn-ids (parallel-friendly, contamination stays local).
  2. ONE **synthesis** over all candidates — code enforces every minted
     opinion's grounding ⊆ the union of its input candidates' ids, so the
     synthesis can never invent grounding, and it sees the whole arc at
     once (the whole-window property a sequential carry would lose).

Opinions change only through this pass (grounded in experience), never
through hygiene (textual redundancy / contradiction) — see the
``self_subject`` rail in :mod:`familiar_connect.sleep.hygiene`.
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
from familiar_connect.structured_output import coerce_json, coerce_positive_int_list

if TYPE_CHECKING:
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import HistoryTurn, SleepWatermark
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)

DEFAULT_TURNS_MAX_PER_DAY = 600
DEFAULT_OPINION_CAP = 60


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
        """Turns the familiar authored (role=assistant) — her own acts."""
        return frozenset(t.id for t in self.turns if t.role == "assistant")


@dataclass(frozen=True)
class DreamWindow:
    """The log slice one dream pass reasons over, bucketed by local day."""

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
) -> DreamWindow:
    """Collect turns since the dream (turn-axis) watermark, bucketed by day."""
    prior = await store.get_sleep_watermark(familiar_id=familiar_id)
    min_turn = prior.last_turn_id if prior is not None else 0
    max_turn_id = await store.latest_fts_id(familiar_id=familiar_id)
    turns = await store.turns_in_id_range(
        familiar_id=familiar_id,
        min_id_exclusive=min_turn,
        max_id_inclusive=max_turn_id,
    )
    return DreamWindow(
        familiar_id=familiar_id,
        days=tuple(bucket_by_day(turns, display_tz)),
        prior_watermark=prior,
        max_turn_id=max_turn_id,
    )


# ---------------------------------------------------------------------------
# stage 1 — per-day candidate stance-moments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A stance-moment from one day, grounded in that day's turn-ids."""

    text: str
    date: str
    turn_ids: tuple[int, ...]


def _render_turn(t: HistoryTurn, self_name: str) -> str:
    """Render a turn for the prompt, marking HER own turns unmistakably.

    Only assistant-role turns are hers; a user posting under her display
    name (impersonation — real in her log) renders with an explicit
    disambiguator so the model never reads it as her own act.
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


def _build_candidate_prompt(
    day: DayBatch, *, self_name: str, denylist: tuple[str, ...]
) -> list[Message]:
    deny = ""
    if denylist:
        deny = (
            "\n\nKNOWN BITS / NOISE (already judged not-real — "
            f"{self_name} may have a TAKE on these, e.g. finding a running "
            "joke tedious, but must never treat them as true events):\n"
            + "\n".join(f"- {d}" for d in denylist)
        )
    system = (
        f"You read one day of chat and surface stance-moments for {self_name}, "
        "an AI character. A stance-moment is a spot where she demonstrated a "
        "feeling, opinion, preference, or reaction — about media, an event, a "
        "person, a bit, anything. Quote nothing; just name the stance and cite "
        "the turn ids that show it.\n\n"
        "Ground every candidate in real turn ids from the transcript below. "
        "Only turns marked '(you)' are hers — never attribute a stance to her "
        "from a turn she did not author, even if a user shares her name. "
        "Prefer turns SHE authored (her own words/acts). Skip small-talk with "
        "no stance. A thin day yields few or zero candidates — that is fine."
        + deny
        + '\n\nReply JSON only: {"candidates": [{"text": "<stance>", '
        '"turn_ids": [<id>...]}]}. Empty list when nothing stands out.'
    )
    body = "\n".join(_render_turn(t, self_name) for t in day.turns)
    return [
        Message(role="system", content=system),
        Message(role="user", content=f"Day {day.date}:\n{body}"),
    ]


async def extract_candidates(
    llm: LLMClient,
    day: DayBatch,
    *,
    self_name: str,
    denylist: tuple[str, ...] = (),
) -> list[Candidate]:
    """Stage 1: pull candidate stances for one day, grounding ⊆ the day."""
    reply = await llm.chat(
        _build_candidate_prompt(day, self_name=self_name, denylist=denylist)
    )
    obj = _extract_object(reply.content_str)
    if obj is None:
        return []
    raw = obj.get("candidates")
    if not isinstance(raw, list):
        return []
    out: list[Candidate] = []
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
        out.append(Candidate(text=text, date=day.date, turn_ids=ids))
    return out


# ---------------------------------------------------------------------------
# stage 2 — synthesis across all candidates
# ---------------------------------------------------------------------------


def _build_synthesis_prompt(
    candidates: list[Candidate],
    *,
    self_name: str,
    prior_self_dossier: str | None,
) -> list[Message]:
    prior = ""
    if prior_self_dossier:
        prior = (
            f"\n\n{self_name}'s existing self-understanding (refine/extend, "
            f"do not blindly repeat):\n{prior_self_dossier}"
        )
    system = (
        f"You synthesize {self_name}'s settled opinions from stance-moments "
        "observed across many days. Merge restatements of the same stance into "
        "one. Keep stances that genuinely differ (including a view she changed "
        "her mind on — record both, they are both true in their time). "
        "An opinion is HERS — phrase it about her. You may mention other "
        "people in passing, but the opinion is her stance, not a claim about "
        "them.\n\n"
        "CRITICAL: every opinion's source_turn_ids MUST be drawn only from the "
        "ids listed on the candidates below. Invent no ids.\n\n"
        "Rate each opinion's IMPORTANCE 1-10 by how durable/central it is to "
        "who she is, NOT how strongly worded:\n"
        "- 7-9 = durable core stance about her identity, values, or a key "
        "relationship (would still be true in a month).\n"
        "- 4-6 = characteristic but situational (recurring tendency tied to a "
        "context).\n"
        "- 2-3 = momentary texture, tied to one specific exchange or event.\n"
        "Reserve 10 for rare bedrock identity; 1 for trivial."
        + prior
        + '\n\nReply JSON only: {"opinions": [{"text": "<her stance>", '
        '"source_turn_ids": [<id>...], "importance": <1-10>, '
        '"reason": "<why>"}]}.'
    )
    lines = [f"- ({c.date}) ids={list(c.turn_ids)}: {c.text}" for c in candidates]
    body = "Stance-moments:\n" + "\n".join(lines)
    return [
        Message(role="system", content=system),
        Message(role="user", content=body),
    ]


async def synthesize(
    llm: LLMClient,
    candidates: list[Candidate],
    *,
    self_name: str,
    prior_self_dossier: str | None = None,
) -> list[dict[str, Any]]:
    """Stage 2: one call → raw opinion dicts (validated separately)."""
    if not candidates:
        return []
    reply = await llm.chat(
        _build_synthesis_prompt(
            candidates, self_name=self_name, prior_self_dossier=prior_self_dossier
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
    """An accepted opinion, ready to mint as a ``self:`` fact."""

    text: str
    source_turn_ids: tuple[int, ...]
    valid_from_date: str  # YYYY-MM-DD, earliest grounding day
    self_grounded: bool  # ≥1 source turn she authored
    importance: int  # LLM-rated 1-10 durability/centrality


@dataclass(frozen=True)
class RejectedOpinion:
    payload: dict[str, Any]
    rail: str
    detail: str


@dataclass(frozen=True)
class DreamPlan:
    familiar_id: str
    opinions: tuple[OpinionFact, ...]
    rejected: tuple[RejectedOpinion, ...]
    flags: tuple[str, ...]
    new_last_turn_id: int
    days_considered: int = 0
    candidates_considered: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


def validate_opinions(
    raw: list[dict[str, Any]],
    *,
    candidates: list[Candidate],
    window: DreamWindow,
    cap: int = DEFAULT_OPINION_CAP,
    notes: tuple[str, ...] = (),
) -> DreamPlan:
    """Filter synthesized opinions through the grounding rails.

    Rails: source ids ⊆ the union of candidate ids (the synthesis can
    not invent grounding); ≥1 id (``ungrounded``); non-empty text
    (``empty_text``); normalized-text dedup across the plan
    (``duplicate``); total ≤ ``cap``. An opinion grounded in no turn she
    authored is accepted but FLAGGED (it may be the room's stance, not
    hers). ``valid_from`` = earliest grounding day (honest timeline).
    """
    candidate_union: set[int] = {i for c in candidates for i in c.turn_ids}
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
        bad = [i for i in ids if i not in candidate_union]
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

    return DreamPlan(
        familiar_id=window.familiar_id,
        opinions=tuple(opinions),
        rejected=tuple(rejected),
        flags=tuple(flags),
        new_last_turn_id=window.max_turn_id,
        days_considered=len(window.days),
        candidates_considered=len(candidates),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# orchestrate
# ---------------------------------------------------------------------------


async def plan_dream(
    store: AsyncHistoryStore,
    llm: LLMClient,
    *,
    familiar_id: str,
    display_tz: str,
    self_name: str,
    denylist: tuple[str, ...] = (),
    prior_self_dossier: str | None = None,
    cap: int = DEFAULT_OPINION_CAP,
) -> DreamPlan:
    """Gather days → per-day candidates → one synthesis → validate."""
    window = await gather_days(store, familiar_id=familiar_id, display_tz=display_tz)
    candidates: list[Candidate] = []
    for day in window.days:
        candidates.extend(
            await extract_candidates(llm, day, self_name=self_name, denylist=denylist)
        )
    raw = await synthesize(
        llm, candidates, self_name=self_name, prior_self_dossier=prior_self_dossier
    )
    plan = validate_opinions(raw, candidates=candidates, window=window, cap=cap)
    _logger.info(
        "dream plan familiar=%s days=%d candidates=%d opinions=%d rejected=%d flags=%d",
        familiar_id,
        plan.days_considered,
        plan.candidates_considered,
        len(plan.opinions),
        len(plan.rejected),
        len(plan.flags),
    )
    return plan


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DreamApplyReport:
    """Outcome of minting a dream plan's opinions."""

    minted: tuple[tuple[str, int], ...]  # (opinion text, fact id)
    watermark: int  # advanced last_turn_id


async def apply_dream(
    store: AsyncHistoryStore,
    plan: DreamPlan,
    *,
    familiar_display_name: str | None = None,
) -> DreamApplyReport:
    """Mint opinions as ``self:`` facts; advance the dream (turn) watermark."""
    fam = plan.familiar_id
    subj = FactSubject(
        canonical_key=self_canonical_key(fam),
        display_at_write=familiar_display_name or fam,
    )
    minted: list[tuple[str, int]] = []
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
        minted.append((op.text, fact.id))
    # dream owns the TURN axis only — leaves hygiene's fact axis intact.
    await store.advance_sleep_watermark(
        familiar_id=fam, last_turn_id=plan.new_last_turn_id
    )
    _logger.info(
        "dream applied familiar=%s minted=%d turn_watermark=%d",
        fam,
        len(minted),
        plan.new_last_turn_id,
    )
    return DreamApplyReport(minted=tuple(minted), watermark=plan.new_last_turn_id)
