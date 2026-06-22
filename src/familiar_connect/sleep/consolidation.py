"""Memory-consolidation pass — propose + validate fact consolidations.

Nightly (manual for now) pass over the day's facts. The LLM sees the
whole window at once — unlike batch-of-10 extraction it can spot
day-level patterns (a claim asserted 9x by one speaker, denied by the
subject every time = a bit, not a fact).

Two action verbs, both supersede-only (never DELETE):
  * ``retire`` — drop junk/noise/bit-asserted-as-fact (no replacement).
  * ``rewrite`` — merge near-dups or re-attribute a misfiled claim into
    one consolidated fact (old facts superseded by the new one).

Every LLM proposal is validated against rails *in code* — the model
advises; it does not decide. Pinned (authored/seed) facts are
untouchable; a per-run retirement cap bounds blast radius; a rewrite
may not introduce a person-subject absent from its source facts (the
extractor's claim-discipline, carried into consolidation). The pass
produces a :class:`ConsolidationPlan`; it never touches the DB itself —
:func:`apply_consolidation` (separate) executes an accepted plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from familiar_connect.history.store import _normalize_fact_text, _subject_key_set
from familiar_connect.identity import self_canonical_key
from familiar_connect.llm import Message
from familiar_connect.structured_output import (
    coerce_json,
    coerce_positive_int_list,
    coerce_str_list,
)

if TYPE_CHECKING:
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.history.store import (
        Fact,
        HistoryTurn,
        SleepWatermark,
    )
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)

DEFAULT_FACTS_MAX = 500
DEFAULT_TURNS_MAX = 400
DEFAULT_RETIRE_CAP = 50


@dataclass(frozen=True)
class ConsolidationWindow:
    """The slice of memory one consolidation pass reasons over.

    ``facts`` are current (non-superseded) facts, capped to
    ``facts_max`` newest. ``turns`` are conversation since the prior
    sleep watermark, capped to ``turns_max`` newest. ``max_fact_id`` /
    ``max_turn_id`` are the true high-water marks (uncapped) the
    watermark advances to. Truncation counts are recorded on the plan so
    a cap never reads as full coverage.
    """

    familiar_id: str
    facts: tuple[Fact, ...]
    turns: tuple[HistoryTurn, ...]
    prior_watermark: SleepWatermark | None
    max_fact_id: int
    max_turn_id: int
    facts_truncated: int
    turns_truncated: int


@dataclass(frozen=True)
class RetireAction:
    """Accepted: retire ``fact_ids`` with no replacement."""

    fact_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class RewriteAction:
    """Accepted: supersede ``old_fact_ids`` with one new consolidated fact."""

    old_fact_ids: tuple[int, ...]
    new_text: str
    subject_keys: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RejectedAction:
    """A proposed action a rail refused. ``payload`` is the raw proposal."""

    kind: str  # "retire" | "rewrite"
    payload: dict[str, Any]
    rail: str
    detail: str


@dataclass(frozen=True)
class ConsolidationPlan:
    """Validated outcome of one pass — accepted actions + rejections.

    Pure description; applying is :func:`apply_consolidation`'s job.
    ``new_last_fact_id`` / ``new_last_turn_id`` are the watermark the
    apply step advances to once the window is consolidated.
    """

    familiar_id: str
    retire: tuple[RetireAction, ...]
    rewrite: tuple[RewriteAction, ...]
    rejected: tuple[RejectedAction, ...]
    new_last_fact_id: int
    new_last_turn_id: int
    facts_considered: int = 0
    facts_truncated: int = 0
    turns_considered: int = 0
    turns_truncated: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def mutated_count(self) -> int:
        """Facts retired + superseded if this plan is applied."""
        r = sum(len(a.fact_ids) for a in self.retire)
        w = sum(len(a.old_fact_ids) for a in self.rewrite)
        return r + w


async def gather_window(
    store: AsyncHistoryStore,
    *,
    familiar_id: str,
    facts_max: int = DEFAULT_FACTS_MAX,
    turns_max: int = DEFAULT_TURNS_MAX,
) -> ConsolidationWindow:
    """Collect current facts + recent turns into a consolidation window."""
    prior = await store.get_sleep_watermark(familiar_id=familiar_id)
    max_fact_id = await store.latest_fact_id(familiar_id=familiar_id)
    max_turn_id = await store.latest_fts_id(familiar_id=familiar_id)

    # all current facts, newest-first; keep newest facts_max, prompt
    # them oldest-first for stable ordering.
    all_current = await store.recent_facts(
        familiar_id=familiar_id, limit=100_000, include_superseded=False
    )
    facts_truncated = max(0, len(all_current) - facts_max)
    kept = list(reversed(all_current[:facts_max]))

    # turns since prior sleep (re-attribution context). bound to newest.
    min_turn = prior.last_turn_id if prior is not None else 0
    all_turns = await store.turns_in_id_range(
        familiar_id=familiar_id,
        min_id_exclusive=min_turn,
        max_id_inclusive=max_turn_id,
    )
    turns_truncated = max(0, len(all_turns) - turns_max)
    kept_turns = all_turns[-turns_max:] if turns_max > 0 else []

    return ConsolidationWindow(
        familiar_id=familiar_id,
        facts=tuple(kept),
        turns=tuple(kept_turns),
        prior_watermark=prior,
        max_fact_id=max_fact_id,
        max_turn_id=max_turn_id,
        facts_truncated=facts_truncated,
        turns_truncated=turns_truncated,
    )


def build_prompt(
    window: ConsolidationWindow, *, self_key: str, system: str = ""
) -> list[Message]:
    """Render the window into the consolidation LLM prompt.

    ``system`` is the static instruction text — config-sourced per
    familiar (prose ships in ``_default/character.toml``; empty Python
    default keeps that profile the single source of truth). The dynamic
    window data (facts, turns, self_key) is assembled here regardless.
    """
    lines: list[str] = [
        f"Self subject key (the character): {self_key}",
        "",
        "Facts (current):",
    ]
    for f in window.facts:
        subj = ", ".join(s.canonical_key for s in f.subjects) or "—"
        lines.append(f"- id={f.id} subjects=[{subj}]: {f.text}")
    if window.turns:
        lines.extend(("", "Recent conversation (for attribution context):"))
        for t in window.turns:
            who = t.author.label if t.author is not None else t.role
            lines.append(f"- {who}: {t.content}")
    return [
        Message(role="system", content=system),
        Message(role="user", content="\n".join(lines)),
    ]


def parse_actions(reply: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract ``(retire, rewrite)`` raw lists from an LLM reply.

    Permissive: bad JSON, missing keys, non-list values all degrade to
    empty lists rather than raising. Non-dict items are dropped.
    """
    parsed = coerce_json(reply, expect="object").value or {}
    # coerce_json only extracts a blob; a non-object payload (e.g. an
    # array reached via the no-match fallback) must still degrade.
    if not isinstance(parsed, dict):
        return [], []

    def _dicts(key: str) -> list[dict[str, Any]]:
        raw = parsed.get(key)
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    return _dicts("retire"), _dicts("rewrite")


def reply_parse_failed(reply: str) -> bool:
    """Report whether a non-empty reply yielded no parseable plan object.

    Distinguishes a model that fumbled the JSON (chatty prose, multiple
    objects the greedy regex spans) from a clean empty plan — the former
    silently zeroes a night's work and is worth a note.
    """
    if not reply or not reply.strip():
        return False
    result = coerce_json(reply, expect="object")
    # a clean plan parses AND is an object; anything else (malformed JSON
    # → parsed_ok=False, or a non-object payload) counts as a failure.
    return not (result.parsed_ok and isinstance(result.value, dict))


def validate(
    window: ConsolidationWindow,
    *,
    retire_raw: list[dict[str, Any]],
    rewrite_raw: list[dict[str, Any]],
    self_key: str,
    cap: int = DEFAULT_RETIRE_CAP,
    notes: tuple[str, ...] = (),
) -> ConsolidationPlan:
    """Filter proposals through the safety rails; build the plan.

    Rails, in order of check per action:
      1. every referenced id exists in the window (else ``unknown_id``)
      2. no referenced fact is a ``self:`` (opinion) fact — consolidation
         never adjudicates feelings (else ``self_subject``)
      3. no fact targeted by more than one action (else ``duplicate_target``)
      4. rewrite: non-empty new text (``empty_text``); subjects not dropped
         when sources had them (``subject_lost``); no subject key absent
         from source facts unless it is ``self_key`` (``subject_introduced``);
         not a no-op restatement (``noop``)
      5. cumulative mutated-fact count ≤ ``cap`` (else ``cap``)
    """
    by_id = {f.id: f for f in window.facts}
    retire: list[RetireAction] = []
    rewrite: list[RewriteAction] = []
    rejected: list[RejectedAction] = []
    claimed: set[int] = set()
    mutated = 0

    def _check_ids(ids: tuple[int, ...]) -> str | None:
        for fid in ids:
            if fid not in by_id:
                return "unknown_id"
            # consolidation does not adjudicate feelings: an opinion only
            # changes through opinion-formation (grounded in experience),
            # never through this pass (textual redundancy / contradiction).
            # leave self: facts alone.
            if any(s.canonical_key == self_key for s in by_id[fid].subjects):
                return "self_subject"
        if any(fid in claimed for fid in ids):
            return "duplicate_target"
        return None

    for payload in retire_raw:
        ids = tuple(coerce_positive_int_list(payload.get("fact_ids")))
        if not ids:
            rejected.append(RejectedAction("retire", payload, "no_ids", "no fact_ids"))
            continue
        rail = _check_ids(ids)
        if rail is not None:
            rejected.append(RejectedAction("retire", payload, rail, str(ids)))
            continue
        if mutated + len(ids) > cap:
            rejected.append(
                RejectedAction("retire", payload, "cap", f"cap={cap} reached")
            )
            continue
        reason = str(payload.get("reason", "")).strip()
        retire.append(RetireAction(fact_ids=ids, reason=reason))
        claimed.update(ids)
        mutated += len(ids)

    for payload in rewrite_raw:
        ids = tuple(coerce_positive_int_list(payload.get("old_fact_ids")))
        if not ids:
            rejected.append(
                RejectedAction("rewrite", payload, "no_ids", "no old_fact_ids")
            )
            continue
        rail = _check_ids(ids)
        if rail is not None:
            rejected.append(RejectedAction("rewrite", payload, rail, str(ids)))
            continue
        new_text = str(payload.get("new_text", "")).strip()
        if not new_text:
            rejected.append(
                RejectedAction("rewrite", payload, "empty_text", "blank new_text")
            )
            continue
        keys = tuple(coerce_str_list(payload.get("subject_keys")))
        source_keys = {s.canonical_key for fid in ids for s in by_id[fid].subjects}
        # subject_lost: dropping all subjects when sources had some silently
        # orphans the fact (NULL subjects_json) and the supersession
        # invalidates the sources' dossiers → they rebuild WITHOUT the
        # consolidated content. Reject — code decides, don't guess keys.
        if source_keys and not keys:
            rejected.append(
                RejectedAction(
                    "rewrite", payload, "subject_lost", str(sorted(source_keys))
                )
            )
            continue
        introduced = [k for k in keys if k not in source_keys and k != self_key]
        if introduced:
            rejected.append(
                RejectedAction(
                    "rewrite", payload, "subject_introduced", str(introduced)
                )
            )
            continue
        # no-op: single source fact, identical normalized text + subject set
        if len(ids) == 1:
            only = by_id[ids[0]]
            same_text = _normalize_fact_text(only.text) == _normalize_fact_text(
                new_text
            )
            same_subj = _subject_key_set(only.subjects) == frozenset(keys)
            if same_text and same_subj:
                rejected.append(
                    RejectedAction("rewrite", payload, "noop", "restatement")
                )
                continue
        if mutated + len(ids) > cap:
            rejected.append(
                RejectedAction("rewrite", payload, "cap", f"cap={cap} reached")
            )
            continue
        reason = str(payload.get("reason", "")).strip()
        rewrite.append(
            RewriteAction(
                old_fact_ids=ids,
                new_text=new_text,
                subject_keys=keys,
                reason=reason,
            )
        )
        claimed.update(ids)
        mutated += len(ids)

    return ConsolidationPlan(
        familiar_id=window.familiar_id,
        retire=tuple(retire),
        rewrite=tuple(rewrite),
        rejected=tuple(rejected),
        new_last_fact_id=window.max_fact_id,
        new_last_turn_id=window.max_turn_id,
        facts_considered=len(window.facts),
        facts_truncated=window.facts_truncated,
        turns_considered=len(window.turns),
        turns_truncated=window.turns_truncated,
        notes=notes,
    )


async def plan_consolidation(
    store: AsyncHistoryStore,
    llm: LLMClient,
    *,
    familiar_id: str,
    facts_max: int = DEFAULT_FACTS_MAX,
    turns_max: int = DEFAULT_TURNS_MAX,
    cap: int = DEFAULT_RETIRE_CAP,
    system: str = "",
) -> ConsolidationPlan:
    """Gather → prompt → LLM → parse → validate. Touches no fact rows.

    ``system`` is the config-sourced static instruction text (prose ships
    in ``_default/character.toml``); rails in :func:`validate` enforce
    safety in code regardless of its phrasing.
    """
    self_key = self_canonical_key(familiar_id)
    window = await gather_window(
        store, familiar_id=familiar_id, facts_max=facts_max, turns_max=turns_max
    )
    prompt = build_prompt(window, self_key=self_key, system=system)
    reply = await llm.chat(prompt)
    retire_raw, rewrite_raw = parse_actions(reply.content_str)
    notes: tuple[str, ...] = ()
    if reply_parse_failed(reply.content_str):
        notes = ("llm reply did not parse to a plan object — treated as empty",)
        _logger.warning("sleep-consolidation parse failure familiar=%s", familiar_id)
    plan = validate(
        window,
        retire_raw=retire_raw,
        rewrite_raw=rewrite_raw,
        self_key=self_key,
        cap=cap,
        notes=notes,
    )
    _logger.info(
        "sleep-consolidation plan familiar=%s retire=%d rewrite=%d rejected=%d "
        "facts=%d(+%d trunc)",
        familiar_id,
        len(plan.retire),
        len(plan.rewrite),
        len(plan.rejected),
        plan.facts_considered,
        plan.facts_truncated,
    )
    return plan
