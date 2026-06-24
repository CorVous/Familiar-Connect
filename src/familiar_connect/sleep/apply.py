"""Apply a validated :class:`ConsolidationPlan` to the DB.

Separate from planning so a dry run can compute the plan without
touching the DB. Apply is supersede-only: retires drop facts with no
replacement, rewrites append one consolidated fact then supersede the
old ones by it. The sleep watermark advances once, to the window's
high-water marks, regardless of how many actions ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from familiar_connect.history.store import FactDraft, FactSubject
from familiar_connect.identity import is_self_key

if TYPE_CHECKING:
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.sleep.consolidation import ConsolidationPlan, RewriteAction

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplyReport:
    """Outcome of applying a plan.

    ``skipped`` records actions a concurrent writer invalidated between
    plan and apply — each ``(kind, fact_id, reason)``. Per-action
    ValueError handling makes apply non-fatal and idempotent: a partial
    or already-applied plan re-runs cleanly instead of crashing.
    """

    retired_fact_ids: tuple[int, ...]
    rewritten: tuple[tuple[tuple[int, ...], int], ...]  # (old_ids, new_id)
    watermark: tuple[int, int]  # (last_fact_id, last_turn_id)
    skipped: tuple[tuple[str, int, str], ...] = ()


def _subjects_for_rewrite(
    action: RewriteAction,
    *,
    by_id: dict[int, Any],
    familiar_display_name: str | None,
) -> list[FactSubject]:
    """Build ``FactSubject`` list for a rewrite's chosen keys.

    display_at_write resolved from the matching source-fact subject
    when present; for the self key falls back to the familiar's display
    name; otherwise the key's local part. Keeps display resolution
    working at read time without a second lookup.
    """
    display_by_key: dict[str, str] = {}
    for fid in action.old_fact_ids:
        for s in by_id[fid].subjects:
            display_by_key.setdefault(s.canonical_key, s.display_at_write)
    out: list[FactSubject] = []
    for key in action.subject_keys:
        if key in display_by_key:
            display = display_by_key[key]
        elif is_self_key(key):
            display = familiar_display_name or key.split(":", 1)[-1]
        else:
            display = key.split(":", 1)[-1]
        out.append(FactSubject(canonical_key=key, display_at_write=display))
    return out


async def apply_consolidation(
    store: AsyncHistoryStore,
    plan: ConsolidationPlan,
    *,
    familiar_display_name: str | None = None,
) -> ApplyReport:
    """Execute a plan's accepted actions; advance the sleep watermark."""
    fam = plan.familiar_id

    # snapshot rewrite sources up front (exact id fetch, includes
    # superseded) to resolve subject displays + the merge channel even as
    # rows get superseded — and so a >100k fact base can't drop a needed
    # row. Provenance union is the store's job now (see supersede).
    needed: set[int] = set()
    for a in plan.rewrite:
        needed.update(a.old_fact_ids)
    by_id = {f.id: f for f in await store.facts_by_ids(familiar_id=fam, ids=needed)}

    skipped: list[tuple[str, int, str]] = []
    retired: list[int] = []
    for action in plan.retire:
        # the store retires each id, recording (not raising on) any a
        # concurrent writer already retired/superseded since plan time.
        result = await store.supersede(
            familiar_id=fam, obsolete_facts=action.fact_ids, new_fact=None
        )
        retired.extend(result.superseded)
        skipped.extend(("retire", fid, reason) for fid, reason in result.skipped)

    rewritten: list[tuple[tuple[int, ...], int]] = []
    for action in plan.rewrite:
        # the store owns the merge: it pre-flights every source, unions
        # their provenance, and mints (dedup=False) only if all are
        # current — declining the whole merge otherwise. Subjects stay
        # caller-prepared; channel follows the first obsolete row.
        result = await store.supersede(
            familiar_id=fam,
            obsolete_facts=action.old_fact_ids,
            new_fact=FactDraft(
                channel_id=by_id[action.old_fact_ids[0]].channel_id,
                text=action.new_text,
                subjects=tuple(
                    _subjects_for_rewrite(
                        action,
                        by_id=by_id,
                        familiar_display_name=familiar_display_name,
                    )
                ),
            ),
        )
        if result.minted is None:
            # a source raced between plan + apply, so the store declined
            # the merge whole — same outcome as the old pre-flight skip:
            # nothing minted, the action recorded as skipped, no raise.
            skipped.extend(("rewrite", fid, reason) for fid, reason in result.skipped)
            continue
        rewritten.append((action.old_fact_ids, result.minted.id))

    # consolidation owns the FACT axis only; the turn axis belongs to the
    # opinion pass. Advancing just last_fact_id leaves its progress untouched.
    await store.advance_sleep_watermark(
        familiar_id=fam,
        last_fact_id=plan.new_last_fact_id,
    )
    _logger.info(
        "sleep-consolidation applied familiar=%s retired=%d rewritten=%d "
        "skipped=%d fact_watermark=%d",
        fam,
        len(retired),
        len(rewritten),
        len(skipped),
        plan.new_last_fact_id,
    )
    return ApplyReport(
        retired_fact_ids=tuple(retired),
        rewritten=tuple(rewritten),
        watermark=(plan.new_last_fact_id, plan.new_last_turn_id),
        skipped=tuple(skipped),
    )
