"""Apply a validated :class:`HygienePlan` + emit its audit artifact.

Separate from planning so a dry run can produce the audit without
touching the DB. Apply is supersede-only: retires drop facts with no
replacement, rewrites append one consolidated fact then supersede the
old ones by it. The sleep watermark advances once, to the window's
high-water marks, regardless of how many actions ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from familiar_connect.history.store import FactSubject
from familiar_connect.identity import is_self_key

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.sleep.hygiene import HygienePlan, RewriteAction

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


async def apply_hygiene(
    store: AsyncHistoryStore,
    plan: HygienePlan,
    *,
    familiar_display_name: str | None = None,
) -> ApplyReport:
    """Execute a plan's accepted actions; advance the sleep watermark."""
    fam = plan.familiar_id

    # snapshot source facts up front (exact id fetch, includes superseded)
    # so rewrites union provenance + resolve subject displays even as rows
    # get superseded — and a >100k fact base can't drop a needed row.
    needed: set[int] = set()
    for a in plan.retire:
        needed.update(a.fact_ids)
    for a in plan.rewrite:
        needed.update(a.old_fact_ids)
    by_id = {f.id: f for f in await store.facts_by_ids(familiar_id=fam, ids=needed)}

    skipped: list[tuple[str, int, str]] = []
    retired: list[int] = []
    for action in plan.retire:
        for fid in action.fact_ids:
            # a concurrent writer may have retired/superseded this since
            # plan time — skip + record rather than crash the whole pass.
            try:
                await store.retire_fact(familiar_id=fam, fact_id=fid)
            except ValueError as exc:
                skipped.append(("retire", fid, str(exc)))
                continue
            retired.append(fid)

    rewritten: list[tuple[tuple[int, ...], int]] = []
    for action in plan.rewrite:
        # pre-flight: if any source is gone/superseded, skip the whole
        # rewrite before appending — never strand a half-merged fact.
        live = await store.facts_by_ids(familiar_id=fam, ids=action.old_fact_ids)
        live_current = {f.id for f in live if f.superseded_at is None}
        stale = [fid for fid in action.old_fact_ids if fid not in live_current]
        if stale:
            skipped.extend(
                ("rewrite", fid, "source no longer current") for fid in stale
            )
            continue
        # union provenance across the merged facts, order-preserving
        turn_ids: list[int] = []
        for fid in action.old_fact_ids:
            for tid in by_id[fid].source_turn_ids:
                if tid not in turn_ids:
                    turn_ids.append(tid)
        channel_id = by_id[action.old_fact_ids[0]].channel_id
        subjects = _subjects_for_rewrite(
            action, by_id=by_id, familiar_display_name=familiar_display_name
        )
        # dedup=False: the consolidated text may equal one of the rows
        # we're about to supersede; force a fresh row so the merge isn't
        # swallowed by near-dup suppression.
        new_fact = await store.append_fact(
            familiar_id=fam,
            channel_id=channel_id,
            text=action.new_text,
            source_turn_ids=turn_ids,
            subjects=subjects,
            dedup=False,
        )
        for fid in action.old_fact_ids:
            try:
                await store.supersede_fact(
                    familiar_id=fam, old_id=fid, new_id=new_fact.id
                )
            except ValueError as exc:
                # lost a race after the pre-flight — new fact stays current
                # (carries the merged content); leave the old be, record it.
                skipped.append(("rewrite", fid, str(exc)))
        rewritten.append((action.old_fact_ids, new_fact.id))

    # hygiene owns the FACT axis only; the turn axis belongs to the dream
    # pass. Advancing just last_fact_id leaves dream's progress untouched.
    await store.advance_sleep_watermark(
        familiar_id=fam,
        last_fact_id=plan.new_last_fact_id,
    )
    _logger.info(
        "sleep-hygiene applied familiar=%s retired=%d rewritten=%d skipped=%d "
        "fact_watermark=%d",
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


# ---------------------------------------------------------------------------
# audit artifact
# ---------------------------------------------------------------------------


def hygiene_audit(
    plan: HygienePlan,
    *,
    applied: bool,
    report: ApplyReport | None = None,
) -> dict[str, Any]:
    """JSON-serializable record of a pass — every change + why.

    Written for every run (dry or applied) so the operator can audit
    what the pass would do / did. ``report`` (when applied) annotates
    rewrites with the new fact id minted.
    """
    new_id_by_olds = (
        {tuple(o): n for o, n in report.rewritten} if report is not None else {}
    )
    return {
        "familiar_id": plan.familiar_id,
        "applied": applied,
        "mutated_count": plan.mutated_count,
        "notes": list(plan.notes),
        "watermark": {
            "last_fact_id": plan.new_last_fact_id,
            "last_turn_id": plan.new_last_turn_id,
        },
        "window": {
            "facts_considered": plan.facts_considered,
            "facts_truncated": plan.facts_truncated,
            "turns_considered": plan.turns_considered,
            "turns_truncated": plan.turns_truncated,
        },
        "retire": [
            {"fact_ids": list(a.fact_ids), "reason": a.reason} for a in plan.retire
        ],
        "rewrite": [
            {
                "old_fact_ids": list(a.old_fact_ids),
                "new_text": a.new_text,
                "subject_keys": list(a.subject_keys),
                "reason": a.reason,
                "new_fact_id": new_id_by_olds.get(a.old_fact_ids),
            }
            for a in plan.rewrite
        ],
        "rejected": [
            {
                "kind": r.kind,
                "rail": r.rail,
                "detail": r.detail,
                "payload": r.payload,
            }
            for r in plan.rejected
        ],
        "skipped": [
            {"kind": k, "fact_id": fid, "reason": reason}
            for (k, fid, reason) in (report.skipped if report is not None else ())
        ],
    }


def write_audit(
    audit: dict[str, Any],
    *,
    audit_dir: Path,
    familiar_id: str,
    when: datetime | None = None,
) -> Path:
    """Write *audit* as timestamped JSON under *audit_dir*; return the path."""
    import json  # noqa: PLC0415 — keep module import surface light

    audit_dir.mkdir(parents=True, exist_ok=True)
    stamp = (when or datetime.now(tz=UTC)).strftime("%Y%m%dT%H%M%SZ")
    path = audit_dir / f"{familiar_id}-{stamp}.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return path
