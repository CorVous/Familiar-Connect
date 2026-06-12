"""Programmatic sleep-pass entry points — hygiene + dream.

Shared by the ``familiar-connect sleep`` CLI (manual operator path)
and the activity engine's lifecycle-coupled passes (fired at sleep
departure). Lives under :mod:`familiar_connect.sleep` so the engine
can import it without pulling the CLI command package (which imports
``run`` and would cycle back into activities).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.identity import self_canonical_key
from familiar_connect.sleep.apply import apply_hygiene, hygiene_audit, write_audit
from familiar_connect.sleep.dream import (
    DEFAULT_OPINION_CAP,
    apply_dream,
    dream_audit,
    plan_dream,
)
from familiar_connect.sleep.hygiene import (
    DEFAULT_FACTS_MAX,
    DEFAULT_RETIRE_CAP,
    DEFAULT_TURNS_MAX,
    plan_hygiene,
)

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.llm import LLMClient
    from familiar_connect.sleep.dream import DreamPlan
    from familiar_connect.sleep.hygiene import HygienePlan

AUDIT_DIRNAME = "sleep_audits"


async def execute_sleep(
    *,
    store: AsyncHistoryStore,
    llm: LLMClient,
    familiar_id: str,
    familiar_display_name: str | None,
    audit_dir: Path,
    apply: bool,
    facts_max: int = DEFAULT_FACTS_MAX,
    turns_max: int = DEFAULT_TURNS_MAX,
    cap: int = DEFAULT_RETIRE_CAP,
) -> tuple[HygienePlan, Path]:
    """Plan (always) → apply (if ``apply``) → write audit. Return plan + path.

    Dependency-injected so the orchestration is testable without
    config/network. ``plan_hygiene`` is read-only; only ``apply_hygiene``
    mutates, so dry-run (``apply=False``) never writes to ``store``.
    """
    plan = await plan_hygiene(
        store,
        llm,
        familiar_id=familiar_id,
        facts_max=facts_max,
        turns_max=turns_max,
        cap=cap,
    )
    report = None
    # try/finally: the run that mutated rows must never be the run with no
    # audit. If apply raises unexpectedly mid-way, the artifact still lands
    # (report may be partial/None) for post-mortem before re-running.
    try:
        if apply:
            report = await apply_hygiene(
                store, plan, familiar_display_name=familiar_display_name
            )
    finally:
        audit = hygiene_audit(plan, applied=apply, report=report)
        path = write_audit(audit, audit_dir=audit_dir, familiar_id=familiar_id)
    return plan, path


def hygiene_denylist_ids(plan: HygienePlan) -> list[int]:
    """Fact ids hygiene judged junk/bits this run — feeds dream's deny-list."""
    ids: list[int] = []
    for a in plan.retire:
        ids.extend(a.fact_ids)
    for a in plan.rewrite:
        ids.extend(a.old_fact_ids)
    return ids


async def execute_dream(
    *,
    store: AsyncHistoryStore,
    llm: LLMClient,
    familiar_id: str,
    familiar_display_name: str | None,
    display_tz: str,
    audit_dir: Path,
    apply: bool,
    denylist: tuple[str, ...] = (),
    cap: int = DEFAULT_OPINION_CAP,
) -> tuple[DreamPlan, Path]:
    """Plan opinions → apply (if ``apply``) → write dream audit + excerpts.

    ``plan_dream`` is read-only; only ``apply_dream`` mints facts, so a
    dry run never writes. ``denylist`` (hygiene's retired-fact texts) is
    fed to the prompt as known-bits context.
    """
    self_key = self_canonical_key(familiar_id)
    prior = await store.get_people_dossier(
        familiar_id=familiar_id, canonical_key=self_key
    )
    plan = await plan_dream(
        store,
        llm,
        familiar_id=familiar_id,
        display_tz=display_tz,
        self_name=familiar_display_name or familiar_id,
        denylist=denylist,
        prior_self_dossier=prior.dossier_text if prior is not None else None,
        cap=cap,
    )
    report = None
    try:
        if apply:
            report = await apply_dream(
                store, plan, familiar_display_name=familiar_display_name
            )
    finally:
        excerpts = await _cited_excerpts(
            store, familiar_id, plan, familiar_display_name
        )
        audit = dream_audit(plan, excerpts=excerpts, applied=apply, report=report)
        path = write_audit(
            audit, audit_dir=audit_dir, familiar_id=f"{familiar_id}-dream"
        )
    return plan, path


async def _cited_excerpts(
    store: AsyncHistoryStore,
    familiar_id: str,
    plan: DreamPlan,
    self_name: str | None,
) -> dict[int, str]:
    """Render ``{turn_id: 'who: content'}`` for every cited turn in the plan."""
    ids = sorted({i for op in plan.opinions for i in op.source_turn_ids})
    if not ids:
        return {}
    turns = await store.turns_by_ids(familiar_id=familiar_id, ids=ids)
    out: dict[int, str] = {}
    for t in turns:
        if t.role == "assistant":
            who = self_name or familiar_id
        elif t.author is not None:
            who = t.author.label
        else:
            who = t.role
        out[t.id] = f"{who}: {t.content}"
    return out
