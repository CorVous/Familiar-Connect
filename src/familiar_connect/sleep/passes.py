"""Maintenance-pass registry — discrete DB-maintenance actions + order.

A *maintenance pass* is one discrete, one-shot consolidation over the
familiar's database (run once per sleep, not a forever-loop). Today's
two: ``hygiene`` (fact retire/rewrite) and ``dream`` (opinion-formation).

Mirrors the projector registry (:mod:`processors.projectors`) Cassidy
cited as precedent — a :class:`MaintenancePass` Protocol for the common
trait, a module ``_REGISTRY``, :func:`register_pass`,
:func:`create_passes` (raises on unknown name), :func:`known_passes`,
and an ordered :data:`DEFAULT_PASSES`. This is the explicit "list of
maintenance actions to run, and when" — configurable, not a hard-coded
inseparable network of modules.

The ONE thing projectors lack: an ordered inter-pass data-flow. Hygiene's
retirements feed dream's known-bits deny-list. Modeled as sequential
:func:`run_passes` where each pass reads/writes a shared
:class:`MaintenanceRun` — hygiene stashes its retired-fact ids, dream
resolves them to texts for its deny-list.

The free functions :func:`execute_sleep` / :func:`execute_dream` remain
the plan→apply orchestrators each pass wraps; ad-hoc callers may still
use them directly. Lives under :mod:`familiar_connect.sleep` so the
engine imports it without pulling the CLI command package.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from familiar_connect import log_style as ls
from familiar_connect.identity import self_canonical_key
from familiar_connect.sleep.apply import apply_hygiene
from familiar_connect.sleep.dream import (
    DEFAULT_OPINION_CAP,
    apply_dream,
    plan_dream,
)
from familiar_connect.sleep.hygiene import (
    DEFAULT_FACTS_MAX,
    DEFAULT_RETIRE_CAP,
    DEFAULT_TURNS_MAX,
    plan_hygiene,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.llm import LLMClient
    from familiar_connect.sleep.dream import DreamPlan
    from familiar_connect.sleep.hygiene import HygienePlan

_logger = logging.getLogger(__name__)

# Registered pass names — also the keys callers pass to ``create_passes``.
HYGIENE_PASS = "hygiene"  # noqa: S105 — pass name, not a secret
DREAM_PASS = "dream"  # noqa: S105 — pass name, not a secret


class _Rejection(Protocol):
    """Shared shape of a rail-blocked proposal across passes."""

    rail: str
    detail: str
    payload: dict[str, Any]


def _log_rejections(pass_name: str, rejected: Sequence[_Rejection]) -> None:
    """Emit each rail-blocked proposal as a WARNING naming the rail.

    The sleep audit JSON is gone; this log is the surviving record of what
    the LLM proposed that the code rails refused (rail + truncated payload).
    """
    for r in rejected:
        _logger.warning(
            f"{ls.tag('Sleep', ls.G)} {pass_name} rejected "
            f"{ls.kv('rail', r.rail, vc=ls.LY)} "
            f"{ls.kv('detail', ls.trunc(r.detail, 80), vc=ls.LW)} "
            f"{ls.kv('payload', ls.trunc(str(r.payload), 160), vc=ls.LW)}"
        )


async def execute_sleep(
    *,
    store: AsyncHistoryStore,
    llm: LLMClient,
    familiar_id: str,
    familiar_display_name: str | None,
    apply: bool,
    facts_max: int = DEFAULT_FACTS_MAX,
    turns_max: int = DEFAULT_TURNS_MAX,
    cap: int = DEFAULT_RETIRE_CAP,
) -> HygienePlan:
    """Plan (always) → apply (if ``apply``). Return the plan.

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
    _log_rejections("hygiene", plan.rejected)
    if apply:
        await apply_hygiene(store, plan, familiar_display_name=familiar_display_name)
    return plan


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
    apply: bool,
    denylist: tuple[str, ...] = (),
    cap: int = DEFAULT_OPINION_CAP,
) -> DreamPlan:
    """Plan opinions → apply (if ``apply``). Return the plan.

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
    _log_rejections("dream", plan.rejected)
    if apply:
        await apply_dream(store, plan, familiar_display_name=familiar_display_name)
    return plan


# ---------------------------------------------------------------------------
# maintenance-pass registry (mirrors processors/projectors.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaintenanceContext:
    """Static inputs every maintenance pass needs.

    Built once per sleep, reused across every factory in
    :data:`_REGISTRY` — replacing the repeated kwarg lists threaded
    through the free functions today. Adding an input here is the
    extension point; callers don't know which pass consumes which field.
    """

    store: AsyncHistoryStore
    llm: LLMClient
    familiar_id: str
    display_name: str | None
    display_tz: str
    apply: bool
    facts_max: int = DEFAULT_FACTS_MAX
    turns_max: int = DEFAULT_TURNS_MAX
    retire_cap: int = DEFAULT_RETIRE_CAP
    opinion_cap: int = DEFAULT_OPINION_CAP


@dataclass
class MaintenanceRun:
    """Mutable accumulator threading one pass's result to the next.

    The ordered data-flow projectors lack: hygiene writes the fact ids it
    retired this run; dream reads them (resolving to texts) for its
    known-bits deny-list. Carrying ids here — not running passes
    independently — keeps the exact engine denylist behavior.

    ``dream_plan`` is the dream pass's product, surfaced for the
    engine-owned prose-gen step (prose stays engine-owned; the run only
    carries the result it consumes).
    """

    denylist_fact_ids: list[int] = field(default_factory=list)
    dream_plan: DreamPlan | None = None


class MaintenancePass(Protocol):
    """One discrete DB-maintenance action, run once per sleep.

    Common trait Cassidy asked for: a stable ``name`` (label for logs /
    selection) and a single one-shot ``run`` — NOT a forever-loop.
    ``run`` reads/writes the shared :class:`MaintenanceRun` so an earlier
    pass's result reaches a later one.
    """

    name: str

    async def run(self, run: MaintenanceRun) -> object: ...


PassFactory = Callable[[MaintenanceContext], MaintenancePass]


_REGISTRY: dict[str, PassFactory] = {}


def register_pass(name: str, factory: PassFactory) -> None:
    """Register *factory* under *name*. Re-registration overwrites."""
    _REGISTRY[name] = factory


def known_passes() -> set[str]:
    """Names registered today."""
    return set(_REGISTRY)


def create_passes(
    *,
    names: Sequence[str],
    context: MaintenanceContext,
) -> list[MaintenancePass]:
    """Instantiate selected passes in *names* order.

    :raises ValueError: when any name is not in :data:`_REGISTRY`.
    """
    out: list[MaintenancePass] = []
    for name in names:
        factory = _REGISTRY.get(name)
        if factory is None:
            valid = ", ".join(sorted(_REGISTRY)) or "(none)"
            msg = f"unknown maintenance pass {name!r}; valid: {valid}"
            raise ValueError(msg)
        out.append(factory(context))
    return out


async def run_passes(
    passes: Sequence[MaintenancePass], run: MaintenanceRun | None = None
) -> MaintenanceRun:
    """Run *passes* in order, threading the shared :class:`MaintenanceRun`.

    Returns the run so a caller can inspect accumulated state. Does NOT
    guard pass failures — the engine owns its own never-blocks-return
    guard; raising here surfaces a bug to that guard unchanged.
    """
    run = run if run is not None else MaintenanceRun()
    for p in passes:
        await p.run(run)
    return run


# ---------------------------------------------------------------------------
# built-in passes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HygienePass:
    """Fact retire/rewrite consolidation. Stashes retired ids for dream."""

    ctx: MaintenanceContext
    name: str = HYGIENE_PASS

    async def run(self, run: MaintenanceRun) -> HygienePlan:
        ctx = self.ctx
        plan = await execute_sleep(
            store=ctx.store,
            llm=ctx.llm,
            familiar_id=ctx.familiar_id,
            familiar_display_name=ctx.display_name,
            apply=ctx.apply,
            facts_max=ctx.facts_max,
            turns_max=ctx.turns_max,
            cap=ctx.retire_cap,
        )
        # thread retirements forward → dream's deny-list (ordered data-flow)
        run.denylist_fact_ids.extend(hygiene_denylist_ids(plan))
        return plan


@dataclass(frozen=True)
class DreamPass:
    """Opinion-formation. Reads hygiene's retired ids as its deny-list."""

    ctx: MaintenanceContext
    name: str = DREAM_PASS

    async def run(self, run: MaintenanceRun) -> DreamPlan:
        ctx = self.ctx
        denylist: tuple[str, ...] = ()
        if run.denylist_fact_ids:
            facts = await ctx.store.facts_by_ids(
                familiar_id=ctx.familiar_id, ids=run.denylist_fact_ids
            )
            denylist = tuple(f.text for f in facts)
        plan = await execute_dream(
            store=ctx.store,
            llm=ctx.llm,
            familiar_id=ctx.familiar_id,
            familiar_display_name=ctx.display_name,
            display_tz=ctx.display_tz,
            apply=ctx.apply,
            denylist=denylist,
            cap=ctx.opinion_cap,
        )
        run.dream_plan = plan  # surface for engine-owned prose-gen
        return plan


def _hygiene_factory(ctx: MaintenanceContext) -> MaintenancePass:
    return HygienePass(ctx)


def _dream_factory(ctx: MaintenanceContext) -> MaintenancePass:
    return DreamPass(ctx)


register_pass(HYGIENE_PASS, _hygiene_factory)
register_pass(DREAM_PASS, _dream_factory)


DEFAULT_PASSES: tuple[str, ...] = (HYGIENE_PASS, DREAM_PASS)
"""Passes run when none explicitly selected — hygiene then dream."""
