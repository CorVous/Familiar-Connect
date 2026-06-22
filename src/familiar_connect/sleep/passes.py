"""Programmatic sleep-pass entry points — hygiene + dream.

Shared by the activity engine's lifecycle-coupled passes (fired at
sleep departure) and any ad-hoc caller. Lives under
:mod:`familiar_connect.sleep` so the engine can import it without
pulling the CLI command package (which imports ``run`` and would cycle
back into activities).
"""

from __future__ import annotations

import logging
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
