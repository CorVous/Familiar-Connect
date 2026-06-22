"""Maintenance-pass registry — list of DB-maintenance actions + order.

Mirrors the projector registry (``processors/projectors.py``): a
``MaintenancePass`` Protocol for the common trait, a module registry,
``create_passes(names, ctx)`` raising on unknown names, ``known_passes``,
and an ordered ``DEFAULT_PASSES``. The one thing projectors lack: an
ordered data-flow — consolidation's retired facts feed the opinion
deny-list,
threaded through a shared run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import familiar_connect.sleep.maintenance as maintenance_mod
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import is_self_key
from familiar_connect.sleep.maintenance import (
    CONSOLIDATION_PASS,
    DEFAULT_PASSES,
    OPINION_PASS,
    MaintenanceContext,
    create_passes,
    known_passes,
    run_passes,
)
from tests.conftest import FakeLLMClient

ARIA = (FactSubject(canonical_key="discord:A", display_at_write="Aria"),)


def _store() -> HistoryStore:
    store = HistoryStore(":memory:")
    store.append_turn(
        familiar_id="fam", channel_id=1, role="user", content="hi", author=None
    )
    store.append_turn(
        familiar_id="fam",
        channel_id=1,
        role="assistant",
        content="lo-fi is real music, fight me",
        author=None,
    )
    store.append_fact(
        familiar_id="fam", channel_id=1, text="noise", source_turn_ids=[1]
    )
    store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria likes tea.",
        source_turn_ids=[1],
        subjects=ARIA,
    )
    return store


def _ctx(store: AsyncHistoryStore, *, apply: bool) -> MaintenanceContext:
    return MaintenanceContext(
        store=store,
        llm=FakeLLMClient(replies=[]),
        familiar_id="fam",
        display_name="Sapphire",
        display_tz="UTC",
        apply=apply,
    )


def _consolidation_reply() -> str:
    # retire fact id 1 ("noise")
    return json.dumps({
        "retire": [{"fact_ids": [1], "reason": "noise"}],
        "rewrite": [],
    })


def _opinion_replies() -> list[str]:
    return [
        json.dumps({"candidates": [{"text": "defends lo-fi", "turn_ids": [2]}]}),
        json.dumps({
            "opinions": [
                {
                    "text": "Sapphire is fiercely protective of lo-fi as real music.",
                    "source_turn_ids": [2],
                    "reason": "defended it",
                }
            ]
        }),
    ]


class TestRegistryShape:
    """Mirror of ``TestProjectorRegistry`` — names, order, unknown raises."""

    def test_create_passes_returns_in_order(self) -> None:
        ctx = _ctx(AsyncHistoryStore(_store()), apply=False)
        passes = create_passes(names=[CONSOLIDATION_PASS, OPINION_PASS], context=ctx)
        assert [p.name for p in passes] == [CONSOLIDATION_PASS, OPINION_PASS]

    def test_create_passes_unknown_name_raises(self) -> None:
        ctx = _ctx(AsyncHistoryStore(_store()), apply=False)
        with pytest.raises(ValueError, match="nope"):
            create_passes(names=["nope"], context=ctx)

    def test_known_passes_lists_registered(self) -> None:
        assert known_passes() == {CONSOLIDATION_PASS, OPINION_PASS}

    def test_default_passes_is_consolidation_then_opinion(self) -> None:
        assert DEFAULT_PASSES == (CONSOLIDATION_PASS, OPINION_PASS)


class TestDenylistThreaded:
    """The ONE way this differs from projectors: ordered data-flow."""

    @pytest.mark.asyncio
    async def test_consolidation_retirement_reaches_opinion_denylist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = _store()
        store = AsyncHistoryStore(raw)
        seen: dict[str, Any] = {}
        orig = maintenance_mod.execute_opinion_formation

        # capture the denylist the opinion pass actually plans against
        async def spy_opinion(**kw: Any) -> Any:  # noqa: ANN401
            seen.update(kw)
            return await orig(**kw)

        monkeypatch.setattr(maintenance_mod, "execute_opinion_formation", spy_opinion)
        ctx = MaintenanceContext(
            store=store,
            llm=FakeLLMClient(replies=[_consolidation_reply(), *_opinion_replies()]),
            familiar_id="fam",
            display_name="Sapphire",
            display_tz="UTC",
            apply=True,
        )
        await run_passes(create_passes(names=DEFAULT_PASSES, context=ctx))

        # fact id 1 ("noise") retired this run reaches the opinion deny-list
        assert seen["denylist"] == ("noise",)


class TestDefaultRunApplies:
    @pytest.mark.asyncio
    async def test_default_run_executes_consolidation_then_opinion(self) -> None:
        raw = _store()
        store = AsyncHistoryStore(raw)
        ctx = MaintenanceContext(
            store=store,
            llm=FakeLLMClient(replies=[_consolidation_reply(), *_opinion_replies()]),
            familiar_id="fam",
            display_name="Sapphire",
            display_tz="UTC",
            apply=True,
        )
        await run_passes(create_passes(names=DEFAULT_PASSES, context=ctx))
        texts = {f.text for f in raw.recent_facts(familiar_id="fam", limit=10)}
        # consolidation retired "noise"
        assert "noise" not in texts
        # opinion pass recorded a self: opinion
        opinions = [
            f
            for f in raw.recent_facts(familiar_id="fam", limit=10)
            if f.subjects and is_self_key(f.subjects[0].canonical_key)
        ]
        assert len(opinions) == 1
