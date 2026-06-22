"""Sleep consolidation — apply path."""

from __future__ import annotations

import json

import pytest

from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import self_canonical_key
from familiar_connect.sleep.apply import apply_consolidation
from familiar_connect.sleep.consolidation import (
    ConsolidationPlan,
    RetireAction,
    RewriteAction,
    plan_consolidation,
)
from tests.conftest import FakeLLMClient

ARIA = (FactSubject(canonical_key="discord:A", display_at_write="Aria"),)


def _store_with_facts() -> tuple[HistoryStore, int, int, int]:
    store = HistoryStore(":memory:")
    for _ in range(3):
        store.append_turn(
            familiar_id="fam", channel_id=1, role="user", content="hi", author=None
        )
    junk = store.append_fact(
        familiar_id="fam", channel_id=1, text="noise", source_turn_ids=[1]
    )
    d1 = store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria likes berries.",
        source_turn_ids=[2],
        subjects=ARIA,
    )
    d2 = store.append_fact(
        familiar_id="fam",
        channel_id=1,
        text="Aria really likes berries.",
        source_turn_ids=[3],
        subjects=ARIA,
    )
    return store, junk.id, d1.id, d2.id


class TestApplyConsolidation:
    @pytest.mark.asyncio
    async def test_retire_marks_superseded(self) -> None:
        raw, junk_id, _, _ = _store_with_facts()
        store = AsyncHistoryStore(raw)
        plan = ConsolidationPlan(
            familiar_id="fam",
            retire=(RetireAction(fact_ids=(junk_id,), reason="noise"),),
            rewrite=(),
            rejected=(),
            new_last_fact_id=3,
            new_last_turn_id=3,
        )
        report = await apply_consolidation(store, plan)
        assert report.retired_fact_ids == (junk_id,)
        current = {f.text for f in raw.recent_facts(familiar_id="fam", limit=10)}
        assert "noise" not in current

    @pytest.mark.asyncio
    async def test_rewrite_merges_with_union_provenance(self) -> None:
        raw, _, d1, d2 = _store_with_facts()
        store = AsyncHistoryStore(raw)
        plan = ConsolidationPlan(
            familiar_id="fam",
            retire=(),
            rewrite=(
                RewriteAction(
                    old_fact_ids=(d1, d2),
                    new_text="Aria likes berries.",
                    subject_keys=("discord:A",),
                    reason="merge",
                ),
            ),
            rejected=(),
            new_last_fact_id=3,
            new_last_turn_id=3,
        )
        report = await apply_consolidation(store, plan)
        assert len(report.rewritten) == 1
        old_ids, new_id = report.rewritten[0]
        assert old_ids == (d1, d2)
        # old facts gone from current, new one present with union provenance
        current = raw.recent_facts(familiar_id="fam", limit=10)
        new_fact = next(f for f in current if f.id == new_id)
        assert new_fact.text == "Aria likes berries."
        assert set(new_fact.source_turn_ids) == {2, 3}
        assert new_fact.subjects[0].canonical_key == "discord:A"
        all_facts = raw.recent_facts(
            familiar_id="fam", limit=10, include_superseded=True
        )
        old1 = next(f for f in all_facts if f.id == d1)
        assert old1.superseded_by == new_id

    @pytest.mark.asyncio
    async def test_rewrite_to_self_subject(self) -> None:
        raw, _, d1, _ = _store_with_facts()
        store = AsyncHistoryStore(raw)
        self_key = self_canonical_key("fam")
        plan = ConsolidationPlan(
            familiar_id="fam",
            retire=(),
            rewrite=(
                RewriteAction(
                    old_fact_ids=(d1,),
                    new_text="Sapphire teases Aria about berries.",
                    subject_keys=(self_key,),
                    reason="bit",
                ),
            ),
            rejected=(),
            new_last_fact_id=3,
            new_last_turn_id=3,
        )
        report = await apply_consolidation(
            store, plan, familiar_display_name="Sapphire"
        )
        _, new_id = report.rewritten[0]
        new_fact = next(
            f for f in raw.recent_facts(familiar_id="fam", limit=10) if f.id == new_id
        )
        assert new_fact.subjects[0].canonical_key == self_key

    @pytest.mark.asyncio
    async def test_skips_concurrently_superseded_fact_without_raising(self) -> None:
        """F1: a planned fact superseded between plan + apply is skipped."""
        raw, junk_id, d1, d2 = _store_with_facts()
        # simulate the live bot retiring the junk fact during the plan→apply gap
        raw.supersede(familiar_id="fam", obsolete_facts=[junk_id], new_fact=None)
        store = AsyncHistoryStore(raw)
        plan = ConsolidationPlan(
            familiar_id="fam",
            retire=(RetireAction(fact_ids=(junk_id,), reason="noise"),),
            rewrite=(
                RewriteAction(
                    old_fact_ids=(d1, d2),
                    new_text="Aria likes berries.",
                    subject_keys=("discord:A",),
                    reason="merge",
                ),
            ),
            rejected=(),
            new_last_fact_id=3,
            new_last_turn_id=3,
        )
        # must not raise; the still-valid rewrite still applies
        report = await apply_consolidation(store, plan)
        assert junk_id not in report.retired_fact_ids
        assert any(fid == junk_id for _, fid, _ in report.skipped)
        assert len(report.rewritten) == 1
        # watermark still advanced despite the skip
        assert raw.get_sleep_watermark(familiar_id="fam") is not None

    @pytest.mark.asyncio
    async def test_advances_sleep_watermark(self) -> None:
        raw, junk_id, _, _ = _store_with_facts()
        store = AsyncHistoryStore(raw)
        plan = ConsolidationPlan(
            familiar_id="fam",
            retire=(RetireAction(fact_ids=(junk_id,), reason="x"),),
            rewrite=(),
            rejected=(),
            new_last_fact_id=3,
            new_last_turn_id=3,
        )
        await apply_consolidation(store, plan)
        wm = raw.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        # consolidation owns the fact axis only — turn axis (opinion's) untouched
        assert (wm.last_fact_id, wm.last_turn_id) == (3, 0)

    @pytest.mark.asyncio
    async def test_full_plan_apply_via_plan_consolidation(self) -> None:
        raw, junk_id, d1, d2 = _store_with_facts()
        store = AsyncHistoryStore(raw)
        reply = json.dumps({
            "retire": [{"fact_ids": [junk_id], "reason": "noise"}],
            "rewrite": [
                {
                    "old_fact_ids": [d1, d2],
                    "new_text": "Aria likes berries.",
                    "subject_keys": ["discord:A"],
                    "reason": "merge dups",
                }
            ],
        })
        plan = await plan_consolidation(
            store, FakeLLMClient(replies=[reply]), familiar_id="fam"
        )
        await apply_consolidation(store, plan)
        current = {f.text for f in raw.recent_facts(familiar_id="fam", limit=10)}
        assert current == {"Aria likes berries."}
