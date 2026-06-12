"""Sleep hygiene — apply path + audit artifact."""

from __future__ import annotations

import json

import pytest

from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import FactSubject, HistoryStore
from familiar_connect.identity import self_canonical_key
from familiar_connect.sleep.apply import apply_hygiene, hygiene_audit, write_audit
from familiar_connect.sleep.hygiene import (
    HygienePlan,
    RejectedAction,
    RetireAction,
    RewriteAction,
    plan_hygiene,
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


class TestApplyHygiene:
    @pytest.mark.asyncio
    async def test_retire_marks_superseded(self) -> None:
        raw, junk_id, _, _ = _store_with_facts()
        store = AsyncHistoryStore(raw)
        plan = HygienePlan(
            familiar_id="fam",
            retire=(RetireAction(fact_ids=(junk_id,), reason="noise"),),
            rewrite=(),
            rejected=(),
            new_last_fact_id=3,
            new_last_turn_id=3,
        )
        report = await apply_hygiene(store, plan)
        assert report.retired_fact_ids == (junk_id,)
        current = {f.text for f in raw.recent_facts(familiar_id="fam", limit=10)}
        assert "noise" not in current

    @pytest.mark.asyncio
    async def test_rewrite_merges_with_union_provenance(self) -> None:
        raw, _, d1, d2 = _store_with_facts()
        store = AsyncHistoryStore(raw)
        plan = HygienePlan(
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
        report = await apply_hygiene(store, plan)
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
        plan = HygienePlan(
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
        report = await apply_hygiene(store, plan, familiar_display_name="Sapphire")
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
        raw.retire_fact(familiar_id="fam", fact_id=junk_id)
        store = AsyncHistoryStore(raw)
        plan = HygienePlan(
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
        report = await apply_hygiene(store, plan)
        assert junk_id not in report.retired_fact_ids
        assert any(fid == junk_id for _, fid, _ in report.skipped)
        assert len(report.rewritten) == 1
        # watermark still advanced despite the skip
        assert raw.get_sleep_watermark(familiar_id="fam") is not None

    @pytest.mark.asyncio
    async def test_advances_sleep_watermark(self) -> None:
        raw, junk_id, _, _ = _store_with_facts()
        store = AsyncHistoryStore(raw)
        plan = HygienePlan(
            familiar_id="fam",
            retire=(RetireAction(fact_ids=(junk_id,), reason="x"),),
            rewrite=(),
            rejected=(),
            new_last_fact_id=3,
            new_last_turn_id=3,
        )
        await apply_hygiene(store, plan)
        wm = raw.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        # hygiene owns the fact axis only — turn axis (dream's) untouched
        assert (wm.last_fact_id, wm.last_turn_id) == (3, 0)

    @pytest.mark.asyncio
    async def test_full_plan_apply_via_plan_hygiene(self) -> None:
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
        plan = await plan_hygiene(
            store, FakeLLMClient(replies=[reply]), familiar_id="fam"
        )
        await apply_hygiene(store, plan)
        current = {f.text for f in raw.recent_facts(familiar_id="fam", limit=10)}
        assert current == {"Aria likes berries."}


class TestAudit:
    def _plan(self) -> HygienePlan:
        return HygienePlan(
            familiar_id="fam",
            retire=(RetireAction(fact_ids=(1,), reason="noise"),),
            rewrite=(
                RewriteAction(
                    old_fact_ids=(2, 3),
                    new_text="merged",
                    subject_keys=("discord:A",),
                    reason="dups",
                ),
            ),
            rejected=(
                RejectedAction("retire", {"fact_ids": [9]}, "self_subject", "(9,)"),
            ),
            new_last_fact_id=3,
            new_last_turn_id=3,
            facts_considered=5,
            facts_truncated=2,
        )

    def test_audit_is_json_serializable_and_complete(self) -> None:
        audit = hygiene_audit(self._plan(), applied=False)
        # round-trips through json
        blob = json.dumps(audit)
        back = json.loads(blob)
        assert back["familiar_id"] == "fam"
        assert back["applied"] is False
        assert back["mutated_count"] == 3
        assert len(back["retire"]) == 1
        assert len(back["rewrite"]) == 1
        assert back["rejected"][0]["rail"] == "self_subject"
        assert back["window"]["facts_truncated"] == 2

    def test_audit_records_applied_flag(self) -> None:
        audit = hygiene_audit(self._plan(), applied=True)
        assert audit["applied"] is True

    def test_write_audit_creates_file(self, tmp_path) -> None:  # noqa: ANN001
        audit = hygiene_audit(self._plan(), applied=False)
        path = write_audit(audit, audit_dir=tmp_path, familiar_id="fam")
        assert path.exists()
        assert path.parent == tmp_path
        loaded = json.loads(path.read_text())
        assert loaded["familiar_id"] == "fam"
