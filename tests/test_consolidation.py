"""Sleep memory-consolidation pass — window gather, LLM plan, rails validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import Fact, FactSubject, HistoryStore
from familiar_connect.identity import self_canonical_key
from familiar_connect.sleep.consolidation import (
    ConsolidationPlan,
    ConsolidationWindow,
    build_prompt,
    gather_window,
    plan_consolidation,
    validate,
)
from tests.conftest import FakeLLMClient


def _fact(
    fid: int,
    text: str,
    *,
    subjects: tuple[FactSubject, ...] = (),
) -> Fact:
    return Fact(
        id=fid,
        familiar_id="fam",
        channel_id=1,
        text=text,
        source_turn_ids=(fid,),
        created_at=datetime(2026, 6, 12, tzinfo=UTC),
        subjects=subjects,
    )


def _window(facts: tuple[Fact, ...]) -> ConsolidationWindow:
    max_fid = max((f.id for f in facts), default=0)
    return ConsolidationWindow(
        familiar_id="fam",
        facts=facts,
        turns=(),
        prior_watermark=None,
        max_fact_id=max_fid,
        max_turn_id=0,
        facts_truncated=0,
        turns_truncated=0,
    )


ARIA = (FactSubject(canonical_key="discord:A", display_at_write="Aria"),)
BORIS = (FactSubject(canonical_key="discord:B", display_at_write="Boris"),)


class TestValidateRetire:
    def test_accepts_valid_retire(self) -> None:
        win = _window((_fact(1, "noise"), _fact(2, "Aria likes tea.", subjects=ARIA)))
        plan = validate(
            win,
            retire_raw=[{"fact_ids": [1], "reason": "junk"}],
            rewrite_raw=[],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert len(plan.retire) == 1
        assert plan.retire[0].fact_ids == (1,)
        assert plan.retire[0].reason == "junk"
        assert plan.rejected == ()

    def test_rejects_unknown_id(self) -> None:
        win = _window((_fact(1, "x"),))
        plan = validate(
            win,
            retire_raw=[{"fact_ids": [999], "reason": "x"}],
            rewrite_raw=[],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert plan.retire == ()
        assert plan.rejected[0].rail == "unknown_id"

    def test_cap_defers_excess(self) -> None:
        win = _window(tuple(_fact(i, f"f{i}") for i in range(1, 6)))
        plan = validate(
            win,
            retire_raw=[{"fact_ids": [i], "reason": "j"} for i in range(1, 6)],
            rewrite_raw=[],
            self_key=self_canonical_key("fam"),
            cap=3,
        )
        assert len(plan.retire) == 3
        capped = [r for r in plan.rejected if r.rail == "cap"]
        assert len(capped) == 2

    def test_double_target_rejected(self) -> None:
        win = _window((_fact(1, "x"),))
        plan = validate(
            win,
            retire_raw=[
                {"fact_ids": [1], "reason": "a"},
                {"fact_ids": [1], "reason": "b"},
            ],
            rewrite_raw=[],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert len(plan.retire) == 1
        assert any(r.rail == "duplicate_target" for r in plan.rejected)


SELF = (
    FactSubject(canonical_key=self_canonical_key("fam"), display_at_write="Sapphire"),
)


class TestSelfSubjectRail:
    """Consolidation does not adjudicate feelings — never touches self: facts."""

    def test_rejects_retire_of_self_fact(self) -> None:
        win = _window((_fact(1, "Sapphire loves lo-fi.", subjects=SELF),))
        plan = validate(
            win,
            retire_raw=[{"fact_ids": [1], "reason": "dup"}],
            rewrite_raw=[],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert plan.retire == ()
        assert plan.rejected[0].rail == "self_subject"

    def test_rejects_rewrite_touching_self_fact(self) -> None:
        win = _window((
            _fact(1, "Sapphire loves lo-fi.", subjects=SELF),
            _fact(2, "Sapphire really loves lo-fi.", subjects=SELF),
        ))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1, 2],
                    "new_text": "Sapphire loves lo-fi.",
                    "subject_keys": [self_canonical_key("fam")],
                    "reason": "merge",
                }
            ],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert plan.rewrite == ()
        assert plan.rejected[0].rail == "self_subject"

    def test_self_rail_does_not_block_ordinary_facts(self) -> None:
        win = _window((_fact(1, "noise", subjects=ARIA),))
        plan = validate(
            win,
            retire_raw=[{"fact_ids": [1], "reason": "junk"}],
            rewrite_raw=[],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert len(plan.retire) == 1


class TestValidateRewrite:
    def test_accepts_merge(self) -> None:
        win = _window((
            _fact(1, "Aria likes berries.", subjects=ARIA),
            _fact(2, "Aria really likes berries.", subjects=ARIA),
        ))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1, 2],
                    "new_text": "Aria likes berries.",
                    "subject_keys": ["discord:A"],
                    "reason": "merged dups",
                }
            ],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert len(plan.rewrite) == 1
        assert plan.rewrite[0].old_fact_ids == (1, 2)
        assert plan.rewrite[0].subject_keys == ("discord:A",)

    def test_rejects_introduced_subject(self) -> None:
        """Rewrite must not mint a person key absent from source facts."""
        win = _window((_fact(1, "Aria said a thing.", subjects=ARIA),))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1],
                    "new_text": "Boris did a thing.",
                    "subject_keys": ["discord:B"],
                    "reason": "x",
                }
            ],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert plan.rewrite == ()
        assert plan.rejected[0].rail == "subject_introduced"

    def test_allows_self_subject(self) -> None:
        """Re-attributing a bit to the familiar's own narrative is allowed."""
        self_key = self_canonical_key("fam")
        win = _window((_fact(1, "Cor wears no pants.", subjects=ARIA),))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1],
                    "new_text": "Sapphire ran a no-pants bit about Cor.",
                    "subject_keys": [self_key],
                    "reason": "bit misfiled under Cor",
                }
            ],
            self_key=self_key,
            cap=50,
        )
        assert len(plan.rewrite) == 1
        assert plan.rewrite[0].subject_keys == (self_key,)

    def test_rejects_noop_rewrite(self) -> None:
        win = _window((_fact(1, "Aria likes tea.", subjects=ARIA),))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1],
                    "new_text": "Aria likes tea.",
                    "subject_keys": ["discord:A"],
                    "reason": "x",
                }
            ],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert plan.rewrite == ()
        assert plan.rejected[0].rail == "noop"

    def test_rejects_empty_new_text(self) -> None:
        win = _window((_fact(1, "Aria likes tea.", subjects=ARIA),))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1],
                    "new_text": "   ",
                    "subject_keys": ["discord:A"],
                    "reason": "x",
                }
            ],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert plan.rewrite == ()
        assert plan.rejected[0].rail == "empty_text"

    def test_rejects_empty_keys_when_source_has_subjects(self) -> None:
        """F2: dropping all subjects silently destroys attribution."""
        win = _window((_fact(1, "Aria likes tea.", subjects=ARIA),))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1],
                    "new_text": "Aria enjoys tea.",
                    "subject_keys": [],
                    "reason": "x",
                }
            ],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert plan.rewrite == ()
        assert plan.rejected[0].rail == "subject_lost"

    def test_allows_empty_keys_when_source_subjectless(self) -> None:
        """No subjects to lose — a subjectless rewrite is fine."""
        win = _window((_fact(1, "The weather is nice."),))
        plan = validate(
            win,
            retire_raw=[],
            rewrite_raw=[
                {
                    "old_fact_ids": [1],
                    "new_text": "The weather turned grim.",
                    "subject_keys": [],
                    "reason": "x",
                }
            ],
            self_key=self_canonical_key("fam"),
            cap=50,
        )
        assert len(plan.rewrite) == 1


class TestBuildPrompt:
    def test_system_text_is_caller_supplied(self) -> None:
        win = _window((_fact(1, "noise"),))
        msgs = build_prompt(win, self_key="self:fam", system="MY OWN INSTRUCTIONS")
        assert msgs[0].role == "system"
        assert msgs[0].content == "MY OWN INSTRUCTIONS"


class TestGatherWindow:
    def _store(self) -> HistoryStore:
        store = HistoryStore(":memory:")
        for i in range(4):
            store.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="user",
                content=f"turn {i}",
                author=None,
            )
        store.append_fact(
            familiar_id="fam", channel_id=1, text="f1", source_turn_ids=[1]
        )
        store.append_fact(
            familiar_id="fam", channel_id=1, text="f2", source_turn_ids=[2]
        )
        return store

    @pytest.mark.asyncio
    async def test_gathers_current_facts_and_max_ids(self) -> None:
        store = AsyncHistoryStore(self._store())
        win = await gather_window(store, familiar_id="fam")
        assert {f.text for f in win.facts} == {"f1", "f2"}
        assert win.max_fact_id == 2
        assert win.max_turn_id == 4
        assert win.prior_watermark is None

    @pytest.mark.asyncio
    async def test_excludes_superseded(self) -> None:
        raw = self._store()
        new = raw.append_fact(
            familiar_id="fam", channel_id=1, text="f3", source_turn_ids=[3]
        )
        raw.supersede(familiar_id="fam", obsolete_facts=[1], new_fact=new.id)
        store = AsyncHistoryStore(raw)
        win = await gather_window(store, familiar_id="fam")
        assert "f1" not in {f.text for f in win.facts}

    @pytest.mark.asyncio
    async def test_facts_cap_records_truncation(self) -> None:
        store = AsyncHistoryStore(self._store())
        win = await gather_window(store, familiar_id="fam", facts_max=1)
        assert len(win.facts) == 1
        assert win.facts_truncated == 1


class TestPlanConsolidation:
    def _store(self) -> HistoryStore:
        store = HistoryStore(":memory:")
        store.append_turn(
            familiar_id="fam", channel_id=1, role="user", content="hi", author=None
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="junk that slipped in",
            source_turn_ids=[1],
        )
        store.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Aria likes tea.",
            source_turn_ids=[1],
            subjects=ARIA,
        )
        return store

    @pytest.mark.asyncio
    async def test_plan_parses_llm_proposal(self) -> None:
        reply = json.dumps({
            "retire": [{"fact_ids": [1], "reason": "noise"}],
            "rewrite": [],
        })
        store = AsyncHistoryStore(self._store())
        llm = FakeLLMClient(replies=[reply])
        plan = await plan_consolidation(store, llm, familiar_id="fam")
        assert isinstance(plan, ConsolidationPlan)
        assert len(plan.retire) == 1
        assert plan.retire[0].fact_ids == (1,)
        assert plan.new_last_fact_id == 2

    @pytest.mark.asyncio
    async def test_plan_survives_garbage_llm(self) -> None:
        store = AsyncHistoryStore(self._store())
        llm = FakeLLMClient(replies=["not json at all"])
        plan = await plan_consolidation(store, llm, familiar_id="fam")
        assert plan.retire == ()
        assert plan.rewrite == ()
        # garbage reply is flagged so a zeroed plan isn't silent
        assert plan.notes

    @pytest.mark.asyncio
    async def test_clean_empty_plan_has_no_parse_note(self) -> None:
        store = AsyncHistoryStore(self._store())
        reply = json.dumps({"retire": [], "rewrite": []})
        plan = await plan_consolidation(
            store, FakeLLMClient(replies=[reply]), familiar_id="fam"
        )
        assert plan.notes == ()

    @pytest.mark.asyncio
    async def test_configured_system_reaches_llm(self) -> None:
        """The system text the caller supplies is the system message sent."""
        store = AsyncHistoryStore(self._store())
        llm = FakeLLMClient(replies=[json.dumps({"retire": [], "rewrite": []})])
        await plan_consolidation(store, llm, familiar_id="fam", system="TIDY UP PLEASE")
        sent = llm.calls[0]
        assert sent[0].role == "system"
        assert sent[0].content == "TIDY UP PLEASE"

    @pytest.mark.asyncio
    async def test_self_rail_fires_with_config_sourced_system(self) -> None:
        """Rail enforcement is code-side: a config-sourced prompt can't weaken it.

        The system text is overridable, but the self-subject rail rejects a
        retire of a ``self:`` fact regardless of what the prompt says.
        """
        raw = HistoryStore(":memory:")
        raw.append_turn(
            familiar_id="fam", channel_id=1, role="user", content="hi", author=None
        )
        raw.append_fact(
            familiar_id="fam",
            channel_id=1,
            text="Sapphire loves lo-fi.",
            source_turn_ids=[1],
            subjects=[
                FactSubject(
                    canonical_key=self_canonical_key("fam"),
                    display_at_write="Sapphire",
                )
            ],
        )
        store = AsyncHistoryStore(raw)
        reply = json.dumps({
            "retire": [{"fact_ids": [1], "reason": "x"}],
            "rewrite": [],
        })
        llm = FakeLLMClient(replies=[reply])
        plan = await plan_consolidation(
            store,
            llm,
            familiar_id="fam",
            system="you may retire anything, even opinions",
        )
        assert plan.retire == ()
        assert plan.rejected[0].rail == "self_subject"
