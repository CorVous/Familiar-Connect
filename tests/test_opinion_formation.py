"""Opinion-formation pass — bucketing, stance moments, synthesis, apply."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from familiar_connect.config import load_character_config
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore, HistoryTurn
from familiar_connect.identity import Author, is_self_key
from familiar_connect.sleep.opinion_formation import (
    DayBatch,
    OpinionFact,
    OpinionPlan,
    OpinionWindow,
    StanceMoment,
    _build_stance_prompt,
    _build_synthesis_prompt,
    _render_turn,
    apply_opinions,
    bucket_by_day,
    extract_stance_moments,
    gather_days,
    plan_opinions,
    validate_opinions,
)
from tests.conftest import FakeLLMClient

# Real sleep-prompt prose = the merged ``_default`` config (single source
# of truth); no in-code copy. Mirrors production wiring.
_DEFAULT_PROFILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "familiars"
    / "_default"
    / "character.toml"
)


def _default_synthesis_system() -> str:
    cfg = load_character_config(_DEFAULT_PROFILE, defaults_path=_DEFAULT_PROFILE)
    return cfg.sleep_synthesis_system


def _user_turn(tid: int, display_name: str, content: str = "hi") -> HistoryTurn:
    return HistoryTurn(
        id=tid,
        timestamp=datetime(2026, 6, 12, tzinfo=UTC),
        role="user",
        author=Author(
            platform="discord",
            user_id=str(tid),
            username="u",
            display_name=display_name,
        ),
        content=content,
        channel_id=1,
    )


class TestRenderTurn:
    def test_self_turn_marked_you(self) -> None:
        r = _render_turn(_turn(1, datetime(2026, 6, 12, tzinfo=UTC)), "Sapphire")
        assert "(you)" in r
        assert "Sapphire" in r

    def test_same_named_user_disambiguated(self) -> None:
        # a user posting under HER name must not read as her own turn
        r = _render_turn(_user_turn(2, "Sapphire"), "Sapphire")
        assert "(you)" not in r
        assert "not you" in r.lower()

    def test_other_user_plain(self) -> None:
        r = _render_turn(_user_turn(3, "Aria"), "Sapphire")
        assert "Aria" in r
        assert "(you)" not in r
        assert "not you" not in r.lower()


def _turn(
    tid: int, when: datetime, role: str = "assistant", content: str = "x"
) -> HistoryTurn:
    return HistoryTurn(
        id=tid,
        timestamp=when,
        role=role,
        author=None,
        content=content,
        channel_id=1,
    )


class TestBucketByDay:
    def test_buckets_by_local_calendar_day(self) -> None:
        # 02:00Z = 2026-06-11 19:00 PT ; 10:00Z = 2026-06-12 03:00 PT
        turns = (
            _turn(1, datetime(2026, 6, 12, 2, 0, tzinfo=UTC)),
            _turn(2, datetime(2026, 6, 12, 10, 0, tzinfo=UTC)),
        )
        days = bucket_by_day(turns, "America/Los_Angeles")
        assert [d.date for d in days] == ["2026-06-11", "2026-06-12"]
        assert days[0].turn_ids == frozenset({1})
        assert days[1].turn_ids == frozenset({2})

    def test_days_ordered_oldest_first(self) -> None:
        turns = (
            _turn(2, datetime(2026, 6, 13, 12, 0, tzinfo=UTC)),
            _turn(1, datetime(2026, 6, 12, 12, 0, tzinfo=UTC)),
        )
        days = bucket_by_day(turns, "UTC")
        assert [d.date for d in days] == ["2026-06-12", "2026-06-13"]

    def test_self_turn_ids_are_assistant_role(self) -> None:
        turns = (
            _turn(1, datetime(2026, 6, 12, 12, 0, tzinfo=UTC), role="assistant"),
            _turn(2, datetime(2026, 6, 12, 12, 1, tzinfo=UTC), role="user"),
        )
        day = bucket_by_day(turns, "UTC")[0]
        assert day.self_turn_ids == frozenset({1})
        assert day.turn_ids == frozenset({1, 2})

    def test_empty(self) -> None:
        assert bucket_by_day((), "UTC") == []


class TestGatherDays:
    @pytest.mark.asyncio
    async def test_gathers_turns_since_watermark(self) -> None:
        raw = HistoryStore(":memory:")
        for _ in range(3):
            raw.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="assistant",
                content="hi",
                author=None,
            )
        store = AsyncHistoryStore(raw)
        win = await gather_days(store, familiar_id="fam", display_tz="UTC")
        # all today (now) -> one day bucket, 3 turns
        assert win.max_turn_id == 3
        total = sum(len(d.turns) for d in win.days)
        assert total == 3

    @pytest.mark.asyncio
    async def test_respects_prior_turn_watermark(self) -> None:
        raw = HistoryStore(":memory:")
        for _ in range(4):
            raw.append_turn(
                familiar_id="fam",
                channel_id=1,
                role="assistant",
                content="hi",
                author=None,
            )
        raw.advance_sleep_watermark(familiar_id="fam", last_turn_id=2)
        store = AsyncHistoryStore(raw)
        win = await gather_days(store, familiar_id="fam", display_tz="UTC")
        ids = {t.id for d in win.days for t in d.turns}
        assert ids == {3, 4}


def _day(date: str, turns: tuple[HistoryTurn, ...]) -> DayBatch:
    return DayBatch(date=date, turns=turns)


def _window(days: tuple[DayBatch, ...]) -> OpinionWindow:
    max_id = max((t.id for d in days for t in d.turns), default=0)
    return OpinionWindow(
        familiar_id="fam", days=days, prior_watermark=None, max_turn_id=max_id
    )


class TestExtractStanceMoments:
    @pytest.mark.asyncio
    async def test_keeps_only_in_day_turn_ids(self) -> None:
        day = _day(
            "2026-06-12",
            (
                _turn(1, datetime(2026, 6, 12, 12, 0, tzinfo=UTC), role="assistant"),
                _turn(2, datetime(2026, 6, 12, 12, 1, tzinfo=UTC), role="user"),
            ),
        )
        # LLM cites a real id (1) and a hallucinated out-of-day id (999)
        reply = json.dumps({
            "candidates": [{"text": "She liked the joke.", "turn_ids": [1, 999]}]
        })
        cands = await extract_stance_moments(
            FakeLLMClient(replies=[reply]), day, self_name="Sapphire", denylist=()
        )
        assert len(cands) == 1
        assert cands[0].turn_ids == (1,)  # 999 dropped

    @pytest.mark.asyncio
    async def test_drops_stance_moment_with_no_valid_ids(self) -> None:
        day = _day(
            "2026-06-12",
            (_turn(1, datetime(2026, 6, 12, 12, 0, tzinfo=UTC)),),
        )
        reply = json.dumps({"candidates": [{"text": "ungrounded", "turn_ids": [999]}]})
        cands = await extract_stance_moments(
            FakeLLMClient(replies=[reply]), day, self_name="Sapphire", denylist=()
        )
        assert cands == []


def _cand(text: str, date: str, turn_ids: tuple[int, ...]) -> StanceMoment:
    return StanceMoment(text=text, date=date, turn_ids=turn_ids)


class TestSynthesisPrompt:
    def test_prompt_instructs_importance_rating(self) -> None:
        # real prose sourced from the merged ``_default`` config (the
        # single source of truth) — production threads it the same way.
        msgs = _build_synthesis_prompt(
            [_cand("likes lo-fi", "2026-06-12", (1,))],
            self_name="Sapphire",
            prior_self_dossier=None,
            system=_default_synthesis_system(),
        )
        system = msgs[0].content
        assert isinstance(system, str)
        assert "importance" in system.lower()
        # rubric anchors present
        assert "1-10" in system

    def test_configured_system_formats_self_name(self) -> None:
        msgs = _build_synthesis_prompt(
            [_cand("likes lo-fi", "2026-06-12", (1,))],
            self_name="Sapphire",
            prior_self_dossier=None,
            system="settle {self_name}'s views",
        )
        # configured persona text appears (self_name interpolated); the
        # machine-parsed JSON reply shape is appended in code, not config.
        assert "settle Sapphire's views" in msgs[0].content
        assert "{self_name}" not in msgs[0].content

    def test_stray_brace_or_unknown_placeholder_does_not_crash(self) -> None:
        """A synthesis override with a stray brace / unknown token degrades.

        An override changes phrasing, never crashes the pass: a literal
        ``{`` and an unknown ``{money}`` pass through verbatim while a
        valid ``{self_name}`` still fills.
        """
        msgs = _build_synthesis_prompt(
            [_cand("likes lo-fi", "2026-06-12", (1,))],
            self_name="Sapphire",
            prior_self_dossier=None,
            system="settle {self_name}'s views {money} a { brace",
        )
        assert "settle Sapphire's views {money} a { brace" in msgs[0].content


class TestStancePrompt:
    def test_configured_system_formats_self_name(self) -> None:
        day = _day(
            "2026-06-12",
            (_turn(1, datetime(2026, 6, 12, 12, 0, tzinfo=UTC), role="assistant"),),
        )
        msgs = _build_stance_prompt(
            day, self_name="Sapphire", denylist=(), system="stances for {self_name}"
        )
        assert "stances for Sapphire" in msgs[0].content
        assert "{self_name}" not in msgs[0].content

    def test_stray_brace_or_unknown_placeholder_does_not_crash(self) -> None:
        """A stance override with a stray brace / unknown token degrades.

        An override changes phrasing, never crashes the pass.
        """
        day = _day(
            "2026-06-12",
            (_turn(1, datetime(2026, 6, 12, 12, 0, tzinfo=UTC), role="assistant"),),
        )
        msgs = _build_stance_prompt(
            day,
            self_name="Sapphire",
            denylist=(),
            system="stances for {self_name} {mood} a { brace",
        )
        assert "stances for Sapphire {mood} a { brace" in msgs[0].content


class TestValidateOpinions:
    def _win(self) -> OpinionWindow:
        return _window((
            _day(
                "2026-06-12",
                (
                    _turn(
                        1, datetime(2026, 6, 12, 12, 0, tzinfo=UTC), role="assistant"
                    ),
                    _turn(2, datetime(2026, 6, 12, 12, 1, tzinfo=UTC), role="user"),
                ),
            ),
            _day(
                "2026-06-20",
                (_turn(5, datetime(2026, 6, 20, 12, 0, tzinfo=UTC), role="assistant"),),
            ),
        ))

    def test_accepts_grounded_and_sets_valid_from_earliest(self) -> None:
        cands = [
            _cand("likes lo-fi", "2026-06-12", (1,)),
            _cand("again", "2026-06-20", (5,)),
        ]
        raw = [
            {"text": "Sapphire loves lo-fi.", "source_turn_ids": [1, 5], "reason": "x"}
        ]
        plan = validate_opinions(
            raw,
            stance_moments=cands,
            window=self._win(),
            cap=50,
        )
        assert len(plan.opinions) == 1
        op = plan.opinions[0]
        assert set(op.source_turn_ids) == {1, 5}
        assert op.valid_from_date == "2026-06-12"  # earliest

    def test_rejects_grounding_outside_candidate_union(self) -> None:
        cands = [_cand("likes lo-fi", "2026-06-12", (1,))]
        raw = [{"text": "x", "source_turn_ids": [1, 5], "reason": "x"}]
        plan = validate_opinions(
            raw,
            stance_moments=cands,
            window=self._win(),
            cap=50,
        )
        assert plan.opinions == ()
        assert plan.rejected[0].rail == "ungrounded"

    def test_rejects_zero_grounding(self) -> None:
        cands = [_cand("likes lo-fi", "2026-06-12", (1,))]
        raw = [{"text": "x", "source_turn_ids": [], "reason": "x"}]
        plan = validate_opinions(
            raw,
            stance_moments=cands,
            window=self._win(),
            cap=50,
        )
        assert plan.opinions == ()
        assert plan.rejected[0].rail == "ungrounded"

    def test_flags_no_self_authored_grounding(self) -> None:
        # turn 2 is a user turn; an opinion grounded only there isn't HER act
        cands = [_cand("room said", "2026-06-12", (2,))]
        raw = [
            {
                "text": "Sapphire thinks tea is nice.",
                "source_turn_ids": [2],
                "reason": "x",
            }
        ]
        plan = validate_opinions(
            raw,
            stance_moments=cands,
            window=self._win(),
            cap=50,
        )
        assert len(plan.opinions) == 1
        assert plan.opinions[0].self_grounded is False
        assert any("no_self_authored" in f for f in plan.flags)

    def test_dedups_restatements(self) -> None:
        cands = [_cand("a", "2026-06-12", (1,)), _cand("b", "2026-06-20", (5,))]
        raw = [
            {"text": "Sapphire loves lo-fi.", "source_turn_ids": [1], "reason": "x"},
            {"text": "Sapphire loves lo-fi!", "source_turn_ids": [5], "reason": "x"},
        ]
        plan = validate_opinions(
            raw,
            stance_moments=cands,
            window=self._win(),
            cap=50,
        )
        assert len(plan.opinions) == 1
        assert any(r.rail == "duplicate" for r in plan.rejected)

    def test_cap(self) -> None:
        cands = [_cand(f"c{i}", "2026-06-12", (1,)) for i in range(5)]
        raw = [
            {"text": f"op {i}", "source_turn_ids": [1], "reason": "x"} for i in range(5)
        ]
        plan = validate_opinions(
            raw,
            stance_moments=cands,
            window=self._win(),
            cap=3,
        )
        assert len(plan.opinions) == 3
        assert sum(1 for r in plan.rejected if r.rail == "cap") == 2

    def test_reads_importance_from_model(self) -> None:
        cands = [_cand("likes lo-fi", "2026-06-12", (1,))]
        raw = [
            {
                "text": "Sapphire loves lo-fi.",
                "source_turn_ids": [1],
                "reason": "x",
                "importance": 8,
            }
        ]
        plan = validate_opinions(raw, stance_moments=cands, window=self._win(), cap=50)
        assert plan.opinions[0].importance == 8

    def test_importance_clamped_high(self) -> None:
        cands = [_cand("likes lo-fi", "2026-06-12", (1,))]
        raw = [
            {
                "text": "Sapphire loves lo-fi.",
                "source_turn_ids": [1],
                "reason": "x",
                "importance": 99,
            }
        ]
        plan = validate_opinions(raw, stance_moments=cands, window=self._win(), cap=50)
        assert plan.opinions[0].importance == 10

    def test_importance_defaults_to_neutral_when_zero(self) -> None:
        # 0 is out-of-range low -> clamp; absent/non-int -> default 5
        cands = [_cand("likes lo-fi", "2026-06-12", (1,))]
        raw = [
            {
                "text": "Sapphire loves lo-fi.",
                "source_turn_ids": [1],
                "reason": "x",
                "importance": 0,
            }
        ]
        plan = validate_opinions(raw, stance_moments=cands, window=self._win(), cap=50)
        assert plan.opinions[0].importance == 1

    def test_importance_missing_defaults_to_five(self) -> None:
        cands = [_cand("likes lo-fi", "2026-06-12", (1,))]
        raw = [{"text": "Sapphire loves lo-fi.", "source_turn_ids": [1], "reason": "x"}]
        plan = validate_opinions(raw, stance_moments=cands, window=self._win(), cap=50)
        # absent importance -> neutral 5, opinion still accepted
        assert len(plan.opinions) == 1
        assert plan.opinions[0].importance == 5

    def test_importance_non_integer_defaults_to_five(self) -> None:
        cands = [_cand("likes lo-fi", "2026-06-12", (1,))]
        raw = [
            {
                "text": "Sapphire loves lo-fi.",
                "source_turn_ids": [1],
                "reason": "x",
                "importance": "very",
            }
        ]
        plan = validate_opinions(raw, stance_moments=cands, window=self._win(), cap=50)
        assert len(plan.opinions) == 1
        assert plan.opinions[0].importance == 5


class TestPlanOpinions:
    @pytest.mark.asyncio
    async def test_end_to_end(self) -> None:
        raw = HistoryStore(":memory:")
        raw.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="lo-fi is real music, fight me",
            author=None,
        )
        store = AsyncHistoryStore(raw)
        # day-1 candidate reply, then synthesis reply
        cand_reply = json.dumps({
            "candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]
        })
        synth_reply = json.dumps({
            "opinions": [
                {
                    "text": "Sapphire is fiercely protective of lo-fi as real music.",
                    "source_turn_ids": [1],
                    "reason": "defended it",
                }
            ]
        })
        llm = FakeLLMClient(replies=[cand_reply, synth_reply])
        plan = await plan_opinions(
            store, llm, familiar_id="fam", display_tz="UTC", self_name="Sapphire"
        )
        assert len(plan.opinions) == 1
        assert plan.opinions[0].source_turn_ids == (1,)
        assert plan.new_last_turn_id == 1

    @pytest.mark.asyncio
    async def test_configured_prompts_reach_llm(self) -> None:
        """Caller-supplied stance + synthesis text are the system messages sent."""
        raw = HistoryStore(":memory:")
        raw.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="lo-fi is real music",
            author=None,
        )
        store = AsyncHistoryStore(raw)
        cand_reply = json.dumps({
            "candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]
        })
        synth_reply = json.dumps({
            "opinions": [{"text": "Sapphire defends lo-fi.", "source_turn_ids": [1]}]
        })
        llm = FakeLLMClient(replies=[cand_reply, synth_reply])
        await plan_opinions(
            store,
            llm,
            familiar_id="fam",
            display_tz="UTC",
            self_name="Sapphire",
            stance_system="STANCE for {self_name}",
            synthesis_system="SYNTH for {self_name}",
        )
        # configured persona text reaches the LLM (self_name interpolated);
        # the JSON reply-shape contract is appended in code.
        stance_sent = llm.calls[0][0].content_str
        synth_sent = llm.calls[1][0].content_str
        assert "STANCE for Sapphire" in stance_sent
        assert "SYNTH for Sapphire" in synth_sent

    @pytest.mark.asyncio
    async def test_ungrounded_rail_fires_with_config_sourced_prompts(self) -> None:
        """Rails are code-enforced: a config prompt can't smuggle invented ids.

        Synthesis cites a turn id absent from any stance-moment — the
        ``ungrounded`` rail rejects it regardless of prompt phrasing.
        """
        raw = HistoryStore(":memory:")
        raw.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="lo-fi is real music",
            author=None,
        )
        store = AsyncHistoryStore(raw)
        cand_reply = json.dumps({
            "candidates": [{"text": "defends lo-fi", "turn_ids": [1]}]
        })
        # synthesis grounds an opinion in id 999 — never a stance-moment id
        synth_reply = json.dumps({
            "opinions": [{"text": "Sapphire loves jazz.", "source_turn_ids": [999]}]
        })
        llm = FakeLLMClient(replies=[cand_reply, synth_reply])
        plan = await plan_opinions(
            store,
            llm,
            familiar_id="fam",
            display_tz="UTC",
            self_name="Sapphire",
            stance_system="say whatever {self_name}",
            synthesis_system="invent ids freely for {self_name}",
        )
        assert plan.opinions == ()
        assert plan.rejected[0].rail == "ungrounded"


def _opinion(
    text: str,
    ids: tuple[int, ...],
    date: str,
    *,
    self_grounded: bool = True,
    importance: int = 5,
) -> OpinionFact:
    return OpinionFact(
        text=text,
        source_turn_ids=ids,
        valid_from_date=date,
        self_grounded=self_grounded,
        importance=importance,
    )


class TestApplyOpinions:
    @pytest.mark.asyncio
    async def test_mints_self_facts_with_provenance_and_valid_from(self) -> None:
        raw = HistoryStore(":memory:")
        raw.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="lo-fi is real music",
            author=None,
        )
        store = AsyncHistoryStore(raw)
        plan = OpinionPlan(
            familiar_id="fam",
            opinions=(
                _opinion("Sapphire defends lo-fi as real music.", (1,), "2026-06-12"),
            ),
            rejected=(),
            flags=(),
            new_last_turn_id=1,
        )
        report = await apply_opinions(store, plan, familiar_display_name="Sapphire")
        assert len(report.recorded) == 1
        fact = raw.recent_facts(familiar_id="fam", limit=10)[0]
        assert fact.source_turn_ids == (1,)
        assert is_self_key(fact.subjects[0].canonical_key)
        assert fact.subjects[0].display_at_write == "Sapphire"
        assert fact.valid_from is not None
        assert fact.valid_from.date().isoformat() == "2026-06-12"

    @pytest.mark.asyncio
    async def test_mints_with_importance(self) -> None:
        raw = HistoryStore(":memory:")
        raw.append_turn(
            familiar_id="fam",
            channel_id=1,
            role="assistant",
            content="lo-fi is real music",
            author=None,
        )
        store = AsyncHistoryStore(raw)
        plan = OpinionPlan(
            familiar_id="fam",
            opinions=(
                _opinion("Sapphire defends lo-fi.", (1,), "2026-06-12", importance=8),
            ),
            rejected=(),
            flags=(),
            new_last_turn_id=1,
        )
        await apply_opinions(store, plan, familiar_display_name="Sapphire")
        fact = raw.recent_facts(familiar_id="fam", limit=10)[0]
        assert fact.importance == 8

    @pytest.mark.asyncio
    async def test_advances_turn_axis_only(self) -> None:
        raw = HistoryStore(":memory:")
        raw.append_turn(
            familiar_id="fam", channel_id=1, role="assistant", content="x", author=None
        )
        # hygiene previously set the fact axis; dream must not stomp it
        raw.advance_sleep_watermark(familiar_id="fam", last_fact_id=77)
        store = AsyncHistoryStore(raw)
        plan = OpinionPlan(
            familiar_id="fam",
            opinions=(_opinion("Sapphire likes tea.", (1,), "2026-06-12"),),
            rejected=(),
            flags=(),
            new_last_turn_id=1,
        )
        await apply_opinions(store, plan, familiar_display_name="Sapphire")
        wm = raw.get_sleep_watermark(familiar_id="fam")
        assert wm is not None
        assert (wm.last_fact_id, wm.last_turn_id) == (77, 1)
