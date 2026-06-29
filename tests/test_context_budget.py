"""Tests for the prompt-assembly budget.

Covers token estimation, the independent per-section caps, and the
derived whole-prompt total. Per-tier defaults live in
``data/familiars/_default/character.toml``; :mod:`tests.test_config`
exercises that path.
"""

from __future__ import annotations

import time
from dataclasses import fields

import pytest

from familiar_connect.budget import (
    ModelBudgetCurve,
    TierBudget,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
)
from familiar_connect.llm import Message


class TestEstimateTokens:
    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_string_rounds_up(self) -> None:
        # ceil(len/4) — 5 chars → 2 tokens
        assert estimate_tokens("hello") == 2

    def test_overcount_safe(self) -> None:
        """Heuristic should over-, not under-count vs typical English."""
        text = "The quick brown fox jumps over the lazy dog."  # 44 chars
        # Real tokenization is ~10 tokens; our heuristic returns 11.
        assert estimate_tokens(text) >= 10

    def test_message_overhead_added(self) -> None:
        """Per-message framing pushes a one-char message above its content."""
        m = Message(role="user", content="x")
        # content = 1 token, +overhead of 4 = 5
        assert estimate_message_tokens(m) >= 5

    def test_message_with_name_costs_more(self) -> None:
        a = Message(role="user", content="x")
        b = Message(role="user", content="x", name="alice_42")
        assert estimate_message_tokens(b) > estimate_message_tokens(a)

    def test_messages_sum(self) -> None:
        msgs = [
            Message(role="user", content="abcd"),
            Message(role="assistant", content="efgh"),
        ]
        assert estimate_messages_tokens(msgs) == sum(
            estimate_message_tokens(m) for m in msgs
        )


class TestTierBudgetFields:
    def test_overriding_one_field_leaves_others_at_default(self) -> None:
        """Each cap is independent — no proportional auto-derivation."""
        a = TierBudget()
        b = TierBudget(rag_tokens=9999)
        assert b.rag_tokens == 9999
        # Other fields untouched.
        assert b.recent_history_tokens == a.recent_history_tokens
        assert b.dossier_tokens == a.dossier_tokens
        assert b.max_dossier_people == a.max_dossier_people

    def test_explicit_subcap_used_directly(self) -> None:
        b = TierBudget(recent_history_tokens=500)
        assert b.recent_history_tokens == 500


class TestTierBudgetDerivedTotal:
    """``total_tokens`` is a derived sum, not a configurable cap."""

    def test_total_is_sum_of_section_caps(self) -> None:
        b = TierBudget(
            recent_history_tokens=1000,
            rag_tokens=200,
            dossier_tokens=200,
            summary_tokens=100,
            reflection_tokens=100,
            lorebook_tokens=100,
        )
        assert b.total_tokens == 1000 + 200 + 200 + 100 + 100 + 100

    def test_total_excludes_count_caps(self) -> None:
        """Count caps (max_*) are not token figures; they don't feed total."""
        base = TierBudget()
        bumped = TierBudget(max_history_turns=base.max_history_turns + 50)
        assert bumped.total_tokens == base.total_tokens

    def test_total_tracks_a_section_cap_change(self) -> None:
        base = TierBudget()
        bumped = TierBudget(rag_tokens=base.rag_tokens + 500)
        assert bumped.total_tokens == base.total_tokens + 500

    def test_total_is_a_derived_property_not_a_field(self) -> None:
        """``total_tokens`` is computed, so it is not a dataclass field."""
        assert "total_tokens" not in {f.name for f in fields(TierBudget)}


class TestModelBudgetCurve:
    def test_defaults_are_all_one(self) -> None:
        c = ModelBudgetCurve()
        for field_val in vars(c).values():
            assert field_val == pytest.approx(1.0)

    def test_partial_override_leaves_others_at_one(self) -> None:
        c = ModelBudgetCurve(recent_history_tokens=2.0, rag_tokens=1.5)
        assert c.recent_history_tokens == pytest.approx(2.0)
        assert c.rag_tokens == pytest.approx(1.5)
        assert c.dossier_tokens == pytest.approx(1.0)
        assert c.summary_tokens == pytest.approx(1.0)

    def test_has_no_total_tokens_field(self) -> None:
        """Total is derived, so there is no multiplier for it."""
        assert "total_tokens" not in {f.name for f in fields(ModelBudgetCurve)}


class TestTierBudgetApplyCurve:
    def test_identity_curve_returns_equivalent_budget(self) -> None:
        b = TierBudget(recent_history_tokens=4000, rag_tokens=500)
        assert b.apply_curve(ModelBudgetCurve()) == b

    def test_scale_recent_history(self) -> None:
        b = TierBudget(recent_history_tokens=1000)
        scaled = b.apply_curve(ModelBudgetCurve(recent_history_tokens=2.0))
        assert scaled.recent_history_tokens == 2000
        assert scaled.rag_tokens == b.rag_tokens  # unchanged

    def test_scale_rounds_to_nearest_int(self) -> None:
        b = TierBudget(rag_tokens=1000)
        scaled = b.apply_curve(ModelBudgetCurve(rag_tokens=1.5))
        assert scaled.rag_tokens == 1500

    def test_scale_minimum_is_one(self) -> None:
        # Near-zero multiplier must not produce 0 or negative tokens.
        b = TierBudget(rag_tokens=1)
        scaled = b.apply_curve(ModelBudgetCurve(rag_tokens=0.001))
        assert scaled.rag_tokens >= 1

    def test_derived_total_follows_scaled_sections(self) -> None:
        b = TierBudget(
            recent_history_tokens=2000,
            rag_tokens=400,
            dossier_tokens=400,
            summary_tokens=200,
            reflection_tokens=200,
            lorebook_tokens=200,
        )
        c = ModelBudgetCurve(
            recent_history_tokens=2.0,
            rag_tokens=1.5,
            dossier_tokens=1.5,
            summary_tokens=1.5,
            reflection_tokens=1.5,
            lorebook_tokens=1.5,
        )
        scaled = b.apply_curve(c)
        assert scaled.recent_history_tokens == 4000
        assert scaled.rag_tokens == 600
        assert scaled.dossier_tokens == 600
        # Derived total reflects the scaled constituents.
        assert scaled.total_tokens == 4000 + 600 + 600 + 300 + 300 + 300

    def test_scale_count_fields(self) -> None:
        b = TierBudget(max_rag_turns=5, max_rag_facts=3, max_reflections=3)
        scaled = b.apply_curve(ModelBudgetCurve(max_rag_turns=2.0, max_rag_facts=2.0))
        assert scaled.max_rag_turns == 10
        assert scaled.max_rag_facts == 6
        assert scaled.max_reflections == b.max_reflections  # unchanged


class TestEstimatorPerf:
    """Hot-path sanity check — ensure the estimator stays microsecond-class."""

    def test_estimate_messages_under_one_ms(self) -> None:
        """30 turns of typical voice content shouldn't approach 1ms."""
        msgs = [
            Message(role="user", content="The quick brown fox jumped over " * 5)
            for _ in range(30)
        ]
        # Warm.
        estimate_messages_tokens(msgs)
        t0 = time.perf_counter()
        for _ in range(1000):
            estimate_messages_tokens(msgs)
        elapsed = time.perf_counter() - t0
        # 1000 iters of 30-msg estimate. 1ms per iter would be 1s total —
        # generous bound; in practice should be sub-millisecond per iter.
        assert elapsed < 1.0, f"estimator too slow: {elapsed:.3f}s for 1000 iters"
