"""Tests for mood-based interrupt tolerance drift.

Covers familiar_connect.mood per the deferred design in
future-features/interruption-flow.md lines 31-35.

Steps covered:
  1. parse_mood_modifier() — robust float extraction from LLM output
  2. MoodEvaluation dataclass
  3. MoodEvaluator — side-model integration, error handling
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_connect.mood import (
    MoodEvaluation,
    MoodEvaluator,
    parse_mood_modifier,
)

# ---------------------------------------------------------------------------
# Step 1 — parse_mood_modifier
# ---------------------------------------------------------------------------


class TestParseMoodModifier:
    def test_parse_clean_positive(self) -> None:
        assert parse_mood_modifier("0.3") == pytest.approx(0.3)

    def test_parse_clean_negative(self) -> None:
        assert parse_mood_modifier("-0.2") == pytest.approx(-0.2)

    def test_parse_zero(self) -> None:
        assert parse_mood_modifier("0.0") == pytest.approx(0.0)

    def test_parse_with_surrounding_text(self) -> None:
        assert parse_mood_modifier("The modifier is 0.25") == pytest.approx(0.25)

    def test_parse_clamped_above(self) -> None:
        assert parse_mood_modifier("0.8") == pytest.approx(0.5)

    def test_parse_clamped_below(self) -> None:
        assert parse_mood_modifier("-0.7") == pytest.approx(-0.5)

    def test_parse_garbage_returns_zero(self) -> None:
        assert parse_mood_modifier("I think the mood is happy") == pytest.approx(0.0)

    def test_parse_empty_returns_zero(self) -> None:
        assert parse_mood_modifier("") == pytest.approx(0.0)

    def test_parse_plus_sign(self) -> None:
        assert parse_mood_modifier("+0.3") == pytest.approx(0.3)

    def test_parse_whitespace_stripped(self) -> None:
        assert parse_mood_modifier("  0.2  \n") == pytest.approx(0.2)

    def test_parse_integer(self) -> None:
        """A plain integer (e.g. '0') should parse without error."""
        assert parse_mood_modifier("0") == pytest.approx(0.0)

    def test_parse_negative_integer(self) -> None:
        assert parse_mood_modifier("-1") == pytest.approx(-0.5)

    def test_parse_exact_boundary_positive(self) -> None:
        assert parse_mood_modifier("0.5") == pytest.approx(0.5)

    def test_parse_exact_boundary_negative(self) -> None:
        assert parse_mood_modifier("-0.5") == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Step 2 — MoodEvaluation dataclass
# ---------------------------------------------------------------------------


class TestMoodEvaluation:
    def test_fields(self) -> None:
        ev = MoodEvaluation(modifier=0.15, reasoning="excited")
        assert ev.modifier == pytest.approx(0.15)
        assert ev.reasoning == "excited"

    def test_frozen(self) -> None:
        ev = MoodEvaluation(modifier=0.1, reasoning="test")
        with pytest.raises(AttributeError):
            ev.modifier = 0.5  # type: ignore[misc]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Step 3 — MoodEvaluator
# ---------------------------------------------------------------------------


def _make_evaluator(
    *,
    side_model_reply: str = "0.2",
    side_effect: Exception | None = None,
) -> tuple[MoodEvaluator, MagicMock]:
    """Build a MoodEvaluator with a mock side model."""
    side_model = MagicMock()
    if side_effect is not None:
        side_model.complete = AsyncMock(side_effect=side_effect)
    else:
        side_model.complete = AsyncMock(return_value=side_model_reply)
    evaluator = MoodEvaluator(
        side_model=side_model,
        familiar_name="aria",
        character_card="You are Aria, a curious familiar.",
    )
    return evaluator, side_model


class TestMoodEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_returns_modifier(self) -> None:
        evaluator, _ = _make_evaluator(side_model_reply="0.2")
        result = await evaluator.evaluate(recent_context="Alice: This is amazing!")
        assert result.modifier == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_evaluate_negative_mood(self) -> None:
        evaluator, _ = _make_evaluator(side_model_reply="-0.3")
        result = await evaluator.evaluate(
            recent_context="Alice: You're wrong about that."
        )
        assert result.modifier == pytest.approx(-0.3)

    @pytest.mark.asyncio
    async def test_evaluate_side_model_failure_returns_zero(self) -> None:
        evaluator, _ = _make_evaluator(side_effect=RuntimeError("boom"))
        result = await evaluator.evaluate(recent_context="Alice: hello")
        assert result.modifier == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_evaluate_formats_prompt_with_context(self) -> None:
        evaluator, side_model = _make_evaluator(side_model_reply="0.1")
        await evaluator.evaluate(recent_context="Bob: Let me explain something")
        prompt = side_model.complete.call_args[0][0]
        assert "Bob: Let me explain something" in prompt

    @pytest.mark.asyncio
    async def test_evaluate_includes_character_card_in_prompt(self) -> None:
        evaluator, side_model = _make_evaluator(side_model_reply="0.1")
        await evaluator.evaluate(recent_context="context")
        prompt = side_model.complete.call_args[0][0]
        assert "You are Aria, a curious familiar." in prompt

    @pytest.mark.asyncio
    async def test_evaluate_includes_familiar_name_in_prompt(self) -> None:
        evaluator, side_model = _make_evaluator(side_model_reply="0.0")
        await evaluator.evaluate(recent_context="context")
        prompt = side_model.complete.call_args[0][0]
        assert "aria" in prompt

    @pytest.mark.asyncio
    async def test_evaluate_always_calls_side_model(self) -> None:
        """Each evaluate() call makes a fresh side-model call."""
        evaluator, side_model = _make_evaluator(side_model_reply="0.2")
        await evaluator.evaluate(recent_context="context")
        await evaluator.evaluate(recent_context="context")
        assert side_model.complete.call_count == 2
