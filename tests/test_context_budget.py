"""Red-first tests for the context-pipeline Budgeter.

Covers familiar_connect.context.budget.Budgeter, which does not exist
yet. The Budgeter walks a flat list of Contributions, groups them by
their declared Layer, sorts each layer's contributions by priority
(higher first), and joins them up to a per-layer token budget. Content
that exceeds the budget is truncated at sentence / word boundaries
where possible and recorded in a BudgetResult.dropped list so the
pipeline can log it.

Token counting for the first pass is a simple character-count
heuristic (~4 chars per token). tiktoken can replace it later without
changing the Budgeter's contract.
"""

from __future__ import annotations

import pytest

from familiar_connect.context.budget import (
    Budgeter,
    BudgetResult,
    DroppedNote,
    estimate_tokens,
)
from familiar_connect.context.types import Contribution, Layer


class TestEstimateTokens:
    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_string_rounds_up(self) -> None:
        """A 5-character string is ~2 tokens under the 4-chars-per-token heuristic."""
        # We only require the estimator is monotonic and nonzero for nonempty;
        # the exact rounding rule is an implementation detail but we check
        # that it returns at least 1 for any nonempty input.
        assert estimate_tokens("hello") >= 1

    def test_longer_string_has_more_tokens(self) -> None:
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        assert long > short


class TestBudgeterEmpty:
    def test_no_contributions_yields_empty_layers(self) -> None:
        b = Budgeter()
        result = b.fill(
            contributions=[],
            budget_by_layer={
                Layer.character: 100,
                Layer.content: 100,
            },
        )
        assert isinstance(result, BudgetResult)
        assert not result.by_layer.get(Layer.character, "")
        assert not result.by_layer.get(Layer.content, "")
        assert result.dropped == []


class TestBudgeterSingleContribution:
    def test_under_budget_passes_through_verbatim(self) -> None:
        b = Budgeter()
        c = Contribution(
            layer=Layer.character,
            priority=10,
            text="A short description.",
            estimated_tokens=10,
            source="char:desc",
        )
        result = b.fill([c], {Layer.character: 1000})
        assert result.by_layer[Layer.character] == "A short description."
        assert result.dropped == []

    def test_over_budget_truncates(self) -> None:
        """A contribution larger than its layer budget is truncated to fit."""
        b = Budgeter()
        long_text = "word " * 1000  # ~5000 chars, ~1250 tokens
        c = Contribution(
            layer=Layer.character,
            priority=10,
            text=long_text,
            estimated_tokens=1250,
            source="char:bloat",
        )
        result = b.fill([c], {Layer.character: 50})  # 50-token budget

        # The resulting text fits within the budget.
        kept_tokens = estimate_tokens(result.by_layer[Layer.character])
        assert kept_tokens <= 50

        # And we report what we dropped.
        assert len(result.dropped) == 1
        note = result.dropped[0]
        assert isinstance(note, DroppedNote)
        assert note.layer is Layer.character
        assert note.source == "char:bloat"
        assert note.reason == "truncated"
        assert note.tokens_dropped > 0


class TestBudgeterMultipleContributionsInOneLayer:
    def test_joined_in_priority_order(self) -> None:
        b = Budgeter()
        high = Contribution(
            layer=Layer.content,
            priority=100,
            text="high-priority snippet",
            estimated_tokens=6,
            source="rag:1",
        )
        low = Contribution(
            layer=Layer.content,
            priority=1,
            text="low-priority snippet",
            estimated_tokens=6,
            source="rag:2",
        )
        # Insert in reverse order to prove the Budgeter sorts rather than
        # preserving insertion order.
        result = b.fill([low, high], {Layer.content: 1000})

        text = result.by_layer[Layer.content]
        assert text.index("high-priority snippet") < text.index("low-priority snippet")
        assert result.dropped == []

    def test_over_budget_drops_lowest_priority_first(self) -> None:
        b = Budgeter()
        high = Contribution(
            layer=Layer.content,
            priority=100,
            text="A" * 40,  # ~10 tokens
            estimated_tokens=10,
            source="rag:high",
        )
        low = Contribution(
            layer=Layer.content,
            priority=1,
            text="B" * 40,  # ~10 tokens
            estimated_tokens=10,
            source="rag:low",
        )
        result = b.fill([high, low], {Layer.content: 10})  # only the high fits

        text = result.by_layer[Layer.content]
        assert "A" * 40 in text
        assert "B" * 40 not in text

        # The low-priority one was fully dropped.
        assert len(result.dropped) == 1
        assert result.dropped[0].source == "rag:low"
        assert result.dropped[0].reason == "dropped"
        assert result.dropped[0].tokens_dropped >= 10

    def test_over_budget_truncates_last_kept_when_partially_over(self) -> None:
        """Partial-fit contributions are truncated, not dropped entirely.

        When the highest-priority contribution fits but the next one
        only partially fits, the next one is truncated rather than
        dropped entirely.
        """
        b = Budgeter()
        high = Contribution(
            layer=Layer.content,
            priority=100,
            text="KEEP. " * 5,  # ~7 tokens
            estimated_tokens=7,
            source="rag:high",
        )
        medium = Contribution(
            layer=Layer.content,
            priority=50,
            text="also keep some of this but not all of it. " * 10,
            estimated_tokens=100,
            source="rag:medium",
        )
        result = b.fill([high, medium], {Layer.content: 30})

        text = result.by_layer[Layer.content]
        assert "KEEP. " in text
        # Some of the medium snippet should remain, but not all of it.
        assert "also keep some of this" in text
        # Total fits in the budget.
        assert estimate_tokens(text) <= 30

        # The medium contribution was truncated, not dropped.
        assert any(
            note.source == "rag:medium" and note.reason == "truncated"
            for note in result.dropped
        )


class TestBudgeterMultipleLayers:
    def test_layers_budgeted_independently(self) -> None:
        b = Budgeter()
        char = Contribution(
            layer=Layer.character,
            priority=10,
            text="Character description.",
            estimated_tokens=5,
            source="char",
        )
        content = Contribution(
            layer=Layer.content,
            priority=10,
            text="Retrieved snippet.",
            estimated_tokens=5,
            source="content",
        )
        result = b.fill(
            [char, content],
            {Layer.character: 1000, Layer.content: 1000},
        )
        assert result.by_layer[Layer.character] == "Character description."
        assert result.by_layer[Layer.content] == "Retrieved snippet."
        assert result.dropped == []

    def test_missing_budget_layer_drops_contribution(self) -> None:
        """A contribution for a layer with no declared budget is dropped and logged."""
        b = Budgeter()
        c = Contribution(
            layer=Layer.content,
            priority=10,
            text="orphan snippet",
            estimated_tokens=3,
            source="rag:orphan",
        )
        result = b.fill([c], {Layer.character: 1000})

        assert not result.by_layer.get(Layer.content, "")
        assert len(result.dropped) == 1
        assert result.dropped[0].source == "rag:orphan"
        assert result.dropped[0].reason == "dropped"


class TestBudgeterZeroBudget:
    def test_zero_budget_drops_all(self) -> None:
        b = Budgeter()
        c = Contribution(
            layer=Layer.content,
            priority=10,
            text="anything",
            estimated_tokens=2,
            source="rag:any",
        )
        result = b.fill([c], {Layer.content: 0})
        assert not result.by_layer.get(Layer.content, "")
        assert len(result.dropped) == 1


class TestBudgeterAcceptsFrozenInput:
    def test_contribution_list_not_mutated(self) -> None:
        b = Budgeter()
        original = [
            Contribution(
                layer=Layer.content,
                priority=p,
                text=f"text{p}",
                estimated_tokens=2,
                source=f"src{p}",
            )
            for p in (1, 10, 5)
        ]
        snapshot = list(original)
        b.fill(original, {Layer.content: 1000})
        assert original == snapshot, "Budgeter must not mutate its input list"


class TestDroppedNote:
    def test_fields(self) -> None:
        note = DroppedNote(
            layer=Layer.content,
            source="rag:x",
            reason="dropped",
            tokens_dropped=5,
        )
        assert note.layer is Layer.content
        assert note.source == "rag:x"
        assert note.reason == "dropped"
        assert note.tokens_dropped == 5

    def test_invalid_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            DroppedNote(
                layer=Layer.content,
                source="x",
                reason="explosion",  # not "dropped" or "truncated"
                tokens_dropped=1,
            )
