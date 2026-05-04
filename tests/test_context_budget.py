"""Tests for the prompt-assembly budgeter.

Covers token estimation and the post-assembly history trimmer.
Per-tier defaults live in ``data/familiars/_default/character.toml``;
:mod:`tests.test_config` exercises that path.
"""

from __future__ import annotations

import time

from familiar_connect.budget import (
    Budgeter,
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
        b = TierBudget(total_tokens=9999)
        assert b.total_tokens == 9999
        # Other fields untouched.
        assert b.recent_history_tokens == a.recent_history_tokens
        assert b.rag_tokens == a.rag_tokens
        assert b.max_dossier_people == a.max_dossier_people

    def test_explicit_subcap_used_directly(self) -> None:
        b = TierBudget(recent_history_tokens=500)
        assert b.recent_history_tokens == 500


class TestBudgeterTrim:
    def test_under_budget_passthrough(self) -> None:
        bud = Budgeter(TierBudget(total_tokens=1000))
        sys = "short prompt"
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="yo"),
        ]
        sys_out, msgs_out = bud.trim(system_prompt=sys, history=msgs)
        assert sys_out == sys
        assert msgs_out == msgs

    def test_drops_oldest_when_over_budget(self) -> None:
        # Tiny budget forces eviction.
        bud = Budgeter(TierBudget(total_tokens=20))
        sys = ""  # no static cost
        msgs = [
            Message(role="user", content="A" * 40),  # ~10 tokens content + 4 overhead
            Message(role="user", content="B" * 40),
            Message(role="user", content="C" * 40),
        ]
        _, kept = bud.trim(system_prompt=sys, history=msgs)
        # Newest survives; oldest is dropped first.
        assert kept[-1].content == "C" * 40
        assert "A" * 40 not in [m.content for m in kept]

    def test_keeps_at_least_newest_turn_even_if_oversize(self) -> None:
        """A single huge turn isn't dropped — the user's last message is sacred."""
        bud = Budgeter(TierBudget(total_tokens=10))
        msgs = [Message(role="user", content="Z" * 10000)]
        _, kept = bud.trim(system_prompt="", history=msgs)
        assert len(kept) == 1

    def test_system_prompt_eats_into_history_budget(self) -> None:
        sys = "S" * 4000  # ~1000 tokens
        bud = Budgeter(TierBudget(total_tokens=1100))
        msgs = [
            Message(role="user", content="A" * 200),  # ~50 tokens + overhead
            Message(role="user", content="B" * 200),
            Message(role="user", content="C" * 200),
            Message(role="user", content="D" * 200),
        ]
        _, kept = bud.trim(system_prompt=sys, history=msgs)
        # Some turns dropped because static prompt ate most of the budget.
        assert len(kept) < len(msgs)
        assert kept[-1].content == "D" * 200

    def test_system_prompt_returned_unchanged(self) -> None:
        bud = Budgeter(TierBudget(total_tokens=10))
        sys_out, _ = bud.trim(system_prompt="LONG " * 1000, history=[])
        assert sys_out == "LONG " * 1000


class TestBudgeterPerf:
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
