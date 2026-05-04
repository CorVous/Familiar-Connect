"""Per-tier prompt assembly budget.

Single primary knob — :attr:`TierBudget.total_tokens` — caps the
soft size of system prompt + recent history. Sub-allocations
(``recent_history_tokens`` etc.) and item caps (``max_history_turns``
etc.) have sensible defaults; override only when tuning one section.

Token accounting uses a fast ``len(text)/4`` heuristic — no real
tokenizer on the hot path. Slightly over-counts (safer for budgets);
adds ~ns per call.

Lives at the package root (not under ``context/``) so
:mod:`familiar_connect.config` can import it without dragging in the
context package — that would trip a circular import through
:mod:`familiar_connect.llm`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.llm import Message


_CHARS_PER_TOKEN = 4
"""OpenAI's well-known English heuristic. Slightly over-counts."""

_MESSAGE_OVERHEAD_TOKENS = 4
"""Per-message chat-format framing (role + delimiters)."""


def estimate_tokens(text: str) -> int:
    """Fast char-based token estimate.

    ``ceil(len / 4)`` — over-counts mildly so budgets stay safe.
    No tokenizer dependency; runs in nanoseconds. See module docstring.
    """
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def estimate_message_tokens(msg: Message) -> int:
    """Chat-format estimate including role/name framing."""
    n = estimate_tokens(msg.content) + _MESSAGE_OVERHEAD_TOKENS
    if msg.name:
        n += estimate_tokens(msg.name)
    return n


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Sum across a message list."""
    return sum(estimate_message_tokens(m) for m in messages)


@dataclass(frozen=True)
class TierBudget:
    """Token budget for one assembly tier (voice / text / background).

    Only :attr:`total_tokens` is the operator's primary knob; the rest
    have sensible defaults derived from the total.

    :param total_tokens: soft cap on system prompt + recent history.
    :param recent_history_tokens: cap on the recent-history block.
        ``None`` → ``total_tokens // 2``.
    :param rag_tokens: cap on the RAG-context block. ``None`` →
        ``total_tokens * 15 // 100``.
    :param dossier_tokens: cap on the people-dossier block. ``None`` →
        ``total_tokens * 15 // 100``.
    :param summary_tokens: cap on the conversation-summary block.
        ``None`` → ``total_tokens * 10 // 100``.
    :param cross_channel_tokens: cap on cross-channel context. ``None`` →
        ``total_tokens * 10 // 100``.
    :param max_history_turns: hard upper bound on recent-history turns
        (safety net before token cap).
    :param max_rag_turns: hard cap on RAG turn results.
    :param max_rag_facts: hard cap on RAG fact results.
    :param max_dossier_people: hard cap on dossier rows.
    """

    total_tokens: int = 3000
    recent_history_tokens: int | None = None
    rag_tokens: int | None = None
    dossier_tokens: int | None = None
    summary_tokens: int | None = None
    cross_channel_tokens: int | None = None
    max_history_turns: int = 100
    max_rag_turns: int = 5
    max_rag_facts: int = 3
    max_dossier_people: int = 8

    def resolved(self) -> ResolvedTierBudget:
        """Fill in ``None`` sub-caps from ``total_tokens``."""
        total = self.total_tokens
        return ResolvedTierBudget(
            total_tokens=total,
            recent_history_tokens=(
                self.recent_history_tokens
                if self.recent_history_tokens is not None
                else total // 2
            ),
            rag_tokens=(
                self.rag_tokens if self.rag_tokens is not None else total * 15 // 100
            ),
            dossier_tokens=(
                self.dossier_tokens
                if self.dossier_tokens is not None
                else total * 15 // 100
            ),
            summary_tokens=(
                self.summary_tokens
                if self.summary_tokens is not None
                else total * 10 // 100
            ),
            cross_channel_tokens=(
                self.cross_channel_tokens
                if self.cross_channel_tokens is not None
                else total * 10 // 100
            ),
            max_history_turns=self.max_history_turns,
            max_rag_turns=self.max_rag_turns,
            max_rag_facts=self.max_rag_facts,
            max_dossier_people=self.max_dossier_people,
        )


@dataclass(frozen=True)
class ResolvedTierBudget:
    """:class:`TierBudget` with all sub-caps filled in."""

    total_tokens: int
    recent_history_tokens: int
    rag_tokens: int
    dossier_tokens: int
    summary_tokens: int
    cross_channel_tokens: int
    max_history_turns: int
    max_rag_turns: int
    max_rag_facts: int
    max_dossier_people: int


# Defaults for each tier — match docs/architecture/tuning.md.
DEFAULT_VOICE_BUDGET: TierBudget = TierBudget(total_tokens=3000)
DEFAULT_TEXT_BUDGET: TierBudget = TierBudget(total_tokens=8000)
DEFAULT_BACKGROUND_BUDGET: TierBudget = TierBudget(total_tokens=24000)


class Budgeter:
    """Total-cap enforcer applied after layer assembly.

    Layers self-truncate to their per-section caps while building.
    The Budgeter then trims oldest history turns until the combined
    ``system_prompt + history`` fits under :attr:`TierBudget.total_tokens`.

    System prompt is *never* truncated here — the static layers (core
    instructions, character card) carry the bot's identity; truncating
    them silently would change behaviour. If they exceed the budget on
    their own, that's an operator-visible misconfiguration; we still
    return them and just let the LLM decide.
    """

    def __init__(self, budget: TierBudget) -> None:
        self._budget = budget
        self._resolved = budget.resolved()

    @property
    def budget(self) -> TierBudget:
        return self._budget

    @property
    def resolved(self) -> ResolvedTierBudget:
        return self._resolved

    def with_overrides(self, **overrides: object) -> Budgeter:
        """Return a new Budgeter with select fields replaced (for tests)."""
        return Budgeter(replace(self._budget, **overrides))  # type: ignore[arg-type]

    def trim(
        self,
        *,
        system_prompt: str,
        history: list[Message],
    ) -> tuple[str, list[Message]]:
        """Drop oldest turns until total token count fits.

        Returns ``(system_prompt, trimmed_history)``. Newest turns
        retained — they're the immediate conversational context the
        model needs most. ``system_prompt`` is returned unchanged.
        """
        sys_tokens = estimate_tokens(system_prompt)
        cap = self._resolved.total_tokens
        # Walk from newest backwards, accumulating until we'd overflow.
        kept_reversed: list[Message] = []
        used = sys_tokens
        for msg in reversed(history):
            cost = estimate_message_tokens(msg)
            if used + cost > cap and kept_reversed:
                break
            kept_reversed.append(msg)
            used += cost
        kept = list(reversed(kept_reversed))
        return system_prompt, kept
