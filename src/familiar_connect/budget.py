"""Per-tier prompt assembly budget.

Token caps for the prompt assembler. Each cap is a hard number — no
proportional derivation, no "auto-fill from total". The operator
sets each value (or accepts the shipped default from
``data/familiars/_default/character.toml``); the Budgeter and
layers consume the values directly.

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

    Every cap is an explicit integer. The shipped per-tier defaults
    live in ``data/familiars/_default/character.toml`` —
    :mod:`familiar_connect.config` deep-merges per-familiar overrides
    on top, so an operator can change one cap without restating the
    rest.

    The dataclass-level defaults below are a programmatic fallback
    for code paths that construct ``CharacterConfig()`` without going
    through TOML (mostly tests). They match the voice tier; tests
    that need other tiers should construct an explicit instance.

    :param total_tokens: post-assembly trim cap (system prompt +
        recent history). The :class:`Budgeter` drops oldest history
        turns to stay under this.
    :param recent_history_tokens: cap on the recent-history block
        while it's being built.
    :param rag_tokens: cap on the RAG-context block.
    :param dossier_tokens: cap on the people-dossier block.
    :param summary_tokens: cap on the conversation-summary block.
    :param cross_channel_tokens: cap on cross-channel context.
    :param reflection_tokens: cap on the reflections block (M3).
    :param max_history_turns: hard upper bound on recent-history turns
        (safety net before the token cap).
    :param max_rag_turns: hard cap on RAG turn results.
    :param max_rag_facts: hard cap on RAG fact results.
    :param max_dossier_people: hard cap on dossier rows.
    :param max_reflections: hard cap on rendered reflection rows (M3).
    """

    total_tokens: int = 3000
    recent_history_tokens: int = 1500
    rag_tokens: int = 450
    dossier_tokens: int = 450
    summary_tokens: int = 300
    cross_channel_tokens: int = 300
    reflection_tokens: int = 300
    max_history_turns: int = 100
    max_rag_turns: int = 5
    max_rag_facts: int = 3
    max_dossier_people: int = 8
    max_reflections: int = 3


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

    @property
    def budget(self) -> TierBudget:
        return self._budget

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
        cap = self._budget.total_tokens
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
