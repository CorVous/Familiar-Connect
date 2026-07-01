"""Per-tier prompt assembly budget.

Token caps for prompt assembler. Each cap is a hard number — no
proportional derivation, no "auto-fill from total". Operator sets
each value (or accepts shipped default from
``data/familiars/_default/character.toml``); the assembly layers
consume values directly, each self-truncating to its own cap.

There is no separate *combined* cap. The whole-prompt ``total_tokens``
is a derived figure (sum of the per-section caps) exposed for
reporting only — see :attr:`TierBudget.total_tokens`.

Token accounting uses fast ``len(text)/4`` heuristic — no real
tokenizer on hot path. Slightly over-counts (safer for budgets);
adds ~ns per call.

Lives at package root (not under ``context/``) so
:mod:`familiar_connect.config` can import without dragging in the
context package — that would trip a circular import through
:mod:`familiar_connect.llm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.llm import Message


_CHARS_PER_TOKEN = 4
"""OpenAI's well-known English heuristic; slightly over-counts."""

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
    n = estimate_tokens(msg.content_str) + _MESSAGE_OVERHEAD_TOKENS
    if msg.name:
        n += estimate_tokens(msg.name)
    return n


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Sum across message list."""
    return sum(estimate_message_tokens(m) for m in messages)


@dataclass(frozen=True)
class ModelBudgetCurve:
    """Per-section multipliers for a specific model.

    All fields default to 1.0 (identity — no change). Operators set
    only sections differing from tier default; unset fields stay at
    base value.

    Field names mirror :class:`TierBudget`'s configurable caps exactly
    so config parsing validates keys via simple set comparison. There
    is no ``total_tokens`` multiplier: the whole-prompt total is
    derived from the per-section caps, so it scales implicitly when
    those caps scale.

    Multipliers must be positive (> 0). Applied via
    :meth:`TierBudget.apply_curve`; each int field scaled + rounded,
    floor of 1.
    """

    recent_history_tokens: float = 1.0
    rag_tokens: float = 1.0
    dossier_tokens: float = 1.0
    summary_tokens: float = 1.0
    reflection_tokens: float = 1.0
    lorebook_tokens: float = 1.0
    max_history_turns: float = 1.0
    max_rag_turns: float = 1.0
    max_rag_facts: float = 1.0
    max_dossier_people: float = 1.0
    max_reflections: float = 1.0
    max_lorebook_entries: float = 1.0


def _scale(base: int, multiplier: float) -> int:
    return max(1, round(base * multiplier))


@dataclass(frozen=True)
class TierBudget:
    """Token budget for one assembly tier (voice / text / background).

    Every cap is an explicit int and is enforced *independently*: each
    assembly layer self-truncates to its own ``*_tokens`` cap while
    building. There is no separate combined cap — the prompt's overall
    size is simply the sum of the section caps, surfaced as the derived
    :attr:`total_tokens` for reporting.

    Shipped per-tier defaults live in
    ``data/familiars/_default/character.toml`` —
    :mod:`familiar_connect.config` deep-merges per-familiar overrides
    on top, so an operator can change one cap without restating rest.

    Dataclass-level defaults below are a programmatic fallback for code
    paths constructing ``CharacterConfig()`` without TOML (mostly
    tests). Match voice tier; tests needing other tiers construct
    explicit instance.

    :param recent_history_tokens: cap on recent-history block during
        build.
    :param rag_tokens: cap on RAG-context block.
    :param dossier_tokens: cap on people-dossier block.
    :param summary_tokens: cap on conversation-summary block.
    :param reflection_tokens: cap on reflections block (M3).
    :param lorebook_tokens: cap on lorebook block (M4).
    :param max_history_turns: hard upper bound on recent-history turns
        (safety net before token cap).
    :param max_rag_turns: hard cap on RAG turn results.
    :param max_rag_facts: hard cap on RAG fact results.
    :param max_dossier_people: hard cap on dossier rows.
    :param max_reflections: hard cap on rendered reflection rows (M3).
    :param max_lorebook_entries: hard cap on rendered lorebook entries (M4).
    """

    recent_history_tokens: int = 3000
    rag_tokens: int = 900
    dossier_tokens: int = 900
    summary_tokens: int = 600
    reflection_tokens: int = 600
    lorebook_tokens: int = 600
    max_history_turns: int = 200
    max_rag_turns: int = 10
    max_rag_facts: int = 6
    max_dossier_people: int = 16
    max_reflections: int = 6
    max_lorebook_entries: int = 12

    @property
    def total_tokens(self) -> int:
        """Derived sum of the per-section token caps.

        Not a configurable knob — the prompt has no separate combined
        cap, and nothing trims against this value. It is the budgeted
        prompt ceiling (excluding the unbudgeted static layers such as
        the character card and core instructions), exposed purely for
        reporting and for eyeballing headroom against a model's context
        window.
        """
        return (
            self.recent_history_tokens
            + self.rag_tokens
            + self.dossier_tokens
            + self.summary_tokens
            + self.reflection_tokens
            + self.lorebook_tokens
        )

    def apply_curve(self, curve: ModelBudgetCurve) -> TierBudget:
        """Return new budget with each field scaled by curve multiplier.

        ``total_tokens`` is derived, so it follows automatically once
        the constituent caps are scaled.
        """
        return TierBudget(
            recent_history_tokens=_scale(
                self.recent_history_tokens, curve.recent_history_tokens
            ),
            rag_tokens=_scale(self.rag_tokens, curve.rag_tokens),
            dossier_tokens=_scale(self.dossier_tokens, curve.dossier_tokens),
            summary_tokens=_scale(self.summary_tokens, curve.summary_tokens),
            reflection_tokens=_scale(self.reflection_tokens, curve.reflection_tokens),
            lorebook_tokens=_scale(self.lorebook_tokens, curve.lorebook_tokens),
            max_history_turns=_scale(self.max_history_turns, curve.max_history_turns),
            max_rag_turns=_scale(self.max_rag_turns, curve.max_rag_turns),
            max_rag_facts=_scale(self.max_rag_facts, curve.max_rag_facts),
            max_dossier_people=_scale(
                self.max_dossier_people, curve.max_dossier_people
            ),
            max_reflections=_scale(self.max_reflections, curve.max_reflections),
            max_lorebook_entries=_scale(
                self.max_lorebook_entries, curve.max_lorebook_entries
            ),
        )
