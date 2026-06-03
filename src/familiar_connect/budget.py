"""Per-tier prompt assembly budget.

Token caps for prompt assembler. Each cap is a hard number — no
proportional derivation, no "auto-fill from total". Operator sets
each value (or accepts shipped default from
``data/familiars/_default/character.toml``); Budgeter and layers
consume values directly.

Token accounting uses fast ``len(text)/4`` heuristic — no real
tokenizer on hot path. Slightly over-counts (safer for budgets);
adds ~ns per call.

Lives at package root (not under ``context/``) so
:mod:`familiar_connect.config` can import without dragging in the
context package — that would trip a circular import through
:mod:`familiar_connect.llm`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

    Field names mirror :class:`TierBudget` exactly so config parsing
    validates keys via simple set comparison.

    Multipliers must be positive (> 0). Applied via
    :meth:`TierBudget.apply_curve`; each int field scaled + rounded,
    floor of 1.
    """

    total_tokens: float = 1.0
    recent_history_tokens: float = 1.0
    rag_tokens: float = 1.0
    dossier_tokens: float = 1.0
    summary_tokens: float = 1.0
    cross_channel_tokens: float = 1.0
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

    Every cap is explicit int. Shipped per-tier defaults live in
    ``data/familiars/_default/character.toml`` —
    :mod:`familiar_connect.config` deep-merges per-familiar overrides
    on top, so operator can change one cap without restating rest.

    Dataclass-level defaults below are programmatic fallback for code
    paths constructing ``CharacterConfig()`` without TOML (mostly
    tests). Match voice tier; tests needing other tiers construct
    explicit instance.

    :param total_tokens: post-assembly trim cap (system prompt +
        recent history). :class:`Budgeter` drops oldest history turns
        to stay under.
    :param recent_history_tokens: cap on recent-history block during
        build.
    :param rag_tokens: cap on RAG-context block.
    :param dossier_tokens: cap on people-dossier block.
    :param summary_tokens: cap on conversation-summary block.
    :param cross_channel_tokens: cap on cross-channel context.
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

    total_tokens: int = 3000
    recent_history_tokens: int = 1500
    rag_tokens: int = 450
    dossier_tokens: int = 450
    summary_tokens: int = 300
    cross_channel_tokens: int = 300
    reflection_tokens: int = 300
    lorebook_tokens: int = 300
    max_history_turns: int = 100
    max_rag_turns: int = 5
    max_rag_facts: int = 3
    max_dossier_people: int = 8
    max_reflections: int = 3
    max_lorebook_entries: int = 6

    def apply_curve(self, curve: ModelBudgetCurve) -> TierBudget:
        """Return new budget with each field scaled by curve multiplier."""
        return TierBudget(
            total_tokens=_scale(self.total_tokens, curve.total_tokens),
            recent_history_tokens=_scale(
                self.recent_history_tokens, curve.recent_history_tokens
            ),
            rag_tokens=_scale(self.rag_tokens, curve.rag_tokens),
            dossier_tokens=_scale(self.dossier_tokens, curve.dossier_tokens),
            summary_tokens=_scale(self.summary_tokens, curve.summary_tokens),
            cross_channel_tokens=_scale(
                self.cross_channel_tokens, curve.cross_channel_tokens
            ),
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


class Budgeter:
    """Total-cap enforcer applied after layer assembly.

    Layers self-truncate to per-section caps while building. Budgeter
    then trims oldest history turns until combined ``system_prompt +
    history`` fits under :attr:`TierBudget.total_tokens`.

    System prompt *never* truncated here — static layers (core
    instructions, character card) carry bot identity; silent truncation
    would change behaviour. When they exceed budget alone, that's
    operator-visible misconfiguration; still return them and let LLM
    decide.
    """

    def __init__(
        self,
        budget: TierBudget,
        channel_total_tokens: dict[int, int] | None = None,
    ) -> None:
        self._budget = budget
        self._channel_total_tokens: dict[int, int] = channel_total_tokens or {}

    @property
    def budget(self) -> TierBudget:
        return self._budget

    def with_overrides(self, **overrides: object) -> Budgeter:
        """Return new Budgeter with select fields replaced (for tests)."""
        return Budgeter(replace(self._budget, **overrides))  # type: ignore[arg-type]

    def trim(
        self,
        *,
        system_prompt: str,
        history: list[Message],
        channel_id: int | None = None,
    ) -> tuple[str, list[Message]]:
        """Drop oldest turns until total token count fits.

        Returns ``(system_prompt, trimmed_history)``. Newest turns
        retained — immediate conversational context model needs most.
        ``system_prompt`` returned unchanged.

        Per-channel ``total_tokens`` overrides tier cap when set.
        """
        sys_tokens = estimate_tokens(system_prompt)
        cap = (
            self._channel_total_tokens[channel_id]
            if channel_id is not None and channel_id in self._channel_total_tokens
            else self._budget.total_tokens
        )
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
