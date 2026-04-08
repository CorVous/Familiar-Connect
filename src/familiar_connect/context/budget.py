"""Token budgeter for the context pipeline.

Walks a list of Contributions, groups them by Layer, sorts each layer's
contributions by priority (higher first), and joins them up to a per-
layer token budget. Over-budget content is truncated at sentence /
word boundaries when possible and reported in BudgetResult.dropped so
the pipeline can log every lost byte rather than silently dropping it.

Token counting for the first pass is a deliberately boring
character-count heuristic (~4 chars per token). tiktoken can replace
:func:`estimate_tokens` later without touching the Budgeter's public
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from familiar_connect.context.types import Contribution, Layer

_CHARS_PER_TOKEN = 4
"""Approximate ratio used by :func:`estimate_tokens`.

Real tokenisers (tiktoken, HF tokenizers) produce more accurate counts
but add a dependency and a warm-up cost. For budgeting we only need a
monotonic, cheap estimate and a known worst-case slack; 4 chars per
token is the OpenAI tokenizer's rule of thumb for English text.
"""

_ALLOWED_REASONS = frozenset({"dropped", "truncated"})

_SECTION_SEPARATOR = "\n\n"


def estimate_tokens(text: str) -> int:
    """Return an approximate token count for *text*.

    Uses a character-count heuristic (``len(text) // 4``, rounded up).
    Returns 0 for the empty string and at least 1 for any non-empty
    input. Callers that need exact counts should swap this for a real
    tokeniser at the call site.
    """
    if not text:
        return 0
    # Round up so short strings don't report 0 tokens.
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


@dataclass(frozen=True)
class DroppedNote:
    """Record of a contribution (or part of one) that didn't make it.

    Produced by the Budgeter for any content it has to truncate or
    drop. The pipeline logs these so we can see at a glance which
    providers are overflowing their budgets.
    """

    layer: Layer
    source: str
    reason: str
    """Either ``"dropped"`` (the whole contribution was cut) or
    ``"truncated"`` (some prefix survived)."""
    tokens_dropped: int

    def __post_init__(self) -> None:
        if self.reason not in _ALLOWED_REASONS:
            msg = (
                f"DroppedNote.reason must be one of {sorted(_ALLOWED_REASONS)}, "
                f"got {self.reason!r}"
            )
            raise ValueError(msg)


@dataclass
class BudgetResult:
    """What the Budgeter produced for a single pipeline run.

    :param by_layer: Final text per layer, ready to be joined into the
        system prompt by the pipeline. Layers not present in the input
        budget map (or with no surviving content) are simply absent.
    :param dropped: One entry per dropped or truncated contribution,
        for logging and dashboard display.
    """

    by_layer: dict[Layer, str] = field(default_factory=dict)
    dropped: list[DroppedNote] = field(default_factory=list)


class Budgeter:
    """Assemble per-layer text from a flat list of Contributions.

    The Budgeter is intentionally stateless — instances don't carry
    configuration — so a single instance can be shared across guilds
    and requests.
    """

    def fill(
        self,
        contributions: Iterable[Contribution],
        budget_by_layer: dict[Layer, int],
    ) -> BudgetResult:
        """Return a BudgetResult for *contributions* under *budget_by_layer*.

        :param contributions: Iterable of Contributions from providers.
            Insertion order does not matter; the budgeter sorts by
            priority (higher first) within each layer.
        :param budget_by_layer: Per-layer token budgets. A layer with no
            entry here has an effective budget of zero — its
            contributions will be dropped and logged.
        :return: A populated :class:`BudgetResult`.
        """
        result = BudgetResult()

        # Group contributions by layer without mutating the input.
        by_layer: dict[Layer, list[Contribution]] = {}
        for c in contributions:
            by_layer.setdefault(c.layer, []).append(c)

        for layer, items in by_layer.items():
            budget = budget_by_layer.get(layer, 0)
            text, dropped = _fit_layer(items, budget)
            if text:
                result.by_layer[layer] = text
            result.dropped.extend(dropped)

        return result


def _fit_layer(
    contributions: list[Contribution],
    budget: int,
) -> tuple[str, list[DroppedNote]]:
    """Fit a single layer's contributions into *budget* tokens.

    Returns the joined text plus any ``DroppedNote``s generated.
    """
    # Highest priority first, stable among equal priorities.
    ordered = sorted(contributions, key=lambda c: -c.priority)

    kept: list[str] = []
    dropped: list[DroppedNote] = []
    remaining = budget
    separator_tokens = estimate_tokens(_SECTION_SEPARATOR)

    for c in ordered:
        if remaining <= 0:
            dropped.append(
                DroppedNote(
                    layer=c.layer,
                    source=c.source,
                    reason="dropped",
                    tokens_dropped=estimate_tokens(c.text),
                )
            )
            continue

        # Every section after the first pays a separator cost.
        sep_cost = separator_tokens if kept else 0
        available = remaining - sep_cost
        if available <= 0:
            dropped.append(
                DroppedNote(
                    layer=c.layer,
                    source=c.source,
                    reason="dropped",
                    tokens_dropped=estimate_tokens(c.text),
                )
            )
            continue

        text_tokens = estimate_tokens(c.text)
        if text_tokens <= available:
            kept.append(c.text)
            remaining -= sep_cost + text_tokens
            continue

        # Truncate this one. Budget only covers part of it.
        truncated = _truncate_to_tokens(c.text, available)
        tokens_kept = estimate_tokens(truncated)
        if tokens_kept <= 0:
            # Truncation couldn't fit even one word; drop entirely.
            dropped.append(
                DroppedNote(
                    layer=c.layer,
                    source=c.source,
                    reason="dropped",
                    tokens_dropped=text_tokens,
                )
            )
            continue

        kept.append(truncated)
        remaining -= sep_cost + tokens_kept
        dropped.append(
            DroppedNote(
                layer=c.layer,
                source=c.source,
                reason="truncated",
                tokens_dropped=max(0, text_tokens - tokens_kept),
            )
        )

    return _SECTION_SEPARATOR.join(kept), dropped


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return the longest prefix of *text* that fits in *max_tokens*.

    Prefers sentence boundaries, falls back to whitespace, and only
    hard-slices as a last resort.
    """
    if max_tokens <= 0:
        return ""

    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text

    candidate = text[:max_chars]

    # Prefer the last sentence-ending punctuation we can see.
    for end in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = candidate.rfind(end)
        if idx > 0:
            return candidate[: idx + len(end)].rstrip()

    # Fall back to the last whitespace.
    idx = candidate.rfind(" ")
    if idx > 0:
        return candidate[:idx].rstrip()

    # No whitespace in the window — hard slice.
    return candidate
