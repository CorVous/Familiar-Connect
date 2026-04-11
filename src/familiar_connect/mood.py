"""Mood-based interrupt tolerance drift.

Evaluates the familiar's emotional state from recent conversation context
and returns a modifier that adjusts :attr:`CharacterConfig.interrupt_tolerance`
before each RNG toll check in the interruption system.

The evaluator uses the existing :class:`SideModel` protocol for cheap
inference. Results are cached per-guild with a configurable TTL so
repeated interruptions within a short window don't trigger redundant
model calls.

Design reference: ``future-features/interruption-flow.md``, lines 31-35.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.context.side_model import SideModel

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoodEvaluation:
    """Result of a mood evaluation.

    :param modifier: Float adjustment to ``interrupt_tolerance``,
        clamped to ``[-0.5, +0.5]``.
    :param reasoning: Raw side-model response (for logging/debugging).
    :param timestamp: ``time.monotonic()`` value when evaluated.
    """

    modifier: float
    reasoning: str
    timestamp: float


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_MOOD_EVAL_PROMPT = """\
You are evaluating the emotional state of {familiar_name} in a conversation.

{character_card}

Recent conversation:
{recent_context}

Based on the conversation above, how would {familiar_name}'s current emotional \
state affect their willingness to keep talking if interrupted?

Return a single number between -0.5 and +0.5:
- Negative values (e.g. -0.3): {familiar_name} would yield more readily \
(just asked a question, got corrected, feeling uncertain)
- Zero (0.0): neutral / no strong emotional pull either way
- Positive values (e.g. +0.3): {familiar_name} would be more stubborn about \
keeping talking (excited, passionate, in a heated debate, making an important point)

Reply with ONLY the number, nothing else. Examples: -0.2, 0.0, 0.15, 0.3"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_mood_modifier(raw: str) -> float:
    """Extract a float modifier from the side-model response.

    Robust to common LLM quirks: surrounding text, sign prefixes,
    whitespace. Returns ``0.0`` on parse failure (safe neutral fallback).

    The result is always clamped to ``[-0.5, +0.5]``.

    :param raw: Raw text from the side model.
    :returns: Clamped float in ``[-0.5, +0.5]``.
    """
    stripped = raw.strip()
    if not stripped:
        return 0.0

    # Fast path: the whole response is a number
    try:
        value = float(stripped)
        return max(-0.5, min(0.5, value))
    except ValueError:
        pass

    # Fallback: find the first decimal number in the text
    match = _FLOAT_RE.search(stripped)
    if match:
        try:
            value = float(match.group())
            return max(-0.5, min(0.5, value))
        except ValueError:
            pass

    _logger.warning("Could not parse mood modifier from: %r", raw[:120])
    return 0.0


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


def effective_tolerance(base: float, mood_modifier: float) -> float:
    """Compute the effective interrupt tolerance after mood drift.

    :param base: The character's base ``interrupt_tolerance`` (0.0-1.0).
    :param mood_modifier: The mood evaluator's output (-0.5 to +0.5).
    :returns: Clamped result in ``[0.0, 1.0]``.
    """
    return max(0.0, min(1.0, base + mood_modifier))


# ---------------------------------------------------------------------------
# MoodEvaluator
# ---------------------------------------------------------------------------


class MoodEvaluator:
    """Evaluates the familiar's emotional state for tolerance drift.

    :param side_model: Cheap model for evaluation.
    :param familiar_name: Character name for prompt context.
    :param character_card: Pre-loaded character card text.
    :param cache_ttl_s: How long a cached evaluation stays valid (seconds).
    """

    def __init__(
        self,
        *,
        side_model: SideModel,
        familiar_name: str,
        character_card: str,
        cache_ttl_s: float = 30.0,
    ) -> None:
        self._side_model = side_model
        self._familiar_name = familiar_name
        self._character_card = character_card
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[int, MoodEvaluation] = {}

    async def evaluate(
        self,
        guild_id: int,
        recent_context: str,
    ) -> MoodEvaluation:
        """Evaluate mood, returning cached result if TTL not expired.

        :param guild_id: Discord guild id for cache partitioning.
        :param recent_context: Pre-formatted recent conversation text.
        :returns: A :class:`MoodEvaluation` with the modifier.
        """
        cached = self.get_cached(guild_id)
        if cached is not None:
            return cached

        prompt = _MOOD_EVAL_PROMPT.format(
            familiar_name=self._familiar_name,
            character_card=self._character_card,
            recent_context=recent_context,
        )

        try:
            raw = await self._side_model.complete(prompt)
        except Exception:
            _logger.exception(
                "Mood evaluation failed for guild %s; defaulting to 0.0",
                guild_id,
            )
            raw = "0.0"

        modifier = parse_mood_modifier(raw)
        evaluation = MoodEvaluation(
            modifier=modifier,
            reasoning=raw,
            timestamp=time.monotonic(),
        )
        self._cache[guild_id] = evaluation

        _logger.debug(
            "Mood evaluation guild=%s modifier=%.2f raw=%r",
            guild_id,
            modifier,
            raw[:120],
        )
        return evaluation

    def get_cached(self, guild_id: int) -> MoodEvaluation | None:
        """Return the cached evaluation if still within TTL, else ``None``."""
        cached = self._cache.get(guild_id)
        if cached is None:
            return None
        if time.monotonic() - cached.timestamp > self._cache_ttl_s:
            del self._cache[guild_id]
            return None
        return cached

    def invalidate(self, guild_id: int) -> None:
        """Remove the cached evaluation for *guild_id*."""
        self._cache.pop(guild_id, None)
