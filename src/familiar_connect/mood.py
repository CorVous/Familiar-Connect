"""Mood-driven tolerance modifier for voice interruption handling.

Emits ``[-0.5, +0.5]`` bias on interrupt tolerance per response.
Positive = more stubborn; negative = more yielding. Cached on
:class:`~familiar_connect.voice.interruption.ResponseTracker` at
``GENERATING`` entry. Currently a stub (always ``0.0``).
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


class MoodEvaluator:
    """Per-response mood modifier (stub: always ``0.0``)."""

    def evaluate(self) -> float:
        """Return mood modifier in ``[-0.5, +0.5]``. Stub — always ``0.0``."""
        modifier = 0.0
        _logger.info("mood_modifier=%.2f (stub)", modifier)
        return modifier
