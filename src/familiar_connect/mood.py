"""Mood-driven tolerance modifier for voice interruption handling.

The mood evaluator inspects recent conversation and emits a single
float in ``[-0.5, +0.5]`` that biases the familiar's interrupt
tolerance for the duration of one response. Positive values make the
familiar more stubborn (push through interruptions); negative values
make it more yielding. The modifier is cached on the
:class:`~familiar_connect.voice.interruption.ResponseTracker` at
``GENERATING`` entry so every decision within that turn uses the same
roll input.

Today this is a **stub**: :meth:`MoodEvaluator.evaluate` always
returns ``0.0``. The real side-model call — a short LLM prompt that
reads the last few user turns and classifies the emotional context
— lands in Step 13 of the voice-interruption roadmap. Splitting the
interface out now lets the detector + tracker treat mood as a
first-class input from Step 6 onward without carrying conditional
"mood is wired yet?" code.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


class MoodEvaluator:
    """Returns the per-response mood modifier (stub: always ``0.0``).

    The public surface is intentionally tiny — a single
    :meth:`evaluate` method — so the Step 13 swap is a drop-in.
    """

    def evaluate(self) -> float:
        """Compute the mood modifier for the upcoming response.

        :returns: A value in ``[-0.5, +0.5]``. Positive = more
            stubborn, negative = more yielding. Currently hard-coded
            to ``0.0``; real implementation in Step 13.
        """
        modifier = 0.0
        _logger.info("mood_modifier=%.2f (stub)", modifier)
        return modifier
