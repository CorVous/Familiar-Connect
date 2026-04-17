"""Mood-driven tolerance modifier for voice interruption handling.

The mood evaluator inspects recent conversation turns and emits a single
float in ``[-0.5, +0.5]`` that biases the familiar's interrupt tolerance
for the duration of one response. Positive values make the familiar more
stubborn (push through interruptions); negative values make it more
yielding. The modifier is cached on the
:class:`~familiar_connect.voice.interruption.ResponseTracker` at
``GENERATING`` entry so every decision within that turn uses the same
roll input.

No-arg construction yields stub mode: :meth:`MoodEvaluator.evaluate`
always returns ``0.0`` without calling the LLM. Pass ``llm_client``
and ``history_store`` for the real side-model call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryStore, HistoryTurn
    from familiar_connect.llm import LLMClient

_logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 6

_PROMPT = """\
Given these recent messages, output a single float between -0.5 and 0.5:
- Positive (up to +0.5): excited, engaged, playful → more stubborn
- Zero: neutral
- Negative (down to -0.5): corrected, confused, asking → more yielding

Recent turns:
{turns}

Output only the number, e.g.: 0.2"""


def _format_turns(turns: list[HistoryTurn]) -> str:
    """Render turns as role-prefixed lines for the mood prompt."""
    lines: list[str] = []
    for t in turns:
        prefix = f"{t.author.label} (user)" if t.role == "user" and t.author else t.role
        lines.append(f"{prefix}: {t.content}")
    return "\n".join(lines)


class MoodEvaluator:
    """Per-response mood modifier via LLM side-call.

    No-arg construction returns stub (always 0.0). Pass ``llm_client``
    and ``history_store`` for a real classification call.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        history_store: HistoryStore | None = None,
    ) -> None:
        self._llm = llm_client
        self._store = history_store

    async def evaluate(self, channel_id: int = 0, familiar_id: str = "") -> float:
        """Compute mood modifier for the upcoming response.

        :returns: Float in ``[-0.5, +0.5]``. Positive = more stubborn,
            negative = more yielding. Returns ``0.0`` when deps absent,
            history empty, or LLM call fails.
        """
        if self._llm is None or self._store is None:
            _logger.info(
                f"{ls.tag('Mood', ls.M)} "
                f"{ls.kv('modifier', '0.00', vc=ls.LW)} "
                f"{ls.kv('reason', 'stub', vc=ls.LW)}"
            )
            return 0.0

        turns = self._store.recent(
            familiar_id=familiar_id,
            channel_id=channel_id,
            limit=_HISTORY_LIMIT,
        )
        if not turns:
            _logger.info(
                f"{ls.tag('Mood', ls.M)} "
                f"{ls.kv('modifier', '0.00', vc=ls.LW)} "
                f"{ls.kv('reason', 'no_history', vc=ls.LW)}"
            )
            return 0.0

        prompt = _PROMPT.format(turns=_format_turns(turns))
        try:
            reply = await self._llm.chat([Message(role="user", content=prompt)])
            modifier = max(-0.5, min(0.5, float(reply.content.strip())))
        except Exception as exc:
            _logger.warning("mood_eval failed: %s", exc, exc_info=True)
            return 0.0

        _logger.info(
            f"{ls.tag('Mood', ls.M)} {ls.kv('modifier', f'{modifier:+.2f}', vc=ls.LM)}"
        )
        return modifier
