"""VoiceParticipantsProvider — surfaces the live voice-channel roster.

Reads ``ContextRequest.voice_participants`` (populated from
``channel.members`` by the voice response path) and emits a single
``Layer.author_note`` Contribution naming everyone else on the call.
Inert for text turns and when no participants are supplied, so safe
to enable in any channel mode.

See docs/architecture/context-pipeline.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer, Modality

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest


_logger = logging.getLogger(__name__)


VOICE_PARTICIPANTS_PRIORITY = 90
"""Priority for the participants note.

Below :data:`CHARACTER_PRIORITY` (100) so the persona wins under
budget pressure, above ``MODE_INSTRUCTION_PRIORITY`` (80) so the
live roster survives ahead of the mode backdrop."""

_DEADLINE_S = 0.05
"""Pure-Python formatting; generous upper bound."""


class VoiceParticipantsProvider:
    """ContextProvider that renders the voice-channel member list."""

    id = "voice_participants"
    deadline_s = _DEADLINE_S

    async def contribute(
        self,
        request: ContextRequest,
    ) -> list[Contribution]:
        """Return one Contribution naming everyone in the voice call.

        Returns ``[]`` for text turns or when ``voice_participants``
        is empty. Duplicate authors (same ``canonical_key``) collapse
        to a single mention; supplied order is otherwise preserved.
        """
        if request.modality is not Modality.voice:
            return []
        if not request.voice_participants:
            return []

        seen: set[str] = set()
        labels: list[str] = []
        for author in request.voice_participants:
            key = author.canonical_key
            if key in seen:
                continue
            seen.add(key)
            labels.append(author.label)

        if not labels:
            return []

        text = _format_participants(labels)
        _logger.info(
            f"{ls.tag('🎙️ Voice Roster', ls.C)} "
            f"{ls.kv('count', str(len(labels)), vc=ls.LC)} "
            f"{ls.kv('names', ', '.join(labels), vc=ls.LC)}"
        )
        return [
            Contribution(
                layer=Layer.author_note,
                priority=VOICE_PARTICIPANTS_PRIORITY,
                text=text,
                estimated_tokens=estimate_tokens(text),
                source="voice_participants",
            )
        ]


def _format_participants(labels: list[str]) -> str:
    """Render the roster as a short English sentence for the prompt."""
    if len(labels) == 1:
        return f"You are currently in a voice call with {labels[0]}."
    if len(labels) == 2:
        return f"You are currently in a voice call with {labels[0]} and {labels[1]}."
    head = ", ".join(labels[:-1])
    return f"You are currently in a voice call with {head}, and {labels[-1]}."
