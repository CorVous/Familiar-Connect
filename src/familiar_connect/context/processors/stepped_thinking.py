"""SteppedThinkingPreProcessor — hidden chain-of-thought via the reasoning slot.

Calls the ``reasoning_context`` slot's LLMClient with a "think step
by step" prompt and stashes the result as a ``Layer.depth_inject``
Contribution on the request. The pipeline picks it up alongside
provider contributions when assembling budgeter input.

When wired with a :class:`HistoryStore`, recent turns and any buffered
pending turns are folded into the prompt so the reasoning model can
actually contextualise the reply. Without a store the processor falls
back to the single trigger utterance (legacy behaviour, kept for
tests and minimal wirings).

Failure-isolated: timeout, exception, or empty response return the
request unchanged. Output is routed to hidden context, never surfaced
to the user. See docs/architecture/context-pipeline.md.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer
from familiar_connect.llm import Message

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest, PendingTurn
    from familiar_connect.history.store import HistoryStore, HistoryTurn
    from familiar_connect.llm import LLMClient


_logger = logging.getLogger(__name__)


STEPPED_THINKING_PRIORITY = 75
"""Priority assigned to the stepped-thinking Contribution.

Higher than the rolling history summary (60) so brief, fresh
reasoning beats stale long-term context under budget pressure;
lower than recent history (80) so the active conversation always
wins."""

DEFAULT_PROCESSOR_TIMEOUT_S = 9.0
"""Soft cap on the reasoning LLM call. Strictly bounded so a
stalled model can never block the pipeline's pre-processor phase."""

_MAX_THINKING_TOKENS_HINT = 256
"""Approximate target length advertised to the model via the prompt."""

DEFAULT_HISTORY_WINDOW = 20
"""Recent-turn fetch size. Mirrors render._DEFAULT_HISTORY_WINDOW so
the reasoning model sees roughly what the main model will see."""

_PROMPT_TEMPLATE = """\
You are about to help a familiar reply to a Discord conversation.
Before the main model speaks, briefly think step by step about what
the speaker is really asking for — given the conversation so far —
and what would actually be useful to mention in the reply. Plain
prose, no headings, no bullet lists. Cap your response at
{max_tokens} tokens — terse is fine.
{history_section}
New message{plural} from {speaker}:
{new_messages}

Your thinking:"""


def _format_turn(turn: HistoryTurn) -> str:
    """Render one stored turn as ``Label: content`` (or ``Familiar: ...``).

    Assistant turns have no author; label them generically so the
    reasoning model can tell its own prior replies apart from users.
    """
    if turn.role == "user" and turn.author is not None:
        return f"{turn.author.label}: {turn.content}"
    if turn.role == "assistant":
        return f"Familiar: {turn.content}"
    return turn.content


def _format_pending(speaker: str, pending: tuple[PendingTurn, ...]) -> str:
    """Render buffered pending turns, one per line, prefixed by speaker."""
    lines: list[str] = []
    for pt in pending:
        label = pt.author.label if pt.author is not None else speaker
        lines.append(f"{label}: {pt.text}")
    return "\n".join(lines)


class SteppedThinkingPreProcessor:
    """PreProcessor that prepends a hidden chain-of-thought to the request."""

    id = "stepped_thinking"

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        history_store: HistoryStore | None = None,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        processor_timeout_s: float = DEFAULT_PROCESSOR_TIMEOUT_S,
    ) -> None:
        self._llm_client = llm_client
        self._history_store = history_store
        self._history_window = history_window
        self._processor_timeout_s = processor_timeout_s

    async def process(self, request: ContextRequest) -> ContextRequest:
        """Run the thinking pass and append a Contribution if it succeeds.

        Returns the request unchanged on any failure (timeout,
        exception, empty response).
        """
        speaker = request.author.label if request.author else "(unknown)"

        # recent history section — only included when a store is wired
        history_section = ""
        if self._history_store is not None and self._history_window > 0:
            turns = self._history_store.recent(
                familiar_id=request.familiar_id,
                channel_id=request.channel_id,
                limit=self._history_window,
            )
            if turns:
                rendered = "\n".join(_format_turn(t) for t in turns)
                history_section = f"\nConversation so far:\n{rendered}\n"

        # new messages: all buffered pending turns, else fall back to
        # the single trigger utterance
        if request.pending_turns:
            new_messages = _format_pending(speaker, request.pending_turns)
            plural = "s" if len(request.pending_turns) > 1 else ""
        else:
            new_messages = request.utterance
            plural = ""

        prompt = _PROMPT_TEMPLATE.format(
            speaker=speaker,
            new_messages=new_messages,
            plural=plural,
            history_section=history_section,
            max_tokens=_MAX_THINKING_TOKENS_HINT,
        )

        try:
            async with asyncio.timeout(self._processor_timeout_s):
                reply = await self._llm_client.chat(
                    [Message(role="user", content=prompt)],
                )
        except TimeoutError:
            _logger.warning(
                "stepped_thinking: LLM call timed out after %.3fs",
                self._processor_timeout_s,
            )
            return request
        except Exception as exc:  # noqa: BLE001 — isolation by design
            _logger.warning(
                "stepped_thinking: LLM call raised %s: %s",
                type(exc).__name__,
                exc,
            )
            return request

        clean = reply.content.strip()
        if not clean:
            return request

        contribution = Contribution(
            layer=Layer.depth_inject,
            priority=STEPPED_THINKING_PRIORITY,
            text=clean,
            estimated_tokens=estimate_tokens(clean),
            source="stepped_thinking",
        )
        return replace(
            request,
            preprocessor_contributions=(
                *request.preprocessor_contributions,
                contribution,
            ),
        )
