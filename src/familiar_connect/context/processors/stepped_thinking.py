"""SteppedThinkingPreProcessor — hidden chain-of-thought via the reasoning slot.

Calls the ``reasoning_context`` slot's LLMClient with a "think step
by step" prompt and stashes the result as a ``Layer.depth_inject``
Contribution on the request. The pipeline picks it up alongside
provider contributions when assembling budgeter input.

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
    from familiar_connect.context.types import ContextRequest
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

_PROMPT_TEMPLATE = """\
You are about to help a familiar reply to a Discord message. Before
the main model speaks, briefly think step by step about what the
speaker is really asking for and what would actually be useful to
mention in the reply. Plain prose, no headings, no bullet lists.
Cap your response at {max_tokens} tokens — terse is fine.

Speaker: {speaker}
Message: {utterance}

Your thinking:"""


class SteppedThinkingPreProcessor:
    """PreProcessor that prepends a hidden chain-of-thought to the request."""

    id = "stepped_thinking"

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        processor_timeout_s: float = DEFAULT_PROCESSOR_TIMEOUT_S,
    ) -> None:
        self._llm_client = llm_client
        self._processor_timeout_s = processor_timeout_s

    async def process(self, request: ContextRequest) -> ContextRequest:
        """Run the thinking pass and append a Contribution if it succeeds.

        Returns the request unchanged on any failure (timeout,
        exception, empty response).
        """
        prompt = _PROMPT_TEMPLATE.format(
            speaker=request.author.label if request.author else "(unknown)",
            utterance=request.utterance,
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
