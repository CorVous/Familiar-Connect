"""SteppedThinkingPreProcessor — hidden chain-of-thought via the reasoning slot.

Step 10 of docs/architecture/context-pipeline.md. Inspired by
SillyTavern's ``st-stepped-thinking``.

Calls the ``reasoning_context`` slot's :class:`LLMClient` with a
focused "think step by step about what the user is really asking
for" prompt and stashes the result on the request's
``preprocessor_contributions`` tuple as a :class:`Contribution` at
:data:`Layer.depth_inject`. The pipeline picks it up automatically
alongside provider contributions when assembling the final
budgeter input.

The processor is **failure-isolated**: a slow LLM call, an
exception, or an empty response all return the request *unchanged*.
A pre-processor that's broken should never block the reply.

The "I'm thinking step by step" output is intentionally not surfaced
to the user — it ends up in :data:`Layer.depth_inject`, which the
budgeter routes into the model's hidden context rather than the
user-visible reply path.
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
    """PreProcessor that prepends a hidden chain-of-thought to the request.

    Conforms to the :class:`PreProcessor` Protocol structurally — no
    inheritance required.

    :param llm_client: The :class:`LLMClient` for the
        ``reasoning_context`` slot.
    :param processor_timeout_s: Soft cap on the LLM call. On
        timeout, the processor returns the request unchanged.
    """

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
            speaker=request.speaker or "(unknown)",
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
