"""RecastPostProcessor — focused cleanup pass on the main LLM reply.

Step 10 of future-features/context-management.md. Inspired by
SillyTavern's ``recast-post-processing``.

Takes the main LLM's reply text and runs a cheap :class:`SideModel`
with a focused rewrite prompt — tighten tone, strip formatting
artefacts, and (for voice turns) rewrite for spoken delivery. The
modality on the :class:`ContextRequest` controls which prompt
variant is used.

The processor is **failure-isolated**: a slow side-model, an
exception, or an empty / whitespace-only response all return the
*original* reply unchanged. A broken cleanup pass should never
break the user-facing reply.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from familiar_connect.context.types import Modality

if TYPE_CHECKING:
    from familiar_connect.context.side_model import SideModel
    from familiar_connect.context.types import ContextRequest


_logger = logging.getLogger(__name__)


DEFAULT_PROCESSOR_TIMEOUT_S = 9.0
"""Soft cap on the side-model call. Strictly bounded so a stalled
cheap model can never block the post-processing phase."""

DEFAULT_MAX_RECAST_TOKENS = 512
"""Approximate target length for the recast output. Should be at
least as large as the typical main-reply length so the processor
isn't accidentally truncating things."""

_TEXT_PROMPT_TEMPLATE = """\
Rewrite the following reply for clarity and tone. Keep the same
meaning and information; remove filler, asides, and any "as an AI"
disclaimers. Do not add new content. Do not wrap the rewrite in
quotes or any prefix — return ONLY the cleaned reply text.

Original reply:
{reply}

Cleaned reply:"""

_VOICE_PROMPT_TEMPLATE = """\
Rewrite the following reply so it sounds natural when read out loud
as speech by a text-to-speech system. Keep the same meaning and
information; remove markdown, asterisks, lists, code blocks, and
anything that doesn't make sense spoken. Use contractions where they
help. Do not add new content. Return ONLY the spoken reply text — no
quotes, no prefix.

Original reply:
{reply}

Spoken reply:"""


class RecastPostProcessor:
    """PostProcessor that runs a cheap-model cleanup pass on the LLM reply.

    Conforms to the :class:`PostProcessor` Protocol structurally —
    no inheritance required.

    :param side_model: The cheap :class:`SideModel` to use for the
        rewrite.
    :param processor_timeout_s: Soft cap on the side-model call.
        On timeout, the processor returns the original reply.
    :param max_recast_tokens: Approximate target length for the
        cleaned-up reply.
    """

    id = "recast"

    def __init__(
        self,
        *,
        side_model: SideModel,
        processor_timeout_s: float = DEFAULT_PROCESSOR_TIMEOUT_S,
        max_recast_tokens: int = DEFAULT_MAX_RECAST_TOKENS,
    ) -> None:
        self._side_model = side_model
        self._processor_timeout_s = processor_timeout_s
        self._max_recast_tokens = max_recast_tokens

    async def process(
        self,
        reply_text: str,
        request: ContextRequest,
    ) -> str:
        """Return a cleaned-up version of *reply_text*, or the original on failure."""
        template = (
            _VOICE_PROMPT_TEMPLATE
            if request.modality is Modality.voice
            else _TEXT_PROMPT_TEMPLATE
        )
        prompt = template.format(reply=reply_text)

        try:
            async with asyncio.timeout(self._processor_timeout_s):
                response = await self._side_model.complete(
                    prompt,
                    max_tokens=self._max_recast_tokens,
                )
        except TimeoutError:
            _logger.warning(
                "recast: side-model timed out after %.3fs",
                self._processor_timeout_s,
            )
            return reply_text
        except Exception as exc:  # noqa: BLE001 — isolation by design
            _logger.warning(
                "recast: side-model raised %s: %s",
                type(exc).__name__,
                exc,
            )
            return reply_text

        clean = response.strip()
        if not clean:
            return reply_text
        return clean
