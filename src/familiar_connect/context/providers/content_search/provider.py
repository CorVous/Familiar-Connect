"""Orchestrator stacking deterministic people lookup on the agent loop.

Tier 1 (``people_lookup``) always runs and emits guaranteed
Contributions for the speaker's file and any mentioned-name files.
Tier 3 (``_agent_loop``) runs after and may emit one additional
Contribution summarising other relevant memory. If the agent loop
raises, the deterministic contributions are still returned — the
never-forget-a-person invariant does not depend on LLM health.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect.context.providers.content_search._agent_loop import (
    DEFAULT_DEADLINE_S,
    DEFAULT_MAX_ITERATIONS,
    _AgentLoop,
)
from familiar_connect.context.providers.content_search.people_lookup import (
    DEFAULT_MAX_TOKENS_PER_FILE,
    lookup,
)

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest, Contribution
    from familiar_connect.llm import LLMClient
    from familiar_connect.memory.store import MemoryStore


_logger = logging.getLogger(__name__)


class ContentSearchProvider:
    """ContextProvider — deterministic people lookup + agent-loop fallback."""

    id = "content_search"
    deadline_s = DEFAULT_DEADLINE_S

    def __init__(
        self,
        *,
        store: MemoryStore,
        llm_client: LLMClient,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        content_cap_tokens: int | None = None,
        max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
    ) -> None:
        self._store = store
        self._agent = _AgentLoop(
            store=store,
            llm_client=llm_client,
            max_iterations=max_iterations,
        )
        self._content_cap_tokens = content_cap_tokens
        self._max_tokens_per_file = max_tokens_per_file

    async def contribute(
        self,
        request: ContextRequest,
    ) -> list[Contribution]:
        deterministic = lookup(
            self._store,
            request,
            content_cap_tokens=self._content_cap_tokens,
            max_tokens_per_file=self._max_tokens_per_file,
        )
        try:
            agent = await self._agent.contribute(request)
        except Exception:
            # never-forget invariant: deterministic tier must still land
            # even if the cheap LLM or its transport fails.
            _logger.exception(
                "content_search: agent loop raised; returning deterministic only"
            )
            agent = []
        return [*deterministic, *agent]
