"""Orchestrator: deterministic people lookup → embedding retrieval → single-shot filter.

Three tiers composed in order (see docs/architecture/context-pipeline.md §8):

1. ``people_lookup.lookup_with_paths`` — always runs. Guaranteed to
   surface the speaker's ``people/<slug>.md`` and any file matching
   a name mentioned in the utterance. Correctness floor.
2. ``retriever.query`` — optional semantic retrieval over the
   embedding index. Excludes any ``rel_path`` already included by
   tier 1 so the filter doesn't see duplicates.
3. ``filter.run`` — one cheap-LLM call (plus at most one grep-
   escalation follow-up) that selects what from the retrieved
   snippets is worth forwarding to the main model.

The never-forget-a-person invariant holds regardless of tier 2/3
health: if the retriever errors or the filter raises, the
deterministic tier's contributions are still returned.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.context.providers.content_search import filter as _filter
from familiar_connect.context.providers.content_search.people_lookup import (
    DEFAULT_MAX_TOKENS_PER_FILE,
    lookup_with_paths,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from familiar_connect.context.providers.content_search.retrieval import (
        Retriever,
    )
    from familiar_connect.context.types import ContextRequest, Contribution
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.llm import LLMClient
    from familiar_connect.memory.store import MemoryStore


_logger = logging.getLogger(__name__)


DEFAULT_DEADLINE_S = 15.0
"""Hard cap the pipeline enforces on this provider's contribute() call."""

DEFAULT_TOP_K = 8
"""Top-K embedding hits requested from the retriever."""

DEFAULT_RECENT_AUTHOR_LIMIT = 5
"""How many most-recently-seen channel users to keep warm via people_lookup."""


class ContentSearchProvider:
    """ContextProvider composing the three tiers."""

    id = "content_search"
    deadline_s = DEFAULT_DEADLINE_S

    def __init__(
        self,
        *,
        store: MemoryStore,
        llm_client: LLMClient,
        history_store: HistoryStore | None = None,
        retriever: Retriever | None = None,
        index_build: Callable[[], Awaitable[None]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        content_cap_tokens: int | None = None,
        max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
        recent_author_limit: int = DEFAULT_RECENT_AUTHOR_LIMIT,
    ) -> None:
        self._store = store
        self._llm_client = llm_client
        self._history_store = history_store
        self._retriever = retriever
        self._index_build = index_build
        self._top_k = top_k
        self._content_cap_tokens = content_cap_tokens
        self._max_tokens_per_file = max_tokens_per_file
        self._recent_author_limit = recent_author_limit
        self._build_task: asyncio.Task[None] | None = None

    async def contribute(
        self,
        request: ContextRequest,
    ) -> list[Contribution]:
        recent_authors = self._fetch_recent_authors(request)
        lookup_result = lookup_with_paths(
            self._store,
            request,
            recent_authors=recent_authors,
            content_cap_tokens=self._content_cap_tokens,
            max_tokens_per_file=self._max_tokens_per_file,
        )
        deterministic = lookup_result.contributions
        exclude = set(lookup_result.rel_paths)
        people_files = (
            ", ".join(PurePosixPath(p).name for p in lookup_result.rel_paths)
            or "(none)"
        )
        _logger.info(
            f"{ls.tag('👥 People', ls.M)} "
            f"{ls.kv('count', str(len(lookup_result.rel_paths)), vc=ls.LM)} "
            f"{ls.kv('files', people_files, vc=ls.LM)}"
        )

        # Fire-and-forget the first-time index build. Uses whatever's
        # already in the index for the current turn.
        self._maybe_start_index_build()

        retrieved = await self._retrieve(request, exclude=exclude)
        retrieved_files = (
            ", ".join(PurePosixPath(c.rel_path).name for c in retrieved) or "(none)"
        )
        _logger.info(
            f"{ls.tag('📚 Content', ls.M)} "
            f"{ls.kv('retrieved', str(len(retrieved)), vc=ls.LM)} "
            f"{ls.kv('files', retrieved_files, vc=ls.LM)}"
        )

        try:
            filtered = await _filter.run(
                llm_client=self._llm_client,
                store=self._store,
                request=request,
                retrieved=retrieved,
                deterministic=deterministic,
            )
        except Exception:
            # never-forget invariant: deterministic tier must still land
            # if the cheap LLM or its transport fails.
            _logger.exception(
                f"{ls.tag('📚 Content', ls.M)} "
                f"{ls.word('filter raised;', ls.LW)} "
                f"{ls.kv('fallback', 'deterministic-only', vc=ls.LY)}"
            )
            filtered = []

        return [*deterministic, *filtered]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_recent_authors(self, request: ContextRequest) -> list:
        """Pull N most-recent distinct users from history, if wired."""
        if self._history_store is None or self._recent_author_limit <= 0:
            return []
        return self._history_store.recent_distinct_authors(
            familiar_id=request.familiar_id,
            channel_id=request.channel_id,
            limit=self._recent_author_limit,
        )

    def _maybe_start_index_build(self) -> None:
        if self._index_build is None or self._build_task is not None:
            return
        self._build_task = asyncio.create_task(self._run_index_build())

    async def _run_index_build(self) -> None:
        assert self._index_build is not None  # noqa: S101 — guarded by caller
        try:
            await self._index_build()
        except Exception:
            _logger.exception(
                f"{ls.tag('📚 Content', ls.M)} {ls.word('index build failed', ls.LW)}"
            )

    async def _retrieve(
        self,
        request: ContextRequest,
        *,
        exclude: set[str],
    ) -> list:
        if self._retriever is None:
            return []
        try:
            return await self._retriever.query(
                request.utterance,
                top_k=self._top_k,
                exclude_paths=exclude,
            )
        except Exception:
            _logger.exception(
                f"{ls.tag('📚 Content', ls.M)} {ls.word('retriever raised', ls.LW)}"
            )
            return []
