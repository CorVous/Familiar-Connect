"""SideModel — the cheap LLM client used by providers and processors.

A deliberately tiny ``Protocol`` so providers can be tested with
scripted stubs and the production implementation (a thin wrapper
around the existing OpenRouter :class:`LLMClient` configured with a
cheaper model) can be swapped without touching them.

The interface is one async method ``complete(prompt, *, max_tokens)
-> str`` plus an ``id`` attribute. Both summarisation (history,
lorebook, recast) and reasoning (stepped thinking) tasks fit naturally
under "give me text, get text back" — chat-style multi-turn input is
not in scope here, because every caller in the context layer assembles
its own prompt out of context they already control.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SideModel(Protocol):
    """A cheap LLM client used by providers and processors.

    Implementors declare a short ``id`` (used for logging and per-
    guild config lookups) and a single async method ``complete``.
    """

    id: str

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
    ) -> str: ...
