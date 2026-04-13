"""Runtime-checkable protocols for context providers and processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest, Contribution


class PreProcessorError(RuntimeError):
    """Signals a ``PreProcessor.process`` failure the pipeline should isolate.

    Only allowed exception from ``process``; anything else is a contract
    violation and propagates intentionally.
    """


@runtime_checkable
class ContextProvider(Protocol):
    """Produces ``Contribution``s for a single pipeline run.

    Pipeline enforces ``deadline_s`` via ``asyncio.timeout``; misses
    are recorded as ``"timeout"`` outcomes, not awaited.
    """

    id: str
    deadline_s: float

    async def contribute(
        self,
        request: ContextRequest,
    ) -> list[Contribution]: ...


@runtime_checkable
class PreProcessor(Protocol):
    """Mutates the ``ContextRequest`` before providers run.

    Canonical example: stepped-thinking pass that appends a hidden
    chain-of-thought. Runs sequentially in registration order.

    Raise contract: ``process`` may raise :class:`PreProcessorError`
    (isolated, logged, skipped). Any other exception propagates.
    """

    id: str

    async def process(self, request: ContextRequest) -> ContextRequest: ...


@runtime_checkable
class PostProcessor(Protocol):
    """Mutates the main LLM's reply before it reaches TTS.

    Wired via :meth:`ContextPipeline.run_post_processors`. Canonical
    examples: recast cleanup pass, voice-friendly rewrites.
    """

    id: str

    async def process(
        self,
        reply_text: str,
        request: ContextRequest,
    ) -> str: ...
