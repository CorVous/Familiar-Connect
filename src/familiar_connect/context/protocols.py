"""Protocols for context providers and processors.

All three protocols are ``@runtime_checkable`` so tests (and guild
configuration) can assert conformance without the implementor having
to declare an explicit base class. They are deliberately tiny — a
single async method plus an ``id`` attribute — because the whole point
of the pipeline is that plugging new providers and processors in is
cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest, Contribution


class PreProcessorError(RuntimeError):
    """Signals a ``PreProcessor.process`` failure the pipeline should isolate.

    The :class:`PreProcessor` protocol permits ``process()`` to raise
    this one type and no other. Any other exception escaping
    ``process`` is a contract violation and will propagate out of the
    pipeline — this is intentional so contract violations surface
    loudly rather than being masked by a blanket ``except Exception``.
    """


@runtime_checkable
class ContextProvider(Protocol):
    """Produces ``Contribution``s for a single pipeline run.

    Implementors declare a short ``id`` (used for logging and per-guild
    config lookups) and a ``deadline_s`` wall-clock cap. The pipeline
    enforces the deadline with ``asyncio.timeout``; providers that miss
    it are dropped and recorded as ``"timeout"`` outcomes rather than
    blocking the reply.
    """

    id: str
    deadline_s: float

    async def contribute(
        self,
        request: ContextRequest,
    ) -> list[Contribution]: ...


@runtime_checkable
class PreProcessor(Protocol):
    """Mutates the outgoing ``ContextRequest`` before providers run.

    The canonical example is a "stepped thinking" pass that calls a
    cheap model to produce a hidden chain-of-thought, then appends it
    to the request so downstream providers and the main LLM can see
    it. Pre-processors run sequentially in registration order; each
    one receives the previous one's output.

    **Raise contract.** ``process`` may raise :class:`PreProcessorError`
    to signal a failure it wants the pipeline to isolate (the pipeline
    will log it at warning level and skip this processor, passing the
    unmodified request to the next stage). Any other exception is a
    protocol violation and will propagate out of
    :meth:`ContextPipeline.assemble`.
    """

    id: str

    async def process(self, request: ContextRequest) -> ContextRequest: ...


@runtime_checkable
class PostProcessor(Protocol):
    """Mutates the main LLM's reply before it reaches TTS.

    Not yet wired into the pipeline — lives here so later roadmap
    steps have a stable shape to target. Canonical examples are the
    "recast" cleanup pass and voice-friendly rewrites.
    """

    id: str

    async def process(
        self,
        reply_text: str,
        request: ContextRequest,
    ) -> str: ...
