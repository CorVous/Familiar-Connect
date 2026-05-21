"""Memory-projector registry (M5).

Lifts watermark-driven writers — :class:`SummaryWorker`,
:class:`FactExtractor`, :class:`PeopleDossierWorker`,
:class:`ReflectionWorker` — behind ``MemoryProjector`` Protocol so
operators can swap or extend strategy via TOML:

.. code:: toml

   [providers.memory]
   projectors = ["rolling_summary", "rich_note", "people_dossier", "reflection"]

Default keeps every shipped projector. Third-party projectors
(Graphiti, Cognee) register alongside via
:func:`register_projector` — same call signature as built-ins.

Design constraints:

* Each projector exposes ``name: str`` (label for logs / TaskGroup
  names) and ``async def run(self) -> None`` (forever loop). That's
  it — registry is intentionally thin.
* Factories take :class:`ProjectorContext` so registry stays
  decoupled from per-projector constructor signatures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from familiar_connect.processors.fact_embedding_worker import FactEmbeddingWorker
from familiar_connect.processors.fact_extractor import FactExtractor
from familiar_connect.processors.fact_supersede_worker import FactSupersedeWorker
from familiar_connect.processors.people_dossier_worker import PeopleDossierWorker
from familiar_connect.processors.reflection_worker import ReflectionWorker
from familiar_connect.processors.summary_worker import SummaryWorker

if TYPE_CHECKING:
    from familiar_connect.embedding.protocol import Embedder
    from familiar_connect.history.async_store import AsyncHistoryStore
    from familiar_connect.llm import LLMClient


class MemoryProjector(Protocol):
    """Backend writer projecting ``turns`` into a side-index.

    Implementations run as long-lived asyncio task. ``run`` is the
    forever loop; cancellation stops the task. Shipped projectors are
    watermark-driven and idempotent — each side-index regenerable
    from ``turns``. Third-party projector (Graphiti, Cognee) plugs in
    here.
    """

    name: str

    async def run(self) -> None: ...


@dataclass(frozen=True)
class ProjectorContext:
    """Inputs available to projector factories.

    Built once in ``commands/run.py``, reused across every factory in
    :data:`_REGISTRY`. Adding a new input here is the extension
    point — callers don't need to know which projector consumes which
    field.

    :param embedder: text → vector seam for M6 ``fact_embedding``
        projector. ``None`` when ``[providers.embedding].backend =
        "off"`` (default); ``fact_embedding`` factory raises config
        error in that case so a typo never silently disables semantic
        recall.
    """

    store: AsyncHistoryStore
    llm_clients: dict[str, LLMClient]
    familiar_id: str
    embedder: Embedder | None = None


ProjectorFactory = Callable[[ProjectorContext], MemoryProjector]


_REGISTRY: dict[str, ProjectorFactory] = {}


def register_projector(name: str, factory: ProjectorFactory) -> None:
    """Register *factory* under *name*. Re-registration overwrites.

    Third-party projectors call at import time;
    ``[providers.memory].projectors`` selector picks them up by name.
    """
    _REGISTRY[name] = factory


def known_projectors() -> set[str]:
    """Names registered today (built-ins plus any third-party additions)."""
    return set(_REGISTRY)


def create_projectors(
    *,
    names: list[str],
    context: ProjectorContext,
) -> list[MemoryProjector]:
    """Instantiate selected projectors in *names* order.

    :raises ValueError: when any name is not in :data:`_REGISTRY`.
    """
    out: list[MemoryProjector] = []
    for name in names:
        factory = _REGISTRY.get(name)
        if factory is None:
            valid = ", ".join(sorted(_REGISTRY)) or "(none)"
            msg = f"unknown memory projector {name!r}; valid: {valid}"
            raise ValueError(msg)
        out.append(factory(context))
    return out


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def _summary_factory(ctx: ProjectorContext) -> MemoryProjector:
    return SummaryWorker(
        store=ctx.store,
        llm_client=ctx.llm_clients["background"],
        familiar_id=ctx.familiar_id,
        turns_threshold=10,
    )


def _rich_note_factory(ctx: ProjectorContext) -> MemoryProjector:
    return FactExtractor(
        store=ctx.store,
        llm_client=ctx.llm_clients["background"],
        familiar_id=ctx.familiar_id,
        batch_size=10,
    )


def _people_dossier_factory(ctx: ProjectorContext) -> MemoryProjector:
    return PeopleDossierWorker(
        store=ctx.store,
        llm_client=ctx.llm_clients["background"],
        familiar_id=ctx.familiar_id,
    )


def _reflection_factory(ctx: ProjectorContext) -> MemoryProjector:
    return ReflectionWorker(
        store=ctx.store,
        llm_client=ctx.llm_clients["background"],
        familiar_id=ctx.familiar_id,
    )


def _fact_supersede_factory(ctx: ProjectorContext) -> MemoryProjector:
    return FactSupersedeWorker(
        store=ctx.store,
        llm_client=ctx.llm_clients["background"],
        familiar_id=ctx.familiar_id,
    )


def _fact_embedding_factory(ctx: ProjectorContext) -> MemoryProjector:
    if ctx.embedder is None:
        msg = (
            "fact_embedding projector requires a configured embedder. "
            'set [providers.embedding].backend to a backend other than "off" '
            '(e.g. "hash") and restart.'
        )
        raise ValueError(msg)
    return FactEmbeddingWorker(
        store=ctx.store,
        embedder=ctx.embedder,
        familiar_id=ctx.familiar_id,
    )


register_projector("rolling_summary", _summary_factory)
register_projector("rich_note", _rich_note_factory)
register_projector("people_dossier", _people_dossier_factory)
register_projector("reflection", _reflection_factory)
register_projector("fact_supersede", _fact_supersede_factory)
register_projector("fact_embedding", _fact_embedding_factory)


DEFAULT_PROJECTORS: tuple[str, ...] = (
    "rolling_summary",
    "rich_note",
    "people_dossier",
    "reflection",
    "fact_supersede",
)
"""Names enabled when ``[providers.memory].projectors`` is unset.

``fact_embedding`` registered but **not** in the default tuple —
M6 stays opt-in so default deployments don't need an embedder
configured. Operators add it to their projector list once they've
picked a backend in ``[providers.embedding]``.
"""
