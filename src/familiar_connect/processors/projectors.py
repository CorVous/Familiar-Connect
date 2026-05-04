"""Memory-projector registry (M5).

Lifts the watermark-driven writers — :class:`SummaryWorker`,
:class:`FactExtractor`, :class:`PeopleDossierWorker`,
:class:`ReflectionWorker` — behind a ``MemoryProjector`` Protocol so
that operators can swap or extend the strategy via TOML:

.. code:: toml

   [providers.memory]
   projectors = ["rolling_summary", "rich_note", "people_dossier", "reflection"]

Default keeps every shipped projector. Third-party projectors
(Graphiti, Cognee) register themselves alongside via
:func:`register_projector` — same call signature as the built-ins.

Design constraints:

* Each projector exposes ``name: str`` (label for logs / TaskGroup
  names) and ``async def run(self) -> None`` (forever loop). That's
  it — the registry is intentionally thin.
* Factories take a :class:`ProjectorContext` so the registry stays
  decoupled from per-projector constructor signatures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from familiar_connect.processors.fact_extractor import FactExtractor
from familiar_connect.processors.people_dossier_worker import PeopleDossierWorker
from familiar_connect.processors.reflection_worker import ReflectionWorker
from familiar_connect.processors.summary_worker import SummaryWorker

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.llm import LLMClient


class MemoryProjector(Protocol):
    """Backend writer projecting ``turns`` into a side-index.

    Implementations run as a long-lived asyncio task. ``run`` is the
    forever loop; cancellation stops the task. The shipped projectors
    are watermark-driven and idempotent — each side-index is
    regenerable from ``turns``. A third-party projector (Graphiti,
    Cognee) plugs in here.
    """

    name: str

    async def run(self) -> None: ...


@dataclass(frozen=True)
class ProjectorContext:
    """Inputs available to projector factories.

    Built once in ``commands/run.py`` and reused across every
    factory in :data:`_REGISTRY`. Adding a new input here is the
    extension point — callers don't need to know which projector
    consumes which field.
    """

    store: HistoryStore
    llm_clients: dict[str, LLMClient]
    familiar_id: str


ProjectorFactory = Callable[[ProjectorContext], MemoryProjector]


_REGISTRY: dict[str, ProjectorFactory] = {}


def register_projector(name: str, factory: ProjectorFactory) -> None:
    """Register *factory* under *name*. Re-registration overwrites.

    Third-party projectors call this at import time; the
    ``[providers.memory].projectors`` selector then picks them up by
    name.
    """
    _REGISTRY[name] = factory


def known_projectors() -> set[str]:
    """Names registered today (built-ins + any third-party additions)."""
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


register_projector("rolling_summary", _summary_factory)
register_projector("rich_note", _rich_note_factory)
register_projector("people_dossier", _people_dossier_factory)
register_projector("reflection", _reflection_factory)


DEFAULT_PROJECTORS: tuple[str, ...] = (
    "rolling_summary",
    "rich_note",
    "people_dossier",
    "reflection",
)
"""Names enabled when ``[providers.memory].projectors`` is unset."""
