"""Structural Protocols for sources, processors, and the bus itself.

Keeping these separate from concrete impls lets a future process-
spanning ``EventBus`` drop in without touching processor code. See
plan § Design.1, § Design.2.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from familiar_connect.bus.envelope import Event


class BackpressurePolicy(Enum):
    """Per-topic behaviour when a subscriber's queue is full.

    - :attr:`BLOCK` — ``publish`` awaits until space frees. Use for
      data that must not be lost.
    - :attr:`DROP_OLDEST` — evict the oldest queued event, enqueue
      the new one. Use for hot streams where freshness matters more
      than completeness (audio).
    - :attr:`DROP_NEWEST` — drop the incoming event. Use when a
      downstream stall should not affect other subscribers and late
      data is worse than missing data.
    - :attr:`UNBOUNDED` — no backpressure; caller accepts the memory
      risk. Use for low-volume channels where dropping is costly.
    """

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    UNBOUNDED = "unbounded"


@runtime_checkable
class StreamSource(Protocol):
    """Produces events onto the bus. Registered by the runtime."""

    name: str

    async def run(self, bus: EventBus) -> None:
        """Run until the bus drains; return cleanly on cancel."""
        ...


@runtime_checkable
class Processor(Protocol):
    """Subscribes to one or more topics; optionally re-publishes.

    ``topics`` is consulted once at registration; dynamic subscription
    is out of scope for v1.
    """

    name: str
    topics: tuple[str, ...]

    async def handle(self, event: Event, bus: EventBus) -> None:
        """Handle a single event.

        Raised exceptions are logged and swallowed by the dispatcher —
        do not rely on them for flow control.
        """
        ...


class EventBus(Protocol):
    """Topic-addressed pub/sub surface."""

    async def start(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def publish(self, event: Event) -> None: ...

    def subscribe(
        self,
        topics: tuple[str, ...],
        *,
        policy: BackpressurePolicy = BackpressurePolicy.BLOCK,
        maxsize: int = 0,
    ) -> AsyncIterator[Event]:
        """Return an async iterator of events for the given topics.

        :param maxsize: ``0`` means default-for-policy.
        """
        ...
