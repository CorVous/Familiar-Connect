"""Structural Protocols for sources, processors, bus itself.

Separate from concrete impls so future process-spanning ``EventBus``
drops in without touching processor code. See plan § Design.1,
§ Design.2.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from familiar_connect.bus.envelope import Event


class BackpressurePolicy(Enum):
    """Per-topic behaviour when subscriber's queue full.

    - :attr:`BLOCK` — ``publish`` awaits until space frees. Use for
      data that must not be lost.
    - :attr:`DROP_OLDEST` — evict oldest queued, enqueue new. Use for
      hot streams where freshness beats completeness (audio).
    - :attr:`DROP_NEWEST` — drop incoming. Use when downstream stall
      must not affect other subscribers and late data worse than
      missing data.
    - :attr:`UNBOUNDED` — no backpressure; caller accepts memory risk.
      Use for low-volume channels where dropping is costly.
    """

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    UNBOUNDED = "unbounded"


@runtime_checkable
class StreamSource(Protocol):
    """Produces events onto bus. Registered by runtime."""

    name: str

    async def run(self, bus: EventBus) -> None:
        """Run until bus drains; return cleanly on cancel."""
        ...


@runtime_checkable
class Processor(Protocol):
    """Subscribes to one or more topics; optionally re-publishes.

    ``topics`` consulted once at registration; dynamic subscription
    out of scope for v1.
    """

    name: str
    topics: tuple[str, ...]

    async def handle(self, event: Event, bus: EventBus) -> None:
        """Handle single event.

        Raised exceptions logged and swallowed by dispatcher — do not
        rely on them for flow control.
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
        """Async iterator of events for given topics.

        :param maxsize: ``0`` means default-for-policy.
        """
        ...
