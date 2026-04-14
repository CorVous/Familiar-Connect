"""Protocol for metrics collection sinks.

``record`` is synchronous and must not block the event loop — implementations
buffer in memory and flush on a background thread, or simply discard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from familiar_connect.metrics.types import TurnTrace


@runtime_checkable
class MetricsCollector(Protocol):
    """Sink for completed :class:`TurnTrace` records."""

    def record(self, trace: TurnTrace) -> None:
        """Record a completed turn trace. Must not block the event loop."""
        ...

    def close(self) -> None:
        """Flush pending writes and release resources."""
        ...


class NullCollector:
    """No-op collector. Default when metrics are disabled."""

    def record(self, trace: TurnTrace) -> None:
        """Discard the trace."""

    def close(self) -> None:
        """No-op."""
