"""TraceBuilder — per-turn timing accumulator with log bridge.

Use :meth:`TraceBuilder.span` as an async context manager around any
stage you want to measure. The span:

- captures wall-clock duration (monotonic)
- yields a mutable ``metadata`` dict for post-hoc enrichment
- emits a DEBUG log line on exit (name, duration, metadata)
- appends a :class:`StageSpan` to the trace

:meth:`TraceBuilder.finalize` freezes the trace and emits an INFO
summary line. Hand the returned :class:`TurnTrace` to a
:class:`MetricsCollector` for persistence.

Never dual-write timing through ``_logger`` — the span handles both
the log line and the persisted record.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect.metrics.types import StageSpan, TurnTrace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from familiar_connect.context.pipeline import ProviderOutcome

_logger = logging.getLogger(__name__)


class TraceBuilder:
    """Mutable accumulator for a single :class:`TurnTrace`.

    Create one per turn. Wrap each stage in :meth:`span`. Call
    :meth:`finalize` to freeze and retrieve the trace.
    """

    def __init__(
        self,
        *,
        familiar_id: str,
        channel_id: int,
        guild_id: int | None,
        modality: str,
    ) -> None:
        self._familiar_id = familiar_id
        self._channel_id = channel_id
        self._guild_id = guild_id
        self._modality = modality
        self._trace_id = uuid.uuid4().hex
        self._timestamp = datetime.now(tz=UTC).isoformat()
        self._turn_start = time.monotonic()
        self._stages: list[StageSpan] = []
        self._tags: dict[str, str] = {}

    @contextlib.asynccontextmanager
    async def span(
        self,
        name: str,
        **metadata: object,
    ) -> AsyncIterator[dict[str, object]]:
        """Time a named stage.

        Caller may add keys after the yield point (e.g. ``meta["tokens_out"]
        = 420`` once the LLM call returns). On exit, a DEBUG log line
        with name, duration, and metadata is emitted, and a
        :class:`StageSpan` is appended to the trace — even if the body raised.

        Yields:
            The mutable metadata dict for post-hoc enrichment.

        """
        start = time.monotonic()
        extra: dict[str, object] = dict(metadata)
        try:
            yield extra
        finally:
            duration = time.monotonic() - start
            offset = start - self._turn_start
            self._stages.append(
                StageSpan(
                    name=name,
                    start_s=offset,
                    duration_s=duration,
                    metadata=extra,
                ),
            )
            _logger.debug(
                "metrics span %s duration=%.3fs %s",
                name,
                duration,
                _format_metadata(extra),
            )

    def tag(self, key: str, value: str) -> None:
        """Attach a string tag to the trace (A/B test groups, quality, etc.)."""
        self._tags[key] = value

    @staticmethod
    def add_provider_outcomes(
        metadata: dict[str, object],
        outcomes: Sequence[ProviderOutcome],
    ) -> None:
        """Embed :class:`ProviderOutcome` list into a span's metadata.

        Stores a plain-dict projection under ``"provider_outcomes"`` so
        the trace is JSON-serializable.
        """
        metadata["provider_outcomes"] = [
            {
                "id": o.provider_id,
                "duration_s": o.duration_s,
                "status": o.status,
                "error": o.error_message,
            }
            for o in outcomes
        ]

    def finalize(self) -> TurnTrace:
        """Freeze and return the trace. Emits an INFO summary log line."""
        total = time.monotonic() - self._turn_start
        trace = TurnTrace(
            trace_id=self._trace_id,
            familiar_id=self._familiar_id,
            channel_id=self._channel_id,
            guild_id=self._guild_id,
            modality=self._modality,
            timestamp=self._timestamp,
            total_duration_s=total,
            stages=tuple(self._stages),
            tags=dict(self._tags),
        )
        _logger.info(
            "metrics turn trace_id=%s channel=%s modality=%s total=%.3fs stages=%d",
            self._trace_id,
            self._channel_id,
            self._modality,
            total,
            len(self._stages),
        )
        return trace


def _format_metadata(metadata: dict[str, object]) -> str:
    """Compact ``key=value`` joining for log lines; empty dict -> ``""``."""
    if not metadata:
        return ""
    parts = [f"{k}={v}" for k, v in metadata.items()]
    return "[" + ", ".join(parts) + "]"
