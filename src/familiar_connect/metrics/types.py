"""Data model for per-turn performance traces.

``TurnTrace`` is the central record. Extensibility via ``tags`` (top-level
KV) and ``StageSpan.metadata`` (per-stage KV) — new dimensions added
without schema migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageSpan:
    """Timing record for one pipeline stage.

    :param name: stage identifier (e.g. ``"pipeline_assembly"``, ``"llm_call"``)
    :param start_s: offset from turn start, monotonic seconds
    :param duration_s: wall-clock duration
    :param metadata: stage-specific KV (tokens, model, status, provider outcomes)
    """

    name: str
    start_s: float
    duration_s: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnTrace:
    """Complete timing record for one reply turn.

    ``tags`` is the open-ended extensibility hook — A/B testing, prompt
    variants, quality scores go here without schema changes.
    """

    trace_id: str
    familiar_id: str
    channel_id: int
    guild_id: int | None
    modality: str
    timestamp: str
    total_duration_s: float
    stages: tuple[StageSpan, ...]
    tags: dict[str, str] = field(default_factory=dict)
