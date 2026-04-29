"""Per-turn voice latency budget recorder.

Stamps phase markers across the voice path (``stt_final`` →
``llm_first_token`` → ``tts_first_audio`` → ``playback_start``) keyed
by ``turn_id``; emits one span per adjacent gap into the shared
:class:`SpanCollector`, plus a cumulative ``voice.total`` when the
funnel completes. ``/diagnostics`` picks them up via the existing
summary table — no new UI plumbing needed.

First record per (turn, phase) wins: sentence streaming records
``tts_first_audio`` ahead of every sentence flush, but only the first
counts as time-to-first-audio.

Implementation: small in-memory ring (``max_turns`` recent turns) so
late phase events on a long-defunct turn don't grow unbounded.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Final

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.collector import get_span_collector

_logger = logging.getLogger("familiar_connect.diagnostics.voice_budget")

PHASE_STT_FINAL: Final = "stt_final"
PHASE_LLM_FIRST_TOKEN: Final = "llm_first_token"  # noqa: S105 — phase label, not a credential
PHASE_TTS_FIRST_AUDIO: Final = "tts_first_audio"
PHASE_PLAYBACK_START: Final = "playback_start"

# Adjacent-phase gaps. ``(prev, curr, span_name)``; emitted when
# ``curr`` is recorded and ``prev`` was already present.
_GAPS: Final[tuple[tuple[str, str, str], ...]] = (
    (PHASE_STT_FINAL, PHASE_LLM_FIRST_TOKEN, "voice.stt_to_ttft"),
    (PHASE_LLM_FIRST_TOKEN, PHASE_TTS_FIRST_AUDIO, "voice.ttft_to_tts"),
    (PHASE_TTS_FIRST_AUDIO, PHASE_PLAYBACK_START, "voice.tts_to_playback"),
)
SPAN_TOTAL: Final = "voice.total"


class VoiceBudgetRecorder:
    """Per-turn phase timestamps; emits gap spans on phase advance."""

    def __init__(self, *, max_turns: int = 32) -> None:
        self._max_turns = max_turns
        self._turns: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._lock = Lock()

    def record(self, *, turn_id: str, phase: str, t: float | None = None) -> None:
        """Stamp ``phase`` for ``turn_id``; emit any newly-reachable gap span.

        ``t`` defaults to :func:`time.perf_counter`. First record per
        (turn, phase) wins — duplicates are dropped.
        """
        if t is None:
            t = time.perf_counter()
        with self._lock:
            phases = self._turns.get(turn_id)
            if phases is None:
                phases = {}
                self._turns[turn_id] = phases
                while len(self._turns) > self._max_turns:
                    self._turns.popitem(last=False)
            else:
                self._turns.move_to_end(turn_id)
            if phase in phases:
                return
            phases[phase] = t
            self._emit_gaps(turn_id, phases, phase)

    def discard(self, turn_id: str) -> None:
        """Drop a turn's phase state. No-op if unknown."""
        with self._lock:
            self._turns.pop(turn_id, None)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _emit_gaps(self, turn_id: str, phases: dict[str, float], current: str) -> None:
        for prev, curr, name in _GAPS:
            if curr != current or prev not in phases:
                continue
            ms = max(0, round((phases[curr] - phases[prev]) * 1000))
            self._emit(turn_id, name, ms)
        # cumulative total fires once on terminal phase if the funnel
        # started cleanly. Useful as the user-perceived latency signal.
        if current == PHASE_PLAYBACK_START and PHASE_STT_FINAL in phases:
            ms = max(
                0,
                round((phases[PHASE_PLAYBACK_START] - phases[PHASE_STT_FINAL]) * 1000),
            )
            self._emit(turn_id, SPAN_TOTAL, ms)

    @staticmethod
    def _emit(turn_id: str, name: str, ms: int) -> None:
        _logger.info(
            f"{ls.tag('budget', ls.LM)} "
            f"{ls.kv('turn', turn_id, vc=ls.LC)} "
            f"{ls.kv('span', name, vc=ls.LM)} "
            f"{ls.kv('ms', str(ms), vc=ls.LC)}"
        )
        # Span recording must never raise into the voice path.
        with contextlib.suppress(Exception):
            get_span_collector().record(name=name, ms=ms, status="ok")


# ---------------------------------------------------------------------------
# Module-level singleton — mirrors :func:`get_span_collector`.
# ---------------------------------------------------------------------------

_recorder: VoiceBudgetRecorder | None = None


def get_voice_budget_recorder() -> VoiceBudgetRecorder:
    """Return the process-wide recorder, creating on first use."""
    global _recorder  # noqa: PLW0603
    if _recorder is None:
        _recorder = VoiceBudgetRecorder()
    return _recorder


def reset_voice_budget_recorder() -> None:
    """Reset the singleton — for tests only."""
    global _recorder  # noqa: PLW0603
    _recorder = None
