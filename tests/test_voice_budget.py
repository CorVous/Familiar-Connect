"""Tests for :class:`VoiceBudgetRecorder` — per-turn voice latency spans.

Stamps phases (``stt_final`` / ``llm_first_token`` / ``tts_first_audio``
/ ``playback_start``) keyed by ``turn_id``; emits one span per
adjacent gap into the shared :class:`SpanCollector`.
"""

from __future__ import annotations

import pytest

from familiar_connect.diagnostics.collector import (
    SpanRecord,
    get_span_collector,
    reset_span_collector,
)
from familiar_connect.diagnostics.voice_budget import (
    PHASE_LLM_FIRST_TOKEN,
    PHASE_PLAYBACK_START,
    PHASE_STT_FINAL,
    PHASE_TTS_FIRST_AUDIO,
    PHASE_VAD_END,
    VoiceBudgetRecorder,
    get_voice_budget_recorder,
    reset_voice_budget_recorder,
)


@pytest.fixture(autouse=True)
def _isolate_singletons() -> None:
    reset_voice_budget_recorder()
    reset_span_collector()


def _names(records: list[SpanRecord]) -> list[str]:
    return [r.name for r in records]


class TestSequentialPhases:
    def test_stt_to_ttft_emits_on_llm_first_token(self) -> None:
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL, t=10.000)
        # Nothing recorded yet — only one phase.
        assert _names(get_span_collector().all()) == []
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=10.250)
        recorded = get_span_collector().all()
        assert _names(recorded) == ["voice.stt_to_ttft"]
        assert recorded[0].ms == 250  # type: ignore[attr-defined]

    def test_full_funnel_emits_three_gaps_plus_total(self) -> None:
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL, t=10.000)
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=10.300)
        rec.record(turn_id="t-1", phase=PHASE_TTS_FIRST_AUDIO, t=10.450)
        rec.record(turn_id="t-1", phase=PHASE_PLAYBACK_START, t=10.700)
        names = _names(get_span_collector().all())
        # Gaps emit on each later phase; total emits with playback_start.
        assert names == [
            "voice.stt_to_ttft",
            "voice.ttft_to_tts",
            "voice.tts_to_playback",
            "voice.total",
        ]
        # Span values.
        by_name = get_span_collector().by_name()
        assert by_name["voice.stt_to_ttft"][0].ms == 300
        assert by_name["voice.ttft_to_tts"][0].ms == 150
        assert by_name["voice.tts_to_playback"][0].ms == 250
        assert by_name["voice.total"][0].ms == 700

    def test_skipped_first_phase_no_emit(self) -> None:
        """``llm_first_token`` without ``stt_final`` produces no gap span."""
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=10.0)
        rec.record(turn_id="t-1", phase=PHASE_TTS_FIRST_AUDIO, t=10.1)
        names = _names(get_span_collector().all())
        assert names == ["voice.ttft_to_tts"]


class TestDeduplication:
    def test_duplicate_phase_first_wins(self) -> None:
        """Re-recording a phase doesn't move the timestamp.

        Sentence streaming records ``tts_first_audio`` before each
        ``speak()``; only the first sentence's timestamp counts.
        """
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=10.000)
        rec.record(turn_id="t-1", phase=PHASE_TTS_FIRST_AUDIO, t=10.100)
        # Later sentence — should NOT overwrite.
        rec.record(turn_id="t-1", phase=PHASE_TTS_FIRST_AUDIO, t=10.500)
        rec.record(turn_id="t-1", phase=PHASE_PLAYBACK_START, t=10.200)
        by_name = get_span_collector().by_name()
        # ttft_to_tts gap reflects the first call (10.000 → 10.100), not
        # the second (would be 500 ms).
        assert by_name["voice.ttft_to_tts"][0].ms == 100
        assert by_name["voice.tts_to_playback"][0].ms == 100


class TestPerTurnIsolation:
    def test_two_turns_compute_independently(self) -> None:
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-A", phase=PHASE_STT_FINAL, t=100.0)
        rec.record(turn_id="t-B", phase=PHASE_STT_FINAL, t=200.0)
        rec.record(turn_id="t-B", phase=PHASE_LLM_FIRST_TOKEN, t=200.05)
        rec.record(turn_id="t-A", phase=PHASE_LLM_FIRST_TOKEN, t=100.30)
        by_name = get_span_collector().by_name()
        gaps = sorted(r.ms for r in by_name["voice.stt_to_ttft"])
        assert gaps == [50, 300]


class TestEviction:
    def test_oldest_turn_dropped_past_capacity(self) -> None:
        """Late phase event on an evicted turn emits no gap.

        Bounds in-memory state without changing the user-visible
        contract — late events on evicted turns just look like turns
        with a missing prior phase.
        """
        rec = VoiceBudgetRecorder(max_turns=2)
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL, t=1.0)
        rec.record(turn_id="t-2", phase=PHASE_STT_FINAL, t=2.0)
        rec.record(turn_id="t-3", phase=PHASE_STT_FINAL, t=3.0)
        # t-1 evicted; its prior phase is gone, so no gap is emitted.
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=4.0)
        assert _names(get_span_collector().all()) == []

    def test_discard_removes_turn(self) -> None:
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL, t=1.0)
        rec.discard("t-1")
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=1.5)
        # No prior phase — no gap emitted.
        assert _names(get_span_collector().all()) == []


class TestTotalSpan:
    def test_total_skipped_if_stt_final_missing(self) -> None:
        """Without an ``stt_final`` timestamp, ``voice.total`` is undefined."""
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_TTS_FIRST_AUDIO, t=10.0)
        rec.record(turn_id="t-1", phase=PHASE_PLAYBACK_START, t=10.2)
        names = _names(get_span_collector().all())
        assert "voice.total" not in names


class TestVadEnd:
    """Local-VAD turn boundary stamps ahead of ``stt_final``.

    V1 phase 2 wires TEN-VAD + Smart Turn into the audio pump; the
    moment Smart Turn classifies a turn complete is the new
    ``vad_end`` mark. ``voice.vad_to_stt`` exposes the STT lag
    between local endpointing and Deepgram's final.
    """

    def test_vad_to_stt_emits_on_stt_final(self) -> None:
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_VAD_END, t=10.000)
        assert _names(get_span_collector().all()) == []
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL, t=10.180)
        recorded = get_span_collector().all()
        assert _names(recorded) == ["voice.vad_to_stt"]
        assert recorded[0].ms == 180

    def test_vad_end_optional_no_emit_without_it(self) -> None:
        """``stt_final`` without ``vad_end`` doesn't emit ``voice.vad_to_stt``."""
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL, t=10.0)
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=10.2)
        names = _names(get_span_collector().all())
        assert "voice.vad_to_stt" not in names
        # Existing chain still works.
        assert names == ["voice.stt_to_ttft"]

    def test_full_funnel_with_vad_end(self) -> None:
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_VAD_END, t=10.000)
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL, t=10.150)
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN, t=10.450)
        rec.record(turn_id="t-1", phase=PHASE_TTS_FIRST_AUDIO, t=10.600)
        rec.record(turn_id="t-1", phase=PHASE_PLAYBACK_START, t=10.850)
        names = _names(get_span_collector().all())
        assert names == [
            "voice.vad_to_stt",
            "voice.stt_to_ttft",
            "voice.ttft_to_tts",
            "voice.tts_to_playback",
            "voice.total",
        ]
        by_name = get_span_collector().by_name()
        assert by_name["voice.vad_to_stt"][0].ms == 150
        # ``voice.total`` keeps its existing ``stt_final`` start so old
        # comparisons stay meaningful.
        assert by_name["voice.total"][0].ms == 700


class TestSingleton:
    def test_singleton_returns_same_instance(self) -> None:
        a = get_voice_budget_recorder()
        b = get_voice_budget_recorder()
        assert a is b

    def test_reset_yields_fresh_instance(self) -> None:
        a = get_voice_budget_recorder()
        reset_voice_budget_recorder()
        b = get_voice_budget_recorder()
        assert a is not b


class TestPerfCounterFallback:
    def test_record_without_explicit_t_uses_clock(self) -> None:
        """Omitting ``t`` stamps the current monotonic clock.

        Real callers (responder, sources) won't pass ``t``; tests above
        do for determinism. Sanity-check the default path emits.
        """
        rec = VoiceBudgetRecorder()
        rec.record(turn_id="t-1", phase=PHASE_STT_FINAL)
        rec.record(turn_id="t-1", phase=PHASE_LLM_FIRST_TOKEN)
        names = _names(get_span_collector().all())
        assert names == ["voice.stt_to_ttft"]
