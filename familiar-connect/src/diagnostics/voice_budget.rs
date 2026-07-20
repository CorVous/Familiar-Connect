//! Per-turn voice latency budget recorder.
//!
//! Port of `familiar_connect/diagnostics/voice_budget.py`. Stamps phase markers
//! across the voice path (`vad_end` → `stt_final` → `llm_first_token` →
//! `tts_first_audio` → `playback_start`) keyed by `turn_id`; emits one span per
//! adjacent gap into the shared [`SpanCollector`](super::collector), plus a
//! cumulative `voice.total` when the funnel completes.
//!
//! `vad_end` is optional (only stamped when local turn detection is wired in);
//! without it the funnel starts at `stt_final`. First record per (turn, phase)
//! wins. State is a small LRU of recent turns so late events on defunct turns
//! don't grow unbounded.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Instant;

use crate::diagnostics::collector::get_span_collector;
use crate::log_style as ls;

/// Local-VAD turn boundary (optional; ahead of `stt_final`).
pub const PHASE_VAD_END: &str = "vad_end";
/// Final transcript available.
pub const PHASE_STT_FINAL: &str = "stt_final";
/// First LLM stream delta.
pub const PHASE_LLM_FIRST_TOKEN: &str = "llm_first_token";
/// First synthesized audio chunk.
pub const PHASE_TTS_FIRST_AUDIO: &str = "tts_first_audio";
/// Playback started.
pub const PHASE_PLAYBACK_START: &str = "playback_start";

/// Cumulative user-perceived latency span name.
pub const SPAN_TOTAL: &str = "voice.total";

/// Default LRU capacity (recent turns kept in memory).
const DEFAULT_MAX_TURNS: usize = 32;

// Adjacent-phase gaps: `(prev, curr, span_name)`; emitted when `curr` is
// recorded and `prev` was already present.
const GAPS: [(&str, &str, &str); 4] = [
    (PHASE_VAD_END, PHASE_STT_FINAL, "voice.vad_to_stt"),
    (PHASE_STT_FINAL, PHASE_LLM_FIRST_TOKEN, "voice.stt_to_ttft"),
    (
        PHASE_LLM_FIRST_TOKEN,
        PHASE_TTS_FIRST_AUDIO,
        "voice.ttft_to_tts",
    ),
    (
        PHASE_TTS_FIRST_AUDIO,
        PHASE_PLAYBACK_START,
        "voice.tts_to_playback",
    ),
];

// Monotonic clock origin for the default `perf_counter()` path (only deltas
// matter, so any fixed origin works).
static PERF_EPOCH: OnceLock<Instant> = OnceLock::new();

fn perf_counter() -> f64 {
    PERF_EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Gap ms = `max(0, round(delta_seconds * 1000))`, banker's rounding
/// (`round_ties_even`, DESIGN §4.3); negative gaps clamp to 0.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn gap_ms(delta_seconds: f64) -> i64 {
    crate::support::round::half_even(delta_seconds * 1000.0).max(0.0) as i64
}

// LRU-ordered map of turn_id -> phase timestamps. Front of `order` is LRU.
#[derive(Debug, Default)]
struct LruTurns {
    max_turns: usize,
    order: Vec<String>,
    map: HashMap<String, HashMap<String, f64>>,
}

impl LruTurns {
    fn new(max_turns: usize) -> Self {
        Self {
            max_turns,
            order: Vec::new(),
            map: HashMap::new(),
        }
    }

    // Move an existing turn to MRU, or create it (evicting LRU past capacity).
    fn touch_or_create(&mut self, turn_id: &str) {
        if self.map.contains_key(turn_id) {
            if let Some(pos) = self.order.iter().position(|k| k == turn_id) {
                let key = self.order.remove(pos);
                self.order.push(key);
            }
        } else {
            self.map.insert(turn_id.to_string(), HashMap::new());
            self.order.push(turn_id.to_string());
            while self.order.len() > self.max_turns {
                let evicted = self.order.remove(0);
                self.map.remove(&evicted);
            }
        }
    }

    fn discard(&mut self, turn_id: &str) {
        if self.map.remove(turn_id).is_some() {
            if let Some(pos) = self.order.iter().position(|k| k == turn_id) {
                self.order.remove(pos);
            }
        }
    }
}

// Gaps newly reachable when `current` is recorded into `phases`.
fn compute_gaps(phases: &HashMap<String, f64>, current: &str) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    for (prev, curr, name) in GAPS {
        if curr != current || !phases.contains_key(prev) {
            continue;
        }
        out.push((name.to_string(), gap_ms(phases[curr] - phases[prev])));
    }
    // Cumulative total fires once on the terminal phase if the funnel started
    // cleanly — deliberately anchored at stt_final even when vad_end exists.
    if current == PHASE_PLAYBACK_START && phases.contains_key(PHASE_STT_FINAL) {
        let ms = gap_ms(phases[PHASE_PLAYBACK_START] - phases[PHASE_STT_FINAL]);
        out.push((SPAN_TOTAL.to_string(), ms));
    }
    out
}

/// Per-turn phase timestamps; emits gap spans on phase advance.
#[derive(Debug)]
pub struct VoiceBudgetRecorder {
    state: Mutex<LruTurns>,
}

impl VoiceBudgetRecorder {
    /// Create a recorder retaining at most `max_turns` recent turns.
    #[must_use]
    pub fn new(max_turns: usize) -> Self {
        Self {
            state: Mutex::new(LruTurns::new(max_turns)),
        }
    }

    /// Create a recorder with the default LRU capacity (32).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_TURNS)
    }

    /// Stamp `phase` for `turn_id`; emit newly-reachable gap spans.
    ///
    /// `t` defaults to a monotonic clock read when `None`. First record per
    /// (turn, phase) wins — duplicates are dropped (but still refresh LRU order,
    /// matching Python's `move_to_end` before the dedupe check). Gaps are
    /// emitted **after** releasing the internal lock, so the lock order is never
    /// budget→collector (DESIGN §4.4 / spec 01 §30).
    // The guard is deliberately block-scoped and dropped before `emit`.
    #[allow(clippy::significant_drop_tightening)]
    pub fn record(&self, turn_id: &str, phase: &str, t: Option<f64>) {
        let t = t.unwrap_or_else(perf_counter);
        let emits = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.touch_or_create(turn_id);
            match state.map.get_mut(turn_id) {
                // First write per (turn, phase) wins — duplicates are dropped.
                Some(phases) if phases.contains_key(phase) => Vec::new(),
                Some(phases) => {
                    phases.insert(phase.to_string(), t);
                    compute_gaps(phases, phase)
                }
                // `max_turns == 0`: `touch_or_create` evicts the turn the instant
                // it is created. Python keeps a now-orphaned local dict after
                // `popitem` and writes/computes on it, so a lone phase emits
                // nothing; mirror that graceful degradation instead of panicking.
                None => {
                    let mut phases = HashMap::new();
                    phases.insert(phase.to_string(), t);
                    compute_gaps(&phases, phase)
                }
            }
        };
        for (name, ms) in emits {
            emit(turn_id, &name, ms);
        }
    }

    /// Drop a turn's phase state; no-op if unknown.
    pub fn discard(&self, turn_id: &str) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .discard(turn_id);
    }
}

/// One gap span: an INFO line on `familiar_connect.diagnostics.voice_budget`
/// plus a `status="ok"` record into the shared collector.
fn emit(turn_id: &str, name: &str, ms: i64) {
    let line = format!(
        "{} {} {} {}",
        ls::tag("budget", ls::LM),
        ls::kv_styled("turn", turn_id, ls::W, ls::LC),
        ls::kv_styled("span", name, ls::W, ls::LM),
        ls::kv_styled("ms", &ms.to_string(), ls::W, ls::LC),
    );
    tracing::info!(target: "familiar_connect.diagnostics.voice_budget", "{line}");
    get_span_collector().record(name, ms, "ok");
}

// ---------------------------------------------------------------------------
// process-wide singleton — mirrors get_span_collector
// ---------------------------------------------------------------------------

static RECORDER: Mutex<Option<Arc<VoiceBudgetRecorder>>> = Mutex::new(None);

/// Return the process-wide recorder, creating it on first use.
#[must_use]
pub fn get_voice_budget_recorder() -> Arc<VoiceBudgetRecorder> {
    let mut guard = RECORDER.lock().unwrap_or_else(PoisonError::into_inner);
    guard
        .get_or_insert_with(|| Arc::new(VoiceBudgetRecorder::with_defaults()))
        .clone()
}

/// Reset the singleton — tests only.
#[cfg(any(test, feature = "test-util"))]
pub fn reset_voice_budget_recorder() {
    *RECORDER.lock().unwrap_or_else(PoisonError::into_inner) = None;
}

#[cfg(test)]
mod tests {
    use super::{
        PHASE_LLM_FIRST_TOKEN, PHASE_PLAYBACK_START, PHASE_STT_FINAL, PHASE_TTS_FIRST_AUDIO,
        PHASE_VAD_END, VoiceBudgetRecorder, get_voice_budget_recorder, reset_voice_budget_recorder,
    };
    use crate::diagnostics::collector::{SpanRecord, get_span_collector, reset_span_collector};
    use crate::diagnostics::testutil::singleton_guard;
    use std::sync::Arc;
    use std::sync::MutexGuard;

    // Mirrors the pytest autouse fixture: isolate both singletons under the
    // shared guard so parallel tests never observe each other's collector.
    fn isolated() -> MutexGuard<'static, ()> {
        let g = singleton_guard();
        reset_voice_budget_recorder();
        reset_span_collector();
        g
    }

    fn names(records: &[SpanRecord]) -> Vec<String> {
        records.iter().map(|r| r.name.clone()).collect()
    }

    fn collected() -> Vec<SpanRecord> {
        get_span_collector().all()
    }

    // --- sequential phases ---

    #[test]
    fn stt_to_ttft_emits_on_llm_first_token() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_STT_FINAL, Some(10.000));
        assert!(names(&collected()).is_empty());
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(10.250));
        let recorded = collected();
        assert_eq!(names(&recorded), vec!["voice.stt_to_ttft"]);
        assert_eq!(recorded[0].ms, 250);
    }

    #[test]
    fn full_funnel_emits_three_gaps_plus_total() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_STT_FINAL, Some(10.000));
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(10.300));
        rec.record("t-1", PHASE_TTS_FIRST_AUDIO, Some(10.450));
        rec.record("t-1", PHASE_PLAYBACK_START, Some(10.700));
        assert_eq!(
            names(&collected()),
            vec![
                "voice.stt_to_ttft",
                "voice.ttft_to_tts",
                "voice.tts_to_playback",
                "voice.total",
            ]
        );
        let by_name = get_span_collector().by_name();
        assert_eq!(by_name["voice.stt_to_ttft"][0].ms, 300);
        assert_eq!(by_name["voice.ttft_to_tts"][0].ms, 150);
        assert_eq!(by_name["voice.tts_to_playback"][0].ms, 250);
        assert_eq!(by_name["voice.total"][0].ms, 700);
    }

    #[test]
    fn skipped_first_phase_no_emit() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(10.0));
        rec.record("t-1", PHASE_TTS_FIRST_AUDIO, Some(10.1));
        assert_eq!(names(&collected()), vec!["voice.ttft_to_tts"]);
    }

    // --- deduplication ---

    #[test]
    fn duplicate_phase_first_wins() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(10.000));
        rec.record("t-1", PHASE_TTS_FIRST_AUDIO, Some(10.100));
        // Later sentence — should NOT overwrite.
        rec.record("t-1", PHASE_TTS_FIRST_AUDIO, Some(10.500));
        rec.record("t-1", PHASE_PLAYBACK_START, Some(10.200));
        let by_name = get_span_collector().by_name();
        assert_eq!(by_name["voice.ttft_to_tts"][0].ms, 100);
        assert_eq!(by_name["voice.tts_to_playback"][0].ms, 100);
    }

    // --- per-turn isolation ---

    #[test]
    fn two_turns_compute_independently() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-A", PHASE_STT_FINAL, Some(100.0));
        rec.record("t-B", PHASE_STT_FINAL, Some(200.0));
        rec.record("t-B", PHASE_LLM_FIRST_TOKEN, Some(200.05));
        rec.record("t-A", PHASE_LLM_FIRST_TOKEN, Some(100.30));
        let by_name = get_span_collector().by_name();
        let mut gaps: Vec<i64> = by_name["voice.stt_to_ttft"].iter().map(|r| r.ms).collect();
        gaps.sort_unstable();
        assert_eq!(gaps, vec![50, 300]);
    }

    // --- eviction ---

    #[test]
    fn oldest_turn_dropped_past_capacity() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::new(2);
        rec.record("t-1", PHASE_STT_FINAL, Some(1.0));
        rec.record("t-2", PHASE_STT_FINAL, Some(2.0));
        rec.record("t-3", PHASE_STT_FINAL, Some(3.0));
        // t-1 evicted; its prior phase is gone, so no gap is emitted.
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(4.0));
        assert!(names(&collected()).is_empty());
    }

    #[test]
    fn max_turns_zero_degrades_without_panic() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::new(0);
        // Every turn is evicted the instant it is created, so each phase lands on
        // a fresh orphan map with no predecessor: no gap ever emits and, crucially,
        // `record` must not panic (Python degrades gracefully; see finding).
        rec.record("t-1", PHASE_STT_FINAL, Some(10.0));
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(10.25));
        rec.record("t-1", PHASE_PLAYBACK_START, Some(10.7));
        assert!(names(&collected()).is_empty());
    }

    #[test]
    fn discard_removes_turn() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_STT_FINAL, Some(1.0));
        rec.discard("t-1");
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(1.5));
        assert!(names(&collected()).is_empty());
    }

    // --- total span ---

    #[test]
    fn total_skipped_if_stt_final_missing() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_TTS_FIRST_AUDIO, Some(10.0));
        rec.record("t-1", PHASE_PLAYBACK_START, Some(10.2));
        assert!(!names(&collected()).contains(&"voice.total".to_string()));
    }

    // --- vad_end chain ---

    #[test]
    fn vad_to_stt_emits_on_stt_final() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_VAD_END, Some(10.000));
        assert!(names(&collected()).is_empty());
        rec.record("t-1", PHASE_STT_FINAL, Some(10.180));
        let recorded = collected();
        assert_eq!(names(&recorded), vec!["voice.vad_to_stt"]);
        assert_eq!(recorded[0].ms, 180);
    }

    #[test]
    fn vad_end_optional_no_emit_without_it() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_STT_FINAL, Some(10.0));
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(10.2));
        let n = names(&collected());
        assert!(!n.contains(&"voice.vad_to_stt".to_string()));
        assert_eq!(n, vec!["voice.stt_to_ttft"]);
    }

    #[test]
    fn full_funnel_with_vad_end() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_VAD_END, Some(10.000));
        rec.record("t-1", PHASE_STT_FINAL, Some(10.150));
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, Some(10.450));
        rec.record("t-1", PHASE_TTS_FIRST_AUDIO, Some(10.600));
        rec.record("t-1", PHASE_PLAYBACK_START, Some(10.850));
        assert_eq!(
            names(&collected()),
            vec![
                "voice.vad_to_stt",
                "voice.stt_to_ttft",
                "voice.ttft_to_tts",
                "voice.tts_to_playback",
                "voice.total",
            ]
        );
        let by_name = get_span_collector().by_name();
        assert_eq!(by_name["voice.vad_to_stt"][0].ms, 150);
        // voice.total keeps its stt_final start (700), not vad_end (850).
        assert_eq!(by_name["voice.total"][0].ms, 700);
    }

    // --- singleton ---

    #[test]
    fn singleton_returns_same_instance() {
        let _g = isolated();
        let a = get_voice_budget_recorder();
        let b = get_voice_budget_recorder();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn reset_yields_fresh_instance() {
        let _g = isolated();
        let a = get_voice_budget_recorder();
        reset_voice_budget_recorder();
        let b = get_voice_budget_recorder();
        assert!(!Arc::ptr_eq(&a, &b));
    }

    // --- default-clock path ---

    #[test]
    fn record_without_explicit_t_uses_clock() {
        let _g = isolated();
        let rec = VoiceBudgetRecorder::with_defaults();
        rec.record("t-1", PHASE_STT_FINAL, None);
        rec.record("t-1", PHASE_LLM_FIRST_TOKEN, None);
        assert_eq!(names(&collected()), vec!["voice.stt_to_ttft"]);
    }
}
