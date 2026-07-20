//! `VoiceSource`: transcription queue → bus (subsystem 09/10; Python `sources/voice.py`).
//!
//! Drains a transcriber result queue onto the bus, publishing four topics per
//! utterance — `voice.activity.start`, `voice.transcript.partial`,
//! `voice.transcript.final`, `voice.activity.end`. All events of one utterance
//! share a `turn_id`; the next utterance mints a fresh one.
//!
//! Discord delivers per-SSRC audio, so the state machine is keyed by `user_id`
//! (`None` = the legacy single-stream slot): two speakers talking concurrently
//! each get a distinct `turn_id` — a single shared slot would drop the second
//! speaker's `activity.start`.
//!
//! The start / final payloads are the responder-facing
//! [`VoiceActivityStart`](crate::processors::VoiceActivityStart) /
//! [`VoiceTranscriptFinal`](crate::processors::VoiceTranscriptFinal) types (the
//! only two the voice responder consumes). See the port summary for the
//! confidence/start/end/speaker fields the landed responder types drop from the
//! final payload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::mpsc::UnboundedReceiver;
use uuid::Uuid;

use crate::bus::envelope::{Event, Payload, payload};
use crate::bus::protocols::EventBus;
use crate::bus::topics::{
    TOPIC_VOICE_ACTIVITY_END, TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_FINAL,
    TOPIC_VOICE_TRANSCRIPT_PARTIAL,
};
use crate::diagnostics::voice_budget::{PHASE_STT_FINAL, PHASE_VAD_END, get_voice_budget_recorder};
use crate::processors::{VoiceActivityStart, VoiceTranscriptFinal};
use crate::stt::TranscriptionResult;

// Monotonic clock origin for the default `vad_end` timestamp path. The recorder
// keeps its own private `perf_counter`; sharing the exact origin would need a
// public accessor (filed as a shared-file request). Deltas dominate, and the
// tests pass an explicit `t`, so this only affects the diagnostic gap value in
// production.
static PERF_EPOCH: OnceLock<Instant> = OnceLock::new();

fn perf_counter() -> f64 {
    PERF_EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// `voice.activity.end` payload — mirrors Python `{"user_id": ...}`.
#[derive(Clone, Debug, Default)]
pub struct VoiceActivityEnd {
    /// Speaker Discord user id, when known.
    pub user_id: Option<i64>,
}

/// `voice.transcript.partial` payload — `{text, confidence, user_id}`.
#[derive(Clone, Debug, Default)]
pub struct VoiceTranscriptPartial {
    /// Interim transcript text.
    pub text: String,
    /// Recognition confidence.
    pub confidence: f64,
    /// Speaker Discord user id, when known.
    pub user_id: Option<i64>,
}

#[derive(Default)]
struct VoiceState {
    seq: u64,
    /// `user_id` → current turn id (`None` key = legacy unattributed slot).
    turn_ids: HashMap<Option<i64>, String>,
    /// Local-endpointer `on_turn_complete` marks awaiting a turn id.
    pending_vad_end: HashMap<i64, f64>,
}

/// Drains a transcription result queue onto the bus.
pub struct VoiceSource {
    bus: Arc<dyn EventBus>,
    channel_id: i64,
    queue: AsyncMutex<UnboundedReceiver<TranscriptionResult>>,
    state: Mutex<VoiceState>,
}

impl VoiceSource {
    /// The stable source name.
    pub const NAME: &'static str = "voice";

    /// Build a source draining `queue` onto `bus` for `voice_channel_id`.
    ///
    /// `familiar_id` is accepted for signature parity with Python; the voice
    /// payloads and envelope key on the channel, not the familiar.
    #[must_use]
    pub fn new(
        bus: Arc<dyn EventBus>,
        familiar_id: impl Into<String>,
        voice_channel_id: i64,
        queue: UnboundedReceiver<TranscriptionResult>,
    ) -> Self {
        let _ = familiar_id.into();
        Self {
            bus,
            channel_id: voice_channel_id,
            queue: AsyncMutex::new(queue),
            state: Mutex::new(VoiceState::default()),
        }
    }

    /// Park a local-endpointer `on_turn_complete` mark for `user_id`.
    ///
    /// The endpointer fires before the transcriber emits its final, so the
    /// `turn_id` may not exist yet. The buffered timestamp is stamped on the
    /// recorder by the next result for `user_id`; the latest fire wins if
    /// several stack up before a result.
    pub fn record_vad_end(&self, user_id: i64, t: Option<f64>) {
        let t = t.unwrap_or_else(perf_counter);
        self.state
            .lock()
            .expect("voice source state mutex poisoned")
            .pending_vad_end
            .insert(user_id, t);
    }

    /// Forever loop: drain the queue, publish. Task cancellation is the only
    /// clean exit (the loop also ends if every sender is dropped).
    pub async fn run(&self) {
        let mut rx = self.queue.lock().await;
        while let Some(result) = rx.recv().await {
            self.handle(result).await;
        }
    }

    async fn handle(&self, result: TranscriptionResult) {
        let user_id = result.user_id;
        let (turn_id, is_new) = {
            use std::collections::hash_map::Entry;
            let mut st = self
                .state
                .lock()
                .expect("voice source state mutex poisoned");
            match st.turn_ids.entry(user_id) {
                Entry::Occupied(e) => (e.get().clone(), false),
                Entry::Vacant(e) => {
                    let t = format!("voice-{}", &Uuid::new_v4().simple().to_string()[..12]);
                    e.insert(t.clone());
                    (t, true)
                }
            }
        };
        if is_new {
            self.publish(
                TOPIC_VOICE_ACTIVITY_START,
                &turn_id,
                payload(VoiceActivityStart { user_id }),
            )
            .await;
        }

        // Drain any buffered vad_end ahead of the other phases so the gap to
        // `stt_final` emits in order. `None` user_id never carries a mark.
        if let Some(uid) = user_id {
            let pending = self
                .state
                .lock()
                .expect("voice source state mutex poisoned")
                .pending_vad_end
                .remove(&uid);
            if let Some(t) = pending {
                get_voice_budget_recorder().record(&turn_id, PHASE_VAD_END, Some(t));
            }
        }

        if result.is_final {
            // Stamp before publishing so the recorder sees stt_final ahead of
            // the responder's llm_first_token mark (preserves gap order).
            get_voice_budget_recorder().record(&turn_id, PHASE_STT_FINAL, None);
            self.publish(
                TOPIC_VOICE_TRANSCRIPT_FINAL,
                &turn_id,
                payload(VoiceTranscriptFinal {
                    text: result.text.clone(),
                    user_id,
                }),
            )
            .await;
            self.publish(
                TOPIC_VOICE_ACTIVITY_END,
                &turn_id,
                payload(VoiceActivityEnd { user_id }),
            )
            .await;
            self.state
                .lock()
                .expect("voice source state mutex poisoned")
                .turn_ids
                .remove(&user_id);
        } else {
            self.publish(
                TOPIC_VOICE_TRANSCRIPT_PARTIAL,
                &turn_id,
                payload(VoiceTranscriptPartial {
                    text: result.text.clone(),
                    confidence: result.confidence,
                    user_id,
                }),
            )
            .await;
        }
    }

    async fn publish(&self, topic: &str, turn_id: &str, event_payload: Payload) {
        let seq = {
            let mut st = self
                .state
                .lock()
                .expect("voice source state mutex poisoned");
            st.seq += 1;
            st.seq
        };
        let event = Event {
            event_id: format!("voice-{seq:08}"),
            turn_id: turn_id.to_owned(),
            session_id: format!("voice:{}", self.channel_id),
            parent_event_ids: Vec::new(),
            topic: topic.to_owned(),
            timestamp: chrono::Utc::now(),
            sequence_number: seq,
            payload: event_payload,
        };
        self.bus.publish(event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::{VoiceSource, VoiceTranscriptPartial};
    use crate::bus::envelope::Event;
    use crate::bus::in_process::InProcessEventBus;
    use crate::bus::protocols::{BackpressurePolicy, EventBus};
    use crate::bus::topics::{
        TOPIC_VOICE_ACTIVITY_END, TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_FINAL,
        TOPIC_VOICE_TRANSCRIPT_PARTIAL,
    };
    use crate::diagnostics::collector::{get_span_collector, reset_span_collector};
    use crate::diagnostics::testutil::singleton_guard;
    use crate::diagnostics::voice_budget::reset_voice_budget_recorder;
    use crate::processors::VoiceTranscriptFinal;
    use crate::stt::TranscriptionResult;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc::{self, UnboundedSender};
    use tokio::time::timeout;

    fn final_res(text: &str, user_id: Option<i64>) -> TranscriptionResult {
        let mut r = TranscriptionResult::new(text, true, 0.0, 1.0);
        r.confidence = 0.9;
        r.user_id = user_id;
        r
    }

    fn partial_res(text: &str, user_id: Option<i64>) -> TranscriptionResult {
        let mut r = TranscriptionResult::new(text, false, 0.0, 0.3);
        r.confidence = 0.5;
        r.user_id = user_id;
        r
    }

    #[allow(clippy::type_complexity)]
    async fn spawn_source(
        channel_id: i64,
    ) -> (
        Arc<InProcessEventBus>,
        Arc<VoiceSource>,
        UnboundedSender<TranscriptionResult>,
        tokio::task::JoinHandle<()>,
    ) {
        let bus = Arc::new(InProcessEventBus::new());
        bus.start().await;
        let (tx, rx) = mpsc::unbounded_channel();
        let source = Arc::new(VoiceSource::new(bus.clone(), "fam", channel_id, rx));
        let producer = tokio::spawn({
            let s = Arc::clone(&source);
            async move { s.run().await }
        });
        tokio::task::yield_now().await;
        (bus, source, tx, producer)
    }

    #[tokio::test]
    async fn final_only_emits_start_transcript_end() {
        let (bus, _source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_ACTIVITY_END,
                TOPIC_VOICE_TRANSCRIPT_FINAL,
            ],
            BackpressurePolicy::Unbounded,
            0,
        );
        tx.send(final_res("hello", None)).unwrap();

        let mut collected = Vec::new();
        for _ in 0..3 {
            collected.push(
                timeout(Duration::from_secs(1), sub.recv())
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
        producer.abort();
        bus.shutdown().await;

        let topics: Vec<&str> = collected.iter().map(|e| e.topic.as_str()).collect();
        assert_eq!(
            topics,
            vec![
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_TRANSCRIPT_FINAL,
                TOPIC_VOICE_ACTIVITY_END,
            ]
        );
        let final_ev = &collected[1];
        assert_eq!(
            final_ev
                .payload
                .downcast_ref::<VoiceTranscriptFinal>()
                .unwrap()
                .text,
            "hello"
        );
        assert_eq!(final_ev.session_id, "voice:123");
        assert_eq!(collected[0].turn_id, final_ev.turn_id);
        assert_eq!(final_ev.turn_id, collected[2].turn_id);
    }

    #[tokio::test]
    async fn partial_emits_start_and_partial_not_end() {
        let (bus, _source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_PARTIAL],
            BackpressurePolicy::Unbounded,
            0,
        );
        tx.send(partial_res("hel", None)).unwrap();
        let mut got = Vec::new();
        for _ in 0..2 {
            got.push(
                timeout(Duration::from_secs(1), sub.recv())
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
        producer.abort();
        bus.shutdown().await;
        let topics: Vec<&str> = got.iter().map(|e| e.topic.as_str()).collect();
        assert_eq!(
            topics,
            vec![TOPIC_VOICE_ACTIVITY_START, TOPIC_VOICE_TRANSCRIPT_PARTIAL]
        );
        assert_eq!(
            got[1]
                .payload
                .downcast_ref::<VoiceTranscriptPartial>()
                .unwrap()
                .text,
            "hel"
        );
    }

    #[tokio::test]
    async fn only_one_activity_start_per_utterance() {
        let (bus, _source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_ACTIVITY_END,
                TOPIC_VOICE_TRANSCRIPT_PARTIAL,
                TOPIC_VOICE_TRANSCRIPT_FINAL,
            ],
            BackpressurePolicy::Unbounded,
            0,
        );
        tx.send(partial_res("hel", None)).unwrap();
        tx.send(partial_res("hello", None)).unwrap();
        tx.send(final_res("hello world", None)).unwrap();
        let mut got = Vec::new();
        loop {
            let ev = timeout(Duration::from_secs(1), sub.recv())
                .await
                .unwrap()
                .unwrap();
            let is_end = ev.topic == TOPIC_VOICE_ACTIVITY_END;
            got.push(ev);
            if is_end {
                break;
            }
        }
        producer.abort();
        bus.shutdown().await;
        let starts = got
            .iter()
            .filter(|e| e.topic == TOPIC_VOICE_ACTIVITY_START)
            .count();
        assert_eq!(starts, 1);
        let turn_ids: HashSet<&str> = got.iter().map(|e| e.turn_id.as_str()).collect();
        assert_eq!(turn_ids.len(), 1);
    }

    #[tokio::test]
    async fn new_utterance_after_final_gets_fresh_turn_id() {
        let (bus, _source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[TOPIC_VOICE_TRANSCRIPT_FINAL],
            BackpressurePolicy::Unbounded,
            0,
        );
        tx.send(final_res("first", None)).unwrap();
        tx.send(final_res("second", None)).unwrap();
        let a = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        let b = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        producer.abort();
        bus.shutdown().await;
        assert_ne!(a.turn_id, b.turn_id);
    }

    #[tokio::test]
    async fn final_payload_carries_user_id() {
        let (bus, _source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[TOPIC_VOICE_TRANSCRIPT_FINAL],
            BackpressurePolicy::Unbounded,
            0,
        );
        tx.send(final_res("hello", Some(42))).unwrap();
        let ev = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        producer.abort();
        bus.shutdown().await;
        assert_eq!(
            ev.payload
                .downcast_ref::<VoiceTranscriptFinal>()
                .unwrap()
                .user_id,
            Some(42)
        );
    }

    #[tokio::test]
    async fn concurrent_speakers_get_independent_turn_ids() {
        let (bus, _source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[
                TOPIC_VOICE_ACTIVITY_START,
                TOPIC_VOICE_ACTIVITY_END,
                TOPIC_VOICE_TRANSCRIPT_FINAL,
            ],
            BackpressurePolicy::Unbounded,
            0,
        );
        tx.send(partial_res("hel", Some(101))).unwrap();
        tx.send(partial_res("hi", Some(202))).unwrap();
        tx.send(final_res("hello", Some(101))).unwrap();
        tx.send(final_res("hi there", Some(202))).unwrap();

        let mut events: Vec<Arc<Event>> = Vec::new();
        while events
            .iter()
            .filter(|e| e.topic == TOPIC_VOICE_ACTIVITY_END)
            .count()
            < 2
        {
            events.push(
                timeout(Duration::from_secs(1), sub.recv())
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
        producer.abort();
        bus.shutdown().await;

        let starts: Vec<&Arc<Event>> = events
            .iter()
            .filter(|e| e.topic == TOPIC_VOICE_ACTIVITY_START)
            .collect();
        assert_eq!(starts.len(), 2);
        assert_ne!(starts[0].turn_id, starts[1].turn_id);
        let finals: Vec<&Arc<Event>> = events
            .iter()
            .filter(|e| e.topic == TOPIC_VOICE_TRANSCRIPT_FINAL)
            .collect();
        let user_of = |e: &Arc<Event>| {
            e.payload
                .downcast_ref::<VoiceTranscriptFinal>()
                .unwrap()
                .user_id
        };
        let turn_101 = finals
            .iter()
            .find(|e| user_of(e) == Some(101))
            .unwrap()
            .turn_id
            .clone();
        let turn_202 = finals
            .iter()
            .find(|e| user_of(e) == Some(202))
            .unwrap()
            .turn_id
            .clone();
        assert_ne!(turn_101, turn_202);
    }

    // --- voice budget (global singleton — serialized + reset) --------------

    #[tokio::test]
    #[allow(
        clippy::await_holding_lock,
        reason = "the singleton guard serialises the whole single-threaded test"
    )]
    async fn final_records_stt_final_phase() {
        let _g = singleton_guard();
        reset_voice_budget_recorder();
        reset_span_collector();
        let (bus, source, tx, producer) = spawn_source(123).await;
        // A lone stt_final emits no gap span; pairing it with a buffered
        // vad_end makes stt_final observable via the `voice.vad_to_stt` gap
        // (which requires stt_final to have been stamped). The exact phase
        // value assertion needs a recorder accessor not exposed publicly.
        source.record_vad_end(7, Some(1.0));
        let mut sub = bus.subscribe(
            &[TOPIC_VOICE_TRANSCRIPT_FINAL],
            BackpressurePolicy::Unbounded,
            0,
        );
        tx.send(final_res("hi", Some(7))).unwrap();
        timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        producer.abort();
        bus.shutdown().await;
        let names: Vec<String> = get_span_collector()
            .all()
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert!(names.iter().any(|n| n == "voice.vad_to_stt"), "{names:?}");
    }

    #[tokio::test]
    #[allow(
        clippy::await_holding_lock,
        reason = "the singleton guard serialises the whole single-threaded test"
    )]
    async fn pending_vad_end_stamped_on_next_result() {
        let _g = singleton_guard();
        reset_voice_budget_recorder();
        reset_span_collector();
        let (bus, source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[TOPIC_VOICE_TRANSCRIPT_FINAL],
            BackpressurePolicy::Unbounded,
            0,
        );
        // Endpointer fires first; final arrives later.
        source.record_vad_end(42, Some(100.0));
        tx.send(final_res("hi", Some(42))).unwrap();
        timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        producer.abort();
        bus.shutdown().await;
        // Adjacent gap span emits (proving vad_end + stt_final both stamped).
        let names: Vec<String> = get_span_collector()
            .all()
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert!(names.iter().any(|n| n == "voice.vad_to_stt"), "{names:?}");
    }

    #[tokio::test]
    #[allow(
        clippy::await_holding_lock,
        reason = "the singleton guard serialises the whole single-threaded test"
    )]
    async fn vad_end_only_applies_to_matching_user() {
        let _g = singleton_guard();
        reset_voice_budget_recorder();
        reset_span_collector();
        let (bus, source, tx, producer) = spawn_source(123).await;
        let mut sub = bus.subscribe(
            &[TOPIC_VOICE_TRANSCRIPT_FINAL],
            BackpressurePolicy::Unbounded,
            0,
        );
        // Endpointer fired for user 101; user 202 happens to final first.
        source.record_vad_end(101, Some(99.0));
        tx.send(final_res("other speaker", Some(202))).unwrap();
        timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        producer.abort();
        bus.shutdown().await;
        // 202's turn must NOT carry 101's vad_end → no gap span emitted.
        let names: Vec<String> = get_span_collector()
            .all()
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert!(!names.iter().any(|n| n == "voice.vad_to_stt"), "{names:?}");
    }
}
