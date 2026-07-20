//! Voice-budget span test — speech path (subsystem 06; Python
//! `tests/test_voice_responder.py::TestVoiceBudgetSpans`).
//!
//! The span collector is a process-wide singleton whose `reset_*` seam is gated
//! behind the non-default `test-util` feature, so this test lives alone in its
//! own test binary (a fresh process → a fresh, empty collector) and its silent
//! counterpart lives in `responders_voice_silent_budget.rs`.

#[path = "responders_support/mod.rs"]
mod support;

use std::sync::Arc;

use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::router::TurnRouter;
use familiar_connect::diagnostics::collector::get_span_collector;
use familiar_connect::processors::voice_responder::VoiceResponder;
use familiar_connect::tts_player::MockTTSPlayer;

use support::{ScriptedLlm, activity_start, make_assembler, store, voice_final};

#[tokio::test]
async fn ttft_to_tts_gap_recorded() {
    let s = store();
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        Arc::new(ScriptedLlm::with_delay(
            &["Hello there. ", "How are you?"],
            20,
        )),
        Arc::new(MockTTSPlayer::new(1, 5)),
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    );
    r.handle(
        &activity_start("voice:1", "t-1", None),
        &InProcessEventBus::new(),
    )
    .await
    .unwrap();
    r.handle(
        &voice_final("hi", "voice:1", "t-1", None),
        &InProcessEventBus::new(),
    )
    .await
    .unwrap();
    r.wait_until_idle().await;

    let names: Vec<String> = get_span_collector()
        .all()
        .into_iter()
        .map(|rec| rec.name)
        .collect();
    assert!(names.iter().any(|n| n == "voice.ttft_to_tts"), "{names:?}");
}
