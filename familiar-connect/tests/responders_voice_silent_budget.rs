//! Voice-budget span test — silent path (subsystem 06; Python
//! `tests/test_voice_responder.py::TestVoiceBudgetSpans`).
//!
//! Alone in its own binary so the process-wide span collector starts empty (see
//! `responders_voice_budget.rs` for the rationale): a silent turn records no
//! TTS phase, so `voice.ttft_to_tts` must be absent.

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
async fn no_budget_spans_on_silent_reply() {
    let s = store();
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        Arc::new(ScriptedLlm::new(&["<silent>"])),
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
    assert!(!names.iter().any(|n| n == "voice.ttft_to_tts"), "{names:?}");
}
