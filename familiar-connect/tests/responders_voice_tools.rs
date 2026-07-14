//! Voice tool-path tests (subsystem 06; Python `tests/test_voice_responder_tools.py`).
//!
//! V15: spoken content reaches TTS before the tool handler runs; the filler
//! backstop speaks a stock phrase before an empty-content tool iteration.
//! (The registry-separation cases belong to subsystem 08 `tools::builtins` and
//! are covered there — see the port summary's skipped list.)

#![allow(clippy::significant_drop_tightening)]

#[path = "responders_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use familiar_connect::bus::envelope::TurnScope;
use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::router::TurnRouter;
use familiar_connect::processors::ToolContextFactory;
use familiar_connect::processors::voice_responder::VoiceResponder;
use familiar_connect::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput, ToolRegistry};
use familiar_connect::tts_player::protocol::TtsPlayer;

use support::{
    ScriptedToolLlm, activity_start, finish, make_assembler, store, tc_delta, text_delta,
    voice_final,
};

const fn bus() -> InProcessEventBus {
    InProcessEventBus::new()
}

/// A TTS player that records the order of spoken texts into a shared buffer.
struct RecordingVoicePlayer {
    spoken: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl TtsPlayer for RecordingVoicePlayer {
    async fn speak(&self, text: &str, _scope: &TurnScope) {
        self.spoken.lock().unwrap().push(text.to_owned());
    }
    async fn stop(&self) {}
}

fn voice_ctx_factory() -> ToolContextFactory {
    Arc::new(|channel_id, turn_id, _images| ToolContext::new("fam", channel_id, "voice", turn_id))
}

#[tokio::test]
async fn speak_completes_before_tool_runs() {
    let spoken = Arc::new(Mutex::new(Vec::<String>::new()));
    let tool_runs = Arc::new(Mutex::new(0usize));
    let runs = Arc::clone(&tool_runs);
    let handler = FnHandler(move |_args: Value, _ctx: ToolContext| {
        let runs = Arc::clone(&runs);
        async move {
            *runs.lock().unwrap() += 1;
            Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
        }
    });
    let mut reg = ToolRegistry::new();
    reg.register(Tool::new("set_alarm", "", json!({}), Arc::new(handler)))
        .unwrap();

    // iteration 1: text + tool_call (text first); iteration 2: short final text.
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![
            text_delta("Sure, one moment."),
            tc_delta(
                "c1",
                "set_alarm",
                json!({"reason": "x", "delay_seconds": 30}),
            ),
            finish("tool_calls"),
        ],
        vec![text_delta("Done."), finish("stop")],
    ]));
    let player = Arc::new(RecordingVoicePlayer {
        spoken: Arc::clone(&spoken),
    });
    let s = store();
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        llm,
        player,
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_tools(Arc::new(reg), voice_ctx_factory())
    .with_tool_filler_phrases(vec!["one sec...".to_owned()]);

    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("set an alarm", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let spoken = spoken.lock().unwrap();
    assert!(!spoken.is_empty(), "expected TTS to have been called");
    assert!(spoken.iter().any(|s| s.contains("Sure")), "{spoken:?}");
    assert_eq!(*tool_runs.lock().unwrap(), 1);
}

#[tokio::test]
async fn filler_spoken_when_tool_call_has_empty_content() {
    let spoken = Arc::new(Mutex::new(Vec::<String>::new()));
    let snapshot_at_tool: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let spoken_for_handler = Arc::clone(&spoken);
    let snap = Arc::clone(&snapshot_at_tool);
    let handler = FnHandler(move |_args: Value, _ctx: ToolContext| {
        let spoken = Arc::clone(&spoken_for_handler);
        let snap = Arc::clone(&snap);
        async move {
            snap.lock().unwrap().push(spoken.lock().unwrap().clone());
            Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
        }
    });
    let mut reg = ToolRegistry::new();
    reg.register(Tool::new("set_alarm", "", json!({}), Arc::new(handler)))
        .unwrap();

    // iteration 1: ONLY a tool_call (no content) → the filler should fire.
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![
            tc_delta(
                "c1",
                "set_alarm",
                json!({"reason": "x", "delay_seconds": 30}),
            ),
            finish("tool_calls"),
        ],
        vec![text_delta("Done."), finish("stop")],
    ]));
    let player = Arc::new(RecordingVoicePlayer {
        spoken: Arc::clone(&spoken),
    });
    let s = store();
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        llm,
        player,
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_tools(Arc::new(reg), voice_ctx_factory())
    .with_tool_filler_phrases(vec!["hang on...".to_owned()]);

    r.handle(&activity_start("voice:1", "t-2", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("set alarm", "voice:1", "t-2", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let snap = snapshot_at_tool.lock().unwrap();
    assert!(!snap.is_empty(), "tool never ran");
    assert!(
        snap[0].iter().any(|s| s.contains("hang on")),
        "filler not spoken before tool: {:?}",
        snap[0]
    );
}
