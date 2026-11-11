//! Voice tool-path tests (subsystem 06).
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
use familiar_connect::llm::LlmDelta;
use familiar_connect::processors::ToolContextFactory;
use familiar_connect::processors::voice_responder::VoiceResponder;
use familiar_connect::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput, ToolRegistry};
use familiar_connect::tts_player::protocol::TtsPlayer;

use support::{
    ScriptedToolLlm, activity_start, default_config, finish, make_assembler, store, tc_delta,
    text_delta, voice_final,
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

    // iteration 1: ONLY a tool_call (no content), and the call opts into speech
    // → the filler should fire.
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![
            tc_delta(
                "c1",
                "set_alarm",
                json!({"reason": "x", "delay_seconds": 30, "silent": false}),
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

/// A silent tool call is going to say nothing at all, so the filler must not
/// fire — it would be the only thing heard.
#[tokio::test]
async fn no_filler_when_the_tool_call_is_silent() {
    let spoken = Arc::new(Mutex::new(Vec::<String>::new()));
    let handler = FnHandler(|_a: Value, _c: ToolContext| async move {
        Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
    });
    let mut reg = ToolRegistry::new();
    reg.register(Tool::new("set_alarm", "", json!({}), Arc::new(handler)))
        .unwrap();
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
    r.handle(&activity_start("voice:1", "t-3", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("set alarm", "voice:1", "t-3", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(spoken.lock().unwrap().is_empty(), "{spoken:?}");
}

#[tokio::test]
async fn leaked_invoke_xml_content_never_reaches_tts_or_history() {
    // Issue #109: the model leaks a tool call as PLAIN CONTENT (no structured
    // tool_calls), split across deltas, inside the agentic tool loop. The
    // streaming gate must suppress it before it reaches TTS, and the raw XML
    // must never land in the persisted assistant turn.
    let spoken = Arc::new(Mutex::new(Vec::<String>::new()));
    let handler = FnHandler(|_a: Value, _c: ToolContext| async move {
        Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
    });
    let mut reg = ToolRegistry::new();
    reg.register(Tool::new("read_channel", "", json!({}), Arc::new(handler)))
        .unwrap();

    let llm = Arc::new(ScriptedToolLlm::new(vec![vec![
        text_delta("<invoke name=\"read_channel\">\n"),
        text_delta("<parameter name=\"limit\">10</parameter>\n"),
        text_delta("</invoke>"),
        finish("stop"),
    ]]));
    let player = Arc::new(RecordingVoicePlayer {
        spoken: Arc::clone(&spoken),
    });
    let s = store();
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        llm,
        player,
        Arc::clone(&s),
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_tools(Arc::new(reg), voice_ctx_factory());

    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(
        &voice_final("read the channel", "voice:1", "t-1", None),
        &bus(),
    )
    .await
    .unwrap();
    r.wait_until_idle().await;

    let spoken = spoken.lock().unwrap();
    assert!(
        spoken
            .iter()
            .all(|s| !s.contains("invoke") && !s.contains('<')),
        "leak reached TTS: {spoken:?}"
    );
    let turns = s.sync().recent("fam", 1, 10, None, None).unwrap();
    assert!(
        turns.iter().all(|t| t.role != "assistant"),
        "assistant turn persisted: {:?}",
        turns
            .iter()
            .map(|t| (t.role.clone(), t.content.clone()))
            .collect::<Vec<_>>()
    );
    assert!(turns.iter().all(|t| !t.content.contains("invoke")));
}

/// Run one voice turn through the tool loop with a single scripted reply and
/// return what reached TTS plus the persisted assistant turns.
async fn voice_tool_turn(deltas: Vec<&str>) -> (Vec<String>, Vec<String>) {
    let mut script: Vec<_> = deltas.into_iter().map(text_delta).collect();
    script.push(finish("stop"));
    voice_tool_turn_scripted(vec![script]).await
}

/// Run one voice turn through the tool loop, one script per message the model
/// writes (a `read_channel` tool and the filler "hang on..." are on hand), and
/// return what reached TTS plus the persisted assistant turns.
async fn voice_tool_turn_scripted(scripts: Vec<Vec<LlmDelta>>) -> (Vec<String>, Vec<String>) {
    let spoken = Arc::new(Mutex::new(Vec::<String>::new()));
    let handler = FnHandler(|_a: Value, _c: ToolContext| async move {
        Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
    });
    let mut reg = ToolRegistry::new();
    reg.register(Tool::new("read_channel", "", json!({}), Arc::new(handler)))
        .unwrap();
    let llm = Arc::new(ScriptedToolLlm::new(scripts));
    let player = Arc::new(RecordingVoicePlayer {
        spoken: Arc::clone(&spoken),
    });
    let s = store();
    let r = VoiceResponder::new(
        make_assembler(Arc::clone(&s)),
        llm,
        player,
        Arc::clone(&s),
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_tools(Arc::new(reg), voice_ctx_factory())
    .with_tool_filler_phrases(vec!["hang on...".to_owned()]);
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(
        &voice_final("anyone there?", "voice:1", "t-1", None),
        &bus(),
    )
    .await
    .unwrap();
    r.wait_until_idle().await;
    let said = spoken.lock().unwrap().clone();
    let recorded = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .filter(|t| t.role == "assistant")
        .map(|t| t.content)
        .collect();
    (said, recorded)
}

/// Gemini's text form of a `silent` call it failed to emit: nothing is
/// spoken and nothing is recorded.
#[tokio::test]
async fn leaked_api_call_text_never_reaches_tts_or_history() {
    let (said, recorded) = voice_tool_turn(vec![
        "CALL:default_",
        "api:silent{reasoning:nobody asked me anything.}",
    ])
    .await;
    assert!(said.is_empty(), "{said:?}");
    assert!(recorded.is_empty(), "{recorded:?}");
}

/// A leaked thought marker ahead of a spoken reply never reaches TTS or the
/// recorded turn, and a leaked call behind the marker is silence.
#[tokio::test]
async fn thought_marker_is_never_spoken() {
    let (said, recorded) = voice_tool_turn(vec![
        "?[thou",
        "ght\n]",
        "Good evening. ",
        "I was listening.",
    ])
    .await;
    assert_eq!(said.concat().replace(' ', ""), "Goodevening.Iwaslistening.");
    assert!(
        said.iter()
            .all(|s| !s.contains("thought") && !s.contains('?')),
        "{said:?}"
    );
    assert_eq!(recorded, vec!["Good evening. I was listening.".to_owned()]);

    let (said, recorded) = voice_tool_turn(vec![
        "?[thought\n]CALL:default_api:silent{reasoning: nobody asked}",
    ])
    .await;
    assert!(said.is_empty(), "{said:?}");
    assert!(recorded.is_empty(), "{recorded:?}");
}

/// Each message the model writes in a turn gets the check: a marker at the
/// start of the message after a tool call is dropped too, though speech is
/// already under way by then.
#[tokio::test]
async fn thought_marker_after_a_tool_call_is_never_spoken() {
    let (said, recorded) = voice_tool_turn_scripted(vec![
        vec![
            text_delta("Let me look. "),
            tc_delta("c1", "read_channel", json!({"silent": false})),
            finish("tool_calls"),
        ],
        vec![
            text_delta("?[thou"),
            text_delta("ght\n]"),
            text_delta("Nothing new there."),
            finish("stop"),
        ],
    ])
    .await;
    assert_eq!(said.concat().replace(' ', ""), "Letmelook.Nothingnewthere.");
    assert!(
        recorded.iter().all(|t| !t.contains("thought")),
        "{recorded:?}"
    );
    assert!(
        recorded.iter().any(|t| t.contains("Nothing new there.")),
        "{recorded:?}"
    );
}

/// A message that is only the marker beside a speaking tool call says
/// nothing of its own, so the filler covers the wait; the marker is neither
/// spoken nor recorded.
#[tokio::test]
async fn thought_marker_alone_beside_a_tool_call_gets_the_filler() {
    let (said, recorded) = voice_tool_turn_scripted(vec![
        vec![
            text_delta("?[thought\n]"),
            tc_delta("c1", "read_channel", json!({"silent": false})),
            finish("tool_calls"),
        ],
        vec![text_delta("Nothing new there."), finish("stop")],
    ])
    .await;
    assert_eq!(said, ["hang on...", "Nothing new there."]);
    assert!(
        recorded.iter().all(|t| !t.contains("thought")),
        "{recorded:?}"
    );
}

/// A reply too short to rule a marker out ("?") is held back while it
/// streams, then spoken and recorded once the message ends.
#[tokio::test]
async fn reply_held_as_a_possible_marker_is_still_recorded() {
    let (_, recorded) = voice_tool_turn(vec!["?"]).await;
    assert_eq!(recorded, vec!["?".to_owned()]);
    let (said, recorded) = voice_tool_turn(vec!["[", "thou", "ght"]).await;
    assert_eq!(said, vec!["[thought".to_owned()]);
    assert_eq!(recorded, vec!["[thought".to_owned()]);
}

/// A one-word reply that is still a prefix of a leak token when the stream
/// ends is speech: it is spoken, not only recorded.
#[tokio::test]
async fn short_reply_left_pending_by_the_gate_is_spoken() {
    for reply in ["Call", "Silent"] {
        let (said, recorded) = voice_tool_turn(vec![reply]).await;
        assert_eq!(said, vec![reply.to_owned()], "{reply}");
        assert_eq!(recorded, vec![reply.to_owned()], "{reply}");
    }
}

// ---------------------------------------------------------------------------
// Voice tool nudge is config (#151)
// ---------------------------------------------------------------------------

/// Drive one tool-enabled voice turn and return the trailing system message.
async fn trailing_with_ack(ack: &str) -> String {
    let mut reg = ToolRegistry::new();
    reg.register(Tool::new(
        "set_alarm",
        "",
        json!({}),
        Arc::new(FnHandler(|_args: Value, _ctx: ToolContext| async move {
            Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
        })),
    ))
    .unwrap();
    let llm = Arc::new(ScriptedToolLlm::new(vec![vec![
        text_delta("Sure."),
        finish("stop"),
    ]]));
    let s = store();
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        Arc::clone(&llm) as Arc<dyn familiar_connect::processors::ResponderLlm>,
        Arc::new(RecordingVoicePlayer {
            spoken: Arc::new(Mutex::new(Vec::new())),
        }),
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_tools(Arc::new(reg), voice_ctx_factory())
    .with_voice_tool_ack(ack);
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    llm.calls()[0].last().unwrap().content_str()
}

#[tokio::test]
async fn shipped_default_voice_tool_ack_reaches_the_trailing_reminder() {
    let trailing = trailing_with_ack(&default_config().voice_tool_ack).await;
    assert!(
        trailing.contains(
            "When you call a tool and still mean to be heard, pass `silent: \
             false` and speak at least a brief acknowledgement in the same \
             reply. A tool call on its own is silent \u{2014} nothing is spoken."
        ),
        "{trailing}"
    );
}

#[tokio::test]
async fn overridden_voice_tool_ack_reaches_the_trailing_reminder() {
    let trailing = trailing_with_ack("ACK_MARKER").await;
    assert!(trailing.contains("ACK_MARKER"), "{trailing}");
    assert!(!trailing.contains("brief acknowledgement"), "{trailing}");
}
