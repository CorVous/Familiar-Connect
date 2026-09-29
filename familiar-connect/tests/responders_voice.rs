//! Integration tests for `VoiceResponder` (subsystem 06). The heart of the
//! concurrency suite: per-user barge-in, the per-channel reply gate, and the
//! sub-200ms cut budget.

#![allow(clippy::significant_drop_tightening, clippy::format_collect)]

#[path = "responders_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use serde_json::Value;

use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::protocols::{BackpressurePolicy, EventBus};
use familiar_connect::bus::router::TurnRouter;
use familiar_connect::diagnostics::llm_mirror::{CallContext, current_call_context};
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::identity::Author;
use familiar_connect::llm::{LlmClient, Message};
use familiar_connect::processors::voice_responder::VoiceResponder;
use familiar_connect::processors::{MemberResolver, ResponderLlm};
use familiar_connect::tts_player::MockTTSPlayer;

use support::{
    CapturingLlm, LogCapture, ModelSpyLayer, ScriptedLlm, TestFocusManager, activity_start,
    default_config, default_modes, make_assembler, spy_assembler, store, text_delta, voice_final,
};

const fn bus() -> InProcessEventBus {
    InProcessEventBus::new()
}

fn voice_responder(
    s: Arc<AsyncHistoryStore>,
    llm: Arc<dyn ResponderLlm>,
    player: Arc<MockTTSPlayer>,
) -> (VoiceResponder, Arc<TurnRouter>) {
    let router = Arc::new(TurnRouter::new());
    let assembler = make_assembler(Arc::clone(&s));
    // Prompt prose is config-sourced; thread the shipped `_default` profile.
    let r = VoiceResponder::new(assembler, llm, player, s, Arc::clone(&router), "fam")
        .with_mode_instructions(default_modes())
        .with_voice_tool_ack(default_config().voice_tool_ack);
    (r, router)
}

// ---------------------------------------------------------------------------
// Activity start
// ---------------------------------------------------------------------------

#[tokio::test]
async fn activity_start_begins_turn() {
    let (r, router) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["hi"])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    r.handle(&activity_start("voice:1", "t-new", None), &bus())
        .await
        .unwrap();
    let scope = router.active_scope("voice:1").unwrap();
    assert_eq!(scope.turn_id, "t-new");
}

#[tokio::test]
async fn second_activity_start_cancels_first() {
    let (r, router) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["hi"])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    let first = router.active_scope("voice:1").unwrap();
    r.handle(&activity_start("voice:1", "t-2", None), &bus())
        .await
        .unwrap();
    let second = router.active_scope("voice:1").unwrap();
    assert_eq!(second.turn_id, "t-2");
    assert!(first.is_cancelled());
}

// ---------------------------------------------------------------------------
// Full reply + gates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_reply_on_final() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["Hello", ", ", "world", "."])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let calls = player.calls();
    assert!(!calls.is_empty());
    assert_eq!(calls[0].0, "Hello, world.");
    assert!(!calls[0].1);
    let contents: Vec<String> = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .map(|t| t.content)
        .collect();
    assert!(contents.iter().any(|c| c.contains("hi there")));
    assert!(contents.iter().any(|c| c.contains("Hello, world.")));
}

// ---------------------------------------------------------------------------
// Speakable-chunk gate
// ---------------------------------------------------------------------------

/// The live Cartesia 400: a reply ending in a trailing emoji leaves that emoji
/// alone as the sentence streamer's flush tail. It must never be synthesized,
/// the sentences before it still play, and the turn persists whole.
#[tokio::test]
async fn trailing_emoji_tail_is_not_spoken() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&[
            "Hey, something's broken! ",
            "I'm being honest here. ",
            "\u{1f605}",
        ])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("what broke?", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let spoken: Vec<String> = player.calls().into_iter().map(|c| c.0).collect();
    assert_eq!(
        spoken,
        vec![
            "Hey, something's broken!".to_owned(),
            "I'm being honest here.".to_owned(),
        ],
    );
    // The turn still records the whole reply, emoji included.
    let contents: Vec<String> = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .map(|t| t.content)
        .collect();
    assert!(contents.iter().any(|c| c.contains('\u{1f605}')));
}

/// An unspeakable chunk in the *middle* of the stream is dropped without
/// disturbing the order of the chunks around it.
#[tokio::test]
async fn unspeakable_chunk_mid_stream_keeps_order() {
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        store(),
        // "... " is a complete "sentence" to the splitter: terminator run then
        // whitespace.
        Arc::new(ScriptedLlm::new(&["First one. ", "... ", "Third one. "])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("go on", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let spoken: Vec<String> = player.calls().into_iter().map(|c| c.0).collect();
    assert_eq!(
        spoken,
        vec!["First one.".to_owned(), "Third one.".to_owned()],
    );
}

/// A reply ending on a boundary leaves an empty flush tail; it is not sent.
#[tokio::test]
async fn empty_flush_tail_is_not_spoken() {
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["All done. "])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("done?", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let spoken: Vec<String> = player.calls().into_iter().map(|c| c.0).collect();
    assert_eq!(spoken, vec!["All done.".to_owned()]);
}

/// Records the [`CallContext`] in scope when the responder calls the LLM.
struct ContextSpyLlm {
    seen: Arc<std::sync::Mutex<Vec<(CallContext, String)>>>,
}

#[async_trait]
impl LlmClient for ContextSpyLlm {
    async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
        Ok(Message::new("assistant", "ok"))
    }
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        _tools: Option<Vec<Value>>,
    ) -> anyhow::Result<familiar_connect::llm::LlmStream> {
        let system = messages
            .iter()
            .filter(|m| m.role == "system")
            .map(Message::content_str)
            .collect::<Vec<_>>()
            .join("\n\n");
        self.seen
            .lock()
            .expect("spy mutex")
            .push((current_call_context(), system));
        Ok(familiar_connect::llm::LlmStream::new(stream::iter(vec![
            Ok(text_delta("ok")),
        ])))
    }
    fn slot(&self) -> Option<&str> {
        Some("fast")
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}

impl familiar_connect::processors::ResponderLlm for ContextSpyLlm {}

#[tokio::test]
async fn the_voice_turn_scopes_the_call_context_for_the_mirror() {
    // What lets a mirrored `llm_calls` row name its turn, channel and scope.
    let s = store();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let llm = Arc::new(ContextSpyLlm {
        seen: Arc::clone(&seen),
    });
    let (r, _) = voice_responder(Arc::clone(&s), llm, Arc::new(MockTTSPlayer::new(5, 5)));
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("who is here?", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let calls = seen.lock().expect("spy mutex").clone();
    assert_eq!(calls.len(), 1);
    let (ctx, system) = &calls[0];
    assert_eq!(ctx.turn_scope.as_deref(), Some("t-1"));
    assert_eq!(ctx.channel_id, Some(1));
    // The anchoring turn is the user turn just appended.
    let user_turn = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .find(|t| t.content.contains("who is here?"))
        .expect("user turn");
    assert_eq!(ctx.turn_id, Some(user_turn.id));
    // And the system prompt the mirror would store is the assembled one.
    assert!(!system.is_empty());
}

#[tokio::test]
async fn trailing_reminder_carries_post_history() {
    let s = store();
    let llm = Arc::new(CapturingLlm::new("ok"));
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let r = {
        let (r, _) = voice_responder(Arc::clone(&s), llm.clone(), player);
        r.with_post_history_instructions("# Etiquette\n\nKeep it brief.")
    };
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let trailing = llm.captured()[0].last().unwrap().content_str();
    assert!(trailing.contains("# Etiquette"));
    assert!(
        trailing.find("You are speaking aloud").unwrap() < trailing.find("# Etiquette").unwrap()
    );
}

#[tokio::test]
async fn respond_decision_logged_once() {
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["Hello", ", ", "world", "."])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    let capture = LogCapture::install();
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let out = capture.contents();
    drop(capture);
    assert_eq!(
        out.lines().filter(|l| l.contains("respond")).count(),
        1,
        "{out}"
    );
    assert!(out.contains("t-1"));
}

/// The tool-less path's only route to silence: a `silent` call the model leaked
/// as plain text. There is no text sentinel.
const SILENT_LEAK: &str = "silent(reasoning=\"not addressed to me\")";

/// `<silent>` used to gate the whole reply. It is ordinary prose now.
#[tokio::test]
async fn literal_silent_string_is_spoken_like_any_other_text() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["<silent>"])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(
        player.calls().iter().any(|c| c.0.contains("<silent>")),
        "{:?}",
        player.calls()
    );
}

#[tokio::test]
async fn leaked_silent_call_skips_tts_and_assistant_turn() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&[SILENT_LEAK])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi nobody", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(player.calls().is_empty());
    let turns = s.sync().recent("fam", 1, 10, None, None).unwrap();
    assert!(turns.iter().all(|t| t.role != "assistant"));
    assert!(
        turns
            .iter()
            .any(|t| t.role == "user" && t.content.contains("hi nobody"))
    );
}

/// Issue #220: the silent decision abandons the stream, but the transport must
/// hear `silent`, not the default `cancelled` (barge-in).
#[tokio::test]
async fn silent_decision_notes_silent_abandon_status() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let llm = Arc::new(ScriptedLlm::new(&[SILENT_LEAK]));
    let (r, _) = voice_responder(Arc::clone(&s), llm.clone(), player.clone());
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi nobody", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert_eq!(llm.abandon_status(), "silent");
}

/// Same seam for the leaked-tool-call arm.
#[tokio::test]
async fn suppressed_leak_notes_suppressed_abandon_status() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let llm = Arc::new(ScriptedLlm::new(&["<invoke name=\"read_channel\">"]));
    let (r, _) = voice_responder(Arc::clone(&s), llm.clone(), player.clone());
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert_eq!(llm.abandon_status(), "suppressed");
}

#[tokio::test]
async fn silent_split_across_deltas_gates() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(1, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["sil", "ent(reasoning=x)"])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(player.calls().is_empty());
    assert!(
        s.sync()
            .recent("fam", 1, 10, None, None)
            .unwrap()
            .iter()
            .all(|t| t.role != "assistant")
    );
}

#[tokio::test]
async fn empty_reply_skips_tts_and_assistant_turn() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["   ", "\n", "\t"])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(player.calls().is_empty());
    let turns = s.sync().recent("fam", 1, 10, None, None).unwrap();
    assert!(turns.iter().all(|t| t.role != "assistant"));
    assert!(
        turns
            .iter()
            .any(|t| t.role == "user" && t.content.contains("hi there"))
    );
}

// ---------------------------------------------------------------------------
// Leaked tool-call XML (issue #109) — must never reach TTS or history
// ---------------------------------------------------------------------------

#[tokio::test]
async fn leaked_invoke_xml_content_never_spoken_bare_path() {
    // The model leaks a tool call as PLAIN CONTENT (no structured tool_calls),
    // split across deltas. The streaming gate must suppress it before TTS, and
    // no assistant turn (raw XML) may land in history.
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&[
            "<invoke name=\"read_channel\">",
            "<parameter name=\"limit\">10</parameter>",
            "</invoke>",
        ])),
        player.clone(),
    );
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
    assert!(
        player.calls().is_empty(),
        "leak spoken: {:?}",
        player.calls()
    );
    let turns = s.sync().recent("fam", 1, 10, None, None).unwrap();
    assert!(turns.iter().all(|t| t.role != "assistant"));
    assert!(turns.iter().all(|t| !t.content.contains("invoke")));
}

#[tokio::test]
async fn leaked_silent_invoke_content_treated_as_silent_bare_path() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["<invoke name=\"silent\">", "</invoke>"])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hush", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(player.calls().is_empty());
    let turns = s.sync().recent("fam", 1, 10, None, None).unwrap();
    assert!(turns.iter().all(|t| t.role != "assistant"));
}

#[tokio::test]
async fn leading_angle_bracket_prose_still_spoken() {
    // False-positive guard: legit `<`-leading prose must open the speak path.
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["<3 you all, ", "friends."])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("love?", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let spoken: Vec<String> = player.calls().into_iter().map(|(t, _)| t).collect();
    assert!(!spoken.is_empty(), "legit <3 prose was suppressed");
    assert!(spoken.join(" ").contains("<3"), "spoken: {spoken:?}");
    let assistants: Vec<String> = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .filter(|t| t.role == "assistant")
        .map(|t| t.content)
        .collect();
    assert_eq!(assistants, ["<3 you all, friends."]);
}

#[tokio::test]
async fn stale_final_ignored() {
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["ignored"])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-new", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("old", "voice:1", "t-OLD", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(player.calls().is_empty());
}

// ---------------------------------------------------------------------------
// Sentence streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_sentence_reply_speaks_each() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(1, 5));
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&[
            "Hello there. ",
            "How are you? ",
            "Nice to meet you.",
        ])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let spoken: Vec<String> = player.calls().into_iter().map(|(t, _)| t).collect();
    assert_eq!(
        spoken,
        ["Hello there.", "How are you?", "Nice to meet you."]
    );
    let assistants: Vec<String> = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .filter(|t| t.role == "assistant")
        .map(|t| t.content)
        .collect();
    assert_eq!(assistants, ["Hello there. How are you? Nice to meet you."]);
}

#[tokio::test]
async fn first_sentence_reaches_tts_before_stream_ends() {
    let player = Arc::new(MockTTSPlayer::new(1, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::with_delay(
            &["First sentence. ", "Second sentence. ", "Third."],
            50,
        )),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    let start = tokio::time::Instant::now();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(player.calls().len() >= 2);
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn partial_buffer_flushed_when_stream_ends() {
    let player = Arc::new(MockTTSPlayer::new(1, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["just a fragment"])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let spoken: Vec<String> = player.calls().into_iter().map(|(t, _)| t).collect();
    assert_eq!(spoken, ["just a fragment"]);
}

// ---------------------------------------------------------------------------
// Barge-in
// ---------------------------------------------------------------------------

#[tokio::test]
async fn barge_in_during_stream_prevents_speech() {
    let deltas: Vec<String> = (0..10).map(|i| format!("d{i} ")).collect();
    let delta_refs: Vec<&str> = deltas.iter().map(String::as_str).collect();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::with_delay(&delta_refs, 50)),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(80)).await;
    r.handle(&activity_start("voice:1", "t-2", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;

    let full: String = (0..10).map(|i| format!("d{i} ")).collect();
    let full = full.trim().to_owned();
    if let Some((spoken, cut)) = player.calls().into_iter().next() {
        assert!(cut || spoken != full);
    }
}

/// A stream gated between the two deltas: it yields `<sil` (an ambiguous
/// prefix, leaving the stream gate undecided), signals `d1`, then parks until
/// `release`, then yields `ent>`. This makes the "cancel lands between deltas" scenario deterministic
/// under any test-runner load (no wall-clock racing).
struct GatedLlm {
    d1_flag: Arc<AtomicBool>,
    d1_notify: Arc<tokio::sync::Notify>,
    proceed_flag: Arc<AtomicBool>,
    proceed_notify: Arc<tokio::sync::Notify>,
}

impl GatedLlm {
    fn new() -> Self {
        Self {
            d1_flag: Arc::new(AtomicBool::new(false)),
            d1_notify: Arc::new(tokio::sync::Notify::new()),
            proceed_flag: Arc::new(AtomicBool::new(false)),
            proceed_notify: Arc::new(tokio::sync::Notify::new()),
        }
    }
    async fn wait_d1(&self) {
        while !self.d1_flag.load(Ordering::SeqCst) {
            self.d1_notify.notified().await;
        }
    }
    fn release(&self) {
        self.proceed_flag.store(true, Ordering::SeqCst);
        self.proceed_notify.notify_waiters();
    }
}

#[async_trait]
impl LlmClient for GatedLlm {
    async fn chat(&self, _m: Vec<Message>) -> anyhow::Result<Message> {
        Ok(Message::new("assistant", "x"))
    }
    async fn stream_completion(
        &self,
        _m: Vec<Message>,
        _t: Option<Vec<Value>>,
    ) -> anyhow::Result<familiar_connect::llm::LlmStream> {
        let d1_flag = Arc::clone(&self.d1_flag);
        let d1_notify = Arc::clone(&self.d1_notify);
        let proceed_flag = Arc::clone(&self.proceed_flag);
        let proceed_notify = Arc::clone(&self.proceed_notify);
        let s = stream::unfold(0usize, move |i| {
            let d1_flag = Arc::clone(&d1_flag);
            let d1_notify = Arc::clone(&d1_notify);
            let proceed_flag = Arc::clone(&proceed_flag);
            let proceed_notify = Arc::clone(&proceed_notify);
            async move {
                match i {
                    0 => Some((Ok(text_delta("<sil")), 1)),
                    1 => {
                        // The responder has consumed delta 1 and is polling for
                        // the next; signal, then park for the barge.
                        d1_flag.store(true, Ordering::SeqCst);
                        d1_notify.notify_waiters();
                        while !proceed_flag.load(Ordering::SeqCst) {
                            proceed_notify.notified().await;
                        }
                        Some((Ok(text_delta("ent>")), 2))
                    }
                    _ => None,
                }
            }
        });
        Ok(familiar_connect::llm::LlmStream::new(s))
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}
impl ResponderLlm for GatedLlm {}

#[tokio::test]
async fn barge_in_logs_preempted_decision() {
    let s = store();
    let llm = Arc::new(GatedLlm::new());
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(Arc::clone(&s), llm.clone(), player);
    let capture = LogCapture::install();
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    // Wait until delta 1 has been consumed and the stream is parked; then barge
    // and release delta 2 so the cancel-check at the loop top wins.
    llm.wait_d1().await;
    r.handle(&activity_start("voice:1", "t-2", None), &bus())
        .await
        .unwrap();
    llm.release();
    r.wait_until_idle().await;
    let out = capture.contents();
    drop(capture);
    let preempted = out
        .lines()
        .filter(|l| l.contains("preempted") && l.contains("t-1"))
        .count();
    let other = out
        .lines()
        .filter(|l| (l.contains("silent") || l.contains("respond")) && l.contains("t-1"))
        .count();
    assert_eq!(other, 0, "unexpected silent/respond for t-1: {out}");
    assert_eq!(preempted, 1, "{out}");
}

#[tokio::test]
async fn barge_in_during_speech_cuts_playback_fast() {
    let words: String = (0..50).map(|i| format!("w{i} ")).collect();
    let player = Arc::new(MockTTSPlayer::new(20, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&[words.trim()])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("go ahead", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), player.await_speak_started())
        .await
        .expect("speak started");
    let cancel_time = tokio::time::Instant::now();
    r.handle(&activity_start("voice:1", "t-2", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let cut_ms = cancel_time.elapsed().as_millis();
    assert!(
        cut_ms < 200,
        "barge-in took {cut_ms}ms; should be under 200"
    );
    let calls = player.calls();
    assert!(!calls.is_empty());
    assert!(calls[0].1, "playback was not cut");
}

// ---------------------------------------------------------------------------
// Per-user barge-in
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cross_user_start_does_not_cancel() {
    let (r, router) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["x"])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    r.handle(&activity_start("voice:1", "t-alice", Some(101)), &bus())
        .await
        .unwrap();
    r.handle(&activity_start("voice:1", "t-bob", Some(202)), &bus())
        .await
        .unwrap();
    let alice = router.active_scope("voice:1:user:101").unwrap();
    let bob = router.active_scope("voice:1:user:202").unwrap();
    assert!(!alice.is_cancelled());
    assert_eq!(alice.turn_id, "t-alice");
    assert_eq!(bob.turn_id, "t-bob");
}

#[tokio::test]
async fn same_user_start_still_cancels() {
    let (r, router) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["x"])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    r.handle(&activity_start("voice:1", "t-1", Some(101)), &bus())
        .await
        .unwrap();
    let first = router.active_scope("voice:1:user:101").unwrap();
    r.handle(&activity_start("voice:1", "t-2", Some(101)), &bus())
        .await
        .unwrap();
    let second = router.active_scope("voice:1:user:101").unwrap();
    assert_eq!(second.turn_id, "t-2");
    assert!(first.is_cancelled());
}

#[tokio::test]
async fn cross_user_does_not_cancel_in_flight_reply() {
    let deltas: Vec<String> = (0..5).map(|i| format!("d{i} ")).collect();
    let refs: Vec<&str> = deltas.iter().map(String::as_str).collect();
    let player = Arc::new(MockTTSPlayer::new(5, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::with_delay(&refs, 30)),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-alice", Some(101)), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-alice", Some(101)), &bus())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    r.handle(&activity_start("voice:1", "t-bob", Some(202)), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let full: String = (0..5).map(|i| format!("d{i} ")).collect();
    let calls = player.calls();
    assert!(!calls.is_empty());
    assert!(!calls[0].1);
    assert_eq!(calls[0].0, full);
}

#[tokio::test]
async fn other_users_continuous_starts_do_not_cut_playback() {
    let player = Arc::new(MockTTSPlayer::new(200, 5));
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["hello "])),
        player.clone(),
    );
    r.handle(&activity_start("voice:1", "t-alice", Some(101)), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-alice", Some(101)), &bus())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), player.await_speak_started())
        .await
        .expect("speak started");
    for i in 1..7 {
        r.handle(
            &activity_start("voice:1", &format!("t-bob-{i}"), Some(202)),
            &bus(),
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    r.wait_until_idle().await;
    let calls = player.calls();
    assert!(!calls.is_empty());
    assert!(!calls[0].1, "Bob's repeated starts cut Alice's playback");
    assert_eq!(calls[0].0, "hello ");
}

// ---------------------------------------------------------------------------
// Cross-user reply gate (barrier LLM: max_active == 1)
// ---------------------------------------------------------------------------

struct BarrierLlm {
    active: AtomicI64,
    max_active: AtomicI64,
    entered: AtomicUsize,
    second_arrived: AtomicBool,
    notify: tokio::sync::Notify,
}

impl BarrierLlm {
    fn new() -> Self {
        Self {
            active: AtomicI64::new(0),
            max_active: AtomicI64::new(0),
            entered: AtomicUsize::new(0),
            second_arrived: AtomicBool::new(false),
            notify: tokio::sync::Notify::new(),
        }
    }
    fn max_active(&self) -> i64 {
        self.max_active.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmClient for BarrierLlm {
    async fn chat(&self, _m: Vec<Message>) -> anyhow::Result<Message> {
        Ok(Message::new("assistant", "x"))
    }
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        _t: Option<Vec<Value>>,
    ) -> anyhow::Result<familiar_connect::llm::LlmStream> {
        let already = messages.iter().any(|m| m.role == "assistant");
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        let entered = self.entered.fetch_add(1, Ordering::SeqCst) + 1;
        if entered >= 2 {
            self.second_arrived.store(true, Ordering::SeqCst);
            self.notify.notify_waiters();
        } else {
            let _ = tokio::time::timeout(Duration::from_millis(300), async {
                while !self.second_arrived.load(Ordering::SeqCst) {
                    self.notify.notified().await;
                }
            })
            .await;
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        let content = if already { SILENT_LEAK } else { "Sure thing." };
        Ok(familiar_connect::llm::LlmStream::new(stream::once(
            async move { Ok(text_delta(content)) },
        )))
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}
impl ResponderLlm for BarrierLlm {}

#[tokio::test]
async fn concurrent_cross_user_finals_serialize_to_one_reply() {
    let s = store();
    let llm = Arc::new(BarrierLlm::new());
    let player = Arc::new(MockTTSPlayer::new(1, 5));
    let (r, _) = voice_responder(Arc::clone(&s), llm.clone(), player.clone());
    r.handle(&activity_start("voice:1", "t-alice", Some(101)), &bus())
        .await
        .unwrap();
    r.handle(&activity_start("voice:1", "t-bob", Some(202)), &bus())
        .await
        .unwrap();
    r.handle(
        &voice_final("did you watch it", "voice:1", "t-alice", Some(101)),
        &bus(),
    )
    .await
    .unwrap();
    r.handle(
        &voice_final("what did you think", "voice:1", "t-bob", Some(202)),
        &bus(),
    )
    .await
    .unwrap();
    r.wait_until_idle().await;

    assert_eq!(llm.max_active(), 1, "reply streams overlapped");
    let calls = player.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "Sure thing.");
    let turns = s.sync().recent("fam", 1, 20, None, None).unwrap();
    assert_eq!(turns.iter().filter(|t| t.role == "assistant").count(), 1);
    let users: Vec<String> = turns
        .iter()
        .filter(|t| t.role == "user")
        .map(|t| t.content.clone())
        .collect();
    assert!(users.iter().any(|c| c.contains("did you watch it")));
    assert!(users.iter().any(|c| c.contains("what did you think")));
}

// ---------------------------------------------------------------------------
// Author resolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn member_resolver_threads_author_into_history() {
    let s = store();
    let resolver: MemberResolver = Arc::new(|channel_id, user_id| {
        assert_eq!(channel_id, 1);
        assert_eq!(user_id, 101);
        Some(Author::new(
            "discord",
            "101",
            Some("cassidy".to_owned()),
            Some("Cor Vous".to_owned()),
        ))
    });
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ack"])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    let r = r.with_member_resolver(resolver);
    r.handle(&activity_start("voice:1", "t-1", Some(101)), &bus())
        .await
        .unwrap();
    r.handle(
        &voice_final("hello there", "voice:1", "t-1", Some(101)),
        &bus(),
    )
    .await
    .unwrap();
    r.wait_until_idle().await;
    let user = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .rfind(|t| t.role == "user")
        .unwrap();
    let author = user.author.unwrap();
    assert_eq!(author.display_name.as_deref(), Some("Cor Vous"));
    assert_eq!(author.user_id, "101");
}

// A resolver miss (speaker absent from the voice-member cache) must still
// persist the numeric user id — an unidentified but *distinct* speaker beats a
// fully anonymous turn (N16).
#[tokio::test]
async fn resolver_miss_keeps_numeric_user_id() {
    let s = store();
    let resolver: MemberResolver = Arc::new(|_c, _u| None);
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ack"])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    let r = r.with_member_resolver(resolver);
    r.handle(&activity_start("voice:1", "t-1", Some(101)), &bus())
        .await
        .unwrap();
    r.handle(
        &voice_final("hello there", "voice:1", "t-1", Some(101)),
        &bus(),
    )
    .await
    .unwrap();
    r.wait_until_idle().await;
    let user = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .rfind(|t| t.role == "user")
        .unwrap();
    assert!(user.content.contains("hello there"));
    let author = user.author.expect("author");
    assert_eq!(author.platform, "discord");
    assert_eq!(author.user_id, "101");
    assert!(author.display_name.is_none());
    assert!(author.username.is_none());
}

// With no resolver wired at all there is still an id to keep.
#[tokio::test]
async fn missing_resolver_keeps_numeric_user_id() {
    let s = store();
    let (r, _) = voice_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ack"])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    r.handle(&activity_start("voice:1", "t-1", Some(202)), &bus())
        .await
        .unwrap();
    r.handle(
        &voice_final("hi again", "voice:1", "t-1", Some(202)),
        &bus(),
    )
    .await
    .unwrap();
    r.wait_until_idle().await;
    let user = s
        .sync()
        .recent("fam", 1, 10, None, None)
        .unwrap()
        .into_iter()
        .rfind(|t| t.role == "user")
        .unwrap();
    assert_eq!(user.author.expect("author").user_id, "202");
}

// ---------------------------------------------------------------------------
// Trailing reminder + server name
// ---------------------------------------------------------------------------

#[tokio::test]
async fn trailing_carries_voice_directive() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let (r, _) = voice_responder(store(), llm.clone(), Arc::new(MockTTSPlayer::new(1, 5)));
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let trailing = llm.captured()[0].last().unwrap().content_str();
    assert!(trailing.contains("You are speaking aloud"));
    assert!(trailing.contains("It is now"));
}

#[tokio::test]
async fn trailing_names_server_and_channel() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let fm = Arc::new(
        TestFocusManager::focused(1)
            .with_channel_name(1, "voice-chan")
            .with_guild_name(1, "My Server"),
    );
    let (r, _) = voice_responder(store(), llm.clone(), Arc::new(MockTTSPlayer::new(5, 5)));
    let r = r.with_focus_manager(fm);
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let trailing = llm.captured()[0].last().unwrap().content_str();
    assert!(trailing.contains("#voice-chan"));
    assert!(trailing.contains("\"My Server\" server"));
}

#[tokio::test]
async fn trailing_omits_focus_without_focus_manager() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let (r, _) = voice_responder(store(), llm.clone(), Arc::new(MockTTSPlayer::new(5, 5)));
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let trailing = llm.captured()[0].last().unwrap().content_str();
    assert!(!trailing.contains("Your attention is currently on"));
}

// ---------------------------------------------------------------------------
// Per-turn origin logging
// ---------------------------------------------------------------------------

#[tokio::test]
async fn respond_log_names_server_and_channel() {
    let fm = Arc::new(
        TestFocusManager::focused(1)
            .with_channel_name(1, "voice-chan")
            .with_guild_name(1, "My Server"),
    );
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["Hello", ", ", "world", "."])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    let r = r.with_focus_manager(fm);
    let capture = LogCapture::install();
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let out = capture.contents();
    drop(capture);
    let line = out.lines().find(|l| l.contains("respond")).unwrap_or("");
    assert!(line.contains("#voice-chan"), "{out}");
    assert!(line.contains("My Server"), "{out}");
}

#[tokio::test]
async fn respond_log_omits_server_without_focus_manager() {
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["Hello", ", ", "world", "."])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    let capture = LogCapture::install();
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let out = capture.contents();
    drop(capture);
    let line = out.lines().find(|l| l.contains("respond")).unwrap_or("");
    assert!(line.contains("#1"), "{out}");
    assert!(!line.contains("srv="), "{out}");
}

#[tokio::test]
async fn silent_log_names_server_and_channel() {
    let fm = Arc::new(
        TestFocusManager::focused(1)
            .with_channel_name(1, "voice-chan")
            .with_guild_name(1, "My Server"),
    );
    let (r, _) = voice_responder(
        store(),
        Arc::new(ScriptedLlm::new(&[SILENT_LEAK])),
        Arc::new(MockTTSPlayer::new(5, 5)),
    );
    let r = r.with_focus_manager(fm);
    let capture = LogCapture::install();
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi nobody", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let out = capture.contents();
    drop(capture);
    let line = out.lines().find(|l| l.contains("silent")).unwrap_or("");
    assert!(line.contains("#voice-chan"), "{out}");
    assert!(line.contains("My Server"), "{out}");
}

// ---------------------------------------------------------------------------
// Dispatch loop — the dispatcher must keep pulling while a final is parked
// ---------------------------------------------------------------------------

struct BlockingLlm {
    started: AtomicBool,
    started_notify: tokio::sync::Notify,
    unblock: AtomicBool,
    unblock_notify: tokio::sync::Notify,
}

impl BlockingLlm {
    fn new() -> Self {
        Self {
            started: AtomicBool::new(false),
            started_notify: tokio::sync::Notify::new(),
            unblock: AtomicBool::new(false),
            unblock_notify: tokio::sync::Notify::new(),
        }
    }
    async fn wait_started(&self) {
        while !self.started.load(Ordering::SeqCst) {
            self.started_notify.notified().await;
        }
    }
    fn release(&self) {
        self.unblock.store(true, Ordering::SeqCst);
        self.unblock_notify.notify_waiters();
    }
}

#[async_trait]
impl LlmClient for BlockingLlm {
    async fn chat(&self, _m: Vec<Message>) -> anyhow::Result<Message> {
        Ok(Message::new("assistant", "x"))
    }
    async fn stream_completion(
        &self,
        _m: Vec<Message>,
        _t: Option<Vec<Value>>,
    ) -> anyhow::Result<familiar_connect::llm::LlmStream> {
        self.started.store(true, Ordering::SeqCst);
        self.started_notify.notify_waiters();
        while !self.unblock.load(Ordering::SeqCst) {
            self.unblock_notify.notified().await;
        }
        Ok(familiar_connect::llm::LlmStream::new(stream::once(
            async move { Ok(text_delta("hello")) },
        )))
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}
impl ResponderLlm for BlockingLlm {}

#[tokio::test]
async fn dispatcher_unblocked_during_in_flight_final() {
    let bus = Arc::new(InProcessEventBus::new());
    bus.start().await;
    let llm = Arc::new(BlockingLlm::new());
    let router = Arc::new(TurnRouter::new());
    let assembler = make_assembler(store());
    let responder = Arc::new(VoiceResponder::new(
        assembler,
        llm.clone(),
        Arc::new(MockTTSPlayer::new(5, 5)),
        store(),
        Arc::clone(&router),
        "fam",
    ));

    let disp_bus = Arc::clone(&bus);
    let disp_r = Arc::clone(&responder);
    let topics = responder.topics();
    let dispatcher = tokio::spawn(async move {
        let mut sub = disp_bus.subscribe(&topics, BackpressurePolicy::Block, 0);
        while let Some(ev) = sub.recv().await {
            let _ = disp_r.handle(&ev, disp_bus.as_ref()).await;
        }
    });

    tokio::task::yield_now().await;
    bus.publish(activity_start("voice:1", "t-1", None)).await;
    bus.publish(voice_final("hi", "voice:1", "t-1", None)).await;
    tokio::time::timeout(Duration::from_secs(1), llm.wait_started())
        .await
        .expect("llm started");
    bus.publish(activity_start("voice:1", "t-2", None)).await;

    let mut ok = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(10)).await;
        if router.active_scope("voice:1").map(|s| s.turn_id.clone()) == Some("t-2".to_owned()) {
            ok = true;
            break;
        }
    }
    assert!(ok, "dispatcher blocked on in-flight final");

    llm.release();
    responder.wait_until_idle().await;
    bus.shutdown().await;
    dispatcher.abort();
}

// ---------------------------------------------------------------------------
// Reminder prose is config (#151)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn trailing_voice_directive_matches_the_shipped_default_byte_for_byte() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let (r, _) = voice_responder(store(), llm.clone(), Arc::new(MockTTSPlayer::new(1, 5)));
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let trailing = llm.captured()[0].last().unwrap().content_str();
    assert!(
        trailing.contains(
            "You are speaking aloud. Keep replies short (one or two sentences). \
             Avoid markdown."
        ),
        "{trailing}"
    );
}

#[tokio::test]
async fn overridden_operating_mode_reaches_the_trailing_reminder() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let (r, _) = voice_responder(store(), llm.clone(), Arc::new(MockTTSPlayer::new(1, 5)));
    let r = r.with_mode_instructions(
        std::iter::once(("voice".to_owned(), "WHISPER_MARKER".to_owned())).collect(),
    );
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let trailing = llm.captured()[0].last().unwrap().content_str();
    assert!(trailing.contains("WHISPER_MARKER"), "{trailing}");
    assert!(!trailing.contains("You are speaking aloud"), "{trailing}");
}

// ---------------------------------------------------------------------------
// Model threading (#183)
// ---------------------------------------------------------------------------

/// Voice assembles with the model too — same seam as text.
#[tokio::test]
async fn assembly_context_carries_the_llm_model() {
    let s = store();
    let spy = Arc::new(ModelSpyLayer::default());
    let router = Arc::new(TurnRouter::new());
    let r = VoiceResponder::new(
        spy_assembler(Arc::clone(&s), &spy),
        Arc::new(ScriptedLlm::new(&["hi"]).with_model("vendor/model-x")),
        Arc::new(MockTTSPlayer::new(5, 5)),
        s,
        router,
        "fam",
    )
    .with_mode_instructions(default_modes())
    .with_voice_tool_ack(default_config().voice_tool_ack);
    r.handle(&activity_start("voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hello", "voice:1", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert_eq!(spy.seen(), vec!["vendor/model-x".to_owned()]);
}
