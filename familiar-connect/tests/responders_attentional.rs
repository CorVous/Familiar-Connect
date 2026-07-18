//! Focus-aware responder tests (subsystem 06; Python
//! `tests/test_attentional_responders.py`).

#[path = "responders_support/mod.rs"]
mod support;

use std::sync::Arc;

use familiar_connect::bus::envelope::{Event, payload as wrap_payload};
use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::router::TurnRouter;
use familiar_connect::bus::topics::TOPIC_DISCORD_TEXT;
use familiar_connect::focus::FocusManager;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::processors::text_responder::TextResponder;
use familiar_connect::processors::voice_responder::VoiceResponder;
use familiar_connect::processors::{
    DiscordTextPayload, FocusManagerApi, ResponderLlm, ToolContextFactory,
};
use familiar_connect::subscriptions::{SubscriptionKind, SubscriptionRegistry, SubscriptionView};
use familiar_connect::tools::registry::{ToolContext, ToolRegistry};
use familiar_connect::tools::shift_focus::build_shift_focus_tool;
use familiar_connect::tts_player::MockTTSPlayer;
use serde_json::json;

use support::{
    CapturingLlm, CapturingSend, RecordingBus, ScriptedLlm, ScriptedToolLlm, TestFocusManager,
    activity_start, discord_text_event, finish, make_assembler, store, tc_delta, text_delta,
    text_payload, voice_final,
};

const fn bus() -> InProcessEventBus {
    InProcessEventBus::new()
}

fn text_responder(
    s: Arc<AsyncHistoryStore>,
    llm: Arc<dyn ResponderLlm>,
    send: Arc<CapturingSend>,
    fm: Option<Arc<dyn FocusManagerApi>>,
) -> TextResponder {
    let assembler = make_assembler(Arc::clone(&s));
    let mut r = TextResponder::new(assembler, llm, send, s, Arc::new(TurnRouter::new()), "fam");
    if let Some(fm) = fm {
        r = r.with_focus_manager(fm);
    }
    r
}

// ---------------------------------------------------------------------------
// TextResponder focus-aware
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unfocused_message_is_staged_no_reply() {
    let s = store();
    let send = Arc::new(CapturingSend::new());
    let fm = Arc::new(TestFocusManager::unfocused());
    let r = text_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["should not be called"])),
        send.clone(),
        Some(fm),
    );
    r.handle(
        &discord_text_event(text_payload(100, "hello"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert!(send.calls().is_empty());
    let turns = s.sync().recent("fam", 100, 10, None, None).unwrap();
    let user = turns.iter().find(|t| t.role == "user").unwrap();
    assert!(user.content.contains("hello"));
    assert!(user.consumed_at.is_none());
}

#[tokio::test]
async fn focused_message_generates_reply_and_calls_end_turn() {
    let s = store();
    let send = Arc::new(CapturingSend::new());
    let fm = Arc::new(TestFocusManager::focused(100));
    let r = text_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["hi there"])),
        send.clone(),
        Some(fm.clone()),
    );
    r.handle(&discord_text_event(text_payload(100, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(send.calls().len(), 1);
    assert_eq!(send.calls()[0].1, "hi there");
    assert_eq!(fm.end_turn_count(), 1);
}

#[tokio::test]
async fn focused_passes_focus_context_to_reminder() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let fm = Arc::new(TestFocusManager::focused(100));
    let r = text_responder(
        store(),
        llm.clone(),
        Arc::new(CapturingSend::new()),
        Some(fm),
    );
    r.handle(&discord_text_event(text_payload(100, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let trailing = llm.captured()[0].last().unwrap().content_str();
    assert!(
        trailing.contains("attention is currently on #100"),
        "{trailing}"
    );
}

#[tokio::test]
async fn no_focus_manager_backward_compat() {
    let send = Arc::new(CapturingSend::new());
    let r = text_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["compat reply"])),
        send.clone(),
        None,
    );
    r.handle(&discord_text_event(text_payload(100, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(send.calls().len(), 1);
    assert_eq!(send.calls()[0].1, "compat reply");
}

#[tokio::test]
async fn unfocused_does_not_call_end_turn() {
    let fm = Arc::new(TestFocusManager::unfocused());
    let r = text_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["x"])),
        Arc::new(CapturingSend::new()),
        Some(fm.clone()),
    );
    r.handle(&discord_text_event(text_payload(100, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(fm.end_turn_count(), 0);
}

// ---------------------------------------------------------------------------
// Idle nudge
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unfocused_arrival_publishes_wake_to_focused_channel() {
    let fm = Arc::new(
        TestFocusManager::unfocused()
            .with_should_wake(true)
            .with_text_focus(Some(555)),
    );
    let r = text_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["x"])),
        Arc::new(CapturingSend::new()),
        Some(fm.clone()),
    );
    let rec = RecordingBus::new();
    r.handle(&discord_text_event(text_payload(100, "hi"), "e-1"), &rec)
        .await
        .unwrap();
    assert_eq!(fm.nudge_count(), 1);
    let published = rec.published();
    let wake_payloads: Vec<&DiscordTextPayload> = published
        .iter()
        .filter_map(|e| e.payload.downcast_ref::<DiscordTextPayload>())
        .filter(|p| p.wake)
        .collect();
    assert_eq!(wake_payloads.len(), 1);
    assert_eq!(wake_payloads[0].channel_id, 555);
}

#[tokio::test]
async fn unfocused_arrival_publishes_nothing_when_no_wake() {
    let fm = Arc::new(TestFocusManager::unfocused());
    let r = text_responder(
        store(),
        Arc::new(ScriptedLlm::new(&["x"])),
        Arc::new(CapturingSend::new()),
        Some(fm.clone()),
    );
    let rec = RecordingBus::new();
    r.handle(&discord_text_event(text_payload(100, "hi"), "e-1"), &rec)
        .await
        .unwrap();
    assert_eq!(fm.nudge_count(), 0);
    let published = rec.published();
    let wakes = published
        .iter()
        .filter(|e| {
            e.payload
                .downcast_ref::<DiscordTextPayload>()
                .is_some_and(|p| p.wake)
        })
        .count();
    assert_eq!(wakes, 0);
}

#[tokio::test]
async fn wake_reply_without_shift_is_suppressed() {
    // Wake = shift-or-silent (#170): a wake turn that produces prose but does
    // NOT shift focus this turn is suppressed entirely — the reply would
    // otherwise post to the stale focus channel. No user turn is persisted
    // either (wake events never stage).
    let s = store();
    let send = Arc::new(CapturingSend::new());
    let fm = Arc::new(TestFocusManager::focused(555));
    let r = text_responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["checking in"])),
        send.clone(),
        Some(fm.clone()),
    );
    let wake = Event {
        event_id: "wake-1".to_owned(),
        turn_id: "wake-1".to_owned(),
        session_id: "discord:555".to_owned(),
        parent_event_ids: Vec::new(),
        topic: TOPIC_DISCORD_TEXT.to_owned(),
        timestamp: chrono::Utc::now(),
        sequence_number: 0,
        payload: wrap_payload(DiscordTextPayload {
            familiar_id: "fam".to_owned(),
            channel_id: 555,
            content: "[idle check]".to_owned(),
            author: None,
            wake: true,
            ..Default::default()
        }),
    };
    r.handle(&wake, &bus()).await.unwrap();
    assert!(send.calls().is_empty());
    assert_eq!(fm.end_turn_count(), 0);
    let user_turns = s
        .sync()
        .recent("fam", 555, 10, None, None)
        .unwrap()
        .into_iter()
        .filter(|t| t.role == "user")
        .count();
    assert_eq!(user_turns, 0);
}

// ---------------------------------------------------------------------------
// VoiceResponder focus-aware
// ---------------------------------------------------------------------------

#[tokio::test]
async fn voice_turn_calls_end_turn() {
    let s = store();
    let fm = Arc::new(TestFocusManager::focused(200));
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        Arc::new(ScriptedLlm::new(&["sure"])),
        Arc::new(MockTTSPlayer::new(1, 5)),
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_focus_manager(fm.clone() as Arc<dyn FocusManagerApi>);
    r.handle(&activity_start("voice:200", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:200", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert_eq!(fm.end_turn_count(), 1);
}

#[tokio::test]
async fn voice_no_focus_manager_backward_compat() {
    let s = store();
    let player = Arc::new(MockTTSPlayer::new(1, 5));
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        Arc::new(ScriptedLlm::new(&["hello there"])),
        player.clone(),
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    );
    r.handle(&activity_start("voice:200", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:200", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    let calls = player.calls();
    assert!(!calls.is_empty());
    assert!(calls[0].0.contains("hello there"));
    assert!(!calls[0].1);
}

#[tokio::test]
async fn voice_end_turn_not_called_on_silent() {
    let s = store();
    let fm = Arc::new(TestFocusManager::focused(200));
    let player = Arc::new(MockTTSPlayer::new(1, 5));
    let assembler = make_assembler(Arc::clone(&s));
    let r = VoiceResponder::new(
        assembler,
        Arc::new(ScriptedLlm::new(&["<silent>"])),
        player.clone(),
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_focus_manager(fm.clone() as Arc<dyn FocusManagerApi>);
    r.handle(&activity_start("voice:200", "t-1", None), &bus())
        .await
        .unwrap();
    r.handle(&voice_final("hi there", "voice:200", "t-1", None), &bus())
        .await
        .unwrap();
    r.wait_until_idle().await;
    assert!(player.calls().is_empty());
    assert_eq!(fm.end_turn_count(), 0);
}

// ---------------------------------------------------------------------------
// Immediate shift_focus (real FocusManager + shift_focus tool)
// ---------------------------------------------------------------------------

fn real_focus_responder(
    llm: Arc<ScriptedToolLlm>,
    send: Arc<CapturingSend>,
) -> (TextResponder, Arc<FocusManager>, Arc<AsyncHistoryStore>) {
    let tmp = tempfile::tempdir().unwrap();
    let mut subs = SubscriptionRegistry::new(tmp.path().join("subscriptions.toml")).unwrap();
    subs.add(100, SubscriptionKind::Text, Some(99), false)
        .unwrap();
    subs.add(200, SubscriptionKind::Text, Some(99), false)
        .unwrap();
    // Keep the temp dir alive for the process lifetime (registry reads lazily).
    std::mem::forget(tmp);
    let subs = Arc::new(subs);

    let s = store();
    let subs_view: Arc<dyn SubscriptionView> = subs;
    let fm = Arc::new(
        FocusManager::new("fam", Arc::clone(&s) as _, subs_view).with_unread_nudge_enabled(false),
    );
    fm.set_focus_immediately(100, "text");

    let mut reg = ToolRegistry::new();
    reg.register(build_shift_focus_tool()).unwrap();

    let fm_ctx = Arc::clone(&fm);
    let store_ctx = Arc::clone(&s);
    let factory: ToolContextFactory = Arc::new(move |channel_id, turn_id, images| {
        ToolContext::new("fam", channel_id, "text", turn_id)
            .with_images(images)
            .with_focus_manager(Arc::clone(&fm_ctx) as _)
            .with_store(Arc::clone(&store_ctx) as _)
    });

    let assembler = make_assembler(Arc::clone(&s));
    let r = TextResponder::new(
        assembler,
        llm,
        send,
        Arc::clone(&s),
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_focus_manager(Arc::clone(&fm) as Arc<dyn FocusManagerApi>)
    .with_tools(Arc::new(reg), factory);
    (r, fm, s)
}

fn shift_tc(channel_id: i64) -> familiar_connect::llm::LlmDelta {
    tc_delta("sf-1", "shift_focus", json!({ "channel_id": channel_id }))
}

#[tokio::test]
async fn silent_shift_focus_moves_immediately() {
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![shift_tc(200), finish("tool_calls")],
        vec![text_delta("<silent>"), finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let (r, fm, _) = real_focus_responder(llm, send.clone());
    r.handle(
        &discord_text_event(text_payload(100, "peek"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert!(send.calls().is_empty());
    assert_eq!(fm.get_focus("text"), Some(200));
}

#[tokio::test]
async fn silent_peek_then_old_channel_stages() {
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![shift_tc(200), finish("tool_calls")],
        vec![text_delta("<silent>"), finish("stop")],
        vec![text_delta("should not send"), finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let (r, fm, s) = real_focus_responder(llm, send.clone());
    r.handle(
        &discord_text_event(text_payload(100, "peek"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    r.handle(
        &discord_text_event(text_payload(100, "ping"), "e-2"),
        &bus(),
    )
    .await
    .unwrap();
    assert_eq!(fm.get_focus("text"), Some(200));
    assert!(send.calls().is_empty());
    let ping = s
        .sync()
        .recent("fam", 100, 10, None, None)
        .unwrap()
        .into_iter()
        .find(|t| t.content == "ping")
        .unwrap();
    assert!(ping.consumed_at.is_none());
}

#[tokio::test]
async fn shift_focus_with_reply_posts_to_new_channel() {
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![shift_tc(200), finish("tool_calls")],
        vec![text_delta("hello over here"), finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let (r, fm, _) = real_focus_responder(llm, send.clone());
    r.handle(&discord_text_event(text_payload(100, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(fm.get_focus("text"), Some(200));
    assert_eq!(send.calls().len(), 1);
    assert_eq!(send.calls()[0].0, 200);
    assert_eq!(send.calls()[0].1, "hello over here");
}

#[tokio::test]
async fn wake_reply_after_shift_posts_to_shifted_channel() {
    // Wake = shift-or-silent (#170): a wake turn that DOES shift focus this turn
    // is delivered — to the channel it shifted to (per-turn routing).
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![shift_tc(200), finish("tool_calls")],
        vec![text_delta("over here now"), finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let (r, fm, _) = real_focus_responder(llm, send.clone());
    let wake = Event {
        event_id: "wake-1".to_owned(),
        turn_id: "wake-1".to_owned(),
        session_id: "discord:100".to_owned(),
        parent_event_ids: Vec::new(),
        topic: TOPIC_DISCORD_TEXT.to_owned(),
        timestamp: chrono::Utc::now(),
        sequence_number: 0,
        payload: wrap_payload(DiscordTextPayload {
            familiar_id: "fam".to_owned(),
            channel_id: 100,
            content: "[idle check]".to_owned(),
            author: None,
            wake: true,
            ..Default::default()
        }),
    };
    r.handle(&wake, &bus()).await.unwrap();
    assert_eq!(fm.get_focus("text"), Some(200));
    assert_eq!(send.calls().len(), 1);
    assert_eq!(send.calls()[0].0, 200);
    assert_eq!(send.calls()[0].1, "over here now");
}

// ---------------------------------------------------------------------------
// Per-turn routing under a concurrent focus shift (#170 race seam)
// ---------------------------------------------------------------------------

/// A tool-less responder wired to a real [`FocusManager`] focused on 100
/// (channels 100 + 200 subscribed). Takes the LLM as a trait object so a
/// delayed [`ScriptedLlm`] can hold the turn open across a concurrent shift.
fn real_focus_bare_responder(
    llm: Arc<dyn ResponderLlm>,
    send: Arc<CapturingSend>,
) -> (TextResponder, Arc<FocusManager>, Arc<AsyncHistoryStore>) {
    let tmp = tempfile::tempdir().unwrap();
    let mut subs = SubscriptionRegistry::new(tmp.path().join("subscriptions.toml")).unwrap();
    subs.add(100, SubscriptionKind::Text, Some(99), false)
        .unwrap();
    subs.add(200, SubscriptionKind::Text, Some(99), false)
        .unwrap();
    std::mem::forget(tmp);
    let subs = Arc::new(subs);

    let s = store();
    let subs_view: Arc<dyn SubscriptionView> = subs;
    let fm = Arc::new(
        FocusManager::new("fam", Arc::clone(&s) as _, subs_view).with_unread_nudge_enabled(false),
    );
    fm.set_focus_immediately(100, "text");

    let assembler = make_assembler(Arc::clone(&s));
    let r = TextResponder::new(
        assembler,
        llm,
        send,
        Arc::clone(&s),
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_focus_manager(Arc::clone(&fm) as Arc<dyn FocusManagerApi>);
    (r, fm, s)
}

#[tokio::test]
async fn reply_routes_to_trigger_channel_despite_concurrent_focus_shift() {
    // Focus is on 100 when the turn starts; the reply streams slowly while a
    // concurrent shift moves the GLOBAL focus to 200 mid-call. Per-turn routing
    // keeps the reply on 100 — the send target is never the global focus read
    // at send time (#170's core race; the old code misrouted to 200).
    let llm: Arc<dyn ResponderLlm> = Arc::new(ScriptedLlm::with_delay(&["still ", "here"], 40));
    let send = Arc::new(CapturingSend::new());
    let (r, fm, _) = real_focus_bare_responder(llm, send.clone());

    let ev = discord_text_event(text_payload(100, "hi"), "e-1");
    let handle = tokio::spawn(async move {
        r.handle(&ev, &bus()).await.unwrap();
    });
    // Let the turn get into its (slow) LLM stream, then move global focus.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    fm.set_focus_immediately(200, "text");
    handle.await.unwrap();

    assert_eq!(fm.get_focus("text"), Some(200));
    assert_eq!(send.calls().len(), 1);
    assert_eq!(send.calls()[0].0, 100);
    assert_eq!(send.calls()[0].1, "still here");
}
