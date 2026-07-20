//! Integration tests for `TextResponder` (subsystem 06; Python
//! `tests/test_text_responder.py`).

#![allow(clippy::significant_drop_tightening)]

#[path = "responders_support/mod.rs"]
mod support;

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde_json::Value;

use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::router::TurnRouter;
use familiar_connect::bus::topics::TOPIC_DISCORD_TEXT;
use familiar_connect::config::DiscordTextConfig;
use familiar_connect::focus::PRIVATE_MESSAGE_GUILD_NAME;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::AppendTurn;
use familiar_connect::identity::Author;
use familiar_connect::llm::{LlmClient, LlmDelta, Message};
use familiar_connect::processors::text_responder::TextResponder;
use familiar_connect::processors::{
    DiscordTextPayload, GateAction, GateDecision, ResponderLlm, SendText,
};
use familiar_connect::typing_interrupt::TypingInterruptHandler;

use support::{
    CapturingLlm, CapturingSend, FakeActivityEngine, LogCapture, RecordingTyping, ScriptedLlm,
    TestFocusManager, discord_text_event, make_assembler, store, text_payload,
};

const fn bus() -> InProcessEventBus {
    InProcessEventBus::new()
}

fn responder(
    s: Arc<AsyncHistoryStore>,
    llm: Arc<dyn ResponderLlm>,
    send: Arc<dyn SendText>,
) -> (TextResponder, Arc<TurnRouter>) {
    let router = Arc::new(TurnRouter::new());
    let assembler = make_assembler(Arc::clone(&s));
    let r = TextResponder::new(assembler, llm, send, s, Arc::clone(&router), "fam");
    (r, router)
}

fn alice_full() -> Author {
    let mut a = Author::new(
        "discord",
        "1",
        Some("alice".to_owned()),
        Some("Alice".to_owned()),
    );
    a.global_name = Some("Alice Liddell".to_owned());
    a.guild_nick = Some("Aria".to_owned());
    a
}

fn full_payload(message_id: &str, content: &str, mentions: Vec<Author>) -> DiscordTextPayload {
    DiscordTextPayload {
        familiar_id: "fam".to_owned(),
        channel_id: 42,
        guild_id: Some(99),
        author: Some(alice_full()),
        content: content.to_owned(),
        message_id: Some(message_id.to_owned()),
        mentions,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Processor surface + basic reply
// ---------------------------------------------------------------------------

#[tokio::test]
async fn processor_surface() {
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["x"])),
        Arc::new(CapturingSend::new()),
    );
    assert_eq!(r.name(), "text-responder");
    assert!(r.topics().contains(&TOPIC_DISCORD_TEXT));
}

#[tokio::test]
async fn streams_reply_and_calls_send_text() {
    let s = store();
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["Hello", ", ", "world", "."])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(text_payload(42, "hi there"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();

    assert_eq!(
        send.calls(),
        vec![(42, "Hello, world.".to_owned(), None, vec![])]
    );
    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert!(
        turns
            .iter()
            .any(|t| t.role == "assistant" && t.content == "Hello, world.")
    );
}

#[tokio::test]
async fn persists_pings_bot_on_user_turn() {
    let s = store();
    let mut payload = text_payload(42, "you there @bot?");
    payload.pings_bot = true;
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
    );
    r.handle(&discord_text_event(payload, "e-1"), &bus())
        .await
        .unwrap();
    let users: Vec<_> = s
        .sync()
        .recent("fam", 42, 10, None, None)
        .unwrap()
        .into_iter()
        .filter(|t| t.role == "user")
        .collect();
    assert_eq!(users.len(), 1);
    assert!(users[0].pings_bot);
}

#[tokio::test]
async fn user_turn_pings_bot_false_without_ping() {
    let s = store();
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
    );
    r.handle(
        &discord_text_event(text_payload(42, "just chatting"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let users: Vec<_> = s
        .sync()
        .recent("fam", 42, 10, None, None)
        .unwrap()
        .into_iter()
        .filter(|t| t.role == "user")
        .collect();
    assert_eq!(users.len(), 1);
    assert!(!users[0].pings_bot);
}

#[tokio::test]
async fn skips_when_payload_missing_content() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(store(), Arc::new(ScriptedLlm::new(&["nope"])), send.clone());
    r.handle(&discord_text_event(text_payload(42, ""), "e-1"), &bus())
        .await
        .unwrap();
    assert!(send.calls().is_empty());
}

#[tokio::test]
async fn skips_other_familiars() {
    let send = Arc::new(CapturingSend::new());
    let mut payload = text_payload(42, "hi");
    payload.familiar_id = "other".to_owned();
    let (r, _) = responder(store(), Arc::new(ScriptedLlm::new(&["nope"])), send.clone());
    r.handle(&discord_text_event(payload, "e-1"), &bus())
        .await
        .unwrap();
    assert!(send.calls().is_empty());
}

#[tokio::test]
async fn begins_turn_for_session_ends_cleanly() {
    let send = Arc::new(CapturingSend::new());
    let (r, router) = responder(store(), Arc::new(ScriptedLlm::new(&["ok"])), send.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(router.active_scope("discord:42").is_none());
    assert!(!send.calls().is_empty());
}

#[tokio::test]
async fn skips_when_llm_returns_empty() {
    let s = store();
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&[])),
        send.clone(),
    );
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(send.calls().is_empty());
    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert!(turns.iter().all(|t| t.role != "assistant"));
}

#[tokio::test]
async fn skips_when_llm_returns_whitespace() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["   ", "\n"])),
        send.clone(),
    );
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(send.calls().is_empty());
}

// ---------------------------------------------------------------------------
// User turn persisted before the LLM stream (observing snapshot)
// ---------------------------------------------------------------------------

struct ObservingLlm {
    store: Arc<AsyncHistoryStore>,
    seen: Arc<std::sync::Mutex<Vec<String>>>,
}

#[async_trait]
impl LlmClient for ObservingLlm {
    async fn chat(&self, _m: Vec<Message>) -> anyhow::Result<Message> {
        Ok(Message::new("assistant", "ack"))
    }
    async fn stream_completion(
        &self,
        _m: Vec<Message>,
        _t: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        let turns = self.store.sync().recent("fam", 42, 10, None, None).unwrap();
        let mut seen = self.seen.lock().unwrap();
        for t in turns.iter().filter(|t| t.role == "user") {
            seen.push(t.content.clone());
        }
        Ok(Box::pin(stream::once(async move {
            Ok(LlmDelta {
                content: "ack".to_owned(),
                ..Default::default()
            })
        })))
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
impl ResponderLlm for ObservingLlm {}

#[tokio::test]
async fn user_turn_persisted_before_llm_stream() {
    let s = store();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let llm = Arc::new(ObservingLlm {
        store: Arc::clone(&s),
        seen: Arc::clone(&seen),
    });
    let (r, _) = responder(Arc::clone(&s), llm, Arc::new(CapturingSend::new()));
    r.handle(
        &discord_text_event(text_payload(42, "hello"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert!(seen.lock().unwrap().iter().any(|c| c == "hello"));
}

// ---------------------------------------------------------------------------
// Silent sentinel
// ---------------------------------------------------------------------------

#[tokio::test]
async fn silent_sentinel_skips_send_and_assistant_turn() {
    let s = store();
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["<silent>"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(text_payload(42, "hi nobody"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert!(send.calls().is_empty());
    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert!(turns.iter().all(|t| t.role != "assistant"));
    assert!(
        turns
            .iter()
            .any(|t| t.role == "user" && t.content.contains("hi nobody"))
    );
}

#[tokio::test]
async fn silent_sentinel_with_leading_whitespace() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["  ", "<silent>"])),
        send.clone(),
    );
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(send.calls().is_empty());
}

#[tokio::test]
async fn sentinel_mid_reply_is_not_a_gate() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["Sure! ", "<silent>", " — kidding."])),
        send.clone(),
    );
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(
        send.calls(),
        vec![(42, "Sure! <silent> — kidding.".to_owned(), None, vec![])]
    );
}

#[tokio::test]
async fn user_turn_recorded_with_author_and_guild() {
    let s = store();
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
    );
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let users: Vec<_> = s
        .sync()
        .recent("fam", 42, 10, None, None)
        .unwrap()
        .into_iter()
        .filter(|t| t.role == "user")
        .collect();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].content, "hi");
    assert_eq!(
        users[0].author.as_ref().unwrap().display_name.as_deref(),
        Some("Alice")
    );
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enters_typing_context_once() {
    let typing = Arc::new(RecordingTyping::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["Hello"])),
        Arc::new(CapturingSend::new()),
    );
    let r = r.with_trigger_typing(typing.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(typing.calls(), vec![42]);
    assert_eq!(typing.entered(), 1);
    assert_eq!(typing.exited(), 1);
}

#[tokio::test]
async fn typing_skipped_on_silent_reply() {
    let typing = Arc::new(RecordingTyping::new());
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["<silent>"])),
        send.clone(),
    );
    let r = r.with_trigger_typing(typing.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(send.calls().is_empty());
    assert_eq!(typing.entered(), 0);
    assert_eq!(typing.exited(), 0);
}

#[tokio::test]
async fn typing_skipped_on_empty_stream() {
    let typing = Arc::new(RecordingTyping::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&[])),
        Arc::new(CapturingSend::new()),
    );
    let r = r.with_trigger_typing(typing.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(typing.entered(), 0);
}

#[tokio::test]
async fn works_without_typing_hook() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(store(), Arc::new(ScriptedLlm::new(&["ok"])), send.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(send.calls(), vec![(42, "ok".to_owned(), None, vec![])]);
}

// ---------------------------------------------------------------------------
// Typing backoff integration
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn waits_for_bot_backoff_before_streaming() {
    let cfg = DiscordTextConfig {
        typing_backoff_initial_s: 0.05,
        typing_backoff_max_s: 0.1,
        ..Default::default()
    };
    let router = Arc::new(TurnRouter::new());
    let handler = Arc::new(TypingInterruptHandler::new(
        cfg,
        Arc::clone(&router),
        Arc::new(|_ch| true),
        Arc::new(|| Some(999)),
    ));
    handler.notify_typing(42, 123, true);

    let s = store();
    let send = Arc::new(CapturingSend::new());
    let assembler = make_assembler(Arc::clone(&s));
    let r = TextResponder::new(
        assembler,
        Arc::new(ScriptedLlm::new(&["ok"])),
        send.clone(),
        s,
        router,
        "fam",
    )
    .with_typing_handler(handler);

    let start = tokio::time::Instant::now();
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(start.elapsed() >= std::time::Duration::from_secs_f64(0.04));
    assert!(!send.calls().is_empty());
}

#[tokio::test]
async fn user_message_clears_backoff() {
    let cfg = DiscordTextConfig {
        typing_backoff_initial_s: 1.0,
        typing_backoff_max_s: 4.0,
        ..Default::default()
    };
    let router = Arc::new(TurnRouter::new());
    let handler = Arc::new(TypingInterruptHandler::new(
        cfg,
        Arc::clone(&router),
        Arc::new(|_ch| true),
        Arc::new(|| Some(999)),
    ));
    handler.notify_typing(42, 123, true);
    handler.notify_typing(42, 123, true);
    assert!((handler.current_backoff_s(42) - 2.0).abs() < 1e-9);

    let s = store();
    let assembler = make_assembler(Arc::clone(&s));
    let r = TextResponder::new(
        assembler,
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
        s,
        Arc::clone(&router),
        "fam",
    )
    .with_typing_handler(Arc::clone(&handler));
    r.handle(
        &discord_text_event(text_payload(42, "hi again"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();

    handler.notify_typing(42, 123, true);
    assert!((handler.current_backoff_s(42) - 1.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Identity write path + threading + pings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn persists_platform_and_reply_message_ids() {
    let s = store();
    let mut payload = full_payload("msg-2", "hi", vec![]);
    payload.reply_to_message_id = Some("msg-1".to_owned());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
    );
    r.handle(&discord_text_event(payload, "e-1"), &bus())
        .await
        .unwrap();
    let looked = s
        .sync()
        .lookup_turn_by_platform_message_id("fam", "msg-2")
        .unwrap()
        .unwrap();
    assert_eq!(looked.role, "user");
    assert_eq!(looked.reply_to_message_id.as_deref(), Some("msg-1"));
}

#[tokio::test]
async fn upserts_author_and_guild_nick() {
    let s = store();
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
    );
    r.handle(
        &discord_text_event(full_payload("msg-1", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let profile = s.sync().get_account_profile("discord:1").unwrap().unwrap();
    assert_eq!(profile.global_name.as_deref(), Some("Alice Liddell"));
    assert_eq!(
        s.sync()
            .resolve_label("discord:1", Some(99), Some("fam"))
            .unwrap(),
        "Aria"
    );
}

#[tokio::test]
async fn upserts_each_mentioned_user_and_records_mentions() {
    let s = store();
    let bob = {
        let mut a = Author::new(
            "discord",
            "2",
            Some("bob".to_owned()),
            Some("Bob".to_owned()),
        );
        a.global_name = Some("Robert".to_owned());
        a
    };
    let carol = Author::new(
        "discord",
        "3",
        Some("carol".to_owned()),
        Some("Carol".to_owned()),
    );
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
    );
    r.handle(
        &discord_text_event(full_payload("msg-1", "hi", vec![bob, carol]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert_eq!(
        s.sync()
            .get_account_profile("discord:2")
            .unwrap()
            .unwrap()
            .global_name
            .as_deref(),
        Some("Robert")
    );
    let user = s
        .sync()
        .recent("fam", 42, 10, None, None)
        .unwrap()
        .into_iter()
        .find(|t| t.role == "user")
        .unwrap();
    let keys = s.sync().mentions_for_turn(user.id).unwrap();
    assert!(keys.contains(&"discord:2".to_owned()));
    assert!(keys.contains(&"discord:3".to_owned()));
}

#[tokio::test]
async fn does_not_thread_by_default() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(store(), Arc::new(ScriptedLlm::new(&["ok"])), send.clone());
    r.handle(
        &discord_text_event(full_payload("incoming-77", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert_eq!(send.calls()[0].2, None);
}

#[tokio::test]
async fn threads_when_llm_emits_marker() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["[↩] sure thing"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("incoming-77", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let (_, content, reply_to, _) = send.calls().remove(0);
    assert_eq!(reply_to.as_deref(), Some("incoming-77"));
    assert_eq!(content, "sure thing");
}

#[tokio::test]
async fn thread_marker_explicit_id() {
    let s = store();
    s.sync()
        .append_turn(AppendTurn::new("fam", 42, "user", "earlier").platform_message_id("older-7"))
        .unwrap();
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["[↩ older-7] sure"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("incoming-99", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let (_, content, reply_to, _) = send.calls().remove(0);
    assert_eq!(reply_to.as_deref(), Some("older-7"));
    assert_eq!(content, "sure");
}

#[tokio::test]
async fn thread_marker_hash_sigil_resolved() {
    let s = store();
    s.sync()
        .append_turn(
            AppendTurn::new("fam", 42, "user", "earlier")
                .platform_message_id("1500691573262782604"),
        )
        .unwrap();
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&[
            "[↩ #1500691573262782604] still thinking",
        ])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("trigger", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let (_, content, reply_to, _) = send.calls().remove(0);
    assert_eq!(reply_to.as_deref(), Some("1500691573262782604"));
    assert_eq!(content, "still thinking");
}

#[tokio::test]
async fn thread_marker_unknown_id_falls_back() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["[↩ no-such] ok"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("incoming-77", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let (_, content, reply_to, _) = send.calls().remove(0);
    assert_eq!(reply_to.as_deref(), Some("incoming-77"));
    assert_eq!(content, "ok");
}

#[tokio::test]
async fn thread_marker_reply_word_form() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["[reply] got it"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("msg-99", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let (_, content, reply_to, _) = send.calls().remove(0);
    assert_eq!(reply_to.as_deref(), Some("msg-99"));
    assert_eq!(content, "got it");
}

#[tokio::test]
async fn rewrites_known_ping_marker() {
    let s = store();
    let bob = {
        let mut a = Author::new(
            "discord",
            "222",
            Some("bob".to_owned()),
            Some("Bob".to_owned()),
        );
        a.global_name = Some("Bob".to_owned());
        a
    };
    s.sync().upsert_account(&bob).unwrap();
    s.sync()
        .append_turn(
            AppendTurn::new("fam", 42, "user", "hi")
                .author(bob)
                .guild_id(99),
        )
        .unwrap();
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["sure thing, [@Bob]"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("m", "hi bob", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let (_, content, _, mentions) = send.calls().remove(0);
    assert!(content.contains("<@222>"));
    assert!(!content.contains("[@Bob]"));
    assert!(mentions.contains(&222));
}

#[tokio::test]
async fn unknown_ping_marker_plain_text() {
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["hi [@Nobody], welcome"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("m", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let (_, content, _, mentions) = send.calls().remove(0);
    assert!(!content.contains("<@"));
    assert!(content.contains("@Nobody"));
    assert!(mentions.is_empty());
}

#[tokio::test]
async fn assistant_turn_persists_sent_id_no_thread() {
    let s = store();
    let send = Arc::new(CapturingSend::with_id("bot-msg-12345"));
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["ok"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("user-msg-1", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let looked = s
        .sync()
        .lookup_turn_by_platform_message_id("fam", "bot-msg-12345")
        .unwrap()
        .unwrap();
    assert_eq!(looked.role, "assistant");
    assert_eq!(looked.reply_to_message_id, None);
}

#[tokio::test]
async fn assistant_turn_records_thread_target() {
    let s = store();
    let send = Arc::new(CapturingSend::with_id("bot-msg-67890"));
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["[↩] yep"])),
        send.clone(),
    );
    r.handle(
        &discord_text_event(full_payload("user-msg-1", "hi", vec![]), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let looked = s
        .sync()
        .lookup_turn_by_platform_message_id("fam", "bot-msg-67890")
        .unwrap()
        .unwrap();
    assert_eq!(looked.reply_to_message_id.as_deref(), Some("user-msg-1"));
}

// ---------------------------------------------------------------------------
// Trailing reminder content
// ---------------------------------------------------------------------------

fn trailing_of(captured: &[Vec<Message>]) -> String {
    captured[0].last().unwrap().content_str()
}

#[tokio::test]
async fn trailing_carries_text_directive() {
    let s = store();
    let llm = Arc::new(CapturingLlm::new("ok"));
    let (r, _) = responder(Arc::clone(&s), llm.clone(), Arc::new(CapturingSend::new()));
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let trailing = trailing_of(&llm.captured());
    assert!(trailing.contains("Markdown"));
    assert!(trailing.contains("It is now"));
    assert!(trailing.contains("[@DisplayName]"));
}

#[tokio::test]
async fn trailing_renders_configured_timezone() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let (r, _) = responder(store(), llm.clone(), Arc::new(CapturingSend::new()));
    let r = r.with_display_tz("America/Los_Angeles");
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let content = trailing_of(&llm.captured());
    assert!(
        content.contains("PST") || content.contains("PDT"),
        "{content}"
    );
    assert!(content.contains("It is now"));
    assert!(!content.contains(" UTC"));
}

#[tokio::test]
async fn trailing_carries_post_history_instructions_deepest() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let (r, _) = responder(store(), llm.clone(), Arc::new(CapturingSend::new()));
    let r = r.with_post_history_instructions("# Etiquette\n\nPrefer <silent>.");
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let trailing = trailing_of(&llm.captured());
    assert!(trailing.contains("# Etiquette"));
    assert!(
        trailing.find("chatting in a text channel").unwrap()
            < trailing.find("# Etiquette").unwrap()
    );
}

#[tokio::test]
async fn trailing_names_server_head_does_not() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let fm = Arc::new(
        TestFocusManager::focused(42)
            .with_channel_name(42, "general")
            .with_guild_name(42, "My Server"),
    );
    let (r, _) = responder(store(), llm.clone(), Arc::new(CapturingSend::new()));
    let r = r.with_focus_manager(fm);
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let cap = llm.captured();
    let head = cap[0][0].content_str();
    let trailing = cap[0].last().unwrap().content_str();
    assert!(trailing.contains("#general"));
    assert!(trailing.contains("\"My Server\" server"));
    assert!(head.contains("#general"));
    assert!(!head.contains("My Server"));
}

#[tokio::test]
async fn trailing_labels_off_focus_dm_unread_as_dm_from() {
    // The DM label only appears once the responder threads the focus manager's
    // guild_names map (marking the channel a private message) into the digest
    // labeler alongside channel_names.
    let llm = Arc::new(CapturingLlm::new("ok"));
    let s = store();
    // Off-focus DM (channel 555) with one unread turn so it lands in the digest.
    s.sync()
        .stage_turn(AppendTurn::new("fam", 555, "user", "hi"))
        .unwrap();
    let fm = Arc::new(
        TestFocusManager::focused(42)
            .with_channel_name(555, "Cor")
            .with_guild_name(555, PRIVATE_MESSAGE_GUILD_NAME),
    );
    let (r, _) = responder(s, llm.clone(), Arc::new(CapturingSend::new()));
    let r = r.with_focus_manager(fm);
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let trailing = trailing_of(&llm.captured());
    assert!(trailing.contains("DM from Cor (id 555)"), "{trailing}");
}

#[tokio::test]
async fn no_server_clause_when_guild_unknown() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let fm = Arc::new(TestFocusManager::focused(42).with_channel_name(42, "general"));
    let (r, _) = responder(store(), llm.clone(), Arc::new(CapturingSend::new()));
    let r = r.with_focus_manager(fm);
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let trailing = trailing_of(&llm.captured());
    assert!(trailing.contains("Your attention is currently on #general."));
    assert!(!trailing.contains("server"));
}

// ---------------------------------------------------------------------------
// Per-turn origin logging
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reply_log_names_server_and_channel() {
    let fm = Arc::new(
        TestFocusManager::focused(42)
            .with_channel_name(42, "general")
            .with_guild_name(42, "My Server"),
    );
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["hello"])),
        Arc::new(CapturingSend::new()),
    );
    let r = r.with_focus_manager(fm);
    let capture = LogCapture::install();
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let out = capture.contents();
    drop(capture);
    assert_eq!(
        out.lines().filter(|l| l.contains("\u{1f4ac} Text")).count(),
        1,
        "{out}"
    );
    assert!(out.contains("#general"));
    assert!(out.contains("My Server"));
}

#[tokio::test]
async fn reply_log_omits_server_without_focus_manager() {
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["hello"])),
        Arc::new(CapturingSend::new()),
    );
    let capture = LogCapture::install();
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    let out = capture.contents();
    drop(capture);
    assert!(out.contains("#42"));
    assert!(!out.contains("srv="));
}

#[tokio::test]
async fn staged_log_names_server_and_channel() {
    let fm = Arc::new(
        TestFocusManager::unfocused()
            .with_channel_name(42, "general")
            .with_guild_name(42, "My Server"),
    );
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["should not stream"])),
        send.clone(),
    );
    let r = r.with_focus_manager(fm);
    let capture = LogCapture::install();
    r.handle(&discord_text_event(text_payload(42, "psst"), "e-1"), &bus())
        .await
        .unwrap();
    let out = capture.contents();
    drop(capture);
    assert!(send.calls().is_empty());
    assert_eq!(
        out.lines()
            .filter(|l| l.contains("\u{1f4e5} Staged"))
            .count(),
        1,
        "{out}"
    );
    assert!(out.contains("#general"));
    assert!(out.contains("My Server"));
}

// ---------------------------------------------------------------------------
// Activity gate
// ---------------------------------------------------------------------------

fn decision(action: GateAction, state_line: Option<&str>) -> GateDecision {
    GateDecision {
        action,
        state_line: state_line.map(str::to_owned),
    }
}

#[tokio::test]
async fn suppress_records_staged_turn_and_skips_reply() {
    let s = store();
    let engine = Arc::new(FakeActivityEngine::new(decision(
        GateAction::Suppress,
        None,
    )));
    let typing = Arc::new(RecordingTyping::new());
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["nope"])),
        send.clone(),
    );
    let r = r
        .with_trigger_typing(typing.clone())
        .with_activity_engine(engine.clone());
    r.handle(
        &discord_text_event(text_payload(42, "anyone home?"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert!(send.calls().is_empty());
    assert_eq!(typing.entered(), 0);
    let turns = s.sync().recent("fam", 42, 10, None, None).unwrap();
    assert!(turns.iter().all(|t| t.role != "assistant"));
    let user = turns.iter().find(|t| t.role == "user").unwrap();
    assert!(user.consumed_at.is_none());
    assert_eq!(engine.replies_notified(), 0);
    assert_eq!(engine.turns_ended(), 0);
    assert_eq!(engine.traffic_notes(), 1);
}

#[tokio::test]
async fn silent_outcome_still_applies_deferred_start() {
    let engine = Arc::new(FakeActivityEngine::new(decision(GateAction::Normal, None)));
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["<silent>"])),
        send.clone(),
    );
    let r = r.with_activity_engine(engine.clone());
    r.handle(
        &discord_text_event(text_payload(42, "[idle: quiet]"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert!(send.calls().is_empty());
    assert_eq!(engine.turns_ended(), 1);
    assert_eq!(engine.replies_notified(), 0);
}

#[tokio::test]
async fn suppressed_ping_noted_as_missed() {
    let s = store();
    let engine = Arc::new(FakeActivityEngine::new(decision(
        GateAction::Suppress,
        None,
    )));
    let mut payload = text_payload(42, "you there?");
    payload.pings_bot = true;
    let (r, _) = responder(
        Arc::clone(&s),
        Arc::new(ScriptedLlm::new(&["nope"])),
        Arc::new(CapturingSend::new()),
    );
    let r = r.with_activity_engine(engine.clone());
    r.handle(&discord_text_event(payload, "e-1"), &bus())
        .await
        .unwrap();
    let user = s
        .sync()
        .recent("fam", 42, 10, None, None)
        .unwrap()
        .into_iter()
        .find(|t| t.role == "user")
        .unwrap();
    assert_eq!(engine.missed_pings(), vec![user.id]);
}

#[tokio::test]
async fn suppressed_non_ping_not_noted() {
    let engine = Arc::new(FakeActivityEngine::new(decision(
        GateAction::Suppress,
        None,
    )));
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["nope"])),
        Arc::new(CapturingSend::new()),
    );
    let r = r.with_activity_engine(engine.clone());
    r.handle(
        &discord_text_event(text_payload(42, "just chatting"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert!(engine.missed_pings().is_empty());
}

#[tokio::test]
async fn note_traffic_on_every_handled_event() {
    let engine = Arc::new(FakeActivityEngine::new(decision(GateAction::Normal, None)));
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["ok"])),
        Arc::new(CapturingSend::new()),
    );
    let r = r.with_activity_engine(engine.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    r.handle(
        &discord_text_event(text_payload(42, "hi again"), "e-2"),
        &bus(),
    )
    .await
    .unwrap();
    assert_eq!(engine.traffic_notes(), 2);
}

#[tokio::test]
async fn judgment_injects_state_line_and_notifies() {
    let llm = Arc::new(CapturingLlm::new("back!"));
    let state = "You are 20 min into a creek walk — Alice pinged you.";
    let engine = Arc::new(FakeActivityEngine::new(decision(
        GateAction::Judgment,
        Some(state),
    )));
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(store(), llm.clone(), send.clone());
    let r = r.with_activity_engine(engine.clone());
    r.handle(
        &discord_text_event(text_payload(42, "hey you there?"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    let trailing = trailing_of(&llm.captured());
    assert!(trailing.contains(state));
    assert!(!send.calls().is_empty());
    assert_eq!(engine.replies_notified(), 1);
    assert_eq!(engine.turns_ended(), 1);
}

#[tokio::test]
async fn judgment_silent_means_stay_out() {
    let engine = Arc::new(FakeActivityEngine::new(decision(
        GateAction::Judgment,
        Some("state"),
    )));
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(
        store(),
        Arc::new(ScriptedLlm::new(&["<silent>"])),
        send.clone(),
    );
    let r = r.with_activity_engine(engine.clone());
    r.handle(&discord_text_event(text_payload(42, "ping"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(send.calls().is_empty());
    assert_eq!(engine.replies_notified(), 0);
}

#[tokio::test]
async fn normal_reply_applies_end_turn_without_notify() {
    let engine = Arc::new(FakeActivityEngine::new(decision(GateAction::Normal, None)));
    let send = Arc::new(CapturingSend::new());
    let (r, _) = responder(store(), Arc::new(ScriptedLlm::new(&["ok"])), send.clone());
    let r = r.with_activity_engine(engine.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(!send.calls().is_empty());
    assert_eq!(engine.replies_notified(), 0);
    assert_eq!(engine.turns_ended(), 1);
}

#[tokio::test]
async fn normal_decision_leaves_prompt_untouched() {
    let llm = Arc::new(CapturingLlm::new("ok"));
    let engine = Arc::new(FakeActivityEngine::new(decision(GateAction::Normal, None)));
    let (r, _) = responder(store(), llm.clone(), Arc::new(CapturingSend::new()));
    let r = r.with_activity_engine(engine.clone());
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert!(!trailing_of(&llm.captured()).contains("pinged"));
}
