//! Tool-calling integration tests for `TextResponder` (subsystem 06; Python
//! `tests/test_text_responder_tools.py`).
//!
//! Port note: `HistoryTurn` exposes no `tool_calls_json` / `tool_call_id`
//! accessor, so these assert the persisted **role sequence** (T14's pinned
//! `user, assistant(tool_calls), tool, assistant(text)`) plus turn contents,
//! rather than the raw column strings.

#![allow(clippy::significant_drop_tightening)]

#[path = "responders_support/mod.rs"]
mod support;

use std::sync::{Arc, Mutex};

use familiar_connect::bus::in_process::InProcessEventBus;
use familiar_connect::bus::router::TurnRouter;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::processors::text_responder::TextResponder;
use familiar_connect::tools::registry::{FnHandler, Tool, ToolContext, ToolOutput, ToolRegistry};
use serde_json::{Value, json};

use support::{
    CapturingSend, ScriptedToolLlm, discord_text_event, finish, make_assembler, simple_ctx_factory,
    store, tc_delta, text_delta, text_payload,
};

const fn bus() -> InProcessEventBus {
    InProcessEventBus::new()
}

fn responder_with_tools(
    s: Arc<AsyncHistoryStore>,
    llm: Arc<ScriptedToolLlm>,
    send: Arc<CapturingSend>,
    registry: Arc<ToolRegistry>,
) -> TextResponder {
    let assembler = make_assembler(Arc::clone(&s));
    TextResponder::new(assembler, llm, send, s, Arc::new(TurnRouter::new()), "fam")
        .with_tools(registry, simple_ctx_factory())
}

fn args_recording_tool(name: &str, sink: Arc<Mutex<Vec<Value>>>) -> Tool {
    let handler = FnHandler(move |args: Value, _ctx: ToolContext| {
        let sink = Arc::clone(&sink);
        async move {
            sink.lock().unwrap().push(args);
            Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
        }
    });
    Tool::new(name, "", json!({}), Arc::new(handler))
}

#[tokio::test]
async fn runs_agentic_loop_full_turn() {
    let s = store();
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let mut reg = ToolRegistry::new();
    reg.register(args_recording_tool("set_alarm", Arc::clone(&seen)))
        .unwrap();
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![
            tc_delta(
                "c1",
                "set_alarm",
                json!({"reason": "ping", "delay_seconds": 30}),
            ),
            finish("tool_calls"),
        ],
        vec![text_delta("Alarm set."), finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let r = responder_with_tools(Arc::clone(&s), llm, send.clone(), Arc::new(reg));
    r.handle(
        &discord_text_event(text_payload(42, "set an alarm"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();

    assert_eq!(
        *seen.lock().unwrap(),
        vec![json!({"reason": "ping", "delay_seconds": 30})]
    );
    let calls = send.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, 42);
    assert_eq!(calls[0].1, "Alarm set.");

    let turns = s.sync().recent("fam", 42, 20, None, None).unwrap();
    let roles: Vec<&str> = turns.iter().map(|t| t.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
    assert_eq!(
        serde_json::from_str::<Value>(&turns[2].content).unwrap(),
        json!({"ok": true})
    );
    assert_eq!(turns[3].content, "Alarm set.");
}

#[tokio::test]
async fn empty_completion_retries_once() {
    let s = store();
    let reg = Arc::new(ToolRegistry::new());
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![finish("stop")],
        vec![text_delta("Better."), finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let r = responder_with_tools(Arc::clone(&s), Arc::clone(&llm), send.clone(), reg);
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(llm.call_count(), 2);
    assert_eq!(send.calls().len(), 1);
    assert_eq!(send.calls()[0].1, "Better.");
}

#[tokio::test]
async fn empty_completion_retries_only_once() {
    let s = store();
    let reg = Arc::new(ToolRegistry::new());
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![finish("stop")],
        vec![finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let r = responder_with_tools(Arc::clone(&s), Arc::clone(&llm), send.clone(), reg);
    r.handle(&discord_text_event(text_payload(42, "hi"), "e-1"), &bus())
        .await
        .unwrap();
    assert_eq!(llm.call_count(), 2);
    assert!(send.calls().is_empty());
}

#[tokio::test]
async fn images_threaded_into_tool_context() {
    let s = store();
    let captured: Arc<Mutex<Vec<std::collections::HashMap<String, String>>>> =
        Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&captured);
    let handler = FnHandler(move |_args: Value, ctx: ToolContext| {
        let sink = Arc::clone(&sink);
        let images = ctx.images;
        async move {
            sink.lock().unwrap().push(images);
            Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
        }
    });
    let mut reg = ToolRegistry::new();
    reg.register(Tool::new("set_alarm", "", json!({}), Arc::new(handler)))
        .unwrap();
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![
            tc_delta(
                "c1",
                "set_alarm",
                json!({"reason": "ping", "delay_seconds": 10}),
            ),
            finish("tool_calls"),
        ],
        vec![text_delta("Done."), finish("stop")],
    ]));
    let send = Arc::new(CapturingSend::new());
    let r = responder_with_tools(Arc::clone(&s), llm, send, Arc::new(reg));

    let mut payload = text_payload(42, "look");
    payload.images.insert(
        "img_0".to_owned(),
        "http://cdn.example.com/cat.png".to_owned(),
    );
    r.handle(&discord_text_event(payload, "e-1"), &bus())
        .await
        .unwrap();

    let cap = captured.lock().unwrap();
    assert_eq!(cap.len(), 1);
    assert_eq!(
        cap[0].get("img_0").map(String::as_str),
        Some("http://cdn.example.com/cat.png")
    );
}

#[tokio::test]
async fn image_tools_only_enters_loop() {
    let s = store();
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let mut reg = ToolRegistry::new();
    reg.register(args_recording_tool("view_image", Arc::clone(&seen)))
        .unwrap();
    // tool_calling disabled, image_tools enabled → still enters the loop.
    let llm = Arc::new(
        ScriptedToolLlm::new(vec![
            vec![
                tc_delta("c1", "view_image", json!({"image_id": "img_0"})),
                finish("tool_calls"),
            ],
            vec![text_delta("Looks like a cat."), finish("stop")],
        ])
        .with_flags(false, true),
    );
    let send = Arc::new(CapturingSend::new());
    let r = responder_with_tools(Arc::clone(&s), llm, send.clone(), Arc::new(reg));
    r.handle(
        &discord_text_event(text_payload(42, "look at img_0"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(send.calls()[0].1, "Looks like a cat.");
}

#[tokio::test]
async fn loop_max_iterations_caps_agentic_loop() {
    let s = store();
    let mut reg = ToolRegistry::new();
    let handler = FnHandler(|_args: Value, _ctx: ToolContext| async move {
        Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
    });
    reg.register(Tool::new("set_alarm", "", json!({}), Arc::new(handler)))
        .unwrap();
    // Every iteration asks for another tool call — only the cap stops it.
    let llm = Arc::new(ScriptedToolLlm::new(vec![
        vec![
            tc_delta(
                "c1",
                "set_alarm",
                json!({"reason": "a", "delay_seconds": 5}),
            ),
            finish("tool_calls"),
        ],
        vec![
            tc_delta(
                "c2",
                "set_alarm",
                json!({"reason": "b", "delay_seconds": 5}),
            ),
            finish("tool_calls"),
        ],
    ]));
    let send = Arc::new(CapturingSend::new());
    let assembler = make_assembler(Arc::clone(&s));
    let llm_dyn: Arc<dyn familiar_connect::processors::ResponderLlm> = llm.clone();
    let r = TextResponder::new(
        assembler,
        llm_dyn,
        send,
        s,
        Arc::new(TurnRouter::new()),
        "fam",
    )
    .with_tools(Arc::new(reg), simple_ctx_factory())
    .with_loop_max_iterations(1);
    r.handle(
        &discord_text_event(text_payload(42, "set an alarm"), "e-1"),
        &bus(),
    )
    .await
    .unwrap();
    assert_eq!(llm.call_count(), 1);
}
