//! Ported from Python `tests/test_agentic_loop.py` + `tests/test_image_serialization.py`
//! — the agentic loop (leak guard, termination, tool execution, guards,
//! callbacks) and `ImageResult` serialization through the loop.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::{Value, json};

use familiar_connect::llm::{Content, LlmClient, LlmDelta, Message};
use familiar_connect::tools::agentic::{
    AgenticHooks, DEFAULT_MAX_ITERATIONS, agentic_loop, serialize_image_result,
    tool_content_as_text,
};
use familiar_connect::tools::registry::{
    FnHandler, ImageResult, Tool, ToolContext, ToolOutput, ToolRegistry,
};

// ---------------------------------------------------------------------------
// Scripted streaming LLM
// ---------------------------------------------------------------------------

struct ScriptedLlm {
    scripts: Mutex<VecDeque<Vec<LlmDelta>>>,
    calls: Mutex<Vec<Vec<Message>>>,
    tool_payloads: Mutex<Vec<Option<Vec<Value>>>>,
    multimodal: bool,
}

impl ScriptedLlm {
    fn new(scripts: Vec<Vec<LlmDelta>>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().collect()),
            calls: Mutex::new(Vec::new()),
            tool_payloads: Mutex::new(Vec::new()),
            multimodal: false,
        }
    }
    const fn multimodal(mut self, m: bool) -> Self {
        self.multimodal = m;
        self
    }
}

#[async_trait]
impl LlmClient for ScriptedLlm {
    async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
        anyhow::bail!("chat not used by the scripted stream mock")
    }
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        self.calls.lock().unwrap().push(messages);
        self.tool_payloads.lock().unwrap().push(tools);
        let deltas = self.scripts.lock().unwrap().pop_front().unwrap_or_default();
        Ok(Box::pin(futures::stream::iter(deltas.into_iter().map(Ok))))
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        self.multimodal
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ctx() -> ToolContext {
    ToolContext::new("fam-1", 42, "text", "turn-1")
}

fn user(text: &str) -> Message {
    Message::new("user", text)
}

fn delta_text(text: &str) -> LlmDelta {
    LlmDelta {
        content: text.to_owned(),
        tool_calls: vec![],
        finish_reason: None,
    }
}

fn delta_tool_call(call_id: &str, name: &str, args: &Value) -> LlmDelta {
    LlmDelta {
        content: String::new(),
        tool_calls: vec![json!({
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args.to_string()},
        })],
        finish_reason: None,
    }
}

fn delta_finish(reason: &str) -> LlmDelta {
    LlmDelta {
        content: String::new(),
        tool_calls: vec![],
        finish_reason: Some(reason.to_owned()),
    }
}

fn named_ok_tool(name: &str) -> Tool {
    Tool::new(
        name,
        "",
        json!({}),
        Arc::new(FnHandler(|_a: Value, _c: ToolContext| async move {
            Ok(ToolOutput::Text("ok".to_owned()))
        })),
    )
}

fn ok_tool() -> Tool {
    named_ok_tool("noop")
}

async fn run(
    scripts: Vec<Vec<LlmDelta>>,
    registry: &ToolRegistry,
    messages: &mut Vec<Message>,
) -> familiar_connect::tools::agentic::AgenticResult {
    let llm = ScriptedLlm::new(scripts);
    agentic_loop(
        &llm,
        messages,
        registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap()
}

fn tool_body(m: &Message) -> Value {
    match &m.content {
        Content::Text(s) => serde_json::from_str(s).unwrap(),
        Content::Blocks(_) => panic!("expected text tool content"),
    }
}

// ---------------------------------------------------------------------------
// Leak guard
// ---------------------------------------------------------------------------

#[tokio::test]
async fn leaked_silent_call_becomes_silent() {
    let leak = "<invoke name=\"silent\">\n<parameter name=\"reasoning\">she stays quiet</parameter>\n</invoke>";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(leak), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn leaked_namespaced_invoke_is_stripped() {
    let leak = "<ns:invoke name=\"silent\">x</ns:invoke>";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(leak), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn leaked_nonsilent_call_stripped_not_silent() {
    let leak = "<invoke name=\"view_image\">x</invoke>";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(leak), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(!result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn python_style_leaked_silent_becomes_silent() {
    let leak = "silent(reasoning=\"not addressed to me; general chat\")";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(leak), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn python_style_leaked_silent_case_insensitive() {
    let leak = "Silent(reasoning=\"sports chat, not aimed at me\")";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(leak), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn python_style_leaked_read_channel_stripped_not_silent() {
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![
            delta_text("read_channel(limit=10)"),
            delta_finish("stop"),
        ]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(!result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn bare_closing_think_tag_stripped() {
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text("</think>"), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(!result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn leading_closing_think_tag_stripped_text_kept() {
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text("</think>\nUmu."), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert_eq!(result.final_content, "Umu.");
}

#[tokio::test]
async fn leading_think_block_stripped_text_kept() {
    let leak = "<think>\nshe weighs the gate\n</think>\nUmu.";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(leak), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert_eq!(result.final_content, "Umu.");
}

#[tokio::test]
async fn think_tag_after_leaked_silent_still_silent() {
    let leak = "</think>\nsilent(reasoning=\"gate unmet\")";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(leak), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(result.is_silent);
    assert!(result.final_content.is_empty());
}

#[tokio::test]
async fn think_mention_mid_prose_untouched() {
    let text = "I think </think> is a vulgar rune.";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(text), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert_eq!(result.final_content, text);
}

#[tokio::test]
async fn normal_reply_with_word_invoke_untouched() {
    let text = "Let me invoke my legendary wit.";
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text(text), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(!result.is_silent);
    assert_eq!(result.final_content, text);
}

// ---------------------------------------------------------------------------
// Termination
// ---------------------------------------------------------------------------

#[tokio::test]
async fn terminates_when_no_tool_calls() {
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text("Hello there"), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert_eq!(result.final_content, "Hello there");
    assert_eq!(result.iterations, 1);
    assert_eq!(result.tool_calls_made, 0);
}

#[tokio::test]
async fn empty_registry_passes_no_tools() {
    let llm = ScriptedLlm::new(vec![vec![delta_text("hi")]]);
    let mut msgs = vec![user("hi")];
    agentic_loop(
        &llm,
        &mut msgs,
        &ToolRegistry::new(),
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    assert_eq!(*llm.tool_payloads.lock().unwrap(), vec![None]);
}

#[tokio::test]
async fn non_empty_registry_passes_tools() {
    let mut registry = ToolRegistry::new();
    registry.register(ok_tool()).unwrap();
    let llm = ScriptedLlm::new(vec![vec![delta_text("hi")]]);
    let mut msgs = vec![user("hi")];
    agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    let first = llm.tool_payloads.lock().unwrap()[0].clone().unwrap();
    assert_eq!(first[0]["function"]["name"], "noop");
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn executes_tool_then_re_calls_llm() {
    let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
    let seen2 = Arc::clone(&seen);
    let mut registry = ToolRegistry::new();
    registry
        .register(Tool::new(
            "set_alarm",
            "",
            json!({"type": "object"}),
            Arc::new(FnHandler(move |args: Value, _c: ToolContext| {
                let seen = Arc::clone(&seen2);
                async move {
                    seen.lock().unwrap().push(args);
                    Ok(ToolOutput::Text(json!({"ok": true}).to_string()))
                }
            })),
        ))
        .unwrap();

    let scripts = vec![
        vec![
            delta_tool_call("c1", "set_alarm", &json!({"reason": "wake"})),
            delta_finish("tool_calls"),
        ],
        vec![delta_text("Done."), delta_finish("stop")],
    ];
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("set alarm")];
    let result = agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![json!({"reason": "wake"})]);
    assert_eq!(result.final_content, "Done.");
    assert_eq!(result.iterations, 2);
    assert_eq!(result.tool_calls_made, 1);

    let second_call = &llm.calls.lock().unwrap()[1];
    let tool_msgs: Vec<&Message> = second_call.iter().filter(|m| m.role == "tool").collect();
    assert_eq!(tool_msgs.len(), 1);
    assert_eq!(tool_msgs[0].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(tool_body(tool_msgs[0]), json!({"ok": true}));
}

#[tokio::test]
async fn handler_exception_surfaced_as_tool_error() {
    let mut registry = ToolRegistry::new();
    registry
        .register(Tool::new(
            "broken",
            "",
            json!({}),
            Arc::new(FnHandler(|_a: Value, _c: ToolContext| async move {
                anyhow::bail!("boom")
            })),
        ))
        .unwrap();
    let scripts = vec![
        vec![
            delta_tool_call("c1", "broken", &json!({})),
            delta_finish("tool_calls"),
        ],
        vec![delta_text("Tool failed, sorry."), delta_finish("stop")],
    ];
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("x")];
    let result = agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    assert_eq!(result.final_content, "Tool failed, sorry.");
    let call = &llm.calls.lock().unwrap()[1];
    let tool_msg = call.iter().find(|m| m.role == "tool").unwrap();
    let body = tool_body(tool_msg);
    assert!(body.get("error").is_some());
    assert!(body["error"].as_str().unwrap().contains("boom"));
}

#[tokio::test]
async fn unknown_tool_returns_error_result() {
    let scripts = vec![
        vec![
            delta_tool_call("c1", "ghost", &json!({})),
            delta_finish("tool_calls"),
        ],
        vec![delta_text("oh well"), delta_finish("stop")],
    ];
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("x")];
    let result = agentic_loop(
        &llm,
        &mut msgs,
        &ToolRegistry::new(),
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    assert_eq!(result.iterations, 2);
    let call = &llm.calls.lock().unwrap()[1];
    let tool_msg = call.iter().find(|m| m.role == "tool").unwrap();
    let body = tool_body(tool_msg);
    assert!(body["error"].as_str().unwrap().contains("ghost"));
}

#[tokio::test]
async fn invalid_args_json_returns_error_result() {
    let mut registry = ToolRegistry::new();
    registry.register(named_ok_tool("t")).unwrap();
    let bad = LlmDelta {
        content: String::new(),
        tool_calls: vec![json!({
            "index": 0, "id": "c1", "type": "function",
            "function": {"name": "t", "arguments": "{not valid json"},
        })],
        finish_reason: None,
    };
    let scripts = vec![
        vec![bad, delta_finish("tool_calls")],
        vec![delta_text("recovered"), delta_finish("stop")],
    ];
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("x")];
    let result = agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    assert_eq!(result.iterations, 2);
    let call = &llm.calls.lock().unwrap()[1];
    let tool_msg = call.iter().find(|m| m.role == "tool").unwrap();
    assert!(tool_body(tool_msg).get("error").is_some());
}

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

#[tokio::test]
async fn caps_at_max_iterations() {
    let mut registry = ToolRegistry::new();
    registry.register(named_ok_tool("t")).unwrap();
    let scripts: Vec<Vec<LlmDelta>> = (0..10)
        .map(|i| {
            vec![
                delta_tool_call(&format!("c{i}"), "t", &json!({})),
                delta_finish("tool_calls"),
            ]
        })
        .collect();
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("loop")];
    let result = agentic_loop(&llm, &mut msgs, &registry, &ctx(), None, 3)
        .await
        .unwrap();
    assert_eq!(result.iterations, 3);
}

#[tokio::test]
async fn handler_timeout_returns_error() {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            Tool::new(
                "slow",
                "",
                json!({}),
                Arc::new(FnHandler(|_a: Value, _c: ToolContext| async move {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    Ok(ToolOutput::Text("never".to_owned()))
                })),
            )
            .with_timeout_s(0.05),
        )
        .unwrap();
    let scripts = vec![
        vec![
            delta_tool_call("c1", "slow", &json!({})),
            delta_finish("tool_calls"),
        ],
        vec![delta_text("ok"), delta_finish("stop")],
    ];
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("x")];
    let result = agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    let call = &llm.calls.lock().unwrap()[1];
    let tool_msg = call.iter().find(|m| m.role == "tool").unwrap();
    let body = tool_body(tool_msg);
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("timeout")
    );
    assert_eq!(result.iterations, 2);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

struct DeltaRecorder {
    seen: Mutex<Vec<LlmDelta>>,
}
#[async_trait]
impl AgenticHooks for DeltaRecorder {
    async fn on_delta(&self, delta: &LlmDelta) {
        self.seen.lock().unwrap().push(delta.clone());
    }
}

#[tokio::test]
async fn on_delta_called_per_chunk() {
    let recorder = DeltaRecorder {
        seen: Mutex::new(Vec::new()),
    };
    let llm = ScriptedLlm::new(vec![vec![
        delta_text("a"),
        delta_text("b"),
        delta_finish("stop"),
    ]]);
    let mut msgs = vec![user("x")];
    agentic_loop(
        &llm,
        &mut msgs,
        &ToolRegistry::new(),
        &ctx(),
        Some(&recorder),
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    let contents: Vec<String> = recorder
        .seen
        .lock()
        .unwrap()
        .iter()
        .filter(|d| !d.content.is_empty())
        .map(|d| d.content.clone())
        .collect();
    assert_eq!(contents, ["a", "b"]);
}

struct IterRecorder {
    events: Mutex<Vec<(Message, Vec<Message>)>>,
}
#[async_trait]
impl AgenticHooks for IterRecorder {
    async fn on_iteration_end(&self, assistant: &Message, tool_msgs: &[Message]) {
        self.events
            .lock()
            .unwrap()
            .push((assistant.clone(), tool_msgs.to_vec()));
    }
}

#[tokio::test]
async fn on_iteration_end_receives_assistant_and_tool_messages() {
    let mut registry = ToolRegistry::new();
    registry
        .register(Tool::new(
            "t",
            "",
            json!({}),
            Arc::new(FnHandler(|_a: Value, _c: ToolContext| async move {
                Ok(ToolOutput::Text("result".to_owned()))
            })),
        ))
        .unwrap();
    let scripts = vec![
        vec![
            delta_text("Working..."),
            delta_tool_call("c1", "t", &json!({})),
            delta_finish("tool_calls"),
        ],
        vec![delta_text("Done"), delta_finish("stop")],
    ];
    let recorder = IterRecorder {
        events: Mutex::new(Vec::new()),
    };
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("x")];
    agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        Some(&recorder),
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    let events = recorder.events.lock().unwrap().clone();
    assert_eq!(events.len(), 2);
    let (asst1, tools1) = &events[0];
    assert_eq!(asst1.content_str(), "Working...");
    assert!(asst1.tool_calls.is_some());
    assert_eq!(tools1.len(), 1);
    assert_eq!(tools1[0].role, "tool");
    let (asst2, tools2) = &events[1];
    assert_eq!(asst2.content_str(), "Done");
    assert!(tools2.is_empty());
}

// ---------------------------------------------------------------------------
// ImageResult serialization (test_image_serialization.py)
// ---------------------------------------------------------------------------

fn image_tool() -> Tool {
    Tool::new(
        "view_image",
        "view an image",
        json!({"type": "object", "properties": {}}),
        Arc::new(FnHandler(|_a: Value, _c: ToolContext| async move {
            Ok(ToolOutput::Image(ImageResult::new("a cat", "abc123")))
        })),
    )
}

#[test]
fn serialize_image_result_text_only() {
    let res = ImageResult::new("a cat", "abc123");
    assert_eq!(
        serialize_image_result(&res, false),
        Content::Text("a cat".to_owned())
    );
}

#[test]
fn serialize_image_result_multimodal() {
    let res = ImageResult::new("a cat", "abc123");
    let Content::Blocks(blocks) = serialize_image_result(&res, true) else {
        panic!("expected blocks");
    };
    assert_eq!(blocks[0], json!({"type": "text", "text": "a cat"}));
    assert_eq!(blocks[1]["type"], "image_url");
    assert_eq!(
        blocks[1]["image_url"]["url"],
        "data:image/jpeg;base64,abc123"
    );
}

#[test]
fn tool_content_as_text_passthrough_and_blocks() {
    assert_eq!(
        tool_content_as_text(&Content::Text("hello".into())),
        "hello"
    );
    let content = Content::Blocks(vec![
        json!({"type": "text", "text": "a cat"}),
        json!({"type": "image_url", "image_url": {"url": "data:..."}}),
    ]);
    assert_eq!(tool_content_as_text(&content), "a cat");
}

#[tokio::test]
async fn loop_serialises_image_result_textonly() {
    let mut registry = ToolRegistry::new();
    registry.register(image_tool()).unwrap();
    let scripts = vec![
        vec![delta_tool_call("call-1", "view_image", &json!({}))],
        vec![delta_text("done")],
    ];
    let llm = ScriptedLlm::new(scripts).multimodal(false);
    let mut msgs = vec![Message::new("system", "sys")];
    let result = agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    let tool_msgs: Vec<&Message> = msgs.iter().filter(|m| m.role == "tool").collect();
    assert_eq!(tool_msgs.len(), 1);
    assert_eq!(tool_msgs[0].content, Content::Text("a cat".to_owned()));
    assert_eq!(result.final_content, "done");
}

#[tokio::test]
async fn loop_serialises_image_result_multimodal() {
    let mut registry = ToolRegistry::new();
    registry.register(image_tool()).unwrap();
    let scripts = vec![
        vec![delta_tool_call("call-1", "view_image", &json!({}))],
        vec![delta_text("done")],
    ];
    let llm = ScriptedLlm::new(scripts).multimodal(true);
    let mut msgs = vec![Message::new("system", "sys")];
    agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        None,
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    let tool_msg = msgs.iter().find(|m| m.role == "tool").unwrap();
    let Content::Blocks(blocks) = &tool_msg.content else {
        panic!("expected multimodal blocks");
    };
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[1]["type"], "image_url");
}

#[test]
fn message_to_dict_passes_list_content() {
    let content = vec![
        json!({"type": "text", "text": "hello"}),
        json!({"type": "image_url", "image_url": {}}),
    ];
    let mut msg = Message::new("tool", Content::Blocks(content.clone()));
    msg.tool_call_id = Some("c1".to_owned());
    let d = msg.to_dict();
    assert_eq!(d["content"], Value::Array(content));
}

// ---------------------------------------------------------------------------
// Silent-tool detection through the loop (test_attentional_tools.py)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn silent_tool_sets_is_silent_true() {
    let mut registry = ToolRegistry::new();
    registry
        .register(familiar_connect::tools::silent::build_silent_tool())
        .unwrap();
    let scripts = vec![vec![
        delta_tool_call("c1", "silent", &json!({"reasoning": "not relevant now"})),
        delta_finish("tool_calls"),
    ]];
    let mut msgs = vec![user("hi")];
    let result = run(scripts, &registry, &mut msgs).await;
    assert!(result.is_silent);
}

#[tokio::test]
async fn normal_tool_does_not_set_is_silent() {
    let mut registry = ToolRegistry::new();
    registry.register(named_ok_tool("noop")).unwrap();
    let scripts = vec![
        vec![
            delta_tool_call("c1", "noop", &json!({})),
            delta_finish("tool_calls"),
        ],
        vec![delta_text("done"), delta_finish("stop")],
    ];
    let mut msgs = vec![user("hi")];
    let result = run(scripts, &registry, &mut msgs).await;
    assert!(!result.is_silent);
}

#[tokio::test]
async fn no_tools_loop_is_not_silent() {
    let mut msgs = vec![user("hi")];
    let result = run(
        vec![vec![delta_text("hello"), delta_finish("stop")]],
        &ToolRegistry::new(),
        &mut msgs,
    )
    .await;
    assert!(!result.is_silent);
}

#[tokio::test]
async fn silent_tool_does_not_call_on_iteration_end() {
    let mut registry = ToolRegistry::new();
    registry
        .register(familiar_connect::tools::silent::build_silent_tool())
        .unwrap();
    let scripts = vec![vec![
        delta_tool_call("c1", "silent", &json!({"reasoning": "not relevant now"})),
        delta_finish("tool_calls"),
    ]];
    let recorder = IterRecorder {
        events: Mutex::new(Vec::new()),
    };
    let llm = ScriptedLlm::new(scripts);
    let mut msgs = vec![user("hi")];
    let result = agentic_loop(
        &llm,
        &mut msgs,
        &registry,
        &ctx(),
        Some(&recorder),
        DEFAULT_MAX_ITERATIONS,
    )
    .await
    .unwrap();
    assert!(result.is_silent);
    assert!(recorder.events.lock().unwrap().is_empty());
}
