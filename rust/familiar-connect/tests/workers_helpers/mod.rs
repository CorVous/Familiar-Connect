//! Shared fixtures for the background-worker integration tests (subsystem 07).
//! Included via `#[path = "workers_helpers/mod.rs"] mod helpers;` — not a test
//! binary itself.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::store::HistoryStore;
use familiar_connect::llm::{LlmClient, LlmDelta, Message};
use futures::stream::BoxStream;
use serde_json::Value;

/// A fresh in-memory store wrapped in the async facade.
#[must_use]
pub fn store() -> Arc<AsyncHistoryStore> {
    let hs = HistoryStore::open(":memory:").expect("open in-memory store");
    Arc::new(AsyncHistoryStore::new(hs))
}

/// Scripted LLM stub: pops one canned reply per `chat` call, records every
/// prompt, and returns a benign default when the script is exhausted. Mirrors
/// the Python `_ScriptedLLM` doubles across the worker test suites.
pub struct ScriptedLlm {
    replies: Mutex<VecDeque<String>>,
    default: String,
    calls: Mutex<Vec<Vec<Message>>>,
}

impl ScriptedLlm {
    /// Build a stub with an initial reply script and the exhausted-default.
    #[must_use]
    pub fn new<I, S>(replies: I, default: impl Into<String>) -> Arc<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Arc::new(Self {
            replies: Mutex::new(replies.into_iter().map(Into::into).collect()),
            default: default.into(),
            calls: Mutex::new(Vec::new()),
        })
    }

    /// Append a reply to the script (mirrors Python `llm._replies.append(...)`).
    pub fn push_reply(&self, reply: impl Into<String>) {
        self.replies.lock().unwrap().push_back(reply.into());
    }

    /// A clone of every prompt sent so far.
    #[must_use]
    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().unwrap().clone()
    }

    /// How many `chat` calls have been made.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl LlmClient for ScriptedLlm {
    async fn chat(&self, messages: Vec<Message>) -> anyhow::Result<Message> {
        self.calls.lock().unwrap().push(messages);
        let content = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| self.default.clone());
        Ok(Message::new("assistant", content))
    }

    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        _tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        Ok(Box::pin(futures::stream::empty()))
    }

    fn slot(&self) -> Option<&str> {
        Some("background")
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}

/// Concatenate every message's text — a proxy for "what the LLM saw".
#[must_use]
pub fn joined(messages: &[Message]) -> String {
    messages
        .iter()
        .map(Message::content_str)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The single system message's text.
#[must_use]
pub fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .find(|m| m.role == "system")
        .map(Message::content_str)
        .unwrap_or_default()
}

/// The single user message's text.
#[must_use]
pub fn user_text(messages: &[Message]) -> String {
    messages
        .iter()
        .find(|m| m.role == "user")
        .map(Message::content_str)
        .unwrap_or_default()
}
