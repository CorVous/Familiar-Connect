//! Shared fixtures for the sleep-pass integration tests (subsystem 04).
//! Included via `#[path = "sleep_helpers/mod.rs"] mod helpers;` — not a test
//! binary itself.
#![allow(dead_code)]

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
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

/// A scripted `LlmClient`: pops canned replies from a queue and records every
/// call for assertions (mirrors the Python `FakeLLMClient`). An exhausted queue
/// returns an empty assistant message, like the Python double.
pub struct ScriptedLlm {
    replies: Mutex<VecDeque<String>>,
    calls: Mutex<Vec<Vec<Message>>>,
}

impl ScriptedLlm {
    #[must_use]
    pub fn new(replies: &[&str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().map(|s| (*s).to_owned()).collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// A shared handle (for injecting into a `MaintenanceContext`).
    #[must_use]
    pub fn shared(replies: &[&str]) -> Arc<Self> {
        Arc::new(Self::new(replies))
    }

    /// A snapshot of every prompt sent so far.
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
        let content = self.replies.lock().unwrap().pop_front().unwrap_or_default();
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
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}

// --- config fixtures (for the `_default` prose thread-through tests) ---------

/// Path to the checked-in `_default/character.toml`.
#[must_use]
pub fn default_profile() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/familiars/_default/character.toml")
}

/// Projector validator set (mirrors the real `known_projectors()` closely enough
/// to load the shipped default profile).
#[must_use]
pub fn projectors() -> BTreeSet<String> {
    [
        "rolling_summary",
        "rich_note",
        "people_dossier",
        "reflection",
        "fact_supersede",
        "fact_embedding",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

/// Embedder validator set.
#[must_use]
pub fn embedders() -> BTreeSet<String> {
    ["off", "hash", "fastembed"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
}
