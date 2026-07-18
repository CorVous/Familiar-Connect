//! Single-writer history persistence processor (subsystem 06; Python
//! `processors/history_writer.py`).
//!
//! Consumes `discord.text` events and writes user turns to the store. This is
//! the legacy standalone writer — **not** wired alongside `TextResponder` in
//! production (which owns the user-turn write for read-after-write
//! consistency), but retained, exported, and fully tested.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::bus::envelope::Event;
use crate::bus::protocols::EventBus;
use crate::bus::topics::TOPIC_DISCORD_TEXT;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::AppendTurn;
use crate::log_style as ls;
use crate::processors::DiscordTextPayload;

/// Persists turn-generating events into the history store.
pub struct HistoryWriter {
    store: Arc<AsyncHistoryStore>,
    familiar_id: String,
    /// In-process dedup set; survives a single run (the bus does not republish).
    seen: Mutex<HashSet<String>>,
}

impl HistoryWriter {
    /// The processor's human label.
    pub const NAME: &'static str = "history-writer";

    /// Construct a writer for `familiar_id` persisting into `store`.
    #[must_use]
    pub fn new(store: Arc<AsyncHistoryStore>, familiar_id: impl Into<String>) -> Self {
        Self {
            store,
            familiar_id: familiar_id.into(),
            seen: Mutex::new(HashSet::new()),
        }
    }

    /// The processor name (`"history-writer"`).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        Self::NAME
    }

    /// The subscribed topics (`discord.text`).
    #[must_use]
    pub const fn topics(&self) -> [&'static str; 1] {
        [TOPIC_DISCORD_TEXT]
    }

    /// Handle one event: dedup + drop rules, then append a user turn.
    ///
    /// # Errors
    /// Propagates a store write failure.
    pub async fn handle(&self, event: &Event, _bus: &dyn EventBus) -> anyhow::Result<()> {
        if event.topic != TOPIC_DISCORD_TEXT {
            return Ok(());
        }
        if self
            .seen
            .lock()
            .expect("history writer seen mutex")
            .contains(&event.event_id)
        {
            return Ok(());
        }
        let Some(payload) = event.payload.downcast_ref::<DiscordTextPayload>() else {
            return Ok(());
        };
        if payload.familiar_id != self.familiar_id {
            return Ok(());
        }
        if payload.content.is_empty() {
            return Ok(());
        }

        self.seen
            .lock()
            .expect("history writer seen mutex")
            .insert(event.event_id.clone());

        let mut append = AppendTurn::new(
            &self.familiar_id,
            payload.channel_id,
            "user",
            &payload.content,
        );
        append = append.pings_bot(payload.pings_bot);
        if let Some(author) = &payload.author {
            append = append.author(author.clone());
        }
        if let Some(guild_id) = payload.guild_id {
            append = append.guild_id(guild_id);
        }
        self.store.append_turn(append).await?;
        tracing::debug!(
            "{} {} {} {}",
            ls::tag("History", ls::LC),
            ls::kv_styled("append", &event.event_id, ls::W, ls::LC),
            ls::kv_styled("channel", &payload.channel_id.to_string(), ls::W, ls::LY),
            ls::kv_styled("text", &ls::trunc(&payload.content, 80), ls::W, ls::LW),
        );
        Ok(())
    }
}
