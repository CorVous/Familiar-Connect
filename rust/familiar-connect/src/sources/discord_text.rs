//! Discord text message → bus event (subsystem 10; Python
//! `sources/discord_text.py`).
//!
//! Not a pull-loop source; the gateway owns the event loop. The bot's
//! `on_message` hands off to [`DiscordTextSource::publish_text`], which builds the
//! envelope and publishes it. Source events root their own turn
//! (`turn_id == event_id`); `session_id = "discord:{channel_id}"`; the
//! `sequence_number` is a per-instance 1-based monotonic counter.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use uuid::Uuid;

use crate::bus::envelope::{Event, payload};
use crate::bus::protocols::EventBus;
use crate::bus::topics::TOPIC_DISCORD_TEXT;
use crate::identity::Author;
use crate::processors::DiscordTextPayload;

/// Parameters for [`DiscordTextSource::publish_text`] — the exact `discord.text`
/// payload fields (spec 10 § Data formats).
#[derive(Clone, Debug)]
pub struct PublishText {
    /// Discord channel snowflake.
    pub channel_id: i64,
    /// Owning guild snowflake, `None` for DMs.
    pub guild_id: Option<i64>,
    /// Message author.
    pub author: Author,
    /// Message body (already merged with embed/image markers).
    pub content: String,
    /// Platform-native message id (Discord snowflake as string).
    pub message_id: Option<String>,
    /// The message this one replies to, when threaded.
    pub reply_to_message_id: Option<String>,
    /// Non-bot user mentions, resolved to [`Author`]s.
    pub mentions: Vec<Author>,
    /// `img_N` → URL map, empty when no images were detected.
    pub images: HashMap<String, String>,
    /// Whether the incoming message pinged the bot (mention or reply-ping).
    pub pings_bot: bool,
}

impl PublishText {
    /// A minimal payload carrying only the required fields (optionals default to
    /// absent, matching the Python keyword defaults).
    #[must_use]
    pub fn new(
        channel_id: i64,
        guild_id: Option<i64>,
        author: Author,
        content: impl Into<String>,
    ) -> Self {
        Self {
            channel_id,
            guild_id,
            author,
            content: content.into(),
            message_id: None,
            reply_to_message_id: None,
            mentions: Vec::new(),
            images: HashMap::new(),
            pings_bot: false,
        }
    }
}

/// Publishes `discord.text` events for messages the bot observes.
pub struct DiscordTextSource {
    bus: Arc<dyn EventBus>,
    familiar_id: String,
    seq: Mutex<u64>,
}

impl DiscordTextSource {
    /// The stable source name.
    pub const NAME: &'static str = "discord-text";

    /// Build a source publishing onto `bus` tagged with `familiar_id`.
    #[must_use]
    pub fn new(bus: Arc<dyn EventBus>, familiar_id: impl Into<String>) -> Self {
        Self {
            bus,
            familiar_id: familiar_id.into(),
            seq: Mutex::new(0),
        }
    }

    /// Construct + publish a text event; return the envelope.
    ///
    /// `event_id = turn_id = "discord-text-" + 12 hex chars`; the payload keys and
    /// optional-field defaults follow spec 10 § Data formats.
    pub async fn publish_text(&self, params: PublishText) -> Event {
        let seq = {
            let mut guard = self
                .seq
                .lock()
                .expect("discord text source seq mutex poisoned");
            *guard += 1;
            *guard
        };
        let event_id = format!(
            "discord-text-{}",
            &Uuid::new_v4().simple().to_string()[..12]
        );
        let session_id = format!("discord:{}", params.channel_id);
        let event = Event {
            event_id: event_id.clone(),
            turn_id: event_id, // source event: turn_id == event_id
            session_id,
            parent_event_ids: Vec::new(),
            topic: TOPIC_DISCORD_TEXT.to_owned(),
            timestamp: chrono::Utc::now(),
            sequence_number: seq,
            payload: payload(DiscordTextPayload {
                familiar_id: self.familiar_id.clone(),
                channel_id: params.channel_id,
                guild_id: params.guild_id,
                author: Some(params.author),
                content: params.content,
                message_id: params.message_id,
                reply_to_message_id: params.reply_to_message_id,
                mentions: params.mentions,
                images: params.images,
                pings_bot: params.pings_bot,
                wake: false,
            }),
        };
        self.bus.publish(event.clone()).await;
        event
    }
}

/// The narrow publish seam the Discord shell (`on_message`) points at.
///
/// Mirrors Python's `ingest_event` → `source.publish_text`. A scripted double
/// satisfies it in the bot tests; production wires [`DiscordTextSource`].
#[async_trait]
pub trait TextPublisher: Send + Sync {
    /// Publish a text event onto the bus.
    async fn publish(&self, params: PublishText);
}

#[async_trait]
impl TextPublisher for DiscordTextSource {
    async fn publish(&self, params: PublishText) {
        self.publish_text(params).await;
    }
}

#[cfg(test)]
mod tests {
    use super::{DiscordTextSource, PublishText};
    use crate::bus::in_process::InProcessEventBus;
    use crate::bus::protocols::{BackpressurePolicy, EventBus};
    use crate::bus::topics::TOPIC_DISCORD_TEXT;
    use crate::identity::Author;
    use crate::processors::DiscordTextPayload;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::timeout;

    fn author(user_id: &str, display_name: &str) -> Author {
        Author::new(
            "discord",
            user_id,
            Some("alice".to_owned()),
            Some(display_name.to_owned()),
        )
    }

    async fn setup() -> (Arc<InProcessEventBus>, DiscordTextSource) {
        let bus = Arc::new(InProcessEventBus::new());
        bus.start().await;
        let source = DiscordTextSource::new(bus.clone(), "fam");
        (bus, source)
    }

    #[tokio::test]
    async fn publishes_on_topic() {
        let (bus, source) = setup().await;
        let mut sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
        source
            .publish_text(PublishText::new(
                111,
                Some(222),
                author("42", "Alice"),
                "hello",
            ))
            .await;
        let ev = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        bus.shutdown().await;

        assert_eq!(ev.topic, TOPIC_DISCORD_TEXT);
        assert_eq!(ev.session_id, "discord:111");
        let p = ev.payload.downcast_ref::<DiscordTextPayload>().unwrap();
        assert_eq!(p.channel_id, 111);
        assert_eq!(p.guild_id, Some(222));
        assert_eq!(p.content, "hello");
        assert_eq!(
            p.author.as_ref().unwrap().display_name.as_deref(),
            Some("Alice")
        );
        assert_eq!(p.familiar_id, "fam");
    }

    #[tokio::test]
    async fn sequence_number_monotonic() {
        let (bus, source) = setup().await;
        let mut sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
        for content in ["a", "b", "c"] {
            source
                .publish_text(PublishText::new(1, None, author("42", "Alice"), content))
                .await;
        }
        let mut seqs = Vec::new();
        for _ in 0..3 {
            let ev = timeout(Duration::from_secs(1), sub.recv())
                .await
                .unwrap()
                .unwrap();
            seqs.push(ev.sequence_number);
        }
        bus.shutdown().await;
        let mut sorted = seqs.clone();
        sorted.sort_unstable();
        assert_eq!(seqs, sorted);
        let distinct: std::collections::HashSet<u64> = seqs.into_iter().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[tokio::test]
    async fn carries_message_id_reply_and_mentions() {
        let (bus, source) = setup().await;
        let mut sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
        let bob = author("222", "Bob");
        let mut params = PublishText::new(111, Some(222), author("42", "Alice"), "hey @bob");
        params.message_id = Some("msg-9999".to_owned());
        params.reply_to_message_id = Some("msg-prev".to_owned());
        params.mentions = vec![bob.clone()];
        source.publish_text(params).await;
        let ev = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        bus.shutdown().await;
        let p = ev.payload.downcast_ref::<DiscordTextPayload>().unwrap();
        assert_eq!(p.message_id.as_deref(), Some("msg-9999"));
        assert_eq!(p.reply_to_message_id.as_deref(), Some("msg-prev"));
        assert_eq!(p.mentions, vec![bob]);
    }

    #[tokio::test]
    async fn carries_pings_bot_flag() {
        let (bus, source) = setup().await;
        let mut sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
        let mut params = PublishText::new(111, Some(222), author("42", "Alice"), "you there?");
        params.pings_bot = true;
        source.publish_text(params).await;
        let ev = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        bus.shutdown().await;
        assert!(
            ev.payload
                .downcast_ref::<DiscordTextPayload>()
                .unwrap()
                .pings_bot
        );
    }

    #[tokio::test]
    async fn message_id_reply_and_mentions_default() {
        let (bus, source) = setup().await;
        let mut sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
        source
            .publish_text(PublishText::new(
                111,
                Some(222),
                author("42", "Alice"),
                "hello",
            ))
            .await;
        let ev = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        bus.shutdown().await;
        let p = ev.payload.downcast_ref::<DiscordTextPayload>().unwrap();
        assert!(p.message_id.is_none());
        assert!(p.reply_to_message_id.is_none());
        assert!(p.mentions.is_empty());
        assert!(!p.pings_bot);
        assert_eq!(p.images, HashMap::new());
    }

    #[tokio::test]
    async fn turn_id_equals_event_id_for_source_events() {
        let (bus, source) = setup().await;
        let mut sub = bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0);
        source
            .publish_text(PublishText::new(1, None, author("42", "Alice"), "x"))
            .await;
        let ev = timeout(Duration::from_secs(1), sub.recv())
            .await
            .unwrap()
            .unwrap();
        bus.shutdown().await;
        // source events are the turn's root; turn_id == event_id
        assert_eq!(ev.turn_id, ev.event_id);
    }
}
