//! Discord bot shell (subsystem 10; Python `bot.py`).
//!
//! Owns the gateway client, the subscribe/unsubscribe/diagnostics slash commands,
//! the `on_message` / reaction / typing / voice-state / ready handlers, the
//! [`BotHandle`] adapter bus-only processors post back through, and the
//! `/subscribe-voice` intake pipeline.
//!
//! ## Port shape (DESIGN §4.8 + spec 10 "Rust port notes")
//!
//! The serenity-independent logic — the [`BotHandle`] policy, image collection,
//! embed composition, reaction/edit dispatch, the DM disclaimer + ingest
//! state-machine, and presence rendering — lives on **default features** and is
//! driven through structural view structs ([`MessageView`], [`EmbedView`],
//! [`AttachmentView`], …) plus small seam traits ([`PresenceSink`],
//! [`InteractionAck`], [`ChannelSender`], [`BotStore`], [`TextPublisher`]). The
//! py-cord duck-typed test doubles become scripted implementations of those
//! traits, so the whole conformance suite runs without serenity. Only the thin
//! adapter that builds the view structs from `serenity` events and drives the
//! gateway is `#[cfg(feature = "discord")]`; the songbird voice join +
//! `VoiceTick` → [`RecordingSink`](crate::voice::recording_sink::RecordingSink)
//! wiring is `#[cfg(feature = "discord-voice")]`.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::BoxFuture;
use regex::Regex;

use crate::activities::engine::{ActivityEngine, PresenceCb};
use crate::focus::{FocusManager, PRIVATE_MESSAGE_GUILD_NAME};
use crate::history::StoreError;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::HistoryStore;
use crate::identity::Author;
use crate::log_style as ls;
use crate::processors::{SendText, TriggerTyping};
use crate::sources::discord_embed_text::{EmbedView, format_embeds};
use crate::sources::discord_text::{PublishText, TextPublisher};
use crate::subscriptions::{SubscriptionKind, SubscriptionRegistry};
use crate::typing_interrupt::TypingInterruptHandler;

// ---------------------------------------------------------------------------
// Constants (byte-exact, test-pinned)
// ---------------------------------------------------------------------------

/// One-time warning sent to a DM channel on a user's first admitted DM. Verbatim
/// per reviewer request (PR #176) — do not reword.
pub const DM_BOT_DISCLAIMER: &str = "\u{26a0} This is a bot, and content may not be isolated solely to this channel- \
     treat messages in this conversation as if they were public.";

/// The checkmark the user reacts with to dismiss the disclaimer (pre-seeded).
pub const DM_BOT_DISCLAIMER_DELETE_EMOJI: &str = "\u{2705}";

/// Dismissal hint appended below the verbatim core (kept separate so PR #176's
/// wording stays byte-for-byte intact).
pub const DM_BOT_DISCLAIMER_DISMISS_HINT: &str = "\n\n_React \u{2705} to delete this disclaimer._";

// ---------------------------------------------------------------------------
// Structural view types (built from serenity by the glue; built literally by tests)
// ---------------------------------------------------------------------------

/// A message attachment as `collect_images` reads it.
#[derive(Clone, Debug, Default)]
pub struct AttachmentView {
    /// Attachment URL.
    pub url: Option<String>,
    /// Original filename.
    pub filename: Option<String>,
    /// MIME content type, when known.
    pub content_type: Option<String>,
}

/// A reaction emoji as `emoji_repr` reads it (Discord `PartialEmoji`).
#[derive(Clone, Debug, Default)]
pub struct EmojiView {
    /// Unicode char or custom-emoji name (`None` for a nameless custom emoji).
    pub name: Option<String>,
    /// Custom-emoji snowflake (`None` for a unicode emoji).
    pub id: Option<u64>,
    /// Whether a custom emoji is animated.
    pub animated: bool,
}

/// A user mention as `on_message` reads it.
#[derive(Clone, Debug)]
pub struct MentionView {
    /// Mentioned user id.
    pub id: i64,
    /// Whether the mentioned user is a bot.
    pub is_bot: bool,
    /// The pre-resolved author (used for non-bot mentions).
    pub author: Author,
}

/// Sends a message into a channel (the DM disclaimer post). A scripted double
/// records the send in tests; the serenity glue wraps `ChannelId::say`.
#[async_trait]
pub trait ChannelSender: Send + Sync {
    /// Post `content`; return a handle to the sent message.
    async fn send(&self, content: &str) -> anyhow::Result<Arc<dyn SentMessage>>;
}

/// A posted message the disclaimer flow can react to / delete.
#[async_trait]
pub trait SentMessage: Send + Sync {
    /// The message's platform id.
    fn id(&self) -> i64;
    /// Pre-seed / add a reaction.
    async fn add_reaction(&self, emoji: &str) -> anyhow::Result<()>;
    /// Delete the message.
    async fn delete(&self) -> anyhow::Result<()>;
}

/// An inbound message as `on_message` reads it.
#[derive(Clone)]
pub struct MessageView {
    /// Author's user id.
    pub author_id: i64,
    /// Whether the author is a bot.
    pub author_is_bot: bool,
    /// The resolved author.
    pub author: Author,
    /// Channel id.
    pub channel_id: i64,
    /// Guild id, `None` for DMs.
    pub guild_id: Option<i64>,
    /// Message body.
    pub content: String,
    /// Message id.
    pub message_id: i64,
    /// The replied-to message id, when this is a reply.
    pub reply_to_message_id: Option<i64>,
    /// All mentions (bot targets included — reply-pings live here too).
    pub mentions: Vec<MentionView>,
    /// Image + non-image attachments.
    pub attachments: Vec<AttachmentView>,
    /// Inbound embeds (usually empty; unfurls arrive via edit).
    pub embeds: Vec<EmbedView>,
    /// The channel sender (for the DM disclaimer post).
    pub channel: Arc<dyn ChannelSender>,
}

/// An edited message as `on_message_edit` reads it.
#[derive(Clone, Debug)]
pub struct MessageEditView {
    /// Author's user id.
    pub author_id: i64,
    /// Whether the author is a bot.
    pub author_is_bot: bool,
    /// Channel id.
    pub channel_id: i64,
    /// Message id.
    pub message_id: i64,
    /// The edited body.
    pub content: String,
    /// Embeds before the edit.
    pub before_embeds: Vec<EmbedView>,
    /// Embeds after the edit.
    pub after_embeds: Vec<EmbedView>,
}

/// A raw reaction add/remove payload.
#[derive(Clone, Debug)]
pub struct ReactionPayloadView {
    /// Reacting user id.
    pub user_id: i64,
    /// Reacted-to message id.
    pub message_id: i64,
    /// Channel id.
    pub channel_id: i64,
    /// The reaction emoji.
    pub emoji: EmojiView,
}

/// A raw reaction clear payload (all emoji, or one scoped emoji).
#[derive(Clone, Debug)]
pub struct ReactionClearPayloadView {
    /// Message id whose reactions clear.
    pub message_id: i64,
    /// Channel id.
    pub channel_id: i64,
    /// The scoped emoji (`None` clears all).
    pub emoji: Option<EmojiView>,
}

/// A typing event.
#[derive(Clone, Copy, Debug)]
pub struct TypingEventView {
    /// Channel id.
    pub channel_id: i64,
    /// Typing user id.
    pub user_id: i64,
    /// Whether the typing user is a bot.
    pub is_bot: bool,
}

/// A voice-state update the shell caches for member resolution.
#[derive(Clone, Debug)]
pub struct VoiceStateUpdateView {
    /// The member whose state changed.
    pub member_id: i64,
    /// The member's guild id.
    pub guild_id: i64,
    /// The joined channel id (`None` when leaving).
    pub after_channel_id: Option<i64>,
    /// The resolved author for the member.
    pub author: Author,
}

/// The `on_ready` snapshot (channel + guild name maps from every guild).
#[derive(Clone, Debug, Default)]
pub struct ReadyInfo {
    /// The logged-in bot user id.
    pub user_id: i64,
    /// `(channel_id, channel_name)` for every named channel.
    pub channel_names: Vec<(i64, String)>,
    /// `(channel_id, guild_name)` for every named channel.
    pub guild_names: Vec<(i64, String)>,
}

// ---------------------------------------------------------------------------
// Seam traits
// ---------------------------------------------------------------------------

/// The synchronous history-store write surface the reaction / edit dispatchers
/// touch.
///
/// Implemented for [`HistoryStore`]; a recorder double stands in for the
/// "must-not-write" disclaimer tests (DESIGN §4.8).
pub trait BotStore: Send + Sync {
    /// Rewrite a stored turn's content by platform message id.
    fn update_turn_content_by_message_id(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        content: &str,
    ) -> Result<(), StoreError>;
    /// Apply a ±delta to one `(message, emoji)` reaction row.
    fn bump_reaction(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: &str,
        delta: i64,
    ) -> Result<(), StoreError>;
    /// Drop reactions on a message (all, or one scoped emoji).
    fn clear_reactions(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: Option<&str>,
    ) -> Result<(), StoreError>;
}

impl BotStore for HistoryStore {
    fn update_turn_content_by_message_id(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        content: &str,
    ) -> Result<(), StoreError> {
        Self::update_turn_content_by_message_id(self, familiar_id, platform_message_id, content)
    }
    fn bump_reaction(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: &str,
        delta: i64,
    ) -> Result<(), StoreError> {
        Self::bump_reaction(self, familiar_id, platform_message_id, emoji, delta)
    }
    fn clear_reactions(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: Option<&str>,
    ) -> Result<(), StoreError> {
        Self::clear_reactions(self, familiar_id, platform_message_id, emoji)
    }
}

/// A [`BotStore`] that pushes each write onto the async store's blocking-thread
/// facade instead of running the synchronous `rusqlite` call inline on the
/// caller's task.
///
/// The serenity gateway handlers (reaction / edit dispatch, B-RX) run on the
/// reactor; a direct synchronous store write there would block a tokio worker on
/// the SQLite lock + disk I/O, which DESIGN §4.4 forbids ("DB work stays off the
/// reactor"). This adapter keeps the cheap subscription / emoji gating inline
/// (`apply_message_edit` / `apply_reaction_*`) and spawns only the DB write,
/// honouring spec 10's guidance: "in Rust use the async store, but keep the
/// no-await-in-gateway-handler spirit by spawning if the write can block."
///
/// Writes are fire-and-forget: the enqueue always succeeds (returns `Ok`), and
/// each spawned task logs its own failure, so the dispatchers' `Err` branch is a
/// no-op in production. `bump_reaction` is additive and per-`(message, emoji)`,
/// so the loss of strict inter-event ordering across concurrent spawns is
/// benign (the net count converges). Must be constructed inside a tokio runtime
/// (it is — the run composition root builds it before the gateway starts).
pub struct AsyncBotStore {
    inner: Arc<AsyncHistoryStore>,
}

impl AsyncBotStore {
    /// Wrap the shared async history store.
    #[must_use]
    pub const fn new(inner: Arc<AsyncHistoryStore>) -> Self {
        Self { inner }
    }
}

impl BotStore for AsyncBotStore {
    fn update_turn_content_by_message_id(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        content: &str,
    ) -> Result<(), StoreError> {
        let inner = Arc::clone(&self.inner);
        let (familiar_id, platform_message_id, content) = (
            familiar_id.to_owned(),
            platform_message_id.to_owned(),
            content.to_owned(),
        );
        tokio::spawn(async move {
            if let Err(err) = inner
                .update_turn_content_by_message_id(familiar_id, platform_message_id, content)
                .await
            {
                tracing::warn!("update_turn_content_by_message_id failed: {err}");
            }
        });
        Ok(())
    }

    fn bump_reaction(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: &str,
        delta: i64,
    ) -> Result<(), StoreError> {
        let inner = Arc::clone(&self.inner);
        let (familiar_id, platform_message_id, emoji) = (
            familiar_id.to_owned(),
            platform_message_id.to_owned(),
            emoji.to_owned(),
        );
        tokio::spawn(async move {
            if let Err(err) = inner
                .bump_reaction(familiar_id, platform_message_id, emoji, delta)
                .await
            {
                tracing::warn!("bump_reaction failed: {err}");
            }
        });
        Ok(())
    }

    fn clear_reactions(
        &self,
        familiar_id: &str,
        platform_message_id: &str,
        emoji: Option<&str>,
    ) -> Result<(), StoreError> {
        let inner = Arc::clone(&self.inner);
        let (familiar_id, platform_message_id, emoji) = (
            familiar_id.to_owned(),
            platform_message_id.to_owned(),
            emoji.map(str::to_owned),
        );
        tokio::spawn(async move {
            if let Err(err) = inner
                .clear_reactions(familiar_id, platform_message_id, emoji)
                .await
            {
                tracing::warn!("clear_reactions failed: {err}");
            }
        });
        Ok(())
    }
}

/// Bot presence status (`change_presence(status=…)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresenceStatus {
    /// Green dot.
    Online,
    /// Yellow dot (reachable-away).
    Idle,
    /// Red dot (do-not-disturb-away).
    Dnd,
}

/// One presence update — status plus either a custom-status `state` line (focus
/// sync) or a `CustomActivity` name (away label).
#[derive(Clone, Debug)]
pub struct Presence {
    /// The online/idle/dnd status.
    pub status: PresenceStatus,
    /// The `CustomActivity` name (away label), when set.
    pub activity_name: Option<String>,
    /// The custom-status `state` line (`✨ …`), when set.
    pub activity_state: Option<String>,
}

/// The bot's presence surface (`is_ready` + `change_presence`). A recorder double
/// captures the update sequence in tests; the serenity glue drives the gateway.
#[async_trait]
pub trait PresenceSink: Send + Sync {
    /// Whether the gateway is up (presence updates are dropped before ready).
    fn is_ready(&self) -> bool;
    /// Apply a presence update.
    async fn set_presence(&self, presence: Presence);
}

/// A slash-command interaction acknowledgement surface.
///
/// Both methods treat a dead (`NotFound 10062`) interaction as benign — the
/// action already ran (spec 10 B-SC1/2). A scripted double drives the defer/reply
/// guard tests.
#[async_trait]
pub trait InteractionAck: Send + Sync {
    /// ACK within Discord's 3 s window; `Err` = the interaction is gone.
    async fn defer(&self) -> Result<(), InteractionGone>;
    /// Send the ephemeral confirmation followup; `Err` = the interaction is gone.
    async fn followup(&self, message: &str) -> Result<(), InteractionGone>;
    /// The command name (for the stale-interaction warning).
    fn command_name(&self) -> Option<String>;
}

/// The interaction is gone (Discord `NotFound 10062`), which is benign.
#[derive(Clone, Copy, Debug)]
pub struct InteractionGone;

/// Re-issue away presence for an in-flight activity (the [`ActivityEngine`]
/// surface `on_ready` needs). A trait so the ordering test injects a recorder.
#[async_trait]
pub trait ActivityResync: Send + Sync {
    /// Re-issue away presence when mid-activity; idle ⇒ no-op.
    async fn resync_presence(&self);
}

#[async_trait]
impl ActivityResync for ActivityEngine {
    async fn resync_presence(&self) {
        Self::resync_presence(self).await;
    }
}

// ---------------------------------------------------------------------------
// Interaction helpers (B-SC1/2)
// ---------------------------------------------------------------------------

/// ACK the interaction ASAP to claim Discord's 3 s response window.
///
/// Returns `false` when the interaction is already gone (`NotFound 10062`) —
/// benign; the handler still performs its action.
pub async fn defer_interaction(ctx: &dyn InteractionAck) -> bool {
    if ctx.defer().await.is_ok() {
        return true;
    }
    let name = ctx.command_name().unwrap_or_else(|| "?".to_owned());
    tracing::warn!(
        "{} {}",
        ls::tag("Discord", ls::Y),
        ls::kv_styled("stale_interaction", &name, ls::W, ls::LY),
    );
    false
}

/// Send the ephemeral confirmation followup; swallow a dead interaction.
pub async fn reply(ctx: &dyn InteractionAck, message: &str) {
    if ctx.followup(message).await.is_err() {
        tracing::warn!(
            "{} {}",
            ls::tag("Discord", ls::Y),
            ls::kv_styled("reply_dropped", "interaction_gone", ls::W, ls::LY),
        );
    }
}

// ---------------------------------------------------------------------------
// Pure content helpers
// ---------------------------------------------------------------------------

/// Real bot ping: the bot user appears in `mentions`.
///
/// The gateway puts both `<@id>` mentions and reply-ping targets in `mentions`;
/// role/@everyone mentions and bare name-strings never count.
#[must_use]
pub fn message_pings_bot(mentions: &[MentionView], bot_user_id: Option<i64>) -> bool {
    bot_user_id.is_some_and(|bot_id| mentions.iter().any(|m| m.id == bot_id))
}

/// Append rendered embed text to `content`.
///
/// Empty embed text → `content` unchanged; empty content → embed text alone (no
/// leading blank line); else `"{content}\n\n{embed_text}"`.
#[must_use]
pub fn compose_content_with_embeds(content: &str, embeds: &[EmbedView]) -> String {
    let embed_text = format_embeds(embeds);
    if embed_text.is_empty() {
        return content.to_owned();
    }
    if content.is_empty() {
        return embed_text;
    }
    format!("{content}\n\n{embed_text}")
}

/// Inline image-URL regex (http(s) URLs ending in an image extension).
static IMAGE_URL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)https?://\S+\.(?:png|jpe?g|gif|webp|bmp|tiff?)(?:\?\S+)?")
        .expect("valid image url regex")
});

/// Image extensions recognised in attachment filenames.
const IMAGE_EXTENSIONS: [&str; 8] = [
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
];

/// Whether an attachment is an image by content-type or filename extension.
fn is_image_attachment(att: &AttachmentView) -> bool {
    if let Some(ct) = att.content_type.as_deref() {
        if ct.to_lowercase().starts_with("image/") {
            return true;
        }
    }
    let filename = att.filename.as_deref().unwrap_or("");
    if filename.contains('.') {
        let ext = format!(
            ".{}",
            filename.rsplit('.').next().unwrap_or("").to_lowercase()
        );
        IMAGE_EXTENSIONS.contains(&ext.as_str())
    } else {
        false
    }
}

/// Last path segment of a URL (`""` when it ends with `/`), optionally with a
/// trailing query string stripped.
fn url_last_segment(url: &str, strip_query: bool) -> &str {
    let seg = url.rsplit('/').next().unwrap_or("");
    if strip_query {
        seg.split('?').next().unwrap_or("")
    } else {
        seg
    }
}

/// Dedupe-aware image registration: assign `img_N`, append a marker.
fn add_image(
    url: &str,
    filename: &str,
    images: &mut Vec<(String, String)>,
    seen: &mut HashSet<String>,
    markers: &mut Vec<String>,
) {
    if !seen.insert(url.to_owned()) {
        return;
    }
    let img_id = format!("img_{}", images.len());
    markers.push(format!("[image: {img_id} ({filename})]"));
    images.push((img_id, url.to_owned()));
}

/// Return `(content_with_placeholders, img_id -> url)`.
///
/// Sources, in order: image attachments, `embed.image` (preferring `proxy_url`),
/// inline image URLs in `content`. Ids assigned `img_0, img_1, …` in discovery
/// order; deduped by exact URL (first source wins). `[image: img_N (filename)]`
/// markers are appended one-per-line after the content.
#[must_use]
pub fn collect_images(
    content: &str,
    attachments: &[AttachmentView],
    embeds: &[EmbedView],
) -> (String, HashMap<String, String>) {
    let mut images: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut markers: Vec<String> = Vec::new();

    for att in attachments {
        if !is_image_attachment(att) {
            continue;
        }
        let url = att.url.as_deref().unwrap_or("");
        if url.is_empty() {
            continue;
        }
        let filename = match att.filename.as_deref() {
            Some(f) if !f.is_empty() => f.to_owned(),
            _ => {
                let seg = url_last_segment(url, false);
                if seg.is_empty() {
                    "image".to_owned()
                } else {
                    seg.to_owned()
                }
            }
        };
        add_image(url, &filename, &mut images, &mut seen, &mut markers);
    }

    for embed in embeds {
        let Some(img) = &embed.image else {
            continue;
        };
        // Prefer proxy_url — Discord's re-hosted copy is more reliably fetchable.
        let url = img
            .proxy_url
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| img.url.as_deref().filter(|s| !s.is_empty()))
            .unwrap_or("");
        if url.is_empty() {
            continue;
        }
        let seg = url_last_segment(url, true);
        let filename = if seg.is_empty() {
            "embed-image".to_owned()
        } else {
            seg.to_owned()
        };
        add_image(url, &filename, &mut images, &mut seen, &mut markers);
    }

    for m in IMAGE_URL_RE.find_iter(content) {
        let url = m.as_str();
        let seg = url_last_segment(url, true);
        let filename = if seg.is_empty() {
            "image".to_owned()
        } else {
            seg.to_owned()
        };
        add_image(url, &filename, &mut images, &mut seen, &mut markers);
    }

    let images_map: HashMap<String, String> = images.into_iter().collect();
    if markers.is_empty() {
        return (content.to_owned(), images_map);
    }
    let marker_text = markers.join("\n");
    let new_content = if content.is_empty() {
        marker_text
    } else {
        format!("{content}\n{marker_text}")
    };
    (new_content, images_map)
}

/// Stable string for a reaction emoji.
///
/// Unicode emoji → the char itself; custom → `<:name:id>` (or `<a:name:id>` for
/// animated); a nameless custom emoji → `""` (the caller short-circuits).
fn emoji_repr(emoji: &EmojiView) -> String {
    let Some(id) = emoji.id else {
        return emoji.name.clone().unwrap_or_default();
    };
    emoji.name.as_ref().map_or_else(String::new, |name| {
        let prefix = if emoji.animated { "a" } else { "" };
        format!("<{prefix}:{name}:{id}>")
    })
}

// ---------------------------------------------------------------------------
// Reaction / edit dispatchers (B-RX)
// ---------------------------------------------------------------------------

/// Refresh a stored turn's content when Discord attaches an embed.
///
/// No-ops when the channel isn't text-subscribed, the edit carries no embed, or
/// no stored turn matches `message_id`.
pub fn apply_message_edit(
    store: &dyn BotStore,
    familiar_id: &str,
    is_subscribed: &dyn Fn(i64) -> bool,
    channel_id: i64,
    message_id: &str,
    content: &str,
    embeds: &[EmbedView],
) -> Result<(), StoreError> {
    if !is_subscribed(channel_id) {
        return Ok(());
    }
    let embed_text = format_embeds(embeds);
    if embed_text.is_empty() {
        return Ok(());
    }
    let merged = compose_content_with_embeds(content, embeds);
    store.update_turn_content_by_message_id(familiar_id, message_id, &merged)
}

/// Apply `delta` to one `(message, emoji)` row (subscription-gated first).
pub fn apply_reaction_delta(
    store: &dyn BotStore,
    familiar_id: &str,
    is_subscribed: &dyn Fn(i64) -> bool,
    channel_id: i64,
    message_id: i64,
    emoji: &EmojiView,
    delta: i64,
) -> Result<(), StoreError> {
    if !is_subscribed(channel_id) {
        return Ok(());
    }
    let name = emoji_repr(emoji);
    if name.is_empty() {
        return Ok(());
    }
    store.bump_reaction(familiar_id, &message_id.to_string(), &name, delta)
}

/// Drop all reactions on a message, optionally scoped to one emoji.
pub fn apply_reaction_clear(
    store: &dyn BotStore,
    familiar_id: &str,
    is_subscribed: &dyn Fn(i64) -> bool,
    channel_id: i64,
    message_id: i64,
    emoji: Option<&EmojiView>,
) -> Result<(), StoreError> {
    if !is_subscribed(channel_id) {
        return Ok(());
    }
    let name = emoji.map(emoji_repr);
    store.clear_reactions(familiar_id, &message_id.to_string(), name.as_deref())
}

// ---------------------------------------------------------------------------
// Presence sync (B-PR)
// ---------------------------------------------------------------------------

/// Update bot presence to reflect the current text focus.
async fn sync_presence(fm: &FocusManager, presence: &dyn PresenceSink) {
    let guild = fm.presence_guild();
    let channel = fm.presence_text();
    let state = match (guild, channel) {
        (Some(g), Some(c)) => Some(format!("\u{2728} {g} -> {c}")),
        (_, Some(c)) => Some(format!("\u{2728} {c}")),
        _ => None,
    };
    presence
        .set_presence(Presence {
            status: PresenceStatus::Online,
            activity_name: None,
            activity_state: state,
        })
        .await;
}

/// Presence callback for the [`ActivityEngine`].
///
/// `("idle", label)` → yellow dot + label; `("dnd", label)` → red dot + label;
/// anything else (`"online"`) → restore focus presence (or a plain online when no
/// focus manager). No-op until the bot is ready (headless tests / pre-login
/// engine start).
#[must_use]
pub fn build_activity_presence_cb(handle: Arc<BotHandle>) -> PresenceCb {
    Arc::new(move |status: String, label: Option<String>| {
        let handle = Arc::clone(&handle);
        Box::pin(async move {
            if !handle.presence.is_ready() {
                return Ok(());
            }
            if status == "idle" || status == "dnd" {
                let st = if status == "idle" {
                    PresenceStatus::Idle
                } else {
                    PresenceStatus::Dnd
                };
                let name = label
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "away".to_owned());
                handle
                    .presence
                    .set_presence(Presence {
                        status: st,
                        activity_name: Some(name),
                        activity_state: None,
                    })
                    .await;
                return Ok(());
            }
            match &handle.focus_manager {
                Some(fm) => sync_presence(fm, handle.presence.as_ref()).await,
                None => {
                    handle
                        .presence
                        .set_presence(Presence {
                            status: PresenceStatus::Online,
                            activity_name: None,
                            activity_state: None,
                        })
                        .await;
                }
            }
            Ok(())
        })
    })
}

// ---------------------------------------------------------------------------
// BotHandle
// ---------------------------------------------------------------------------

/// Synchronous voice-member resolver seam (`(channel_id, user_id) -> Author`).
/// Must never block — the audio path cannot tolerate REST round-trips (B-VM29).
pub type ResolveMember = Arc<dyn Fn(i64, i64) -> Option<Author> + Send + Sync>;

/// Bot + outbound seams for bus processors.
///
/// The closures the Python dataclass stored (`send_text`, `trigger_typing`,
/// `resolve_member`) become trait objects; the mutable side caches
/// (`voice_members`, the late-set `activity_engine`, and — under `discord-voice`
/// — `voice_runtime`) sit behind mutexes so many tasks can touch them.
pub struct BotHandle {
    /// Outbound text seam consumed by the text responder (06).
    pub send_text: Arc<dyn SendText>,
    /// Typing-indicator factory consumed by the text responder (06).
    pub trigger_typing: Option<Arc<dyn TriggerTyping>>,
    /// Typing-event policy consumed by `on_typing` + the text responder (06).
    pub typing_interrupt: Option<Arc<TypingInterruptHandler>>,
    /// Attentional focus controller (05).
    pub focus_manager: Option<Arc<FocusManager>>,
    /// The bot presence surface.
    pub presence: Arc<dyn PresenceSink>,
    /// Voice-only member side cache (`user_id -> Author`).
    pub voice_members: Mutex<HashMap<i64, Author>>,
    /// Voice-turn member resolver consumed by the voice responder (06).
    pub resolve_member: Mutex<Option<ResolveMember>>,
    /// Absence controller (11); `on_ready` resyncs away presence via it.
    pub activity_engine: Mutex<Option<Arc<dyn ActivityResync>>>,
    /// Per-voice-channel intake pipeline state (voice feature only). A
    /// `BTreeMap` (not `HashMap`) so the TTS voice-client getter's
    /// `.values().next()` is deterministic + stable (Python `_first_voice_client`).
    #[cfg(feature = "discord-voice")]
    pub voice_runtime:
        Mutex<std::collections::BTreeMap<i64, crate::bot::voice_intake::VoiceRuntime>>,
    /// Active voice-channel ids (the default-feature proxy for `voice_runtime`;
    /// feeds the activity engine's `voice_active_fn`).
    pub voice_channels: Mutex<HashSet<i64>>,
}

impl BotHandle {
    /// A handle with the required `send_text` + `presence` seams; everything else
    /// defaults to absent.
    #[must_use]
    pub fn new(send_text: Arc<dyn SendText>, presence: Arc<dyn PresenceSink>) -> Self {
        Self {
            send_text,
            trigger_typing: None,
            typing_interrupt: None,
            focus_manager: None,
            presence,
            voice_members: Mutex::new(HashMap::new()),
            resolve_member: Mutex::new(None),
            activity_engine: Mutex::new(None),
            #[cfg(feature = "discord-voice")]
            voice_runtime: Mutex::new(std::collections::BTreeMap::new()),
            voice_channels: Mutex::new(HashSet::new()),
        }
    }

    /// Builder: attach the focus manager.
    #[must_use]
    pub fn with_focus_manager(mut self, fm: Arc<FocusManager>) -> Self {
        self.focus_manager = Some(fm);
        self
    }

    /// Builder: attach the typing-indicator factory.
    #[must_use]
    pub fn with_trigger_typing(mut self, tt: Arc<dyn TriggerTyping>) -> Self {
        self.trigger_typing = Some(tt);
        self
    }

    /// Builder: attach the typing-interrupt policy.
    #[must_use]
    pub fn with_typing_interrupt(mut self, ti: Arc<TypingInterruptHandler>) -> Self {
        self.typing_interrupt = Some(ti);
        self
    }

    /// Install the (late-set) activity engine.
    pub fn set_activity_engine(&self, engine: Arc<dyn ActivityResync>) {
        *self
            .activity_engine
            .lock()
            .expect("activity engine mutex poisoned") = Some(engine);
    }

    /// Resolve a voice member from the side cache alone (no I/O).
    #[must_use]
    pub fn voice_member_cached(&self, user_id: i64) -> Option<Author> {
        self.voice_members
            .lock()
            .expect("voice members mutex poisoned")
            .get(&user_id)
            .cloned()
    }
}

// ---------------------------------------------------------------------------
// Event ingest
// ---------------------------------------------------------------------------

/// Publish a text event onto the bus (the one seam `on_message` points at).
pub async fn ingest_event(source: &dyn TextPublisher, params: PublishText) {
    source.publish(params).await;
}

// ---------------------------------------------------------------------------
// Event dispatch core (default-feature; the serenity glue adapts events to it)
// ---------------------------------------------------------------------------

/// The Discord event-handler core, decoupled from serenity.
///
/// Holds the shared mutable ingest state (bot user id, the subscription registry,
/// the once-per-user disclaimer set + live disclaimer messages) and the outbound
/// seams. Each `on_*` method takes a structural view struct the serenity glue
/// builds from a gateway event (and tests build literally).
pub struct BotEvents {
    familiar_id: String,
    bot_user_id: Arc<Mutex<Option<i64>>>,
    subscriptions: Arc<Mutex<SubscriptionRegistry>>,
    dm_allowlist: Vec<i64>,
    store: Arc<dyn BotStore>,
    text_source: Arc<dyn TextPublisher>,
    handle: Arc<BotHandle>,
    disclaimed_dm_users: Mutex<HashSet<i64>>,
    disclaimer_messages: Mutex<HashMap<i64, Arc<dyn SentMessage>>>,
}

impl BotEvents {
    /// Build the dispatch core over its shared state + seams.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "the dispatch core threads the shared ingest state + seams"
    )]
    pub fn new(
        familiar_id: impl Into<String>,
        bot_user_id: Arc<Mutex<Option<i64>>>,
        subscriptions: Arc<Mutex<SubscriptionRegistry>>,
        dm_allowlist: Vec<i64>,
        store: Arc<dyn BotStore>,
        text_source: Arc<dyn TextPublisher>,
        handle: Arc<BotHandle>,
    ) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            bot_user_id,
            subscriptions,
            dm_allowlist,
            store,
            text_source,
            handle,
            disclaimed_dm_users: Mutex::new(HashSet::new()),
            disclaimer_messages: Mutex::new(HashMap::new()),
        }
    }

    fn bot_user_id(&self) -> Option<i64> {
        *self.bot_user_id.lock().expect("bot_user_id mutex poisoned")
    }

    fn text_subscribed(&self, channel_id: i64) -> bool {
        u64::try_from(channel_id)
            .ok()
            .and_then(|c| {
                self.subscriptions
                    .lock()
                    .expect("subscriptions mutex poisoned")
                    .get(c, SubscriptionKind::Text)
            })
            .is_some()
    }

    /// Register a DM channel as an ephemeral (never-persisted) text subscription
    /// and seed a text focus when none exists.
    fn register_dm_channel(&self, channel_id: i64) {
        if let Ok(cid) = u64::try_from(channel_id) {
            let _ = self
                .subscriptions
                .lock()
                .expect("subscriptions mutex poisoned")
                .add(cid, SubscriptionKind::Text, None, false);
        }
        if let Some(fm) = &self.handle.focus_manager {
            fm.set_guild_name(channel_id, PRIVATE_MESSAGE_GUILD_NAME);
            if fm.get_focus("text").is_none() {
                fm.set_focus_immediately(channel_id, "text");
            }
        }
    }

    /// `/subscribe-voice` registry + shared-state mutation (default-feature; the
    /// songbird join + intake pipeline is layered on by the `discord-voice` glue
    /// in [`create_bot`]'s dispatcher).
    ///
    /// Registers a persisted voice subscription and marks the channel active in
    /// the `voice_channels` proxy the activity engine's `voice_active_fn` reads.
    /// The focus manager sees the new subscription immediately through the
    /// shared registry (no restart), matching Python's single-registry
    /// semantics.
    pub fn on_subscribe_voice(&self, channel_id: i64, guild_id: Option<i64>) {
        if let Ok(cid) = u64::try_from(channel_id) {
            let _ = self
                .subscriptions
                .lock()
                .expect("subscriptions mutex poisoned")
                .add(
                    cid,
                    SubscriptionKind::Voice,
                    guild_id.and_then(|g| u64::try_from(g).ok()),
                    true,
                );
        }
        self.handle
            .voice_channels
            .lock()
            .expect("voice channels mutex poisoned")
            .insert(channel_id);
    }

    /// `/unsubscribe-voice` registry + shared-state mutation. Finds the guild's
    /// voice subscription, removes it, and clears the active-channel proxy.
    /// Returns the unsubscribed voice channel id (the `discord-voice` glue tears
    /// down the intake pipeline + leaves the channel with it), or `None` when
    /// the guild has no voice subscription.
    pub fn on_unsubscribe_voice(&self, guild_id: i64) -> Option<i64> {
        let sub = u64::try_from(guild_id).ok().and_then(|g| {
            self.subscriptions
                .lock()
                .expect("subscriptions mutex poisoned")
                .voice_in_guild(g)
        })?;
        let channel_id = i64::try_from(sub.channel_id).ok()?;
        let _ = self
            .subscriptions
            .lock()
            .expect("subscriptions mutex poisoned")
            .remove(sub.channel_id, SubscriptionKind::Voice);
        self.handle
            .voice_channels
            .lock()
            .expect("voice channels mutex poisoned")
            .remove(&channel_id);
        Some(channel_id)
    }

    /// `on_message` ingest (B-OM). Guard order is load-bearing: own echo, then any
    /// bot author, then the DM-allowlist / subscription gates.
    pub async fn on_message(&self, message: MessageView) {
        let bot_user_id = self.bot_user_id();
        if bot_user_id == Some(message.author_id) {
            return;
        }
        if message.author_is_bot {
            return;
        }
        if message.guild_id.is_none() {
            // DM: admit only allowlisted users.
            if !self.dm_allowlist.contains(&message.author_id) {
                return;
            }
            // First admitted DM from this user: warn DMs aren't private.
            let first = self
                .disclaimed_dm_users
                .lock()
                .expect("disclaimed users mutex poisoned")
                .insert(message.author_id);
            if first {
                let body = format!("{DM_BOT_DISCLAIMER}{DM_BOT_DISCLAIMER_DISMISS_HINT}");
                match message.channel.send(&body).await {
                    Ok(sent) => {
                        self.disclaimer_messages
                            .lock()
                            .expect("disclaimer messages mutex poisoned")
                            .insert(sent.id(), Arc::clone(&sent));
                        if let Err(err) = sent.add_reaction(DM_BOT_DISCLAIMER_DELETE_EMOJI).await {
                            tracing::debug!("disclaimer add_reaction failed: {err}");
                        }
                    }
                    Err(err) => {
                        tracing::warn!("disclaimer send failed: {err}");
                    }
                }
            }
            self.register_dm_channel(message.channel_id);
        } else if !self.text_subscribed(message.channel_id) {
            return;
        }

        let reply_to = message.reply_to_message_id.map(|id| id.to_string());
        let mention_authors: Vec<Author> = message
            .mentions
            .iter()
            .filter(|m| !m.is_bot)
            .map(|m| m.author.clone())
            .collect();
        let pings_bot = message_pings_bot(&message.mentions, bot_user_id);
        let text = compose_content_with_embeds(&message.content, &message.embeds);
        let (text, images) = collect_images(&text, &message.attachments, &message.embeds);

        ingest_event(
            self.text_source.as_ref(),
            PublishText {
                channel_id: message.channel_id,
                guild_id: message.guild_id,
                author: message.author.clone(),
                content: text,
                message_id: Some(message.message_id.to_string()),
                reply_to_message_id: reply_to,
                mentions: mention_authors,
                images,
                pings_bot,
            },
        )
        .await;
    }

    /// `on_message_edit` (B-RX16/17): act only when the edit *added* embed content.
    pub fn on_message_edit(&self, edit: &MessageEditView) {
        if self.bot_user_id() == Some(edit.author_id) {
            return;
        }
        if edit.author_is_bot {
            return;
        }
        if edit.after_embeds.is_empty() || edit.before_embeds == edit.after_embeds {
            return;
        }
        let is_sub = |ch: i64| self.text_subscribed(ch);
        if let Err(err) = apply_message_edit(
            self.store.as_ref(),
            &self.familiar_id,
            &is_sub,
            edit.channel_id,
            &edit.message_id.to_string(),
            &edit.content,
            &edit.after_embeds,
        ) {
            tracing::warn!("apply_message_edit failed: {err}");
        }
    }

    /// `on_typing` (B-EV22): forward to the typing-interrupt policy when wired.
    pub fn on_typing(&self, ev: TypingEventView) {
        if let Some(ti) = &self.handle.typing_interrupt {
            ti.notify_typing(ev.channel_id, ev.user_id, ev.is_bot);
        }
    }

    /// `on_voice_state_update` (B-EV23): cache voice-only members joining the
    /// subscribed voice channel.
    pub fn on_voice_state_update(&self, ev: VoiceStateUpdateView) {
        if self.bot_user_id() == Some(ev.member_id) {
            return;
        }
        let Some(after_channel) = ev.after_channel_id else {
            return;
        };
        let matches = u64::try_from(ev.guild_id).ok().and_then(|g| {
            self.subscriptions
                .lock()
                .expect("subscriptions mutex poisoned")
                .voice_in_guild(g)
        });
        let matches = matches
            .is_some_and(|sub| u64::try_from(after_channel).is_ok_and(|c| sub.channel_id == c));
        if !matches {
            return;
        }
        self.handle
            .voice_members
            .lock()
            .expect("voice members mutex poisoned")
            .insert(ev.member_id, ev.author);
    }

    /// `on_ready` (B-PR24): set the bot user id, bulk-populate focus name caches,
    /// install the presence hook, sync presence, then resync the away presence so
    /// a mid-activity idle/dnd wins over the online focus presence.
    pub async fn on_ready(&self, ready: ReadyInfo) {
        *self.bot_user_id.lock().expect("bot_user_id mutex poisoned") = Some(ready.user_id);
        if let Some(fm) = &self.handle.focus_manager {
            for (cid, name) in &ready.channel_names {
                fm.set_channel_name(*cid, name.clone());
            }
            for (cid, gname) in &ready.guild_names {
                fm.set_guild_name(*cid, gname.clone());
            }
            let weak = Arc::downgrade(fm);
            let presence = Arc::clone(&self.handle.presence);
            fm.set_on_shift(Arc::new(move || {
                let weak = weak.clone();
                let presence = Arc::clone(&presence);
                Box::pin(async move {
                    if let Some(fm) = weak.upgrade() {
                        sync_presence(&fm, presence.as_ref()).await;
                    }
                }) as BoxFuture<'static, ()>
            }));
            sync_presence(fm, self.handle.presence.as_ref()).await;
        }
        let engine = self
            .handle
            .activity_engine
            .lock()
            .expect("activity engine mutex poisoned")
            .clone();
        if let Some(engine) = engine {
            engine.resync_presence().await;
        }
    }

    /// `on_raw_reaction_add` (B-RX18): the disclaimer's checkmark-dismiss
    /// lifecycle, else a `+1` reaction delta.
    pub async fn on_raw_reaction_add(&self, payload: &ReactionPayloadView) {
        let is_disclaimer = self
            .disclaimer_messages
            .lock()
            .expect("disclaimer messages mutex poisoned")
            .contains_key(&payload.message_id);
        if is_disclaimer {
            let is_user_checkmark = Some(payload.user_id) != self.bot_user_id()
                && emoji_repr(&payload.emoji) == DM_BOT_DISCLAIMER_DELETE_EMOJI;
            if is_user_checkmark {
                let sent = self
                    .disclaimer_messages
                    .lock()
                    .expect("disclaimer messages mutex poisoned")
                    .remove(&payload.message_id);
                if let Some(sent) = sent {
                    if let Err(err) = sent.delete().await {
                        tracing::debug!("disclaimer delete failed: {err}");
                    }
                }
            }
            return;
        }
        self.reaction_delta(payload, 1);
    }

    /// `on_raw_reaction_remove` (B-RX19): disclaimer reactions are a total no-op;
    /// else a `-1` reaction delta.
    pub fn on_raw_reaction_remove(&self, payload: &ReactionPayloadView) {
        let is_disclaimer = self
            .disclaimer_messages
            .lock()
            .expect("disclaimer messages mutex poisoned")
            .contains_key(&payload.message_id);
        if is_disclaimer {
            return;
        }
        self.reaction_delta(payload, -1);
    }

    fn reaction_delta(&self, payload: &ReactionPayloadView, delta: i64) {
        let is_sub = |ch: i64| self.text_subscribed(ch);
        if let Err(err) = apply_reaction_delta(
            self.store.as_ref(),
            &self.familiar_id,
            &is_sub,
            payload.channel_id,
            payload.message_id,
            &payload.emoji,
            delta,
        ) {
            tracing::warn!("apply_reaction_delta failed: {err}");
        }
    }

    /// `on_raw_reaction_clear`: drop every reaction on the message.
    pub fn on_raw_reaction_clear(&self, payload: &ReactionClearPayloadView) {
        self.reaction_clear(payload.channel_id, payload.message_id, None);
    }

    /// `on_raw_reaction_clear_emoji`: drop one emoji's reactions on the message.
    pub fn on_raw_reaction_clear_emoji(&self, payload: &ReactionClearPayloadView) {
        self.reaction_clear(
            payload.channel_id,
            payload.message_id,
            payload.emoji.as_ref(),
        );
    }

    fn reaction_clear(&self, channel_id: i64, message_id: i64, emoji: Option<&EmojiView>) {
        let is_sub = |ch: i64| self.text_subscribed(ch);
        if let Err(err) = apply_reaction_clear(
            self.store.as_ref(),
            &self.familiar_id,
            &is_sub,
            channel_id,
            message_id,
            emoji,
        ) {
            tracing::warn!("apply_reaction_clear failed: {err}");
        }
    }
}

// ---------------------------------------------------------------------------
// Serenity gateway glue (thin adapter — cfg(feature = "discord"))
// ---------------------------------------------------------------------------
//
// This is the only part of the subsystem that names `serenity`. It builds the
// structural view structs from gateway events, drives them through [`BotEvents`],
// and implements the outbound seam traits over `serenity::Http` / the gateway
// shard. Everything policy-shaped lives above, on default features. Compiled +
// clippy-checked under `--features discord`; its behaviour is exercised through
// the default-feature core.
#[cfg(feature = "discord")]
pub use serenity_glue::{CreateBotDeps, create_bot};

#[cfg(feature = "discord")]
#[allow(
    clippy::wildcard_imports,
    clippy::cast_possible_wrap,
    clippy::significant_drop_tightening,
    reason = "serenity's prelude is idiomatic; snowflake casts are lossless in practice"
)]
mod serenity_glue {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serenity::all::{
        ActivityData, Attachment, ChannelId, Client, Command, Context, CreateAllowedMentions,
        CreateCommand, CreateInteractionResponseFollowup, CreateMessage, Embed, EventHandler,
        GatewayIntents, Interaction, Message, MessageId, MessageUpdateEvent, OnlineStatus,
        Reaction, ReactionType, Ready, TypingStartEvent, User, VoiceState,
    };

    use super::{
        AttachmentView, BotEvents, BotHandle, BotStore, ChannelSender, EmojiView, InteractionAck,
        InteractionGone, MentionView, MessageEditView, MessageView, Presence, PresenceSink,
        PresenceStatus, ReactionClearPayloadView, ReactionPayloadView, ReadyInfo, ResolveMember,
        SentMessage, TypingEventView, VoiceStateUpdateView, defer_interaction, reply,
    };
    use crate::diagnostics::collector::get_span_collector;
    use crate::diagnostics::report::render_summary_table;
    use crate::focus::FocusManager;
    use crate::history::async_store::AsyncHistoryStore;
    use crate::identity::Author;
    use crate::processors::{SendText, TriggerTyping, TypingIndicator};
    use crate::sources::discord_embed_text::{EmbedFieldView, EmbedImageView, EmbedView};
    use crate::sources::discord_text::DiscordTextSource;
    use crate::subscriptions::SubscriptionKind;

    // -- view builders ---------------------------------------------------

    fn author_from_user(user: &User) -> Author {
        let display = user
            .global_name
            .clone()
            .unwrap_or_else(|| user.name.clone());
        Author::new(
            "discord",
            user.id.get().to_string(),
            Some(user.name.clone()),
            Some(display),
        )
    }

    fn attachment_view(att: &Attachment) -> AttachmentView {
        AttachmentView {
            url: Some(att.url.clone()),
            filename: Some(att.filename.clone()),
            content_type: att.content_type.clone(),
        }
    }

    fn embed_view(embed: &Embed) -> EmbedView {
        EmbedView {
            provider_name: embed.provider.as_ref().and_then(|p| p.name.clone()),
            author_name: embed.author.as_ref().map(|a| a.name.clone()),
            title: embed.title.clone(),
            description: embed.description.clone(),
            footer_text: embed.footer.as_ref().map(|f| f.text.clone()),
            url: embed.url.clone(),
            fields: embed
                .fields
                .iter()
                .map(|f| EmbedFieldView {
                    name: Some(f.name.clone()),
                    value: Some(f.value.clone()),
                })
                .collect(),
            image: embed.image.as_ref().map(|i| EmbedImageView {
                url: Some(i.url.clone()),
                proxy_url: i.proxy_url.clone(),
            }),
        }
    }

    fn emoji_view(emoji: &ReactionType) -> EmojiView {
        match emoji {
            ReactionType::Custom { animated, id, name } => EmojiView {
                name: name.clone(),
                id: Some(id.get()),
                animated: *animated,
            },
            ReactionType::Unicode(s) => EmojiView {
                name: Some(s.clone()),
                id: None,
                animated: false,
            },
            _ => EmojiView::default(),
        }
    }

    fn message_view(ctx: &Context, msg: &Message) -> MessageView {
        MessageView {
            author_id: msg.author.id.get() as i64,
            author_is_bot: msg.author.bot,
            author: author_from_user(&msg.author),
            channel_id: msg.channel_id.get() as i64,
            guild_id: msg.guild_id.map(|g| g.get() as i64),
            content: msg.content.clone(),
            message_id: msg.id.get() as i64,
            reply_to_message_id: msg
                .message_reference
                .as_ref()
                .and_then(|r| r.message_id)
                .map(|m| m.get() as i64),
            mentions: msg
                .mentions
                .iter()
                .map(|u| MentionView {
                    id: u.id.get() as i64,
                    is_bot: u.bot,
                    author: author_from_user(u),
                })
                .collect(),
            attachments: msg.attachments.iter().map(attachment_view).collect(),
            embeds: msg.embeds.iter().map(embed_view).collect(),
            channel: Arc::new(SerenityChannel {
                http: ctx.http.clone(),
                channel_id: msg.channel_id,
            }),
        }
    }

    fn reaction_payload(reaction: &Reaction) -> ReactionPayloadView {
        ReactionPayloadView {
            user_id: reaction.user_id.map_or(0, |u| u.get() as i64),
            message_id: reaction.message_id.get() as i64,
            channel_id: reaction.channel_id.get() as i64,
            emoji: emoji_view(&reaction.emoji),
        }
    }

    // -- outbound seam adapters ------------------------------------------

    /// The gateway presence surface, backed by a [`Context`] captured on ready.
    struct SerenityPresence {
        ctx: Mutex<Option<Context>>,
    }

    #[async_trait]
    impl PresenceSink for SerenityPresence {
        fn is_ready(&self) -> bool {
            self.ctx.lock().expect("presence ctx mutex").is_some()
        }
        async fn set_presence(&self, presence: Presence) {
            let ctx = self.ctx.lock().expect("presence ctx mutex").clone();
            let Some(ctx) = ctx else {
                return;
            };
            let status = match presence.status {
                PresenceStatus::Online => OnlineStatus::Online,
                PresenceStatus::Idle => OnlineStatus::Idle,
                PresenceStatus::Dnd => OnlineStatus::DoNotDisturb,
            };
            let activity = presence
                .activity_name
                .or(presence.activity_state)
                .map(ActivityData::custom);
            ctx.set_presence(activity, status);
        }
    }

    /// The DM-disclaimer channel sender.
    struct SerenityChannel {
        http: Arc<serenity::all::Http>,
        channel_id: ChannelId,
    }

    #[async_trait]
    impl ChannelSender for SerenityChannel {
        async fn send(&self, content: &str) -> anyhow::Result<Arc<dyn SentMessage>> {
            let msg = self.channel_id.say(&self.http, content).await?;
            Ok(Arc::new(SerenitySent {
                http: self.http.clone(),
                message: msg,
            }))
        }
    }

    struct SerenitySent {
        http: Arc<serenity::all::Http>,
        message: Message,
    }

    #[async_trait]
    impl SentMessage for SerenitySent {
        fn id(&self) -> i64 {
            self.message.id.get() as i64
        }
        async fn add_reaction(&self, emoji: &str) -> anyhow::Result<()> {
            self.message
                .react(&self.http, ReactionType::Unicode(emoji.to_owned()))
                .await?;
            Ok(())
        }
        async fn delete(&self) -> anyhow::Result<()> {
            self.message.delete(&self.http).await?;
            Ok(())
        }
    }

    /// Outbound text seam (`send_text`) over `serenity::Http`.
    struct SerenitySendText {
        http: Arc<serenity::all::Http>,
    }

    #[async_trait]
    impl SendText for SerenitySendText {
        async fn send(
            &self,
            channel_id: i64,
            content: &str,
            reply_to_message_id: Option<&str>,
            mention_user_ids: &[i64],
        ) -> anyhow::Result<Option<String>> {
            let Ok(cid) = u64::try_from(channel_id) else {
                return Ok(None);
            };
            let channel = ChannelId::new(cid);
            let users: Vec<serenity::all::UserId> = mention_user_ids
                .iter()
                .filter_map(|&u| u64::try_from(u).ok())
                .map(serenity::all::UserId::new)
                .collect();
            let allowed = CreateAllowedMentions::new()
                .everyone(false)
                .all_roles(false)
                .users(users);
            let mut builder = CreateMessage::new()
                .content(content)
                .allowed_mentions(allowed);
            if let Some(rid) = reply_to_message_id.and_then(|r| r.parse::<u64>().ok()) {
                builder = builder.reference_message((channel, MessageId::new(rid)));
            }
            match channel.send_message(&self.http, builder).await {
                Ok(sent) => Ok(Some(sent.id.get().to_string())),
                Err(err) => {
                    tracing::warn!("send_text failed: channel={channel_id} err={err}");
                    Ok(None)
                }
            }
        }
    }

    /// Typing-indicator factory.
    struct SerenityTyping {
        http: Arc<serenity::all::Http>,
    }

    struct SerenityTypingIndicator;

    #[async_trait]
    impl TypingIndicator for SerenityTypingIndicator {
        async fn close(&self) {}
    }

    #[async_trait]
    impl TriggerTyping for SerenityTyping {
        async fn open(&self, channel_id: i64) -> Box<dyn TypingIndicator> {
            if let Ok(cid) = u64::try_from(channel_id) {
                let _ = ChannelId::new(cid).broadcast_typing(&self.http).await;
            }
            Box::new(SerenityTypingIndicator)
        }
    }

    // -- slash-command interaction adapter -------------------------------

    struct SlashCtx {
        command: serenity::all::CommandInteraction,
        http: Arc<serenity::all::Http>,
    }

    #[async_trait]
    impl InteractionAck for SlashCtx {
        async fn defer(&self) -> Result<(), InteractionGone> {
            self.command
                .defer_ephemeral(&self.http)
                .await
                .map_err(|_| InteractionGone)
        }
        async fn followup(&self, message: &str) -> Result<(), InteractionGone> {
            self.command
                .create_followup(
                    &self.http,
                    CreateInteractionResponseFollowup::new()
                        .content(message)
                        .ephemeral(true),
                )
                .await
                .map(|_| ())
                .map_err(|_| InteractionGone)
        }
        fn command_name(&self) -> Option<String> {
            Some(self.command.data.name.clone())
        }
    }

    // -- gateway event handler -------------------------------------------

    struct Handler {
        events: Arc<BotEvents>,
        slash: Arc<SlashContext>,
        presence: Arc<SerenityPresence>,
    }

    /// The per-command dispatch context (subscriptions + focus + diagnostics,
    /// plus the voice-intake seams `/subscribe-voice` needs under
    /// `discord-voice`).
    struct SlashContext {
        familiar_id: String,
        subscriptions: Arc<Mutex<crate::subscriptions::SubscriptionRegistry>>,
        history_store: Arc<AsyncHistoryStore>,
        focus_manager: Option<Arc<FocusManager>>,
        // Read only by the voice dispatch paths; without discord-voice the
        // field still exists (constructor stays feature-agnostic) but is dead.
        #[cfg_attr(not(feature = "discord-voice"), allow(dead_code))]
        handle: Arc<BotHandle>,
        #[cfg(feature = "discord-voice")]
        bus: Arc<dyn crate::bus::protocols::EventBus>,
        #[cfg(feature = "discord-voice")]
        transcriber_template: Option<Arc<Mutex<Box<dyn crate::stt::Transcriber>>>>,
        #[cfg(feature = "discord-voice")]
        local_turn_detector: Option<Arc<crate::voice::turn_detection::LocalTurnDetector>>,
    }

    impl Handler {
        async fn dispatch_command(
            &self,
            ctx: &Context,
            command: serenity::all::CommandInteraction,
        ) {
            let name = command.data.name.clone();
            let guild_id = command.guild_id.map(|g| g.get() as i64);
            let channel_id = command.channel_id.get() as i64;
            let ack = SlashCtx {
                command,
                http: ctx.http.clone(),
            };
            match name.as_str() {
                "subscribe-text" => {
                    defer_interaction(&ack).await;
                    if let Ok(cid) = u64::try_from(channel_id) {
                        let _ = self.slash.subscriptions.lock().expect("subs").add(
                            cid,
                            SubscriptionKind::Text,
                            guild_id.and_then(|g| u64::try_from(g).ok()),
                            true,
                        );
                        reply(&ack, "Listening in this channel.").await;
                    } else {
                        reply(&ack, "No channel in context.").await;
                    }
                }
                "unsubscribe-text" => {
                    defer_interaction(&ack).await;
                    if let Ok(cid) = u64::try_from(channel_id) {
                        let _ = self
                            .slash
                            .subscriptions
                            .lock()
                            .expect("subs")
                            .remove(cid, SubscriptionKind::Text);
                    }
                    reply(&ack, "No longer listening here.").await;
                }
                #[cfg(feature = "discord-voice")]
                "subscribe-voice" => {
                    self.dispatch_subscribe_voice(ctx, &ack).await;
                }
                #[cfg(feature = "discord-voice")]
                "unsubscribe-voice" => {
                    self.dispatch_unsubscribe_voice(ctx, &ack).await;
                }
                "diagnostics" => {
                    defer_interaction(&ack).await;
                    let summary = get_span_collector().summary();
                    let mut text = render_summary_table(&summary);
                    if let Some(fm) = &self.slash.focus_manager {
                        text.push_str(&self.focus_line(fm).await);
                    }
                    reply(&ack, &text).await;
                }
                other => {
                    tracing::warn!("unknown slash command: {other}");
                }
            }
        }

        async fn focus_line(&self, fm: &FocusManager) -> String {
            use std::fmt::Write as _;
            let tf = fm
                .get_focus("text")
                .map_or_else(|| "unset".to_owned(), |c| format!("#{c}"));
            let vf = fm
                .get_focus("voice")
                .map_or_else(|| "unset".to_owned(), |c| format!("#{c}"));
            let mut line = format!("\nFocus: text={tf} voice={vf}");
            let staged = self
                .slash
                .history_store
                .staged_channels(self.slash.familiar_id.clone())
                .await
                .unwrap_or_default();
            if !staged.is_empty() {
                let mut ids: Vec<i64> = staged.keys().copied().collect();
                ids.sort_unstable();
                let unreads: Vec<String> = ids
                    .iter()
                    .map(|c| format!("#{c} ({})", staged[c].unread()))
                    .collect();
                let _ = write!(line, "\nUnreads: {}", unreads.join(", "));
            }
            line
        }

        /// `/subscribe-voice`: join the caller's voice channel via songbird
        /// (songbird owns the DAVE/MLS handshake), wire the [`RecordingSink`] +
        /// per-speaker intake pipeline, populate the `voice_runtime` map, and
        /// register the voice subscription (Python `bot.py::subscribe_voice`).
        ///
        /// [`RecordingSink`]: crate::voice::recording_sink::RecordingSink
        #[cfg(feature = "discord-voice")]
        async fn dispatch_subscribe_voice(&self, ctx: &Context, ack: &SlashCtx) {
            use crate::bot::voice_intake::{join_voice, start_voice_intake, stop_voice_intake};
            use std::collections::btree_map::Entry;

            defer_interaction(ack).await;
            let Some(gid) = ack.command.guild_id else {
                reply(ack, "You must be in a guild voice channel.").await;
                return;
            };
            let guild_id_u64 = gid.get();
            let guild_id = guild_id_u64 as i64;
            let user_id = ack.command.user.id;
            // Resolve the caller's current voice channel from the gateway cache
            // (GUILD_VOICE_STATES). The dashmap ref is dropped before any await.
            let resolved: Option<(u64, Option<String>)> = ctx.cache.guild(gid).and_then(|guild| {
                let cid = guild
                    .voice_states
                    .get(&user_id)
                    .and_then(|vs| vs.channel_id)?;
                let name = guild.channels.get(&cid).map(|ch| ch.name.clone());
                Some((cid.get(), name))
            });
            let Some((channel_id_u64, channel_name)) = resolved else {
                reply(ack, "You must be in a voice channel.").await;
                return;
            };
            let channel_id = channel_id_u64 as i64;
            let display = channel_name.unwrap_or_else(|| format!("#{channel_id}"));

            // Idempotent: a second subscribe for the same live channel re-affirms.
            if self
                .slash
                .handle
                .voice_runtime
                .lock()
                .expect("voice_runtime mutex poisoned")
                .contains_key(&channel_id)
            {
                self.events.on_subscribe_voice(channel_id, Some(guild_id));
                reply(ack, &format!("Already listening in {display}.")).await;
                return;
            }

            let Some(manager) = songbird::get(ctx).await else {
                reply(ack, "Voice runtime unavailable.").await;
                return;
            };
            let (voice_client, audio_rx) =
                match join_voice(&manager, guild_id_u64, channel_id_u64).await {
                    Ok(pair) => pair,
                    Err(err) => {
                        tracing::warn!("voice connect failed: {err}");
                        reply(ack, "Could not join voice.").await;
                        return;
                    }
                };

            let template = self.slash.transcriber_template.clone();
            let has_transcriber = template.is_some();
            let runtime = start_voice_intake(
                voice_client,
                template,
                self.slash.local_turn_detector.clone(),
                self.slash.bus.clone(),
                self.slash.familiar_id.clone(),
                channel_id,
                audio_rx,
            );
            // Re-check under the lock before inserting: `join_voice` awaited
            // above, so a concurrent `/subscribe-voice` for this same channel
            // could have won the race and already inserted. Keep the
            // first-joined runtime (Python's idempotent `subscribe_voice`
            // intent, bot.py:262) and tear down the one we just built rather
            // than overwriting — a bare `insert` would drop the live runtime
            // without `stop_voice_intake`, orphaning its intake tasks (which
            // would keep publishing transcripts) and leaking the songbird call.
            let displaced = match self
                .slash
                .handle
                .voice_runtime
                .lock()
                .expect("voice_runtime mutex poisoned")
                .entry(channel_id)
            {
                Entry::Occupied(_) => Some(runtime),
                Entry::Vacant(slot) => {
                    slot.insert(runtime);
                    None
                }
            };
            if let Some(orphan) = displaced {
                // Loser of the race: stop only our intake tasks. Do NOT leave
                // the songbird call — it is shared per-guild with the winner.
                stop_voice_intake(orphan).await;
                self.events.on_subscribe_voice(channel_id, Some(guild_id));
                reply(ack, &format!("Already listening in {display}.")).await;
                return;
            }
            self.events.on_subscribe_voice(channel_id, Some(guild_id));

            let suffix = if has_transcriber {
                ""
            } else {
                " (playback only — no transcriber)"
            };
            reply(ack, &format!("Joined {display}.{suffix}")).await;
        }

        /// `/unsubscribe-voice`: tear down the intake pipeline (cancel the
        /// router / source / per-speaker pumps + fan-ins, close the
        /// transcribers), leave the songbird call, and drop the voice
        /// subscription (Python `bot.py::unsubscribe_voice`).
        #[cfg(feature = "discord-voice")]
        async fn dispatch_unsubscribe_voice(&self, ctx: &Context, ack: &SlashCtx) {
            use crate::bot::voice_intake::stop_voice_intake;

            defer_interaction(ack).await;
            let Some(gid) = ack.command.guild_id else {
                reply(ack, "Not in a guild.").await;
                return;
            };
            let guild_id = gid.get() as i64;
            let Some(channel_id) = self.events.on_unsubscribe_voice(guild_id) else {
                reply(ack, "Not in a voice channel here.").await;
                return;
            };

            let runtime = self
                .slash
                .handle
                .voice_runtime
                .lock()
                .expect("voice_runtime mutex poisoned")
                .remove(&channel_id);
            if let Some(runtime) = runtime {
                stop_voice_intake(runtime).await;
            }
            if let Some(manager) = songbird::get(ctx).await {
                let _ = manager.remove(gid).await;
            }
            reply(ack, "Left voice channel.").await;
        }
    }

    #[async_trait]
    impl EventHandler for Handler {
        async fn ready(&self, ctx: Context, ready: Ready) {
            *self.presence.ctx.lock().expect("presence ctx mutex") = Some(ctx.clone());
            let mut channel_names = Vec::new();
            let mut guild_names = Vec::new();
            for guild in &ready.guilds {
                if let Some(channels) = ctx.cache.guild(guild.id).map(|g| g.channels.clone()) {
                    for (cid, ch) in channels {
                        channel_names.push((cid.get() as i64, ch.name.clone()));
                        if let Some(name) = ctx.cache.guild(guild.id).map(|g| g.name.clone()) {
                            guild_names.push((cid.get() as i64, name));
                        }
                    }
                }
            }
            // Register the slash commands (best-effort).
            #[cfg_attr(not(feature = "discord-voice"), allow(unused_mut))]
            let mut commands: Vec<(&str, &str)> = vec![
                (
                    "subscribe-text",
                    "Listen for text messages in this channel.",
                ),
                (
                    "unsubscribe-text",
                    "Stop listening for text messages in this channel.",
                ),
                ("diagnostics", "Show span timings (last p50/p95 per span)."),
            ];
            #[cfg(feature = "discord-voice")]
            commands.extend([
                ("subscribe-voice", "Join your voice channel and listen."),
                (
                    "unsubscribe-voice",
                    "Leave the voice channel in this guild.",
                ),
            ]);
            for (cmd, desc) in commands {
                let _ = Command::create_global_command(
                    &ctx.http,
                    CreateCommand::new(cmd).description(desc),
                )
                .await;
            }
            self.events
                .on_ready(ReadyInfo {
                    user_id: ready.user.id.get() as i64,
                    channel_names,
                    guild_names,
                })
                .await;
        }

        async fn message(&self, ctx: Context, msg: Message) {
            self.events.on_message(message_view(&ctx, &msg)).await;
        }

        async fn message_update(
            &self,
            _ctx: Context,
            old: Option<Message>,
            new: Option<Message>,
            event: MessageUpdateEvent,
        ) {
            // Prefer serenity's cache-merged `after`. When the pre-edit message
            // has fallen out of cache (`new == None` — e.g. a late URL-unfurl
            // edit on an older message after a restart, the exact B-RX16 case
            // on_message_edit exists to merge), fall back to the raw
            // MESSAGE_UPDATE payload, which still carries the embeds. The gateway
            // payload is partial, so bail unless it also carries the author and
            // content: without `author` the own/bot-author guards can't run, and
            // without `content` an embeds-only rewrite would clobber the stored
            // turn's original text.
            let (author_id, author_is_bot, channel_id, message_id, content, after_embeds) =
                if let Some(after) = new.as_ref() {
                    (
                        after.author.id.get() as i64,
                        after.author.bot,
                        after.channel_id.get() as i64,
                        after.id.get() as i64,
                        after.content.clone(),
                        after.embeds.iter().map(embed_view).collect::<Vec<_>>(),
                    )
                } else if let (Some(author), Some(content), Some(embeds)) = (
                    event.author.as_ref(),
                    event.content.as_ref(),
                    event.embeds.as_ref(),
                ) {
                    (
                        author.id.get() as i64,
                        author.bot,
                        event.channel_id.get() as i64,
                        event.id.get() as i64,
                        content.clone(),
                        embeds.iter().map(embed_view).collect::<Vec<_>>(),
                    )
                } else {
                    return;
                };
            let before_embeds = old
                .as_ref()
                .map(|m| m.embeds.iter().map(embed_view).collect())
                .unwrap_or_default();
            self.events.on_message_edit(&MessageEditView {
                author_id,
                author_is_bot,
                channel_id,
                message_id,
                content,
                before_embeds,
                after_embeds,
            });
        }

        async fn reaction_add(&self, _ctx: Context, reaction: Reaction) {
            self.events
                .on_raw_reaction_add(&reaction_payload(&reaction))
                .await;
        }

        async fn reaction_remove(&self, _ctx: Context, reaction: Reaction) {
            self.events
                .on_raw_reaction_remove(&reaction_payload(&reaction));
        }

        async fn reaction_remove_all(
            &self,
            _ctx: Context,
            channel_id: ChannelId,
            message_id: MessageId,
        ) {
            self.events
                .on_raw_reaction_clear(&ReactionClearPayloadView {
                    channel_id: channel_id.get() as i64,
                    message_id: message_id.get() as i64,
                    emoji: None,
                });
        }

        async fn reaction_remove_emoji(&self, _ctx: Context, reaction: Reaction) {
            self.events
                .on_raw_reaction_clear_emoji(&ReactionClearPayloadView {
                    channel_id: reaction.channel_id.get() as i64,
                    message_id: reaction.message_id.get() as i64,
                    emoji: Some(emoji_view(&reaction.emoji)),
                });
        }

        async fn typing_start(&self, ctx: Context, event: TypingStartEvent) {
            // Discord attaches the full `Member` (carrying the user's `bot`
            // flag) on guild typing events; prefer it so the bot flag is read
            // reliably (Python `on_typing` reads `user.bot` directly). Fall back
            // to the user cache for DM typing, then to non-bot when uncached.
            let is_bot = event.member.as_ref().map_or_else(
                || ctx.cache.user(event.user_id).is_some_and(|u| u.bot),
                |m| m.user.bot,
            );
            self.events.on_typing(TypingEventView {
                channel_id: event.channel_id.get() as i64,
                user_id: event.user_id.get() as i64,
                is_bot,
            });
        }

        async fn voice_state_update(
            &self,
            _ctx: Context,
            _old: Option<VoiceState>,
            new: VoiceState,
        ) {
            let Some(guild_id) = new.guild_id else {
                return;
            };
            let author = new.member.as_ref().map_or_else(
                || Author::new("discord", new.user_id.get().to_string(), None, None),
                |m| author_from_user(&m.user),
            );
            self.events.on_voice_state_update(VoiceStateUpdateView {
                member_id: new.user_id.get() as i64,
                guild_id: guild_id.get() as i64,
                after_channel_id: new.channel_id.map(|c| c.get() as i64),
                author,
            });
        }

        async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
            if let Interaction::Command(command) = interaction {
                self.dispatch_command(&ctx, command).await;
            }
        }
    }

    // -- construction ----------------------------------------------------

    /// The dependencies [`create_bot`] wires into the gateway shell.
    pub struct CreateBotDeps {
        /// The Discord bot token.
        pub token: String,
        /// The familiar id.
        pub familiar_id: String,
        /// The shared bot-user-id cell (set on ready).
        pub bot_user_id: Arc<Mutex<Option<i64>>>,
        /// The subscription registry.
        pub subscriptions: Arc<Mutex<crate::subscriptions::SubscriptionRegistry>>,
        /// The DM allowlist.
        pub dm_allowlist: Vec<i64>,
        /// The synchronous store write surface (reactions / edits).
        pub store: Arc<dyn BotStore>,
        /// The async history store (diagnostics unread counts).
        pub history_store: Arc<AsyncHistoryStore>,
        /// The event bus the text source (and voice source) publish onto.
        pub bus: Arc<dyn crate::bus::protocols::EventBus>,
        /// The attentional focus controller.
        pub focus_manager: Option<Arc<FocusManager>>,
        /// The turn router the typing-interrupt policy cancels active turns on.
        pub router: Arc<crate::bus::router::TurnRouter>,
        /// `[discord.text]` config driving the typing-interrupt policy (the
        /// `respond_to_typing` switch + backoff knobs).
        pub discord_text: crate::config::DiscordTextConfig,
        /// The transcriber prototype `/subscribe-voice` clones per speaker
        /// (`None` degrades a join to playback-only). Cloned from the familiar's
        /// own template so the familiar's teardown copy is unaffected.
        pub transcriber_template: Option<Arc<Mutex<Box<dyn crate::stt::Transcriber>>>>,
        /// Optional local turn detector for per-speaker endpointing.
        pub local_turn_detector: Option<Arc<crate::voice::turn_detection::LocalTurnDetector>>,
    }

    /// Build the gateway client + [`BotHandle`], mirroring Python `create_bot`.
    ///
    /// Pure construction — no login. The caller drives `client.start()`.
    ///
    /// # Errors
    /// Propagates a `serenity` client-build failure.
    pub async fn create_bot(deps: CreateBotDeps) -> anyhow::Result<(Arc<BotHandle>, Client)> {
        let http = Arc::new(serenity::all::Http::new(&deps.token));
        let presence = Arc::new(SerenityPresence {
            ctx: Mutex::new(None),
        });
        let send_text = Arc::new(SerenitySendText { http: http.clone() });
        let trigger_typing = Arc::new(SerenityTyping { http: http.clone() });

        let mut handle = BotHandle::new(send_text, presence.clone());
        handle.trigger_typing = Some(trigger_typing);
        handle.focus_manager = deps.focus_manager.clone();
        // Typing-interrupt policy: cancel the active turn when a real user types,
        // back off when another bot types (Python `bot.py` constructs it here and
        // stores it on the handle; the text responder + `on_typing` consume it).
        let typing_subs = deps.subscriptions.clone();
        let is_subscribed: crate::typing_interrupt::IsSubscribed = Arc::new(move |ch: i64| {
            u64::try_from(ch)
                .ok()
                .and_then(|cid| {
                    typing_subs
                        .lock()
                        .expect("subscriptions mutex poisoned")
                        .get(cid, SubscriptionKind::Text)
                })
                .is_some()
        });
        let typing_bot_id = deps.bot_user_id.clone();
        let bot_user_id_provider: crate::typing_interrupt::BotUserIdProvider =
            Arc::new(move || *typing_bot_id.lock().expect("bot_user_id mutex poisoned"));
        handle.typing_interrupt = Some(Arc::new(
            crate::typing_interrupt::TypingInterruptHandler::new(
                deps.discord_text.clone(),
                deps.router.clone(),
                is_subscribed,
                bot_user_id_provider,
            ),
        ));
        let handle = Arc::new(handle);
        // Cache-only resolver (no I/O on the audio path, B-VM29); the gateway
        // cache warm-up + background fetch land the side cache.
        let weak = Arc::downgrade(&handle);
        let resolve: ResolveMember = Arc::new(move |_channel_id, user_id| {
            weak.upgrade().and_then(|h| h.voice_member_cached(user_id))
        });
        *handle.resolve_member.lock().expect("resolve_member mutex") = Some(resolve);

        let text_source = Arc::new(DiscordTextSource::new(
            deps.bus.clone(),
            deps.familiar_id.clone(),
        ));
        let events = Arc::new(BotEvents::new(
            deps.familiar_id.clone(),
            deps.bot_user_id.clone(),
            deps.subscriptions.clone(),
            deps.dm_allowlist.clone(),
            deps.store.clone(),
            text_source,
            handle.clone(),
        ));
        let slash = Arc::new(SlashContext {
            familiar_id: deps.familiar_id.clone(),
            subscriptions: deps.subscriptions.clone(),
            history_store: deps.history_store.clone(),
            focus_manager: deps.focus_manager.clone(),
            handle: handle.clone(),
            #[cfg(feature = "discord-voice")]
            bus: deps.bus.clone(),
            #[cfg(feature = "discord-voice")]
            transcriber_template: deps.transcriber_template.clone(),
            #[cfg(feature = "discord-voice")]
            local_turn_detector: deps.local_turn_detector.clone(),
        });
        let handler = Handler {
            events,
            slash,
            presence,
        };
        let intents = GatewayIntents::non_privileged()
            | GatewayIntents::MESSAGE_CONTENT
            | GatewayIntents::GUILD_VOICE_STATES
            | GatewayIntents::GUILD_MESSAGE_TYPING;
        let builder = Client::builder(&deps.token, intents).event_handler(handler);
        #[cfg(feature = "discord-voice")]
        let builder = {
            use songbird::serenity::SerenityInit as _;
            builder.register_songbird()
        };
        let client = builder.await?;
        Ok((handle, client))
    }
}

// ---------------------------------------------------------------------------
// Voice intake pipeline + songbird join (cfg(feature = "discord-voice"))
// ---------------------------------------------------------------------------
//
// The router → per-user-pump → transcriber-clone → fan-in → VoiceSource topology
// (spec 10 B-VI34-41) is serenity-free: it drains the `(user_id, mono_pcm)`
// channel the landed [`RecordingSink`](crate::voice::recording_sink) fills.
// Songbird owns only the voice connection + DAVE/MLS; the `VoiceTick` receiver
// converts songbird's decoded per-SSRC audio into a landed `VoiceTick` and hands
// it to `RecordingSink::on_tick`, which fans it onto that channel.
#[cfg(feature = "discord-voice")]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::significant_drop_tightening,
    reason = "snowflake/sample casts are lossless in practice; intake guards are short-lived"
)]
pub mod voice_intake {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use tokio::sync::Mutex as AsyncMutex;
    use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
    use tokio::task::JoinHandle;
    use tokio::time::{Duration, Instant, timeout};

    use crate::bus::protocols::EventBus;
    use crate::log_style as ls;
    use crate::sources::voice::VoiceSource;
    use crate::stt::{Transcriber, TranscriptionResult};
    use crate::tts_player::VoiceClientLike;
    use crate::voice::recording_sink::{AudioChunk, RecordingSink, SsrcResolver, VoiceTick};
    use crate::voice::turn_detection::{LocalTurnDetector, UtteranceEndpointer};

    /// Idle-finalize gap (s) before a forced `finalize` when no local endpointer
    /// owns endpointing (Python `stt.deepgram.DEFAULT_IDLE_FINALIZE_S`).
    pub const DEFAULT_IDLE_FINALIZE_S: f64 = 0.5;

    /// A per-user transcriber clone, shared between its pump and teardown.
    type SharedTranscriber = Arc<AsyncMutex<Box<dyn Transcriber>>>;
    /// The template, cloned per user (Python `familiar.transcriber`).
    type Template = Arc<Mutex<Box<dyn Transcriber>>>;

    /// Per-voice-channel intake state (Python `VoiceRuntime`), all per-user maps
    /// keyed by Discord user id.
    #[derive(Default)]
    struct IntakeState {
        transcribers: HashMap<u64, SharedTranscriber>,
        endpointers: HashMap<u64, Arc<AsyncMutex<UtteranceEndpointer>>>,
        fanin_tasks: HashMap<u64, JoinHandle<()>>,
        user_pump_tasks: HashMap<u64, JoinHandle<()>>,
        user_audio_tx: HashMap<u64, UnboundedSender<Vec<u8>>>,
        last_audio_time: HashMap<u64, Instant>,
    }

    /// Live intake pipeline for one voice channel.
    pub struct VoiceRuntime {
        /// The voice channel id.
        pub channel_id: i64,
        /// The live voice client, read by [`DiscordVoicePlayer`] for TTS
        /// playback (populated by [`join_voice`]).
        ///
        /// [`DiscordVoicePlayer`]: crate::tts_player::DiscordVoicePlayer
        pub voice_client: Arc<dyn VoiceClientLike>,
        // The intake tasks are `None` on a playback-only join (no transcriber
        // configured) — the bot still joined so TTS can play out.
        router_task: Option<JoinHandle<()>>,
        source_task: Option<JoinHandle<()>>,
        watchdog_task: Option<JoinHandle<()>>,
        state: Arc<Mutex<IntakeState>>,
    }

    /// Shared context threaded into every pump / fan-in task.
    struct IntakeCtx {
        template: Template,
        detector: Option<Arc<LocalTurnDetector>>,
        idle_finalize_s: f64,
        source: Arc<VoiceSource>,
        result_tx: UnboundedSender<TranscriptionResult>,
        state: Arc<Mutex<IntakeState>>,
    }

    impl IntakeCtx {
        async fn ensure_transcriber(&self, user_id: u64) -> Option<SharedTranscriber> {
            if let Some(existing) = self
                .state
                .lock()
                .expect("intake state")
                .transcribers
                .get(&user_id)
                .cloned()
            {
                return Some(existing);
            }
            self.state
                .lock()
                .expect("intake state")
                .last_audio_time
                .insert(user_id, Instant::now());
            let mut clone = { self.template.lock().expect("template").clone_transcriber() };
            if self.detector.is_some() {
                clone.set_endpointing_ms(10);
            }
            let (per_user_tx, per_user_rx) = unbounded_channel();
            if clone.start(per_user_tx).await.is_err() {
                return None;
            }
            let shared: SharedTranscriber = Arc::new(AsyncMutex::new(clone));
            self.state
                .lock()
                .expect("intake state")
                .transcribers
                .insert(user_id, shared.clone());

            if let Some(detector) = &self.detector {
                let source = self.source.clone();
                let transcribers = self.state.clone();
                let cb = Box::new(move |_audio: Vec<u8>| {
                    let source = source.clone();
                    let transcribers = transcribers.clone();
                    Box::pin(async move {
                        source.record_vad_end(user_id as i64, None);
                        let target = transcribers
                            .lock()
                            .expect("intake state")
                            .transcribers
                            .get(&user_id)
                            .cloned();
                        if let Some(target) = target {
                            target.lock().await.finalize().await;
                        }
                    })
                        as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
                });
                if let Ok(ep) = detector.make_endpointer(cb) {
                    self.state
                        .lock()
                        .expect("intake state")
                        .endpointers
                        .insert(user_id, Arc::new(AsyncMutex::new(ep)));
                }
            }

            let fanin = tokio::spawn(fanin(user_id, per_user_rx, self.result_tx.clone()));
            self.state
                .lock()
                .expect("intake state")
                .fanin_tasks
                .insert(user_id, fanin);
            tracing::info!(
                "{} {} {}",
                ls::tag("\u{1f399}\u{fe0f}  Voice", ls::G),
                ls::kv_styled("user", &user_id.to_string(), ls::W, ls::LC),
                ls::kv_styled("transcriber", "opened", ls::W, ls::LG),
            );
            Some(shared)
        }
    }

    /// Tag every result with `user_id` and forward it to the shared queue.
    async fn fanin(
        user_id: u64,
        mut rx: UnboundedReceiver<TranscriptionResult>,
        out: UnboundedSender<TranscriptionResult>,
    ) {
        while let Some(mut result) = rx.recv().await {
            result.user_id = Some(user_id as i64);
            if out.send(result).is_err() {
                break;
            }
        }
    }

    /// Drain one user's audio: `send_audio` + endpointer feed, with the idle-flush
    /// fallback (Discord halts RTP during silence, B-VI36).
    async fn user_pump(ctx: Arc<IntakeCtx>, user_id: u64, mut rx: UnboundedReceiver<Vec<u8>>) {
        let Some(transcriber) = ctx.ensure_transcriber(user_id).await else {
            let mut st = ctx.state.lock().expect("intake state");
            st.user_pump_tasks.remove(&user_id);
            st.user_audio_tx.remove(&user_id);
            return;
        };
        let endpointer = ctx
            .state
            .lock()
            .expect("intake state")
            .endpointers
            .get(&user_id)
            .cloned();
        let idle_flush_s = if endpointer.is_some() {
            ctx.detector
                .as_ref()
                .map_or(ctx.idle_finalize_s, |d| d.idle_fallback_s)
        } else {
            ctx.idle_finalize_s
        };
        let mut dirty = false;
        loop {
            let pcm = if dirty {
                match timeout(Duration::from_secs_f64(idle_flush_s), rx.recv()).await {
                    Ok(Some(pcm)) => pcm,
                    Ok(None) => break,
                    Err(_) => {
                        if let Some(ep) = &endpointer {
                            ep.lock().await.force_complete_if_pending().await;
                        } else {
                            transcriber.lock().await.finalize().await;
                        }
                        dirty = false;
                        continue;
                    }
                }
            } else {
                match rx.recv().await {
                    Some(pcm) => pcm,
                    None => break,
                }
            };
            let _ = transcriber.lock().await.send_audio(&pcm).await;
            dirty = true;
            if let Some(ep) = &endpointer {
                ep.lock().await.feed_audio(&pcm).await;
            }
        }
    }

    /// Demux the sink channel into per-user pumps (no per-chunk awaits, B-VI34).
    async fn route_audio(ctx: Arc<IntakeCtx>, mut audio_rx: UnboundedReceiver<AudioChunk>) {
        while let Some((user_id, pcm)) = audio_rx.recv().await {
            let tx = {
                let mut st = ctx.state.lock().expect("intake state");
                st.last_audio_time.insert(user_id, Instant::now());
                let existing = st.user_audio_tx.get(&user_id).cloned();
                existing.unwrap_or_else(|| {
                    let (tx, rx) = unbounded_channel();
                    st.user_audio_tx.insert(user_id, tx.clone());
                    let pump = tokio::spawn(user_pump(ctx.clone(), user_id, rx));
                    st.user_pump_tasks.insert(user_id, pump);
                    tx
                })
            };
            let _ = tx.send(pcm);
        }
    }

    /// Close idle streams before Deepgram's server-side silence close (B-VI39).
    async fn idle_watchdog(ctx: Arc<IntakeCtx>, idle_close_s: f64) {
        let interval = Duration::from_secs_f64((idle_close_s / 4.0).max(0.01));
        loop {
            tokio::time::sleep(interval).await;
            let now = Instant::now();
            let stale: Vec<u64> = {
                let st = ctx.state.lock().expect("intake state");
                st.last_audio_time
                    .iter()
                    .filter(|(_, last)| now.duration_since(**last).as_secs_f64() > idle_close_s)
                    .map(|(&uid, _)| uid)
                    .collect()
            };
            for uid in stale {
                close_user_stream(&ctx.state, uid).await;
            }
        }
    }

    /// Tear down one user's stream (producers before the transcriber, B-VI40).
    async fn close_user_stream(state: &Arc<Mutex<IntakeState>>, user_id: u64) {
        let (transcriber, fanin, pump) = {
            let mut st = state.lock().expect("intake state");
            let t = st.transcribers.remove(&user_id);
            let f = st.fanin_tasks.remove(&user_id);
            let p = st.user_pump_tasks.remove(&user_id);
            st.user_audio_tx.remove(&user_id);
            st.endpointers.remove(&user_id);
            st.last_audio_time.remove(&user_id);
            (t, f, p)
        };
        if let Some(pump) = pump {
            pump.abort();
        }
        if let Some(fanin) = fanin {
            fanin.abort();
        }
        if let Some(transcriber) = transcriber {
            transcriber.lock().await.stop().await;
        }
    }

    /// Bring up the intake pipeline draining `audio_rx` for `channel_id`.
    ///
    /// With no transcriber configured the returned runtime is **playback-only**
    /// (B-VI31): it carries the `voice_client` (so TTS still plays out) but
    /// spawns no intake tasks. The `template` is cloned per user on first audio.
    pub fn start_voice_intake(
        voice_client: Arc<dyn VoiceClientLike>,
        template: Option<Template>,
        detector: Option<Arc<LocalTurnDetector>>,
        bus: Arc<dyn EventBus>,
        familiar_id: String,
        channel_id: i64,
        audio_rx: UnboundedReceiver<AudioChunk>,
    ) -> VoiceRuntime {
        let state = Arc::new(Mutex::new(IntakeState::default()));
        let Some(template) = template else {
            // Playback-only join: keep the voice client, spawn nothing.
            drop(audio_rx);
            return VoiceRuntime {
                channel_id,
                voice_client,
                router_task: None,
                source_task: None,
                watchdog_task: None,
                state,
            };
        };
        let idle_close_s = { template.lock().expect("template").idle_close_s() };
        let (result_tx, result_rx) = unbounded_channel();
        let source = Arc::new(VoiceSource::new(bus, familiar_id, channel_id, result_rx));
        let ctx = Arc::new(IntakeCtx {
            template,
            detector,
            idle_finalize_s: DEFAULT_IDLE_FINALIZE_S,
            source: source.clone(),
            result_tx,
            state: state.clone(),
        });
        let router_task = tokio::spawn(route_audio(ctx.clone(), audio_rx));
        let source_task = tokio::spawn(async move { source.run().await });
        let watchdog_task =
            (idle_close_s > 0.0).then(|| tokio::spawn(idle_watchdog(ctx.clone(), idle_close_s)));
        tracing::info!(
            "{} {} {}",
            ls::tag("\u{1f399}\u{fe0f}  Voice", ls::G),
            ls::kv_styled("intake", "started", ls::W, ls::LG),
            ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LC),
        );
        VoiceRuntime {
            channel_id,
            voice_client,
            router_task: Some(router_task),
            source_task: Some(source_task),
            watchdog_task,
            state,
        }
    }

    /// Tear down the intake pipeline; per-user WS closes run in parallel (B-VI41).
    pub async fn stop_voice_intake(rt: VoiceRuntime) {
        if let Some(t) = &rt.router_task {
            t.abort();
        }
        if let Some(t) = &rt.source_task {
            t.abort();
        }
        if let Some(w) = &rt.watchdog_task {
            w.abort();
        }
        let (pumps, fanins, transcribers) = {
            let mut st = rt.state.lock().expect("intake state");
            let pumps: Vec<JoinHandle<()>> = st.user_pump_tasks.drain().map(|(_, t)| t).collect();
            let fanins: Vec<JoinHandle<()>> = st.fanin_tasks.drain().map(|(_, t)| t).collect();
            let transcribers: Vec<SharedTranscriber> =
                st.transcribers.drain().map(|(_, t)| t).collect();
            st.user_audio_tx.clear();
            st.endpointers.clear();
            st.last_audio_time.clear();
            (pumps, fanins, transcribers)
        };
        for p in pumps {
            p.abort();
        }
        for f in fanins {
            f.abort();
        }
        // Parallel WS closes — sequential would multiply unsubscribe latency.
        let closes = transcribers
            .into_iter()
            .map(|t| async move { t.lock().await.stop().await });
        futures::future::join_all(closes).await;
        tracing::info!(
            "{} {} {}",
            ls::tag("\u{1f399}\u{fe0f}  Voice", ls::Y),
            ls::kv_styled("intake", "stopped", ls::W, ls::LY),
            ls::kv_styled("channel", &rt.channel_id.to_string(), ls::W, ls::LC),
        );
    }

    // -- songbird join + VoiceTick → RecordingSink wiring ----------------

    /// SSRC → Discord user id map, fed by songbird speaking-state events.
    #[derive(Default)]
    pub struct SsrcMap {
        inner: Mutex<HashMap<u32, u64>>,
    }

    impl SsrcMap {
        /// Record an SSRC → user binding.
        pub fn insert(&self, ssrc: u32, user_id: u64) {
            self.inner.lock().expect("ssrc map").insert(ssrc, user_id);
        }
    }

    impl SsrcResolver for SsrcMap {
        fn user_id(&self, ssrc: u32) -> Option<u64> {
            self.inner.lock().expect("ssrc map").get(&ssrc).copied()
        }
    }

    /// Songbird receiver: converts decoded per-SSRC ticks into a landed
    /// [`VoiceTick`] and fans it onto the sink's `(user_id, mono)` channel, and
    /// tracks SSRC ownership from speaking-state updates.
    pub struct TickReceiver {
        sink: Arc<RecordingSink>,
        ssrc_map: Arc<SsrcMap>,
    }

    impl TickReceiver {
        /// Build a receiver over the sink + SSRC map.
        #[must_use]
        pub const fn new(sink: Arc<RecordingSink>, ssrc_map: Arc<SsrcMap>) -> Self {
            Self { sink, ssrc_map }
        }
    }

    #[async_trait::async_trait]
    impl songbird::EventHandler for TickReceiver {
        async fn act(&self, ctx: &songbird::EventContext<'_>) -> Option<songbird::Event> {
            match ctx {
                songbird::EventContext::VoiceTick(tick) => {
                    let mut vt = VoiceTick::default();
                    for (ssrc, data) in &tick.speaking {
                        if let Some(samples) = &data.decoded_voice {
                            let mut bytes = Vec::with_capacity(samples.len() * 2);
                            for s in samples {
                                bytes.extend_from_slice(&s.to_le_bytes());
                            }
                            vt.speaking.insert(*ssrc, bytes);
                        }
                    }
                    vt.silent = tick.silent.iter().copied().collect();
                    self.sink.on_tick(&vt, self.ssrc_map.as_ref());
                }
                songbird::EventContext::SpeakingStateUpdate(speaking) => {
                    if let Some(user_id) = speaking.user_id {
                        self.ssrc_map.insert(speaking.ssrc, user_id.0);
                    }
                }
                _ => {}
            }
            None
        }
    }

    /// Join `channel_id` in `guild_id` via songbird (songbird owns the DAVE/MLS
    /// handshake) and wire the [`RecordingSink`] to its `VoiceTick` stream.
    ///
    /// Returns the live voice client (for TTS playback) and the sink's audio
    /// channel receiver — the caller passes both to [`start_voice_intake`].
    ///
    /// # Errors
    /// Propagates a songbird join failure.
    pub async fn join_voice(
        manager: &songbird::Songbird,
        guild_id: u64,
        channel_id: u64,
    ) -> Result<(Arc<dyn VoiceClientLike>, UnboundedReceiver<AudioChunk>), songbird::error::JoinError>
    {
        let gid = songbird::id::GuildId(
            std::num::NonZeroU64::new(guild_id).unwrap_or(std::num::NonZeroU64::MIN),
        );
        let cid = songbird::id::ChannelId(
            std::num::NonZeroU64::new(channel_id).unwrap_or(std::num::NonZeroU64::MIN),
        );
        let call_lock = manager.join(gid, cid).await?;
        let (audio_tx, audio_rx) = unbounded_channel();
        let sink = Arc::new(RecordingSink::new(audio_tx));
        let ssrc_map = Arc::new(SsrcMap::default());
        {
            let mut call = call_lock.lock().await;
            call.add_global_event(
                songbird::CoreEvent::VoiceTick.into(),
                TickReceiver::new(sink.clone(), ssrc_map.clone()),
            );
            call.add_global_event(
                songbird::CoreEvent::SpeakingStateUpdate.into(),
                TickReceiver::new(sink, ssrc_map),
            );
        }
        let voice_client: Arc<dyn VoiceClientLike> = Arc::new(SongbirdVoiceClient::new(call_lock));
        Ok((voice_client, audio_rx))
    }

    // -- songbird playback bridge (VoiceClientLike over a `Call`) -----------

    /// [`VoiceClientLike`] adapter over a songbird [`Call`](songbird::Call).
    ///
    /// Bridges the synchronous 4-method player surface (DESIGN §4.8) onto
    /// songbird's async call + [`TrackHandle`](songbird::tracks::TrackHandle):
    /// [`DiscordVoicePlayer`](crate::tts_player::DiscordVoicePlayer) hands us
    /// Discord-format stereo s16le @ 48 kHz PCM, which we convert to the
    /// interleaved `f32` stream songbird's [`RawAdapter`](songbird::input::RawAdapter)
    /// consumes. `is_playing` is tracked by an atomic flipped false by a
    /// [`TrackEvent::End`](songbird::TrackEvent::End) handler.
    struct SongbirdVoiceClient {
        call: Arc<AsyncMutex<songbird::Call>>,
        track: Mutex<Option<songbird::tracks::TrackHandle>>,
        playing: Arc<std::sync::atomic::AtomicBool>,
    }

    impl SongbirdVoiceClient {
        fn new(call: Arc<AsyncMutex<songbird::Call>>) -> Self {
            Self {
                call,
                track: Mutex::new(None),
                playing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }
        }
    }

    /// Flip the shared `is_playing` flag false when a track ends or is stopped.
    struct TrackEndFlag(Arc<std::sync::atomic::AtomicBool>);

    #[async_trait::async_trait]
    impl songbird::EventHandler for TrackEndFlag {
        async fn act(&self, _ctx: &songbird::EventContext<'_>) -> Option<songbird::Event> {
            self.0.store(false, std::sync::atomic::Ordering::SeqCst);
            None
        }
    }

    impl VoiceClientLike for SongbirdVoiceClient {
        fn is_connected(&self) -> bool {
            // Best-effort: if the call lock is momentarily held (e.g. mid-play),
            // assume still connected rather than reporting a spurious drop.
            self.call
                .try_lock()
                .map_or(true, |call| call.current_connection().is_some())
        }

        fn is_playing(&self) -> bool {
            self.playing.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn play(
            &self,
            source: crate::tts_player::AudioSource,
        ) -> Result<(), crate::tts_player::PlayError> {
            use std::sync::atomic::Ordering;

            if self.playing.swap(true, Ordering::SeqCst) {
                return Err(crate::tts_player::PlayError::AlreadyPlaying);
            }
            let input: songbird::input::Input = match source {
                crate::tts_player::AudioSource::Buffered(bytes) => {
                    let f32_bytes = s16le_stereo_to_f32_bytes(&bytes);
                    songbird::input::RawAdapter::new(std::io::Cursor::new(f32_bytes), 48_000, 2)
                        .into()
                }
                crate::tts_player::AudioSource::Streaming(src) => {
                    songbird::input::RawAdapter::new(StreamingF32Reader::new(src), 48_000, 2).into()
                }
            };
            // Briefly lock the async call to hand songbird the track. The lock is
            // uncontended at play time (the driver runs on its own task), so this
            // resolves immediately; `block_in_place` lets peers progress if not.
            let handle = tokio::task::block_in_place(|| {
                futures::executor::block_on(async { self.call.lock().await.play_input(input) })
            });
            let _ = handle.add_event(
                songbird::Event::Track(songbird::TrackEvent::End),
                TrackEndFlag(self.playing.clone()),
            );
            *self.track.lock().expect("track handle mutex poisoned") = Some(handle);
            Ok(())
        }

        fn stop(&self) {
            let track = self
                .track
                .lock()
                .expect("track handle mutex poisoned")
                .take();
            if let Some(handle) = track {
                let _ = handle.stop();
            }
            self.playing
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Convert interleaved s16le PCM bytes to interleaved `f32` LE PCM bytes
    /// (songbird's `RawAdapter` reads `f32`; the player emits s16le). A trailing
    /// odd byte (never produced by our 2-byte-aligned pipeline) is dropped.
    fn s16le_stereo_to_f32_bytes(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bytes.len() * 2);
        for frame in bytes.chunks_exact(2) {
            let sample = i16::from_le_bytes([frame[0], frame[1]]);
            let f = f32::from(sample) / 32768.0;
            out.extend_from_slice(&f.to_le_bytes());
        }
        out
    }

    /// A [`MediaSource`](songbird::input::core::io::MediaSource) that pulls
    /// stereo s16le frames from a [`StreamingPcmSource`] and yields interleaved
    /// `f32` LE bytes for songbird's `RawAdapter` (non-seekable, unbounded).
    ///
    /// [`StreamingPcmSource`]: crate::voice::audio::StreamingPcmSource
    struct StreamingF32Reader {
        src: Arc<crate::voice::audio::StreamingPcmSource>,
        buf: Vec<u8>,
        pos: usize,
        done: bool,
    }

    impl StreamingF32Reader {
        const fn new(src: Arc<crate::voice::audio::StreamingPcmSource>) -> Self {
            Self {
                src,
                buf: Vec::new(),
                pos: 0,
                done: false,
            }
        }
    }

    impl std::io::Read for StreamingF32Reader {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            loop {
                if self.pos < self.buf.len() {
                    let n = (self.buf.len() - self.pos).min(out.len());
                    out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
                    self.pos += n;
                    return Ok(n);
                }
                if self.done {
                    return Ok(0);
                }
                // Blocking drain of the next 20 ms frame (empty == end of stream);
                // songbird reads inputs on a dedicated blocking thread.
                let frame = self.src.read();
                if frame.is_empty() {
                    self.done = true;
                    return Ok(0);
                }
                self.buf = s16le_stereo_to_f32_bytes(&frame);
                self.pos = 0;
            }
        }
    }

    impl std::io::Seek for StreamingF32Reader {
        fn seek(&mut self, _pos: std::io::SeekFrom) -> std::io::Result<u64> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "voice stream is not seekable",
            ))
        }
    }

    impl songbird::input::core::io::MediaSource for StreamingF32Reader {
        fn is_seekable(&self) -> bool {
            false
        }
        fn byte_len(&self) -> Option<u64> {
            None
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::too_many_lines,
    clippy::significant_drop_tightening,
    reason = "the conformance suite ports many cases; test guards are short-lived"
)]
mod tests {
    use super::{
        Author, BotEvents, BotHandle, DM_BOT_DISCLAIMER, DM_BOT_DISCLAIMER_DELETE_EMOJI,
        DM_BOT_DISCLAIMER_DISMISS_HINT, EmbedView, EmojiView, InteractionAck, InteractionGone,
        MentionView, MessageView, Presence, PresenceSink, PresenceStatus, ReactionPayloadView,
        ReadyInfo, SentMessage, TypingEventView, apply_message_edit, apply_reaction_clear,
        apply_reaction_delta, build_activity_presence_cb, collect_images,
        compose_content_with_embeds, defer_interaction, emoji_repr, message_pings_bot, reply,
    };
    use crate::bot::{ActivityResync, ChannelSender};
    use crate::focus::{FocusManager, FocusStore};
    use crate::history::StoreError;
    use crate::history::store::{AppendTurn, FocusPointers, HistoryStore, Promotion};
    use crate::processors::SendText;
    use crate::sources::discord_embed_text::{EmbedFieldView, EmbedImageView};
    use crate::sources::discord_text::{PublishText, TextPublisher};
    use crate::subscriptions::{SubscriptionKind, SubscriptionRegistry};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // --- shared doubles ----------------------------------------------------

    fn author(user_id: &str, display_name: &str) -> Author {
        Author::new(
            "discord",
            user_id,
            Some(display_name.to_lowercase()),
            Some(display_name.to_owned()),
        )
    }

    fn emoji(name: &str) -> EmojiView {
        EmojiView {
            name: Some(name.to_owned()),
            id: None,
            animated: false,
        }
    }

    fn mention(id: i64) -> MentionView {
        MentionView {
            id,
            is_bot: false,
            author: author(&id.to_string(), "x"),
        }
    }

    struct NoopSendText;
    #[async_trait]
    impl SendText for NoopSendText {
        async fn send(
            &self,
            _channel_id: i64,
            _content: &str,
            _reply_to_message_id: Option<&str>,
            _mention_user_ids: &[i64],
        ) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
    }

    #[derive(Default)]
    struct RecordingPresence {
        ready: AtomicBool,
        calls: Mutex<Vec<Presence>>,
    }
    #[async_trait]
    impl PresenceSink for RecordingPresence {
        fn is_ready(&self) -> bool {
            self.ready.load(Ordering::SeqCst)
        }
        async fn set_presence(&self, presence: Presence) {
            self.calls.lock().unwrap().push(presence);
        }
    }

    struct NullFocusStore;
    #[async_trait]
    impl FocusStore for NullFocusStore {
        async fn get_focus_pointers(
            &self,
            _familiar_id: &str,
        ) -> Result<Option<FocusPointers>, StoreError> {
            Ok(None)
        }
        async fn set_focus_pointers(
            &self,
            _familiar_id: &str,
            _text_channel_id: Option<i64>,
            _voice_channel_id: Option<i64>,
        ) -> Result<(), StoreError> {
            Ok(())
        }
        async fn promote_staged_turns(
            &self,
            _familiar_id: &str,
            _channel_id: i64,
            _catch_up_limit: usize,
        ) -> Result<Promotion, StoreError> {
            Ok(Promotion {
                consumed: 0,
                missed: 0,
            })
        }
    }

    fn empty_registry() -> Arc<SubscriptionRegistry> {
        let dir = tempfile::tempdir().unwrap();
        Arc::new(SubscriptionRegistry::new(dir.path().join("subs.toml")).unwrap())
    }

    fn focus_manager() -> Arc<FocusManager> {
        Arc::new(FocusManager::new(
            "fam",
            Arc::new(NullFocusStore),
            empty_registry(),
        ))
    }

    // --- defer / reply guards (B-SC1/2) ------------------------------------

    struct ScriptedCtx {
        defer_ok: bool,
        followup_ok: bool,
        defer_calls: AtomicUsize,
        followups: Mutex<Vec<String>>,
    }
    #[async_trait]
    impl InteractionAck for ScriptedCtx {
        async fn defer(&self) -> Result<(), InteractionGone> {
            self.defer_calls.fetch_add(1, Ordering::SeqCst);
            if self.defer_ok {
                Ok(())
            } else {
                Err(InteractionGone)
            }
        }
        async fn followup(&self, message: &str) -> Result<(), InteractionGone> {
            self.followups.lock().unwrap().push(message.to_owned());
            if self.followup_ok {
                Ok(())
            } else {
                Err(InteractionGone)
            }
        }
        fn command_name(&self) -> Option<String> {
            Some("subscribe-text".to_owned())
        }
    }

    fn ctx(defer_ok: bool, followup_ok: bool) -> ScriptedCtx {
        ScriptedCtx {
            defer_ok,
            followup_ok,
            defer_calls: AtomicUsize::new(0),
            followups: Mutex::new(Vec::new()),
        }
    }

    #[tokio::test]
    async fn defer_returns_true_on_success() {
        let c = ctx(true, true);
        assert!(defer_interaction(&c).await);
        assert_eq!(c.defer_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn defer_returns_false_on_dead_interaction() {
        let c = ctx(false, true);
        assert!(!defer_interaction(&c).await);
    }

    #[tokio::test]
    async fn reply_sends_followup() {
        let c = ctx(true, true);
        reply(&c, "ok").await;
        assert_eq!(c.followups.lock().unwrap().as_slice(), &["ok".to_owned()]);
    }

    #[tokio::test]
    async fn reply_swallows_dead_interaction() {
        let c = ctx(true, false);
        reply(&c, "ok").await; // must not panic
    }

    // --- message_pings_bot truth table (B-OM13) ----------------------------

    #[test]
    fn pings_true_when_bot_in_mentions() {
        assert!(message_pings_bot(&[mention(7), mention(99)], Some(99)));
    }

    #[test]
    fn pings_false_when_bot_absent() {
        assert!(!message_pings_bot(&[mention(7)], Some(99)));
    }

    #[test]
    fn pings_false_when_bot_user_id_unknown() {
        assert!(!message_pings_bot(&[mention(99)], None));
    }

    #[test]
    fn pings_false_on_empty_mentions() {
        assert!(!message_pings_bot(&[], Some(99)));
    }

    // --- emoji_repr forms (B-RX21) -----------------------------------------

    #[test]
    fn unicode_emoji_returns_name() {
        assert_eq!(emoji_repr(&emoji("\u{1f44d}")), "\u{1f44d}");
    }

    #[test]
    fn custom_emoji_returns_tagged_form() {
        assert_eq!(
            emoji_repr(&EmojiView {
                name: Some("party_blob".to_owned()),
                id: Some(12345),
                animated: false,
            }),
            "<:party_blob:12345>"
        );
    }

    #[test]
    fn animated_custom_emoji_uses_a_prefix() {
        assert_eq!(
            emoji_repr(&EmojiView {
                name: Some("dance".to_owned()),
                id: Some(999),
                animated: true,
            }),
            "<a:dance:999>"
        );
    }

    #[test]
    fn empty_name_returns_empty_string() {
        assert!(emoji_repr(&EmojiView::default()).is_empty());
    }

    // --- compose_content_with_embeds (B-DS49) ------------------------------

    fn embed_desc(text: &str) -> EmbedView {
        EmbedView {
            description: Some(text.to_owned()),
            ..Default::default()
        }
    }

    #[test]
    fn compose_no_embeds_returns_original() {
        assert_eq!(compose_content_with_embeds("hi", &[]), "hi");
    }

    #[test]
    fn compose_appends_embed_text_to_content() {
        assert_eq!(
            compose_content_with_embeds("look at this", &[embed_desc("body")]),
            "look at this\n\n[embed]\nbody"
        );
    }

    #[test]
    fn compose_url_only_message_yields_just_embed_text() {
        assert_eq!(
            compose_content_with_embeds("", &[embed_desc("body")]),
            "[embed]\nbody"
        );
    }

    #[test]
    fn compose_blank_embed_keeps_content_only() {
        assert_eq!(
            compose_content_with_embeds("hi", &[EmbedView::default()]),
            "hi"
        );
    }

    // --- collect_images (B-DS50) -------------------------------------------

    fn attachment(url: &str, filename: &str, content_type: &str) -> super::AttachmentView {
        super::AttachmentView {
            url: Some(url.to_owned()),
            filename: Some(filename.to_owned()),
            content_type: Some(content_type.to_owned()),
        }
    }

    fn embed_image(url: &str) -> EmbedView {
        EmbedView {
            image: Some(EmbedImageView {
                url: Some(url.to_owned()),
                proxy_url: None,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn collects_attachment_and_injects_placeholder() {
        let (content, images) = collect_images(
            "hello",
            &[attachment(
                "http://cdn.example.com/cat.png",
                "cat.png",
                "image/png",
            )],
            &[],
        );
        assert_eq!(
            images,
            HashMap::from([(
                "img_0".to_owned(),
                "http://cdn.example.com/cat.png".to_owned()
            )])
        );
        assert!(content.contains("[image: img_0 (cat.png)]"));
    }

    #[test]
    fn non_image_attachment_excluded() {
        let (content, images) = collect_images(
            "hi",
            &[attachment(
                "http://x.com/doc.pdf",
                "doc.pdf",
                "application/pdf",
            )],
            &[],
        );
        assert!(images.is_empty());
        assert_eq!(content, "hi");
    }

    #[test]
    fn collects_embed_image() {
        let (content, images) = collect_images(
            "",
            &[],
            &[embed_image("http://cdn.example.com/preview.jpg")],
        );
        assert_eq!(images.len(), 1);
        assert!(images.values().next().unwrap().contains("preview.jpg"));
        assert!(content.contains("[image: img_0"));
    }

    #[test]
    fn embed_without_image_ignored() {
        let (content, images) = collect_images("hello", &[], &[EmbedView::default()]);
        assert!(images.is_empty());
        assert_eq!(content, "hello");
    }

    #[test]
    fn collects_inline_url() {
        let url = "https://i.imgur.com/abc.jpg";
        let (content, images) = collect_images(&format!("check this {url}"), &[], &[]);
        assert_eq!(images.len(), 1);
        assert!(images.values().any(|v| v == url));
        assert!(content.contains("[image: img_0 (abc.jpg)]"));
    }

    #[test]
    fn no_images_returns_content_unchanged_empty_dict() {
        let (content, images) = collect_images("just text", &[], &[]);
        assert_eq!(content, "just text");
        assert!(images.is_empty());
    }

    #[test]
    fn dedupes_url_present_as_attachment_and_inline() {
        let url = "http://cdn.example.com/cat.png";
        let (_content, images) = collect_images(
            &format!("look: {url}"),
            &[attachment(url, "cat.png", "image/png")],
            &[],
        );
        assert_eq!(images.len(), 1);
        assert!(images.values().any(|v| v == url));
    }

    #[test]
    fn multiple_images_get_sequential_ids() {
        let (content, images) = collect_images(
            "hello",
            &[
                attachment("http://cdn.example.com/a.png", "a.png", "image/png"),
                attachment("http://cdn.example.com/b.jpeg", "b.jpeg", "image/png"),
            ],
            &[],
        );
        let keys: std::collections::HashSet<&str> = images.keys().map(String::as_str).collect();
        assert_eq!(keys, ["img_0", "img_1"].into_iter().collect());
        assert!(content.contains("[image: img_0 (a.png)]"));
        assert!(content.contains("[image: img_1 (b.jpeg)]"));
    }

    // --- apply_message_edit (B-RX17) with a real store ---------------------

    fn store_with_turn(content: &str, message_id: &str) -> HistoryStore {
        let store = HistoryStore::open(":memory:").unwrap();
        store
            .append_turn(
                AppendTurn::new("fam", 10, "user", content)
                    .author(author("111", "Alice"))
                    .platform_message_id(message_id),
            )
            .unwrap();
        store
    }

    #[test]
    fn edit_updates_stored_turn_when_embed_added() {
        let store = store_with_turn("check this", "m1");
        apply_message_edit(
            &store,
            "fam",
            &|_| true,
            10,
            "m1",
            "check this",
            &[embed_desc("body")],
        )
        .unwrap();
        let turn = store
            .lookup_turn_by_platform_message_id("fam", "m1")
            .unwrap()
            .unwrap();
        assert_eq!(turn.content, "check this\n\n[embed]\nbody");
    }

    #[test]
    fn edit_skips_when_channel_not_subscribed() {
        let store = store_with_turn("check this", "m1");
        apply_message_edit(
            &store,
            "fam",
            &|_| false,
            10,
            "m1",
            "check this",
            &[embed_desc("body")],
        )
        .unwrap();
        let turn = store
            .lookup_turn_by_platform_message_id("fam", "m1")
            .unwrap()
            .unwrap();
        assert_eq!(turn.content, "check this");
    }

    #[test]
    fn edit_no_op_when_no_embeds() {
        let store = store_with_turn("check this", "m1");
        apply_message_edit(
            &store,
            "fam",
            &|_| true,
            10,
            "m1",
            "check this (edited)",
            &[],
        )
        .unwrap();
        let turn = store
            .lookup_turn_by_platform_message_id("fam", "m1")
            .unwrap()
            .unwrap();
        assert_eq!(turn.content, "check this");
    }

    #[test]
    fn edit_no_row_for_message_is_silent() {
        let store = HistoryStore::open(":memory:").unwrap();
        apply_message_edit(
            &store,
            "fam",
            &|_| true,
            10,
            "m-unknown",
            "hi",
            &[embed_desc("body")],
        )
        .unwrap();
    }

    // --- reaction dispatch (B-RX20) with a real store ----------------------

    #[test]
    fn reaction_add_then_remove_returns_to_zero() {
        let store = HistoryStore::open(":memory:").unwrap();
        let e = emoji("\u{1f44d}");
        apply_reaction_delta(&store, "fam", &|_| true, 1, 42, &e, 1).unwrap();
        apply_reaction_delta(&store, "fam", &|_| true, 1, 42, &e, 1).unwrap();
        apply_reaction_delta(&store, "fam", &|_| true, 1, 42, &e, -1).unwrap();
        let out = store.reactions_for_messages("fam", &["42"]).unwrap();
        assert_eq!(
            out,
            HashMap::from([("42".to_owned(), vec![("\u{1f44d}".to_owned(), 1)])])
        );
    }

    #[test]
    fn reaction_unsubscribed_channel_writes_nothing() {
        let store = HistoryStore::open(":memory:").unwrap();
        apply_reaction_delta(&store, "fam", &|_| false, 1, 42, &emoji("\u{1f44d}"), 1).unwrap();
        let out = store.reactions_for_messages("fam", &["42"]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn reaction_clear_drops_everything() {
        let store = HistoryStore::open(":memory:").unwrap();
        store.set_reaction("fam", "42", "\u{1f44d}", 2).unwrap();
        store
            .set_reaction("fam", "42", "\u{2764}\u{fe0f}", 1)
            .unwrap();
        apply_reaction_clear(&store, "fam", &|_| true, 1, 42, None).unwrap();
        let out = store.reactions_for_messages("fam", &["42"]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn reaction_clear_one_emoji_keeps_others() {
        let store = HistoryStore::open(":memory:").unwrap();
        store.set_reaction("fam", "42", "\u{1f44d}", 2).unwrap();
        store
            .set_reaction("fam", "42", "\u{2764}\u{fe0f}", 1)
            .unwrap();
        apply_reaction_clear(&store, "fam", &|_| true, 1, 42, Some(&emoji("\u{1f44d}"))).unwrap();
        let out = store.reactions_for_messages("fam", &["42"]).unwrap();
        assert_eq!(
            out,
            HashMap::from([("42".to_owned(), vec![("\u{2764}\u{fe0f}".to_owned(), 1)])])
        );
    }

    // --- build_activity_presence_cb (B-PR26) -------------------------------

    fn presence_handle(ready: bool) -> (Arc<BotHandle>, Arc<RecordingPresence>) {
        let presence = Arc::new(RecordingPresence::default());
        presence.ready.store(ready, Ordering::SeqCst);
        let fm = focus_manager();
        fm.set_channel_name(5, "general");
        fm.set_guild_name(5, "Guild");
        fm.set_focus_immediately(5, "text");
        let handle = Arc::new(
            BotHandle::new(
                Arc::new(NoopSendText),
                presence.clone() as Arc<dyn PresenceSink>,
            )
            .with_focus_manager(fm),
        );
        (handle, presence)
    }

    #[tokio::test]
    async fn idle_sets_idle_status_with_label() {
        let (handle, presence) = presence_handle(true);
        let cb = build_activity_presence_cb(handle);
        cb("idle".to_owned(), Some("creek walk".to_owned()))
            .await
            .unwrap();
        let calls = presence.calls.lock().unwrap();
        assert_eq!(calls[0].status, PresenceStatus::Idle);
        assert_eq!(calls[0].activity_name.as_deref(), Some("creek walk"));
    }

    #[tokio::test]
    async fn dnd_sets_dnd_status_with_label() {
        let (handle, presence) = presence_handle(true);
        let cb = build_activity_presence_cb(handle);
        cb("dnd".to_owned(), Some("hatbox tending".to_owned()))
            .await
            .unwrap();
        let calls = presence.calls.lock().unwrap();
        assert_eq!(calls[0].status, PresenceStatus::Dnd);
        assert_eq!(calls[0].activity_name.as_deref(), Some("hatbox tending"));
    }

    #[tokio::test]
    async fn online_restores_focus_presence() {
        let (handle, presence) = presence_handle(true);
        let cb = build_activity_presence_cb(handle);
        cb("online".to_owned(), None).await.unwrap();
        let calls = presence.calls.lock().unwrap();
        assert_eq!(calls[0].status, PresenceStatus::Online);
        assert!(
            calls[0]
                .activity_state
                .as_deref()
                .unwrap()
                .contains("general")
        );
    }

    #[tokio::test]
    async fn presence_noop_when_bot_not_ready() {
        let (handle, presence) = presence_handle(false);
        let cb = build_activity_presence_cb(handle);
        cb("idle".to_owned(), Some("creek walk".to_owned()))
            .await
            .unwrap();
        assert!(presence.calls.lock().unwrap().is_empty());
    }

    // --- on_ready presence resync ordering (B-PR24) ------------------------

    struct AwayResync {
        presence: Arc<dyn PresenceSink>,
    }
    #[async_trait]
    impl ActivityResync for AwayResync {
        async fn resync_presence(&self) {
            self.presence
                .set_presence(Presence {
                    status: PresenceStatus::Dnd,
                    activity_name: Some("hatbox tending".to_owned()),
                    activity_state: None,
                })
                .await;
        }
    }

    fn empty_subs_mut() -> Arc<Mutex<SubscriptionRegistry>> {
        let dir = tempfile::tempdir().unwrap();
        let reg = SubscriptionRegistry::new(dir.path().join("subs.toml")).unwrap();
        std::mem::forget(dir);
        Arc::new(Mutex::new(reg))
    }

    #[tokio::test]
    async fn ready_after_reload_ends_with_away_presence() {
        let presence = Arc::new(RecordingPresence::default());
        presence.ready.store(true, Ordering::SeqCst);
        let fm = focus_manager();
        fm.set_channel_name(5, "general");
        fm.set_focus_immediately(5, "text");
        let handle = Arc::new(
            BotHandle::new(
                Arc::new(NoopSendText),
                presence.clone() as Arc<dyn PresenceSink>,
            )
            .with_focus_manager(fm),
        );
        handle.set_activity_engine(Arc::new(AwayResync {
            presence: presence.clone() as Arc<dyn PresenceSink>,
        }));
        let subs = empty_subs_mut();
        let events = BotEvents::new(
            "fam",
            Arc::new(Mutex::new(None)),
            subs,
            vec![],
            Arc::new(RecordingStore::default()),
            Arc::new(RecordingPublisher::default()),
            handle,
        );
        events
            .on_ready(ReadyInfo {
                user_id: 99,
                ..Default::default()
            })
            .await;
        let calls = presence.calls.lock().unwrap();
        assert_eq!(calls.first().unwrap().status, PresenceStatus::Online);
        assert_eq!(calls.last().unwrap().status, PresenceStatus::Dnd);
        assert_eq!(
            calls.last().unwrap().activity_name.as_deref(),
            Some("hatbox tending")
        );
    }

    #[tokio::test]
    async fn ready_without_engine_skips_resync() {
        let presence = Arc::new(RecordingPresence::default());
        presence.ready.store(true, Ordering::SeqCst);
        let fm = focus_manager();
        fm.set_focus_immediately(5, "text");
        let handle = Arc::new(
            BotHandle::new(
                Arc::new(NoopSendText),
                presence.clone() as Arc<dyn PresenceSink>,
            )
            .with_focus_manager(fm),
        );
        let subs = empty_subs_mut();
        let events = BotEvents::new(
            "fam",
            Arc::new(Mutex::new(None)),
            subs,
            vec![],
            Arc::new(RecordingStore::default()),
            Arc::new(RecordingPublisher::default()),
            handle,
        );
        events
            .on_ready(ReadyInfo {
                user_id: 99,
                ..Default::default()
            })
            .await;
        let calls = presence.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, PresenceStatus::Online);
    }

    // --- on_message DM allowlist + disclaimer + reactions (B-OM/B-RX) ------

    #[derive(Default)]
    struct RecordingStore {
        bumps: Mutex<Vec<(String, i64)>>,
    }
    impl super::BotStore for RecordingStore {
        fn update_turn_content_by_message_id(
            &self,
            _familiar_id: &str,
            _platform_message_id: &str,
            _content: &str,
        ) -> Result<(), StoreError> {
            Ok(())
        }
        fn bump_reaction(
            &self,
            _familiar_id: &str,
            _platform_message_id: &str,
            emoji: &str,
            delta: i64,
        ) -> Result<(), StoreError> {
            self.bumps.lock().unwrap().push((emoji.to_owned(), delta));
            Ok(())
        }
        fn clear_reactions(
            &self,
            _familiar_id: &str,
            _platform_message_id: &str,
            _emoji: Option<&str>,
        ) -> Result<(), StoreError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingPublisher {
        calls: Mutex<Vec<PublishText>>,
    }
    #[async_trait]
    impl TextPublisher for RecordingPublisher {
        async fn publish(&self, params: PublishText) {
            self.calls.lock().unwrap().push(params);
        }
    }

    struct FakeSent {
        id: i64,
        reactions: Mutex<Vec<String>>,
        deleted: AtomicBool,
    }
    #[async_trait]
    impl SentMessage for FakeSent {
        fn id(&self) -> i64 {
            self.id
        }
        async fn add_reaction(&self, emoji: &str) -> anyhow::Result<()> {
            self.reactions.lock().unwrap().push(emoji.to_owned());
            Ok(())
        }
        async fn delete(&self) -> anyhow::Result<()> {
            self.deleted.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    struct FakeChannel {
        sent: Mutex<Vec<String>>,
        last: Mutex<Option<Arc<FakeSent>>>,
        sent_id: i64,
    }
    impl FakeChannel {
        fn new(sent_id: i64) -> Arc<Self> {
            Arc::new(Self {
                sent: Mutex::new(Vec::new()),
                last: Mutex::new(None),
                sent_id,
            })
        }
    }
    #[async_trait]
    impl ChannelSender for FakeChannel {
        async fn send(&self, content: &str) -> anyhow::Result<Arc<dyn SentMessage>> {
            self.sent.lock().unwrap().push(content.to_owned());
            let s = Arc::new(FakeSent {
                id: self.sent_id,
                reactions: Mutex::new(Vec::new()),
                deleted: AtomicBool::new(false),
            });
            *self.last.lock().unwrap() = Some(s.clone());
            Ok(s)
        }
    }

    struct DmFixture {
        events: Arc<BotEvents>,
        store: Arc<RecordingStore>,
        publisher: Arc<RecordingPublisher>,
        subs: Arc<Mutex<SubscriptionRegistry>>,
        fm: Arc<FocusManager>,
        subs_path: std::path::PathBuf,
    }

    fn dm_fixture(allowlist: Vec<i64>) -> DmFixture {
        let dir = tempfile::tempdir().unwrap();
        let subs_path = dir.path().join("subs.toml");
        // Leak the tempdir so the path survives the fixture (test-scoped).
        std::mem::forget(dir);
        let subs = Arc::new(Mutex::new(
            SubscriptionRegistry::new(subs_path.clone()).unwrap(),
        ));
        let store = Arc::new(RecordingStore::default());
        let publisher = Arc::new(RecordingPublisher::default());
        let presence = Arc::new(RecordingPresence::default());
        let fm = focus_manager();
        let handle = Arc::new(
            BotHandle::new(Arc::new(NoopSendText), presence as Arc<dyn PresenceSink>)
                .with_focus_manager(fm.clone()),
        );
        let events = Arc::new(BotEvents::new(
            "fam",
            Arc::new(Mutex::new(Some(99))),
            subs.clone(),
            allowlist,
            store.clone(),
            publisher.clone(),
            handle,
        ));
        DmFixture {
            events,
            store,
            publisher,
            subs,
            fm,
            subs_path,
        }
    }

    fn dm_message(
        author_id: i64,
        channel_id: i64,
        guild_id: Option<i64>,
        is_bot: bool,
    ) -> (MessageView, Arc<FakeChannel>) {
        let channel = FakeChannel::new(777);
        let msg = MessageView {
            author_id,
            author_is_bot: is_bot,
            author: author(&author_id.to_string(), "X"),
            channel_id,
            guild_id,
            content: "hi".to_owned(),
            message_id: 42,
            reply_to_message_id: None,
            mentions: Vec::new(),
            attachments: Vec::new(),
            embeds: Vec::new(),
            channel: channel.clone(),
        };
        (msg, channel)
    }

    #[tokio::test]
    async fn allowlisted_dm_registers_ephemeral_and_ingests() {
        let fx = dm_fixture(vec![123]);
        let (msg, _ch) = dm_message(123, 555, None, false);
        fx.events.on_message(msg).await;
        assert_eq!(fx.publisher.calls.lock().unwrap().len(), 1);
        assert!(
            fx.subs
                .lock()
                .unwrap()
                .get(555, SubscriptionKind::Text)
                .is_some()
        );
        assert_eq!(
            fx.fm.guild_name_for(Some(555)).as_deref(),
            Some("Private Message")
        );
        assert_eq!(fx.fm.get_focus("text"), Some(555));
        // ephemeral row must never touch the sidecar
        assert!(!fx.subs_path.exists());
    }

    #[tokio::test]
    async fn non_allowlisted_dm_ignored() {
        let fx = dm_fixture(vec![123]);
        let (msg, _ch) = dm_message(999, 555, None, false);
        fx.events.on_message(msg).await;
        assert!(fx.publisher.calls.lock().unwrap().is_empty());
        assert!(
            fx.subs
                .lock()
                .unwrap()
                .get(555, SubscriptionKind::Text)
                .is_none()
        );
        assert!(fx.fm.guild_name_for(Some(555)).is_none());
    }

    #[tokio::test]
    async fn bot_authored_dm_ignored_even_if_allowlisted() {
        let fx = dm_fixture(vec![123]);
        let (msg, ch) = dm_message(123, 555, None, true);
        fx.events.on_message(msg).await;
        assert!(fx.publisher.calls.lock().unwrap().is_empty());
        assert!(
            fx.subs
                .lock()
                .unwrap()
                .get(555, SubscriptionKind::Text)
                .is_none()
        );
        assert!(ch.sent.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn own_dm_echo_ignored_even_if_allowlisted() {
        let fx = dm_fixture(vec![99]);
        let (msg, ch) = dm_message(99, 555, None, false);
        fx.events.on_message(msg).await;
        assert!(fx.publisher.calls.lock().unwrap().is_empty());
        assert!(ch.sent.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn allowlisted_dm_keeps_existing_focus() {
        let fx = dm_fixture(vec![123]);
        fx.fm.set_focus_immediately(777, "text");
        let (msg, _ch) = dm_message(123, 555, None, false);
        fx.events.on_message(msg).await;
        assert_eq!(fx.fm.get_focus("text"), Some(777));
        assert_eq!(fx.publisher.calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn subscribed_guild_channel_ingests() {
        let fx = dm_fixture(vec![123]);
        fx.subs
            .lock()
            .unwrap()
            .add(888, SubscriptionKind::Text, Some(7), true)
            .unwrap();
        let (msg, _ch) = dm_message(123, 888, Some(7), false);
        fx.events.on_message(msg).await;
        assert_eq!(fx.publisher.calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn unsubscribed_guild_channel_ignored() {
        let fx = dm_fixture(vec![123]);
        let (msg, _ch) = dm_message(123, 888, Some(7), false);
        fx.events.on_message(msg).await;
        assert!(fx.publisher.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn first_allowlisted_dm_sends_disclaimer_once() {
        let fx = dm_fixture(vec![123]);
        let (msg, ch) = dm_message(123, 555, None, false);
        fx.events.on_message(msg).await;
        assert_eq!(
            ch.sent.lock().unwrap().as_slice(),
            &[format!(
                "{DM_BOT_DISCLAIMER}{DM_BOT_DISCLAIMER_DISMISS_HINT}"
            )]
        );
        let sent = ch.last.lock().unwrap().clone().unwrap();
        assert_eq!(
            sent.reactions.lock().unwrap().as_slice(),
            &[DM_BOT_DISCLAIMER_DELETE_EMOJI.to_owned()]
        );
    }

    #[tokio::test]
    async fn second_dm_same_user_does_not_resend_disclaimer() {
        let fx = dm_fixture(vec![123]);
        let (first, ch1) = dm_message(123, 555, None, false);
        fx.events.on_message(first).await;
        assert_eq!(ch1.sent.lock().unwrap().len(), 1);
        let (second, ch2) = dm_message(123, 555, None, false);
        fx.events.on_message(second).await;
        assert!(ch2.sent.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn non_allowlisted_dm_never_sends_disclaimer() {
        let fx = dm_fixture(vec![123]);
        let (msg, ch) = dm_message(999, 555, None, false);
        fx.events.on_message(msg).await;
        assert!(ch.sent.lock().unwrap().is_empty());
    }

    async fn send_disclaimer(fx: &DmFixture) -> Arc<FakeSent> {
        let (msg, ch) = dm_message(123, 555, None, false);
        fx.events.on_message(msg).await;
        ch.last.lock().unwrap().clone().unwrap()
    }

    fn reaction_payload(
        user_id: i64,
        message_id: i64,
        channel_id: i64,
        emoji_name: &str,
    ) -> ReactionPayloadView {
        ReactionPayloadView {
            user_id,
            message_id,
            channel_id,
            emoji: emoji(emoji_name),
        }
    }

    #[tokio::test]
    async fn user_checkmark_deletes_disclaimer() {
        let fx = dm_fixture(vec![123]);
        let sent = send_disclaimer(&fx).await;
        fx.events
            .on_raw_reaction_add(&reaction_payload(
                123,
                sent.id(),
                555,
                DM_BOT_DISCLAIMER_DELETE_EMOJI,
            ))
            .await;
        assert!(sent.deleted.load(Ordering::SeqCst));
        assert!(fx.store.bumps.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn bot_own_checkmark_does_not_delete_disclaimer() {
        let fx = dm_fixture(vec![123]);
        let sent = send_disclaimer(&fx).await;
        fx.events
            .on_raw_reaction_add(&reaction_payload(
                99,
                sent.id(),
                555,
                DM_BOT_DISCLAIMER_DELETE_EMOJI,
            ))
            .await;
        assert!(!sent.deleted.load(Ordering::SeqCst));
        assert!(fx.store.bumps.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn non_checkmark_reaction_does_not_delete_disclaimer() {
        let fx = dm_fixture(vec![123]);
        let sent = send_disclaimer(&fx).await;
        fx.events
            .on_raw_reaction_add(&reaction_payload(123, sent.id(), 555, "\u{1f44d}"))
            .await;
        assert!(!sent.deleted.load(Ordering::SeqCst));
        assert!(fx.store.bumps.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unreacting_disclaimer_writes_no_history() {
        let fx = dm_fixture(vec![123]);
        let sent = send_disclaimer(&fx).await;
        fx.events.on_raw_reaction_remove(&reaction_payload(
            123,
            sent.id(),
            555,
            DM_BOT_DISCLAIMER_DELETE_EMOJI,
        ));
        assert!(fx.store.bumps.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn checkmark_on_unknown_message_does_not_delete() {
        let fx = dm_fixture(vec![123]);
        let sent = send_disclaimer(&fx).await;
        fx.events
            .on_raw_reaction_add(&reaction_payload(
                123,
                999_999,
                555,
                DM_BOT_DISCLAIMER_DELETE_EMOJI,
            ))
            .await;
        assert!(!sent.deleted.load(Ordering::SeqCst));
        assert_eq!(fx.store.bumps.lock().unwrap().len(), 1);
    }

    // duck-typed embed rendering is covered in sources::discord_embed_text;
    // this asserts the field-view re-export is reachable from bot for callers.
    #[test]
    fn embed_field_view_reexport() {
        let f = EmbedFieldView {
            name: Some("k".to_owned()),
            value: Some("v".to_owned()),
        };
        assert_eq!(f.name.as_deref(), Some("k"));
    }

    // --- composition-root wiring seams (parity-audit §3a/§3b + spec 10) -----

    fn wiring_events(subs: Arc<Mutex<SubscriptionRegistry>>, handle: Arc<BotHandle>) -> BotEvents {
        BotEvents::new(
            "fam",
            Arc::new(Mutex::new(Some(999))),
            subs,
            vec![],
            Arc::new(RecordingStore::default()),
            Arc::new(RecordingPublisher::default()),
            handle,
        )
    }

    fn wiring_handle_inner() -> BotHandle {
        BotHandle::new(
            Arc::new(NoopSendText),
            Arc::new(RecordingPresence::default()) as Arc<dyn PresenceSink>,
        )
    }

    fn wiring_handle() -> Arc<BotHandle> {
        Arc::new(wiring_handle_inner())
    }

    // §3b — the composed `on_typing` path cancels an in-flight reply when a real
    // user types. The unit ladder is pinned in `typing_interrupt.rs`; this proves
    // the seam is actually wired (handler installed on the handle + reached by
    // `on_typing`), which the audit flagged dead in production.
    #[tokio::test]
    async fn on_typing_cancels_active_turn_through_installed_handler() {
        use crate::bus::router::TurnRouter;
        use crate::config::DiscordTextConfig;
        use crate::typing_interrupt::{BotUserIdProvider, IsSubscribed, TypingInterruptHandler};

        let router = Arc::new(TurnRouter::new());
        let subs = empty_subs_mut();
        subs.lock()
            .unwrap()
            .add(42, SubscriptionKind::Text, None, true)
            .unwrap();
        let is_subscribed: IsSubscribed = {
            let subs = subs.clone();
            Arc::new(move |ch: i64| {
                u64::try_from(ch)
                    .ok()
                    .and_then(|c| subs.lock().unwrap().get(c, SubscriptionKind::Text))
                    .is_some()
            })
        };
        let bot_id: BotUserIdProvider = Arc::new(|| Some(999));
        let handler = Arc::new(TypingInterruptHandler::new(
            DiscordTextConfig::default(),
            router.clone(),
            is_subscribed,
            bot_id,
        ));
        let handle = Arc::new(wiring_handle_inner().with_typing_interrupt(handler));
        let events = wiring_events(subs, handle);

        let scope = router.begin_turn("discord:42", "t-1");
        assert!(!scope.is_cancelled());
        events.on_typing(TypingEventView {
            channel_id: 42,
            user_id: 7,
            is_bot: false,
        });
        assert!(scope.is_cancelled());
    }

    // §3b — the `[discord.text].respond_to_typing` switch flows through the seam:
    // disabled → the composed `on_typing` leaves the turn running.
    #[tokio::test]
    async fn on_typing_respects_disabled_config_through_seam() {
        use crate::bus::router::TurnRouter;
        use crate::config::DiscordTextConfig;
        use crate::typing_interrupt::{BotUserIdProvider, IsSubscribed, TypingInterruptHandler};

        let router = Arc::new(TurnRouter::new());
        let subs = empty_subs_mut();
        subs.lock()
            .unwrap()
            .add(42, SubscriptionKind::Text, None, true)
            .unwrap();
        let is_subscribed: IsSubscribed = Arc::new(|_| true);
        let bot_id: BotUserIdProvider = Arc::new(|| Some(999));
        let handler = Arc::new(TypingInterruptHandler::new(
            DiscordTextConfig {
                respond_to_typing: false,
                ..Default::default()
            },
            router.clone(),
            is_subscribed,
            bot_id,
        ));
        let handle = Arc::new(wiring_handle_inner().with_typing_interrupt(handler));
        let events = wiring_events(subs, handle);

        let scope = router.begin_turn("discord:42", "t-1");
        events.on_typing(TypingEventView {
            channel_id: 42,
            user_id: 7,
            is_bot: false,
        });
        assert!(!scope.is_cancelled());
    }

    // With no handler installed (the shape a non-text build could take),
    // `on_typing` is an inert no-op rather than a panic.
    #[test]
    fn on_typing_without_handler_is_noop() {
        let events = wiring_events(empty_subs_mut(), wiring_handle());
        events.on_typing(TypingEventView {
            channel_id: 42,
            user_id: 7,
            is_bot: false,
        });
    }

    // §3a — a runtime `/subscribe-voice` mutation is visible to the FocusManager
    // through the shared `SubscriptionView` registry, with no restart.
    #[test]
    fn runtime_voice_subscribe_is_visible_to_focus_manager() {
        let subs = empty_subs_mut();
        let subs_view: Arc<dyn crate::subscriptions::SubscriptionView> = subs.clone();
        let fm = Arc::new(FocusManager::new(
            "fam",
            Arc::new(NullFocusStore),
            subs_view,
        ));
        let handle = Arc::new(wiring_handle_inner().with_focus_manager(fm.clone()));
        let events = wiring_events(subs, handle);

        assert!(!fm.is_subscribed(555));
        events.on_subscribe_voice(555, Some(7));
        assert!(fm.is_subscribed(555));
    }

    // spec 10 §B — `/subscribe-voice` registers a persisted voice row and marks
    // the channel active in the `voice_channels` proxy the activity engine reads.
    #[test]
    fn subscribe_voice_registers_and_marks_active() {
        let subs = empty_subs_mut();
        let handle = wiring_handle();
        let events = wiring_events(subs.clone(), handle.clone());

        events.on_subscribe_voice(555, Some(7));
        let row = subs.lock().unwrap().get(555, SubscriptionKind::Voice);
        assert_eq!(row.map(|s| s.guild_id), Some(Some(7)));
        assert!(handle.voice_channels.lock().unwrap().contains(&555));
    }

    // spec 10 §B — `/unsubscribe-voice` finds the guild's voice sub, removes it,
    // clears the proxy, and returns the channel id for pipeline teardown.
    #[test]
    fn unsubscribe_voice_removes_and_returns_channel() {
        let subs = empty_subs_mut();
        let handle = wiring_handle();
        let events = wiring_events(subs.clone(), handle.clone());

        events.on_subscribe_voice(555, Some(7));
        assert_eq!(events.on_unsubscribe_voice(7), Some(555));
        assert!(
            subs.lock()
                .unwrap()
                .get(555, SubscriptionKind::Voice)
                .is_none()
        );
        assert!(!handle.voice_channels.lock().unwrap().contains(&555));
        // An unknown guild is a no-op returning `None`.
        assert_eq!(events.on_unsubscribe_voice(9999), None);
    }
}
