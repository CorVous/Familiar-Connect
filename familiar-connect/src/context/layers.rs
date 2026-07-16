//! Prompt layer implementations (subsystem 05; Python `context/layers.py`).
//!
//! Each layer owns one segment of the system prompt with its own invalidation
//! signal. The [`Layer`] trait is the swappable seam (Python's `Protocol`); the
//! eight concrete layers below implement it — except [`RecentHistoryLayer`],
//! which is a distinct assembler slot (DESIGN D15), not a `Layer`.
//!
//! Store access goes through the async facade (`build` and `invalidation_key` are
//! both `async` so neither blocks the reactor — DESIGN D16). All truncation caps
//! count Unicode scalars via a `limit-1 + "…"` helper (DESIGN §4.9, spec 05 port
//! notes); chars-per-token is 4.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use blake2::Blake2bVar;
use blake2::digest::{Update, VariableOutput};
use chrono::{DateTime, Utc};
use chrono_tz::Tz;
use regex::Regex;

use super::assembler::AssemblyContext;
use crate::budget::{estimate_message_tokens, estimate_tokens};
use crate::embedding::protocol::Embedder;
use crate::history::FOCUS_STREAM_CHANNEL_ID;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{AccountProfile, Fact, HistoryTurn};
use crate::identity::{ego_canonical_key, is_ego_key};
use crate::llm::{Message, sanitize_name};

/// A shared async history store handle every layer holds.
type Store = Arc<AsyncHistoryStore>;

/// `channel_id -> name` resolver used for channel-change markers.
pub type ChannelResolver = Arc<dyn Fn(i64) -> Option<String> + Send + Sync>;

/// Discord mention syntax: `<@USER_ID>`, `!` variant for the nick form.
static DISCORD_MENTION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<@!?(\d+)>").expect("valid discord mention regex"));

/// Soft cap on a parent reply's full content inlined into the child's prefix.
const REPLY_PARENT_FULL_CAP: usize = 400;
/// Snippet cap when the parent is already in the recent window.
const REPLY_PARENT_SNIPPET_CAP: usize = 80;
/// Bio cap for the dossier prompt header.
const BIO_CHAR_CAP: usize = 240;

/// Single prompt-layer seam.
///
/// [`build`](Layer::build) returns the layer's text contribution to the system
/// prompt (empty string opts out); [`invalidation_key`](Layer::invalidation_key)
/// is a short string used for in-process caching. Both are `async` so neither
/// blocks the reactor (DESIGN D16).
#[async_trait]
pub trait Layer: Send + Sync {
    /// Stable layer name (the cache key namespace).
    fn name(&self) -> &'static str;
    /// Render the layer's system-prompt contribution (empty opts out).
    async fn build(&self, ctx: &AssemblyContext) -> String;
    /// Short cache key; equal keys reuse the prior [`build`](Layer::build) output.
    async fn invalidation_key(&self, ctx: &AssemblyContext) -> String;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Lowercase hex of a byte slice.
fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Short content hash of a file (BLAKE2b, 8-byte digest); `"missing"` when the
/// file is absent or unreadable. Content-based, not mtime — sub-second edits
/// flip the key (behavior 7).
fn content_hash(path: &Path) -> String {
    std::fs::read(path).map_or_else(
        |_| "missing".to_owned(),
        |bytes| {
            let mut hasher = Blake2bVar::new(8).expect("blake2b-8 is a valid output size");
            hasher.update(&bytes);
            let mut out = [0u8; 8];
            hasher
                .finalize_variable(&mut out)
                .expect("8-byte finalize buffer");
            hex_lower(&out)
        },
    )
}

/// Hard cap on a string; `…` (U+2026) suffix when truncated. Keeps `limit - 1`
/// scalars then the ellipsis so the result is at most `limit` scalars — the
/// context module's convention (Python `_truncate`), distinct from
/// [`crate::support::text::truncate`] (which keeps `limit` then appends).
fn truncate_cap(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let keep = limit.saturating_sub(1);
    let mut out: String = text.chars().take(keep).collect();
    out.push('\u{2026}');
    out
}

/// Truncate so the estimated token count fits `max_tokens` (char/4 heuristic).
fn truncate_to_tokens(text: &str, max_tokens: i64) -> String {
    if max_tokens <= 0 {
        return String::new();
    }
    if estimate_tokens(text) <= max_tokens {
        return text.to_owned();
    }
    let char_cap = usize::try_from(max_tokens.saturating_mul(4)).unwrap_or(usize::MAX);
    truncate_cap(text, char_cap)
}

/// `[reactions: 👍 x3 ❤️ x1]`; empty input → empty string.
fn format_reactions(reactions: &[(String, i64)]) -> String {
    if reactions.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = reactions
        .iter()
        .map(|(emoji, count)| format!("{emoji} x{count}"))
        .collect();
    format!("[reactions: {}]", parts.join(" "))
}

/// Non-empty string test mirroring Python truthiness of an optional string.
fn is_nonempty(value: Option<&str>) -> bool {
    value.is_some_and(|s| !s.is_empty())
}

/// Time in `tz` as `2:29PM` (no leading zero on the hour).
fn format_clock_12h(ts: DateTime<Utc>, tz: Tz) -> String {
    let local = ts.with_timezone(&tz);
    let raw = local.format("%I:%M%p").to_string();
    raw.trim_start_matches('0').to_owned()
}

/// Date in `tz` as `YYYY-MM-DD`.
fn format_date_iso(ts: DateTime<Utc>, tz: Tz) -> String {
    ts.with_timezone(&tz).format("%Y-%m-%d").to_string()
}

/// `HH:MM` in `tz`.
fn format_hhmm(ts: DateTime<Utc>, tz: Tz) -> String {
    ts.with_timezone(&tz).format("%H:%M").to_string()
}

/// Rewrite `<@USER_ID>` / `<@!USER_ID>` mentions to `[@DisplayName]`.
///
/// Resolution via `resolve_label` (same per-guild preference order as speaker
/// names); unknown ids fall back to the bare user id, never raising.
async fn rewrite_mentions(
    store: &AsyncHistoryStore,
    content: &str,
    familiar_id: &str,
    guild_id: Option<i64>,
) -> String {
    let ids: Vec<String> = DISCORD_MENTION_RE
        .captures_iter(content)
        .map(|c| c[1].to_owned())
        .collect();
    if ids.is_empty() {
        return content.to_owned();
    }
    let mut display: HashMap<String, String> = HashMap::new();
    for id in ids {
        if display.contains_key(&id) {
            continue;
        }
        let resolved = store
            .resolve_label(
                format!("discord:{id}"),
                guild_id,
                Some(familiar_id.to_owned()),
            )
            .await
            .unwrap_or_else(|_| id.clone());
        display.insert(id, resolved);
    }
    DISCORD_MENTION_RE
        .replace_all(content, |caps: &regex::Captures| {
            let id = &caps[1];
            let name = display.get(id).cloned().unwrap_or_else(|| id.to_owned());
            format!("[@{name}]")
        })
        .into_owned()
}

/// Speaker label for a turn (author's `resolve_label`, or the role when
/// authorless).
async fn resolve_turn_label(
    store: &AsyncHistoryStore,
    ctx: &AssemblyContext,
    turn: &HistoryTurn,
) -> String {
    match &turn.author {
        None => turn.role.clone(),
        Some(author) => store
            .resolve_label(
                author.canonical_key(),
                ctx.guild_id,
                Some(ctx.familiar_id.clone()),
            )
            .await
            .unwrap_or_else(|_| author.canonical_key()),
    }
}

// ---------------------------------------------------------------------------
// Static / file-sourced layers
// ---------------------------------------------------------------------------

/// Per-familiar persona text from a `character.md` sidecar.
pub struct CharacterCardLayer {
    path: PathBuf,
}

impl CharacterCardLayer {
    /// New layer reading `card_path`.
    #[must_use]
    pub fn new(card_path: impl Into<PathBuf>) -> Self {
        Self {
            path: card_path.into(),
        }
    }
}

#[async_trait]
impl Layer for CharacterCardLayer {
    fn name(&self) -> &'static str {
        "character_card"
    }

    async fn build(&self, _ctx: &AssemblyContext) -> String {
        if !self.path.exists() {
            return String::new();
        }
        std::fs::read_to_string(&self.path)
            .map(|s| s.trim().to_owned())
            .unwrap_or_default()
    }

    async fn invalidation_key(&self, _ctx: &AssemblyContext) -> String {
        content_hash(&self.path)
    }
}

/// Per-viewer-mode directive block.
pub struct OperatingModeLayer {
    modes: HashMap<String, String>,
}

impl OperatingModeLayer {
    /// New layer with the given per-mode strings.
    #[must_use]
    pub const fn new(modes: HashMap<String, String>) -> Self {
        Self { modes }
    }
}

#[async_trait]
impl Layer for OperatingModeLayer {
    fn name(&self) -> &'static str {
        "operating_mode"
    }

    async fn build(&self, ctx: &AssemblyContext) -> String {
        self.modes
            .get(&ctx.viewer_mode)
            .cloned()
            .unwrap_or_default()
    }

    async fn invalidation_key(&self, ctx: &AssemblyContext) -> String {
        ctx.viewer_mode.clone()
    }
}

// ---------------------------------------------------------------------------
// Recent history (assembler slot, not a Layer — DESIGN D15)
// ---------------------------------------------------------------------------

/// Collapse consecutive same-speaker voice fragments into one rendered message.
fn coalesce_voice_fragments(turns: Vec<HistoryTurn>, max_gap_seconds: f64) -> Vec<HistoryTurn> {
    if max_gap_seconds <= 0.0 || turns.is_empty() {
        return turns;
    }
    let mut merged: Vec<HistoryTurn> = Vec::with_capacity(turns.len());
    for turn in turns {
        let coalesce = merged
            .last()
            .is_some_and(|prev| can_coalesce(prev, &turn, max_gap_seconds));
        if coalesce {
            let prev = merged
                .last_mut()
                .expect("last exists when coalesce is true");
            prev.content = format!("{} {}", prev.content, turn.content);
            prev.platform_message_id = None;
            prev.reply_to_message_id = None;
        } else {
            merged.push(turn);
        }
    }
    merged
}

/// Gap in seconds between two turns (microsecond precision, matching Python
/// `timedelta.total_seconds()` over microsecond-granularity store timestamps —
/// `layers.py:392`/`:333`).
fn gap_seconds(earlier: DateTime<Utc>, later: DateTime<Utc>) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "turn gaps are small; microsecond counts never approach f64 precision limits"
    )]
    let micros = (later - earlier).num_microseconds().unwrap_or(i64::MAX) as f64;
    micros / 1_000_000.0
}

fn can_coalesce(prev: &HistoryTurn, curr: &HistoryTurn, max_gap_seconds: f64) -> bool {
    if prev.role != curr.role {
        return false;
    }
    let (Some(prev_author), Some(curr_author)) = (&prev.author, &curr.author) else {
        return false;
    };
    if prev_author.canonical_key() != curr_author.canonical_key() {
        return false;
    }
    if is_nonempty(prev.platform_message_id.as_deref())
        || is_nonempty(curr.platform_message_id.as_deref())
    {
        return false;
    }
    if is_nonempty(prev.reply_to_message_id.as_deref())
        || is_nonempty(curr.reply_to_message_id.as_deref())
    {
        return false;
    }
    let gap = gap_seconds(prev.timestamp, curr.timestamp);
    (0.0..=max_gap_seconds).contains(&gap)
}

/// Index of the first turn to keep after the last qualifying silence gap (0 =
/// keep all / disabled).
fn silence_fold_index(turns: &[HistoryTurn], min_gap_seconds: f64) -> usize {
    if min_gap_seconds <= 0.0 || turns.len() < 2 {
        return 0;
    }
    let mut fold_idx = 0;
    for i in 1..turns.len() {
        if gap_seconds(turns[i - 1].timestamp, turns[i].timestamp) >= min_gap_seconds {
            fold_idx = i;
        }
    }
    fold_idx
}

/// Drop oldest messages until the total estimated tokens fit, always keeping the
/// newest message even if it alone exceeds the cap.
fn trim_messages_to_token_cap(messages: Vec<Message>, max_tokens: i64) -> Vec<Message> {
    if messages.is_empty() {
        return messages;
    }
    let mut kept_rev: Vec<Message> = Vec::new();
    let mut used: i64 = 0;
    for msg in messages.into_iter().rev() {
        let cost = estimate_message_tokens(&msg);
        if used + cost > max_tokens && !kept_rev.is_empty() {
            break;
        }
        used += cost;
        kept_rev.push(msg);
    }
    kept_rev.reverse();
    kept_rev
}

/// Render a channel-change separator: `{server}/{channel}` (or the bare
/// `#channel` / `#<id>` fallback when no guild / name resolves).
fn format_channel_marker(
    channel_id: i64,
    channel_resolver: Option<&ChannelResolver>,
    guild_resolver: Option<&ChannelResolver>,
) -> String {
    // Mirror Python's empty-string-is-falsy truthiness (`layers.py` lines
    // 276/278): an empty resolver result is treated as absent, not rendered.
    let name = channel_resolver
        .and_then(|r| r(channel_id))
        .filter(|s| !s.is_empty());
    let channel = name.map_or_else(|| format!("#{channel_id}"), |n| format!("#{n}"));
    let guild = guild_resolver
        .and_then(|r| r(channel_id))
        .filter(|s| !s.is_empty());
    match guild {
        Some(g) => format!("{g}/{channel}"),
        None => channel,
    }
}

/// Interleave channel markers when the surviving window spans more than one
/// channel; single-channel windows pass through byte-for-byte.
fn insert_channel_markers(
    turns: &[HistoryTurn],
    rendered: Vec<Message>,
    channel_resolver: Option<&ChannelResolver>,
    guild_resolver: Option<&ChannelResolver>,
) -> Vec<Message> {
    let distinct: HashSet<i64> = turns.iter().map(|t| t.channel_id).collect();
    if distinct.len() <= 1 {
        return rendered;
    }
    let mut out: Vec<Message> = Vec::with_capacity(rendered.len() + distinct.len());
    let mut prev_channel: Option<i64> = None;
    for (turn, msg) in turns.iter().zip(rendered) {
        if Some(turn.channel_id) != prev_channel {
            out.push(Message::new(
                "user",
                format_channel_marker(turn.channel_id, channel_resolver, guild_resolver),
            ));
            prev_channel = Some(turn.channel_id);
        }
        out.push(msg);
    }
    out
}

/// `[↩ Bob (HH:MM): …]` prefix body for a child reply turn (no trailing space).
async fn reply_prefix_body(
    store: &AsyncHistoryStore,
    parent: &HistoryTurn,
    parent_in_window: bool,
    familiar_id: &str,
    guild_id: Option<i64>,
    tz: Tz,
) -> String {
    let parent_label = match &parent.author {
        Some(author) => store
            .resolve_label(
                author.canonical_key(),
                guild_id,
                Some(familiar_id.to_owned()),
            )
            .await
            .unwrap_or_else(|_| author.canonical_key()),
        None => parent.role.clone(),
    };
    let parent_text = rewrite_mentions(store, &parent.content, familiar_id, guild_id).await;
    if parent_in_window {
        let snippet = truncate_cap(&parent_text, REPLY_PARENT_SNIPPET_CAP);
        format!("\u{21a9} {parent_label}: {snippet}")
    } else {
        let parent_ts = format_hhmm(parent.timestamp, tz);
        let full = truncate_cap(&parent_text, REPLY_PARENT_FULL_CAP);
        format!("\u{21a9} {parent_label} ({parent_ts}): {full}")
    }
}

/// Render one [`HistoryTurn`] into an LLM [`Message`] with all enrichment
/// (behaviors 13–17). `pub(crate)` so subsystem 06 can reuse it.
pub(crate) async fn turn_to_message_with_context(
    store: &AsyncHistoryStore,
    turn: &HistoryTurn,
    familiar_id: &str,
    guild_id: Option<i64>,
    in_window_msg_ids: &HashSet<String>,
    reactions: &[(String, i64)],
    tz: Tz,
) -> Message {
    let content = rewrite_mentions(store, &turn.content, familiar_id, guild_id).await;
    let reactions_suffix = format_reactions(reactions);

    if turn.role == "tool" {
        return Message::new("user", format!("[tool result] {content}"));
    }
    if turn.role == "assistant" || turn.author.is_none() {
        let body = if reactions_suffix.is_empty() {
            content
        } else {
            format!("{content} {reactions_suffix}")
        };
        return Message::new(turn.role.clone(), body);
    }

    let author = turn.author.as_ref().expect("user turn has an author");
    let msg_id_tag = turn
        .platform_message_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .map_or_else(String::new, |id| format!(" #{id}"));
    let label = store
        .resolve_label(
            author.canonical_key(),
            guild_id,
            Some(familiar_id.to_owned()),
        )
        .await
        .unwrap_or_else(|_| author.canonical_key());
    let name = sanitize_name(&author.canonical_key());
    let ts = format_hhmm(turn.timestamp, tz);

    let mut reply_prefix = String::new();
    if let Some(reply_id) = turn
        .reply_to_message_id
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        if let Ok(Some(parent)) = store
            .lookup_turn_by_platform_message_id(familiar_id.to_owned(), reply_id.to_owned())
            .await
        {
            let parent_in_window = parent
                .platform_message_id
                .as_deref()
                .is_some_and(|pid| in_window_msg_ids.contains(pid));
            let body =
                reply_prefix_body(store, &parent, parent_in_window, familiar_id, guild_id, tz)
                    .await;
            reply_prefix = format!("{body} ");
        }
    }

    let mut prefixed = format!(
        "[{ts} {label} #{}{msg_id_tag}] {reply_prefix}{content}",
        turn.channel_id
    );
    if !reactions_suffix.is_empty() {
        prefixed = format!("{prefixed} {reactions_suffix}");
    }
    let mut msg = Message::new(turn.role.clone(), prefixed);
    msg.name = name;
    msg
}

/// Verbatim tail of the consumed cross-channel stream for the active familiar.
///
/// Not a [`Layer`]: the assembler holds it as a distinct slot (DESIGN D15) and
/// consumes [`recent_messages`](RecentHistoryLayer::recent_messages).
pub struct RecentHistoryLayer {
    store: Store,
    tz: Tz,
    window_size: i64,
    max_tokens: Option<i64>,
    coalesce_max_gap_seconds: f64,
    silence_gap_fold_seconds: f64,
    channel_name_resolver: Option<ChannelResolver>,
    guild_name_resolver: Option<ChannelResolver>,
}

impl RecentHistoryLayer {
    /// Start building a recent-history layer over `store`.
    #[must_use]
    pub fn builder(store: Store) -> RecentHistoryBuilder {
        RecentHistoryBuilder::new(store)
    }

    /// Always the empty string — this slot contributes to `recent_history`, not
    /// the system prompt (behavior 19).
    #[allow(
        clippy::unused_async,
        reason = "mirrors the layer build contract; the slot opts out of the system prompt"
    )]
    pub async fn build(&self, _ctx: &AssemblyContext) -> String {
        String::new()
    }

    /// The last `window_size` turns of the consumed cross-channel stream as LLM
    /// messages (behaviors 9–18).
    pub async fn recent_messages(&self, ctx: &AssemblyContext) -> Vec<Message> {
        let mut turns = self
            .store
            .recent_cross_channel(ctx.familiar_id.clone(), self.window_size, true)
            .await
            .unwrap_or_default();
        turns = coalesce_voice_fragments(turns, self.coalesce_max_gap_seconds);
        let fold_idx = silence_fold_index(&turns, self.silence_gap_fold_seconds);
        if fold_idx > 0 {
            turns = turns.split_off(fold_idx);
        }
        let pmids: Vec<String> = turns
            .iter()
            .filter_map(|t| t.platform_message_id.as_deref().filter(|s| !s.is_empty()))
            .map(str::to_owned)
            .collect();
        let in_window_msg_ids: HashSet<String> = pmids.iter().cloned().collect();
        let reactions = self
            .store
            .reactions_for_messages(ctx.familiar_id.clone(), pmids)
            .await
            .unwrap_or_default();

        let mut rendered: Vec<Message> = Vec::with_capacity(turns.len());
        for turn in &turns {
            let react = turn
                .platform_message_id
                .as_deref()
                .and_then(|id| reactions.get(id))
                .cloned()
                .unwrap_or_default();
            let msg = turn_to_message_with_context(
                &self.store,
                turn,
                &ctx.familiar_id,
                ctx.guild_id,
                &in_window_msg_ids,
                &react,
                self.tz,
            )
            .await;
            rendered.push(msg);
        }

        if let Some(max_tokens) = self.max_tokens {
            rendered = trim_messages_to_token_cap(rendered, max_tokens);
            let keep = rendered.len();
            let drop = turns.len().saturating_sub(keep);
            turns = turns.split_off(drop);
        }

        insert_channel_markers(
            &turns,
            rendered,
            self.channel_name_resolver.as_ref(),
            self.guild_name_resolver.as_ref(),
        )
    }
}

/// Builder for [`RecentHistoryLayer`].
pub struct RecentHistoryBuilder {
    store: Store,
    window_size: i64,
    max_tokens: Option<i64>,
    coalesce_max_gap_seconds: f64,
    silence_gap_fold_seconds: f64,
    display_tz: String,
    channel_name_resolver: Option<ChannelResolver>,
    guild_name_resolver: Option<ChannelResolver>,
}

impl RecentHistoryBuilder {
    fn new(store: Store) -> Self {
        Self {
            store,
            window_size: 20,
            max_tokens: None,
            coalesce_max_gap_seconds: 45.0,
            silence_gap_fold_seconds: 0.0,
            display_tz: "UTC".to_owned(),
            channel_name_resolver: None,
            guild_name_resolver: None,
        }
    }

    /// Turn window size.
    #[must_use]
    pub const fn window_size(mut self, window_size: i64) -> Self {
        self.window_size = window_size;
        self
    }
    /// Token cap for the rendered window (`None` disables).
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: Option<i64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }
    /// Voice-fragment coalescing gap (`<= 0` disables).
    #[must_use]
    pub const fn coalesce_max_gap_seconds(mut self, secs: f64) -> Self {
        self.coalesce_max_gap_seconds = secs;
        self
    }
    /// Silence-gap fold threshold (`<= 0` disables).
    #[must_use]
    pub const fn silence_gap_fold_seconds(mut self, secs: f64) -> Self {
        self.silence_gap_fold_seconds = secs;
        self
    }
    /// IANA display timezone for rendered clocks.
    #[must_use]
    pub fn display_tz(mut self, tz: impl Into<String>) -> Self {
        self.display_tz = tz.into();
        self
    }
    /// Channel-name resolver for markers.
    #[must_use]
    pub fn channel_name_resolver(mut self, resolver: ChannelResolver) -> Self {
        self.channel_name_resolver = Some(resolver);
        self
    }
    /// Guild-name resolver for markers.
    #[must_use]
    pub fn guild_name_resolver(mut self, resolver: ChannelResolver) -> Self {
        self.guild_name_resolver = Some(resolver);
        self
    }

    /// Build the layer, resolving the display timezone (falls back to UTC).
    #[must_use]
    pub fn build(self) -> RecentHistoryLayer {
        RecentHistoryLayer {
            store: self.store,
            tz: self.display_tz.parse().unwrap_or(Tz::UTC),
            window_size: self.window_size,
            max_tokens: self.max_tokens,
            coalesce_max_gap_seconds: self.coalesce_max_gap_seconds,
            silence_gap_fold_seconds: self.silence_gap_fold_seconds,
            channel_name_resolver: self.channel_name_resolver,
            guild_name_resolver: self.guild_name_resolver,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversation summary
// ---------------------------------------------------------------------------

/// Read-only layer over the per-familiar focus-stream summary.
pub struct ConversationSummaryLayer {
    store: Store,
    max_tokens: Option<i64>,
}

impl ConversationSummaryLayer {
    /// New layer over `store`.
    #[must_use]
    pub const fn new(store: Store) -> Self {
        Self {
            store,
            max_tokens: None,
        }
    }

    /// Set the token cap.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: i64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

#[async_trait]
impl Layer for ConversationSummaryLayer {
    fn name(&self) -> &'static str {
        "conversation_summary"
    }

    async fn build(&self, ctx: &AssemblyContext) -> String {
        let Some(entry) = self
            .store
            .get_summary(ctx.familiar_id.clone(), FOCUS_STREAM_CHANNEL_ID)
            .await
            .ok()
            .flatten()
        else {
            return String::new();
        };
        let body = entry.summary_text.trim();
        if body.is_empty() {
            return String::new();
        }
        let body = self.max_tokens.map_or_else(
            || body.to_owned(),
            |max_tokens| truncate_to_tokens(body, max_tokens),
        );
        format!("## Conversation so far\n\n{body}")
    }

    async fn invalidation_key(&self, ctx: &AssemblyContext) -> String {
        match self
            .store
            .get_summary(ctx.familiar_id.clone(), FOCUS_STREAM_CHANNEL_ID)
            .await
            .ok()
            .flatten()
        {
            None => "none".to_owned(),
            Some(entry) => format!(
                "focus:{}:{}",
                entry.last_consumed_at.as_deref().unwrap_or("None"),
                entry.last_summarised_id
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// People dossier
// ---------------------------------------------------------------------------

/// Simple ASCII-ish title-case fallback for the ego header display.
///
/// Mirrors Python `str.title()` (`layers.py:1001`): word boundaries fall on
/// Unicode *cased* characters only. Digits and punctuation are uncased, so a
/// letter following one starts a new word and is capitalized
/// (`agent007bond` -> `Agent007Bond`, `3cats` -> `3Cats`). Uncased characters
/// pass through their (identity) case mapping unchanged.
fn title_case(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_is_cased = false;
    for ch in text.chars() {
        if previous_is_cased {
            out.extend(ch.to_lowercase());
        } else {
            out.extend(ch.to_uppercase());
        }
        previous_is_cased = ch.is_lowercase() || ch.is_uppercase();
    }
    out
}

/// Per-person header block (`### display` + optional `@user · pronouns` and
/// `Bio:` lines).
fn format_profile_header(display: &str, profile: Option<&AccountProfile>) -> String {
    let mut lines: Vec<String> = vec![format!("### {display}")];
    if let Some(profile) = profile {
        let username = profile.username.as_deref().unwrap_or("").trim();
        let pronouns = profile.pronouns.as_deref().unwrap_or("").trim();
        let mut meta: Vec<String> = Vec::new();
        if !username.is_empty() {
            meta.push(format!("@{username}"));
        }
        if !pronouns.is_empty() {
            meta.push(pronouns.to_owned());
        }
        if !meta.is_empty() {
            lines.push(meta.join(" \u{b7} "));
        }
        let bio = profile.bio.as_deref().unwrap_or("").trim();
        if !bio.is_empty() {
            lines.push(format!("Bio: {}", truncate_cap(bio, BIO_CHAR_CAP)));
        }
    }
    lines.join("\n")
}

/// Per-person dossier block for people active in the channel.
pub struct PeopleDossierLayer {
    store: Store,
    window_size: i64,
    max_people: i64,
    max_tokens: Option<i64>,
    familiar_display_name: Option<String>,
}

impl PeopleDossierLayer {
    /// Start building a dossier layer over `store`.
    #[must_use]
    pub const fn builder(store: Store) -> PeopleDossierBuilder {
        PeopleDossierBuilder::new(store)
    }

    async fn candidate_keys(&self, ctx: &AssemblyContext) -> Vec<String> {
        let mut ordered: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let ego = ego_canonical_key(&ctx.familiar_id);
        seen.insert(ego.clone());
        ordered.push(ego);

        let Some(channel_id) = ctx.channel_id else {
            return ordered;
        };
        let turns = self
            .store
            .recent(
                ctx.familiar_id.clone(),
                channel_id,
                self.window_size,
                None,
                None,
            )
            .await
            .unwrap_or_default();
        let cap = usize::try_from(self.max_people).unwrap_or(0);
        let mut people: Vec<String> = Vec::new();
        let mut people_seen: HashSet<String> = HashSet::new();
        for turn in turns.iter().rev() {
            if let Some(author) = &turn.author {
                let key = author.canonical_key();
                if people_seen.insert(key.clone()) {
                    people.push(key);
                }
            }
            let mentions = self
                .store
                .mentions_for_turn(turn.id)
                .await
                .unwrap_or_default();
            for key in mentions {
                if people_seen.insert(key.clone()) {
                    people.push(key);
                }
            }
            if people.len() >= cap {
                break;
            }
        }
        for key in people.into_iter().take(cap) {
            if seen.insert(key.clone()) {
                ordered.push(key);
            }
        }
        ordered
    }
}

#[async_trait]
impl Layer for PeopleDossierLayer {
    fn name(&self) -> &'static str {
        "people_dossier"
    }

    async fn build(&self, ctx: &AssemblyContext) -> String {
        let candidates = self.candidate_keys(ctx).await;
        if candidates.is_empty() {
            return String::new();
        }
        let mut sections: Vec<String> = Vec::new();
        let mut remaining = self.max_tokens;
        for key in candidates {
            let Some(entry) = self
                .store
                .get_people_dossier(ctx.familiar_id.clone(), key.clone())
                .await
                .ok()
                .flatten()
            else {
                continue;
            };
            if entry.dossier_text.trim().is_empty() {
                continue;
            }
            let header = if is_ego_key(&key) {
                let display = self
                    .familiar_display_name
                    .clone()
                    .unwrap_or_else(|| title_case(&ctx.familiar_id));
                format_profile_header(&display, None)
            } else {
                let display = self
                    .store
                    .resolve_label(key.clone(), ctx.guild_id, Some(ctx.familiar_id.clone()))
                    .await
                    .unwrap_or_else(|_| key.clone());
                let profile = self
                    .store
                    .get_account_profile(key.clone())
                    .await
                    .ok()
                    .flatten();
                format_profile_header(&display, profile.as_ref())
            };
            let mut section = format!("{header}\n\n{}", entry.dossier_text.trim());
            if let Some(rem) = remaining {
                let cost = estimate_tokens(&section);
                if cost > rem && !sections.is_empty() {
                    break;
                }
                section = truncate_to_tokens(&section, rem);
                remaining = Some(rem - estimate_tokens(&section));
            }
            sections.push(section);
        }
        if sections.is_empty() {
            return String::new();
        }
        format!(
            "## People in this conversation\n\n{}",
            sections.join("\n\n")
        )
    }

    async fn invalidation_key(&self, ctx: &AssemblyContext) -> String {
        let candidates = self.candidate_keys(ctx).await;
        let latest = self
            .store
            .latest_id(ctx.familiar_id.clone(), Some(ctx.channel_id.unwrap_or(0)))
            .await
            .ok()
            .flatten()
            .unwrap_or(0);
        let mut parts: Vec<String> = vec![format!("t{latest}"), format!("cap{}", self.max_people)];
        for key in candidates {
            let entry = self
                .store
                .get_people_dossier(ctx.familiar_id.clone(), key.clone())
                .await
                .ok()
                .flatten();
            parts.push(match entry {
                None => format!("{key}:none"),
                Some(entry) => format!("{key}:f{}", entry.last_fact_id),
            });
        }
        parts.join("|")
    }
}

/// Builder for [`PeopleDossierLayer`].
pub struct PeopleDossierBuilder {
    store: Store,
    window_size: i64,
    max_people: i64,
    max_tokens: Option<i64>,
    familiar_display_name: Option<String>,
}

impl PeopleDossierBuilder {
    const fn new(store: Store) -> Self {
        Self {
            store,
            window_size: 20,
            max_people: 8,
            max_tokens: None,
            familiar_display_name: None,
        }
    }

    /// Channel scan window size.
    #[must_use]
    pub const fn window_size(mut self, window_size: i64) -> Self {
        self.window_size = window_size;
        self
    }
    /// Maximum non-ego people rendered.
    #[must_use]
    pub const fn max_people(mut self, max_people: i64) -> Self {
        self.max_people = max_people;
        self
    }
    /// Token cap.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: Option<i64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }
    /// Familiar display name for the ego header.
    #[must_use]
    pub fn familiar_display_name(mut self, name: impl Into<String>) -> Self {
        self.familiar_display_name = Some(name.into());
        self
    }

    /// Build the layer.
    #[must_use]
    pub fn build(self) -> PeopleDossierLayer {
        PeopleDossierLayer {
            store: self.store,
            window_size: self.window_size,
            max_people: self.max_people,
            max_tokens: self.max_tokens,
            familiar_display_name: self.familiar_display_name,
        }
    }
}

// ---------------------------------------------------------------------------
// RAG context
// ---------------------------------------------------------------------------

/// Cosine similarity over equal-length float vectors; `0.0` on mismatch / zero
/// norm.
#[allow(
    clippy::suboptimal_flops,
    reason = "mirror Python's plain +/* float arithmetic for bit-parity of the cosine score"
)]
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb).sqrt()
}

/// Fuse BM25 / recency / importance / embedding signals into one rank
/// (behavior 31); each signal is normalized to `[0, 1]` within the batch.
fn rerank_fact_candidates(
    scored: Vec<(Fact, f32)>,
    limit: i64,
    bm25_weight: f64,
    recency_weight: f64,
    importance_weight: f64,
    embedding_weight: f64,
    sims: &HashMap<i64, f64>,
) -> Vec<Fact> {
    if scored.is_empty() {
        return Vec::new();
    }
    let bm25_scores: Vec<f64> = scored.iter().map(|(_, s)| f64::from(*s)).collect();
    let bm25_min = bm25_scores.iter().copied().fold(f64::INFINITY, f64::min);
    let bm25_max = bm25_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let bm25_span = bm25_max - bm25_min;

    let mut fact_ids: Vec<i64> = scored.iter().map(|(f, _)| f.id).collect();
    fact_ids.sort_unstable();
    fact_ids.dedup();
    let mut recency_rank: HashMap<i64, f64> = HashMap::new();
    if fact_ids.len() == 1 {
        recency_rank.insert(fact_ids[0], 1.0);
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "candidate batches are tiny; index counts never approach f64 precision limits"
        )]
        let last = (fact_ids.len() - 1) as f64;
        for (i, fid) in fact_ids.iter().enumerate() {
            #[allow(
                clippy::cast_precision_loss,
                reason = "candidate batches are tiny; index counts never approach f64 precision limits"
            )]
            let rank = i as f64 / last;
            recency_rank.insert(*fid, rank);
        }
    }

    let mut ranked: Vec<(f64, usize, Fact)> = Vec::with_capacity(scored.len());
    for (idx, (fact, bm25)) in scored.into_iter().enumerate() {
        let bm25_q = if bm25_span > 0.0 {
            (f64::from(bm25) - bm25_min) / bm25_span
        } else {
            1.0
        };
        let recency_q = recency_rank.get(&fact.id).copied().unwrap_or(1.0);
        #[allow(
            clippy::cast_precision_loss,
            reason = "importance is clamped 1..=10; the cast is exact"
        )]
        let importance_q = fact.importance.map_or(0.5, |i| i as f64 / 10.0);
        let embedding_q = sims.get(&fact.id).map_or(0.5, |cos| (cos + 1.0) / 2.0);
        #[allow(
            clippy::suboptimal_flops,
            reason = "mirror Python's plain +/* float arithmetic for bit-parity of the rerank score"
        )]
        let score = bm25_weight * bm25_q
            + recency_weight * recency_q
            + importance_weight * importance_q
            + embedding_weight * embedding_q;
        ranked.push((score, idx, fact));
    }
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let keep = usize::try_from(limit).unwrap_or(0);
    ranked.into_iter().take(keep).map(|(_, _, f)| f).collect()
}

/// Render one fact line with optional rename annotations (behavior 33).
async fn render_fact_line(
    store: &AsyncHistoryStore,
    familiar_id: &str,
    fact: &Fact,
    guild_id: Option<i64>,
) -> String {
    if fact.subjects.is_empty() {
        return fact.text.clone();
    }
    let mut notes: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for subject in &fact.subjects {
        if !seen.insert(subject.canonical_key.clone()) {
            continue;
        }
        let current = store
            .resolve_label(
                subject.canonical_key.clone(),
                guild_id,
                Some(familiar_id.to_owned()),
            )
            .await
            .unwrap_or_else(|_| subject.canonical_key.clone());
        let tail = subject
            .canonical_key
            .split_once(':')
            .map_or("", |(_, tail)| tail);
        if current == tail {
            continue;
        }
        if current == subject.display_at_write {
            continue;
        }
        notes.push(format!(
            "{} is now known as {current}",
            subject.display_at_write
        ));
    }
    if notes.is_empty() {
        fact.text.clone()
    } else {
        format!("{} ({})", fact.text, notes.join("; "))
    }
}

/// Cap RAG fact + turn lines together; facts win ties (behavior 35).
fn trim_rag_lines_to_tokens(
    fact_lines: Vec<String>,
    turn_lines: Vec<String>,
    max_tokens: i64,
) -> (Vec<String>, Vec<String>) {
    let mut used: i64 = 0;
    let mut kept_facts: Vec<String> = Vec::new();
    for line in fact_lines {
        let cost = estimate_tokens(&line);
        if used + cost > max_tokens && !kept_facts.is_empty() {
            break;
        }
        used += cost;
        kept_facts.push(line);
    }
    let mut kept_turns: Vec<String> = Vec::new();
    for line in turn_lines {
        let cost = estimate_tokens(&line);
        if used + cost > max_tokens && (!kept_turns.is_empty() || !kept_facts.is_empty()) {
            break;
        }
        used += cost;
        kept_turns.push(line);
    }
    (kept_facts, kept_turns)
}

/// FTS-backed retrieval of relevant historical turns *and* facts.
pub struct RagContextLayer {
    store: Store,
    tz: Tz,
    max_results: i64,
    max_facts: i64,
    recent_window_size: i64,
    max_tokens: Option<i64>,
    context_window: i64,
    bm25_weight: f64,
    recency_weight: f64,
    importance_weight: f64,
    embedding_weight: f64,
    embedder: Option<Arc<dyn Embedder>>,
    fact_overfetch: i64,
    cue: Mutex<String>,
    embedder_warned: AtomicBool,
}

impl RagContextLayer {
    /// Start building a RAG layer over `store`.
    #[must_use]
    pub fn builder(store: Store) -> RagContextBuilder {
        RagContextBuilder::new(store)
    }

    /// Set the retrieval cue (stripped; empty opts the layer out).
    pub fn set_current_cue(&self, cue: &str) {
        cue.trim()
            .clone_into(&mut self.cue.lock().expect("rag cue mutex"));
    }

    fn cue(&self) -> String {
        self.cue.lock().expect("rag cue mutex").clone()
    }

    fn rerank_facts(&self) -> bool {
        self.recency_weight > 0.0
            || self.importance_weight > 0.0
            || (self.embedding_weight > 0.0 && self.embedder.is_some())
    }

    async fn embedding_similarities(&self, scored: &[(Fact, f32)], cue: &str) -> HashMap<i64, f64> {
        let Some(embedder) = &self.embedder else {
            return HashMap::new();
        };
        if self.embedding_weight <= 0.0 || scored.is_empty() {
            return HashMap::new();
        }
        let ids: Vec<i64> = scored.iter().map(|(f, _)| f.id).collect();
        let stored_vecs = self
            .store
            .get_fact_embeddings(ids, embedder.name().to_owned())
            .await
            .unwrap_or_default();
        if stored_vecs.is_empty() {
            return HashMap::new();
        }
        let Ok(vectors) = embedder.embed(&[cue.to_owned()]).await else {
            return HashMap::new();
        };
        let Some(cue_vec) = vectors.into_iter().next() else {
            return HashMap::new();
        };
        stored_vecs
            .into_iter()
            .map(|(fid, vec)| (fid, cosine(&cue_vec, &vec)))
            .collect()
    }

    /// Retrieve the fact results for `cue` — BM25-only (default) or the
    /// over-fetch-and-rerank path when any non-BM25 weight is set (behavior 30).
    async fn retrieve_facts(&self, ctx: &AssemblyContext, cue: &str) -> Vec<Fact> {
        if !self.rerank_facts() {
            return self
                .store
                .search_facts(
                    ctx.familiar_id.clone(),
                    cue.to_owned(),
                    self.max_facts,
                    false,
                    None,
                )
                .await
                .unwrap_or_default();
        }
        let fetch = self
            .fact_overfetch
            .min(self.max_facts * 4)
            .max(self.max_facts);
        let scored = self
            .store
            .search_facts_scored(ctx.familiar_id.clone(), cue.to_owned(), fetch, false, None)
            .await
            .unwrap_or_default();
        let sims = self.embedding_similarities(&scored, cue).await;
        let embedding_weight = if self.embedder.is_some() {
            self.embedding_weight
        } else {
            0.0
        };
        rerank_fact_candidates(
            scored,
            self.max_facts,
            self.bm25_weight,
            self.recency_weight,
            self.importance_weight,
            embedding_weight,
            &sims,
        )
    }

    async fn render_turn_lines(
        &self,
        ctx: &AssemblyContext,
        hits: &[HistoryTurn],
        max_id: Option<i64>,
    ) -> Vec<String> {
        if hits.is_empty() {
            return Vec::new();
        }
        let mut wanted: HashSet<i64> = HashSet::new();
        for hit in hits {
            for d in -self.context_window..=self.context_window {
                wanted.insert(hit.id + d);
            }
        }
        let ids: Vec<i64> = wanted.into_iter().collect();
        let expanded = self
            .store
            .turns_by_ids(ctx.familiar_id.clone(), ids)
            .await
            .unwrap_or_default();
        let hit_channels: HashSet<i64> = hits.iter().map(|h| h.channel_id).collect();
        let mut kept: Vec<HistoryTurn> = Vec::new();
        for turn in expanded {
            if !hit_channels.contains(&turn.channel_id) {
                continue;
            }
            if let (Some(max_id), Some(active)) = (max_id, ctx.channel_id) {
                if turn.channel_id == active && turn.id > max_id {
                    continue;
                }
            }
            kept.push(turn);
        }

        let mut by_date: BTreeMap<String, Vec<HistoryTurn>> = BTreeMap::new();
        for turn in kept {
            by_date
                .entry(format_date_iso(turn.timestamp, self.tz))
                .or_default()
                .push(turn);
        }

        let mut lines: Vec<String> = Vec::new();
        for (date_label, turns) in by_date {
            lines.push(format!("{date_label}:"));
            for turn in &turns {
                let label = resolve_turn_label(&self.store, ctx, turn).await;
                let rewritten =
                    rewrite_mentions(&self.store, &turn.content, &ctx.familiar_id, ctx.guild_id)
                        .await;
                let clock = format_clock_12h(turn.timestamp, self.tz);
                let mut content_lines = rewritten.split('\n');
                let first = content_lines.next().unwrap_or("");
                lines.push(format!("> [{clock} {label}]: {first}"));
                for cont in content_lines {
                    lines.push(if cont.is_empty() {
                        ">".to_owned()
                    } else {
                        format!("> {cont}")
                    });
                }
            }
            lines.push(String::new());
        }
        if lines.last().is_some_and(String::is_empty) {
            lines.pop();
        }
        lines
    }
}

#[async_trait]
impl Layer for RagContextLayer {
    fn name(&self) -> &'static str {
        "rag_context"
    }

    async fn build(&self, ctx: &AssemblyContext) -> String {
        let cue = self.cue();
        if cue.is_empty() {
            return String::new();
        }
        if self.embedding_weight > 0.0
            && self.embedder.is_none()
            && !self.embedder_warned.swap(true, Ordering::Relaxed)
        {
            tracing::warn!(
                "RagContextLayer: embedding_weight={:.2} but no embedder configured \
                 (set [providers.embedding].backend); falling back to BM25-only ranking.",
                self.embedding_weight
            );
        }

        let mut max_id: Option<i64> = None;
        if self.recent_window_size > 0 {
            if let Some(channel_id) = ctx.channel_id {
                if let Some(latest) = self
                    .store
                    .latest_id(ctx.familiar_id.clone(), Some(channel_id))
                    .await
                    .ok()
                    .flatten()
                {
                    max_id = Some(latest - self.recent_window_size);
                }
            }
        }

        let turn_results = self
            .store
            .search_turns(
                ctx.familiar_id.clone(),
                cue.clone(),
                self.max_results,
                None,
                max_id,
            )
            .await
            .unwrap_or_default();

        let fact_results = self.retrieve_facts(ctx, &cue).await;

        if turn_results.is_empty() && fact_results.is_empty() {
            return String::new();
        }

        let mut fact_lines: Vec<String> = Vec::with_capacity(fact_results.len());
        for fact in &fact_results {
            let line = render_fact_line(&self.store, &ctx.familiar_id, fact, ctx.guild_id).await;
            fact_lines.push(format!("- {line}"));
        }
        let mut turn_lines = self.render_turn_lines(ctx, &turn_results, max_id).await;

        if let Some(max_tokens) = self.max_tokens {
            let (facts, turns) = trim_rag_lines_to_tokens(fact_lines, turn_lines, max_tokens);
            fact_lines = facts;
            turn_lines = turns;
        }

        let mut sections: Vec<String> = Vec::new();
        if !fact_lines.is_empty() {
            let mut section = String::from("## Possibly relevant facts\n");
            for line in &fact_lines {
                section.push('\n');
                section.push_str(line);
            }
            sections.push(section);
        }
        if !turn_lines.is_empty() {
            let mut section = String::from("## Possibly relevant earlier turns\n");
            for line in &turn_lines {
                section.push('\n');
                section.push_str(line);
            }
            sections.push(section);
        }
        sections.join("\n\n")
    }

    async fn invalidation_key(&self, ctx: &AssemblyContext) -> String {
        let cue = self.cue();
        let latest_turn = self
            .store
            .latest_fts_id(ctx.familiar_id.clone())
            .await
            .unwrap_or(0);
        let latest_fact = self
            .store
            .latest_fact_id(ctx.familiar_id.clone())
            .await
            .unwrap_or(0);
        format!("{cue}|t{latest_turn}|f{latest_fact}")
    }
}

/// Builder for [`RagContextLayer`].
pub struct RagContextBuilder {
    store: Store,
    max_results: i64,
    max_facts: i64,
    recent_window_size: i64,
    max_tokens: Option<i64>,
    context_window: i64,
    bm25_weight: f64,
    recency_weight: f64,
    importance_weight: f64,
    embedding_weight: f64,
    embedder: Option<Arc<dyn Embedder>>,
    fact_overfetch: i64,
    display_tz: String,
}

impl RagContextBuilder {
    fn new(store: Store) -> Self {
        Self {
            store,
            max_results: 5,
            max_facts: 3,
            recent_window_size: 0,
            max_tokens: None,
            context_window: 1,
            bm25_weight: 1.0,
            recency_weight: 0.0,
            importance_weight: 0.0,
            embedding_weight: 0.0,
            embedder: None,
            fact_overfetch: 12,
            display_tz: "UTC".to_owned(),
        }
    }

    /// Max retrieved turn results.
    #[must_use]
    pub const fn max_results(mut self, n: i64) -> Self {
        self.max_results = n;
        self
    }
    /// Max retrieved fact results.
    #[must_use]
    pub const fn max_facts(mut self, n: i64) -> Self {
        self.max_facts = n;
        self
    }
    /// Recent-window exclusion size (`0` disables).
    #[must_use]
    pub const fn recent_window_size(mut self, n: i64) -> Self {
        self.recent_window_size = n;
        self
    }
    /// Joint fact+turn token cap.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: Option<i64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }
    /// Neighbour expansion window per hit.
    #[must_use]
    pub const fn context_window(mut self, n: i64) -> Self {
        self.context_window = n;
        self
    }
    /// BM25 rank weight.
    #[must_use]
    pub const fn bm25_weight(mut self, w: f64) -> Self {
        self.bm25_weight = w;
        self
    }
    /// Recency rank weight.
    #[must_use]
    pub const fn recency_weight(mut self, w: f64) -> Self {
        self.recency_weight = w;
        self
    }
    /// Importance rank weight.
    #[must_use]
    pub const fn importance_weight(mut self, w: f64) -> Self {
        self.importance_weight = w;
        self
    }
    /// Embedding rank weight.
    #[must_use]
    pub const fn embedding_weight(mut self, w: f64) -> Self {
        self.embedding_weight = w;
        self
    }
    /// Fact over-fetch cap for reranking.
    #[must_use]
    pub const fn fact_overfetch(mut self, n: i64) -> Self {
        self.fact_overfetch = n;
        self
    }
    /// Embedder for the embedding signal.
    #[must_use]
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }
    /// IANA display timezone for date headers / clocks.
    #[must_use]
    pub fn display_tz(mut self, tz: impl Into<String>) -> Self {
        self.display_tz = tz.into();
        self
    }

    /// Build the layer (`context_window`/`fact_overfetch` are floored as in
    /// Python).
    #[must_use]
    pub fn build(self) -> RagContextLayer {
        RagContextLayer {
            store: self.store,
            tz: self.display_tz.parse().unwrap_or(Tz::UTC),
            max_results: self.max_results,
            max_facts: self.max_facts,
            recent_window_size: self.recent_window_size,
            max_tokens: self.max_tokens,
            context_window: self.context_window.max(0),
            bm25_weight: self.bm25_weight,
            recency_weight: self.recency_weight,
            importance_weight: self.importance_weight,
            embedding_weight: self.embedding_weight,
            embedder: self.embedder,
            fact_overfetch: self.fact_overfetch.max(1),
            cue: Mutex::new(String::new()),
            embedder_warned: AtomicBool::new(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Reflections
// ---------------------------------------------------------------------------

/// Recent reflections block with citation breadcrumbs and a `(stale)` flag.
pub struct ReflectionLayer {
    store: Store,
    max_reflections: i64,
    max_tokens: Option<i64>,
}

impl ReflectionLayer {
    /// New layer over `store` (default `max_reflections = 3`).
    #[must_use]
    pub const fn new(store: Store) -> Self {
        Self {
            store,
            max_reflections: 3,
            max_tokens: None,
        }
    }

    /// Set the reflection cap (clamped at `0`).
    #[must_use]
    pub const fn with_max_reflections(mut self, n: i64) -> Self {
        self.max_reflections = if n < 0 { 0 } else { n };
        self
    }

    /// Set the token cap.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: i64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// `Option<i64>` rendered as Python would render `ctx.channel_id` in an f-string.
fn opt_int_display(value: Option<i64>) -> String {
    value.map_or_else(|| "None".to_owned(), |v| v.to_string())
}

#[async_trait]
impl Layer for ReflectionLayer {
    fn name(&self) -> &'static str {
        "reflection"
    }

    async fn build(&self, ctx: &AssemblyContext) -> String {
        if self.max_reflections <= 0 {
            return String::new();
        }
        let rows = self
            .store
            .recent_reflections(
                ctx.familiar_id.clone(),
                ctx.channel_id,
                self.max_reflections,
            )
            .await
            .unwrap_or_default();
        if rows.is_empty() {
            return String::new();
        }
        let mut all_cited: Vec<i64> = Vec::new();
        for row in &rows {
            all_cited.extend(&row.cited_fact_ids);
        }
        let stale = self
            .store
            .superseded_fact_ids(ctx.familiar_id.clone(), all_cited)
            .await
            .unwrap_or_default();

        let mut sections: Vec<String> = Vec::new();
        let mut remaining = self.max_tokens;
        for row in &rows {
            let mut citations: Vec<String> = Vec::new();
            for tid in &row.cited_turn_ids {
                citations.push(format!("T#{tid}"));
            }
            for fid in &row.cited_fact_ids {
                citations.push(format!("F#{fid}"));
            }
            let cite_block = if citations.is_empty() {
                String::new()
            } else {
                format!(" [{}]", citations.join(", "))
            };
            let stale_block = if row.cited_fact_ids.iter().any(|fid| stale.contains(fid)) {
                " (stale)"
            } else {
                ""
            };
            let mut line = format!("- {}{cite_block}{stale_block}", row.text.trim());
            if let Some(rem) = remaining {
                let cost = estimate_tokens(&line);
                if cost > rem && !sections.is_empty() {
                    break;
                }
                line = truncate_to_tokens(&line, rem);
                remaining = Some(rem - estimate_tokens(&line));
            }
            sections.push(line);
        }
        if sections.is_empty() {
            return String::new();
        }
        format!("## Recent reflections\n\n{}", sections.join("\n"))
    }

    async fn invalidation_key(&self, ctx: &AssemblyContext) -> String {
        let rows = self
            .store
            .recent_reflections(ctx.familiar_id.clone(), ctx.channel_id, 1)
            .await
            .unwrap_or_default();
        let latest_id = rows.first().map_or(0, |r| r.id);
        format!(
            "ch{}|r{latest_id}|cap{}",
            opt_int_display(ctx.channel_id),
            self.max_reflections
        )
    }
}

// ---------------------------------------------------------------------------
// Lorebook
// ---------------------------------------------------------------------------

/// One `[[entries]]` row from `lorebook.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LorebookEntry {
    /// Substrings that activate this entry (matched case-insensitively).
    pub keys: Vec<String>,
    /// Text inserted into the system prompt on hit.
    pub content: String,
    /// Higher renders first; ties keep file order.
    pub priority: i64,
    /// `true` → all keys must match (AND); default any (OR).
    pub selective: bool,
}

/// Stringify a TOML scalar the way Python `str(...)` would for lorebook content.
fn toml_value_to_string(value: &toml::Value) -> String {
    match value {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => f.to_string(),
        toml::Value::Boolean(true) => "True".to_owned(),
        toml::Value::Boolean(false) => "False".to_owned(),
        other => other.to_string(),
    }
}

/// Python-style truthiness of a TOML scalar (for the `selective` flag).
fn toml_truthy(value: &toml::Value) -> bool {
    match value {
        toml::Value::Boolean(b) => *b,
        toml::Value::Integer(i) => *i != 0,
        toml::Value::Float(f) => *f != 0.0,
        toml::Value::String(s) => !s.is_empty(),
        toml::Value::Array(a) => !a.is_empty(),
        toml::Value::Table(t) => !t.is_empty(),
        toml::Value::Datetime(_) => true,
    }
}

/// Keyword-activated authored canon (M4).
pub struct LorebookLayer {
    store: Store,
    path: PathBuf,
    recent_window: i64,
    max_entries: i64,
    max_tokens: Option<i64>,
}

impl LorebookLayer {
    /// Start building a lorebook layer over `store` reading `path`.
    #[must_use]
    pub fn builder(store: Store, path: impl Into<PathBuf>) -> LorebookBuilder {
        LorebookBuilder::new(store, path.into())
    }

    fn load_entries(&self) -> Vec<LorebookEntry> {
        let Ok(text) = std::fs::read_to_string(&self.path) else {
            return Vec::new();
        };
        let Ok(value) = toml::from_str::<toml::Value>(&text) else {
            return Vec::new();
        };
        let Some(raw_entries) = value.get("entries").and_then(toml::Value::as_array) else {
            return Vec::new();
        };
        let mut out: Vec<LorebookEntry> = Vec::new();
        for raw in raw_entries {
            let Some(table) = raw.as_table() else {
                continue;
            };
            let Some(keys_raw) = table.get("keys").and_then(toml::Value::as_array) else {
                continue;
            };
            let keys: Vec<String> = keys_raw
                .iter()
                .filter_map(toml::Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            if keys.is_empty() {
                continue;
            }
            let content = table
                .get("content")
                .map(toml_value_to_string)
                .unwrap_or_default();
            let content = content.trim().to_owned();
            if content.is_empty() {
                continue;
            }
            let priority = match table.get("priority") {
                Some(toml::Value::Integer(i)) => *i,
                _ => 0,
            };
            let selective = table.get("selective").is_some_and(toml_truthy);
            out.push(LorebookEntry {
                keys,
                content,
                priority,
                selective,
            });
        }
        out
    }

    async fn scan_text(&self, ctx: &AssemblyContext) -> String {
        let Some(channel_id) = ctx.channel_id else {
            return String::new();
        };
        let turns = self
            .store
            .recent(
                ctx.familiar_id.clone(),
                channel_id,
                self.recent_window,
                None,
                None,
            )
            .await
            .unwrap_or_default();
        turns
            .iter()
            .map(|t| t.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
            .to_lowercase()
    }

    fn matched_indices(entries: &[LorebookEntry], scan: &str) -> Vec<usize> {
        if scan.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<usize> = Vec::new();
        for (idx, entry) in entries.iter().enumerate() {
            let keys_lc: Vec<String> = entry.keys.iter().map(|k| k.to_lowercase()).collect();
            let hit = if entry.selective {
                keys_lc.iter().all(|k| scan.contains(k.as_str()))
            } else {
                keys_lc.iter().any(|k| scan.contains(k.as_str()))
            };
            if hit {
                out.push(idx);
            }
        }
        out
    }
}

#[async_trait]
impl Layer for LorebookLayer {
    fn name(&self) -> &'static str {
        "lorebook"
    }

    async fn build(&self, ctx: &AssemblyContext) -> String {
        if self.max_entries <= 0 {
            return String::new();
        }
        let entries = self.load_entries();
        if entries.is_empty() {
            return String::new();
        }
        let scan = self.scan_text(ctx).await;
        let mut idxs = Self::matched_indices(&entries, &scan);
        if idxs.is_empty() {
            return String::new();
        }
        idxs.sort_by(|&a, &b| {
            entries[b]
                .priority
                .cmp(&entries[a].priority)
                .then(a.cmp(&b))
        });
        idxs.truncate(usize::try_from(self.max_entries).unwrap_or(0));

        let mut sections: Vec<String> = Vec::new();
        let mut remaining = self.max_tokens;
        for &i in &idxs {
            let mut section = entries[i].content.clone();
            if let Some(rem) = remaining {
                let cost = estimate_tokens(&section);
                if cost > rem && !sections.is_empty() {
                    break;
                }
                section = truncate_to_tokens(&section, rem);
                remaining = Some(rem - estimate_tokens(&section));
            }
            sections.push(section);
        }
        if sections.is_empty() {
            return String::new();
        }
        format!("## Lorebook\n\n{}", sections.join("\n\n"))
    }

    async fn invalidation_key(&self, ctx: &AssemblyContext) -> String {
        let file_hash = content_hash(&self.path);
        let entries = self.load_entries();
        let scan = self.scan_text(ctx).await;
        let idxs = Self::matched_indices(&entries, &scan);
        let matched = idxs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "f{file_hash}|ch{}|m{matched}|cap{}",
            opt_int_display(ctx.channel_id),
            self.max_entries
        )
    }
}

/// Builder for [`LorebookLayer`].
pub struct LorebookBuilder {
    store: Store,
    path: PathBuf,
    recent_window: i64,
    max_entries: i64,
    max_tokens: Option<i64>,
}

impl LorebookBuilder {
    const fn new(store: Store, path: PathBuf) -> Self {
        Self {
            store,
            path,
            recent_window: 20,
            max_entries: 10,
            max_tokens: None,
        }
    }

    /// Channel scan window (floored at 1).
    #[must_use]
    pub const fn recent_window(mut self, n: i64) -> Self {
        self.recent_window = n;
        self
    }
    /// Max rendered entries (floored at 0).
    #[must_use]
    pub const fn max_entries(mut self, n: i64) -> Self {
        self.max_entries = n;
        self
    }
    /// Token cap.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: Option<i64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Build the layer.
    #[must_use]
    pub fn build(self) -> LorebookLayer {
        LorebookLayer {
            store: self.store,
            path: self.path,
            recent_window: self.recent_window.max(1),
            max_entries: self.max_entries.max(0),
            max_tokens: self.max_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Parity with Python `str.title()` (`layers.py:1001`): word boundaries fall
    // on cased characters only, so a letter after a digit begins a new word.
    #[test]
    fn title_case_matches_python_str_title() {
        assert_eq!(title_case("familiar"), "Familiar");
        assert_eq!(title_case("agent007bond"), "Agent007Bond");
        assert_eq!(title_case("3cats"), "3Cats");
        assert_eq!(title_case("my-familiar_name"), "My-Familiar_Name");
        assert_eq!(title_case(""), "");
    }

    // Parity with Python `timedelta.total_seconds()`: sub-millisecond fractions
    // of a second must survive so a gap just above a whole-second cap is not
    // truncated down onto the cap (`layers.py:392`).
    #[test]
    fn gap_seconds_keeps_microsecond_precision() {
        let base =
            DateTime::<Utc>::from_timestamp_micros(1_000_000_000_000_000).expect("valid timestamp");
        // 45.0004 s == 45_000_400 microseconds.
        let later = base + chrono::Duration::microseconds(45_000_400);
        let gap = gap_seconds(base, later);
        assert!((gap - 45.0004).abs() < 1e-9, "gap was {gap}");
        assert!(
            gap > 45.0,
            "sub-second fraction must not truncate onto the cap"
        );
    }

    // Parity with Python's empty-string-is-falsy handling in
    // `_format_channel_marker` (`layers.py:276`/`:278`).
    #[test]
    fn format_channel_marker_treats_empty_resolver_result_as_absent() {
        let empty: ChannelResolver = Arc::new(|_| Some(String::new()));
        // Empty channel name falls back to `#<id>`, not `#`.
        assert_eq!(format_channel_marker(42, Some(&empty), None), "#42");
        // Empty guild name is dropped entirely, leaving the bare channel.
        let named: ChannelResolver = Arc::new(|_| Some("general".to_owned()));
        assert_eq!(
            format_channel_marker(42, Some(&named), Some(&empty)),
            "#general"
        );
        // Non-empty guild + name renders `guild/#channel`.
        let guild: ChannelResolver = Arc::new(|_| Some("My Server".to_owned()));
        assert_eq!(
            format_channel_marker(42, Some(&named), Some(&guild)),
            "My Server/#general"
        );
        // No resolvers -> `#<id>`.
        assert_eq!(format_channel_marker(42, None, None), "#42");
    }
}
