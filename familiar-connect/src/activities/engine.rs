//! `ActivityEngine` — the global-absence state machine (subsystem 11; Python
//! `activities/engine.py`).
//!
//! `idle → active → returning → idle`. The active row is persisted in the
//! activities table (restart-safe); [`ActivityEngine::start`] reloads it and
//! re-arms the return timer (past-due returns fire at `now + floor`, never inline
//! at boot). Start is deferred: the `start_activity` tool calls
//! [`ActivityEngine::defer_start`], the responder applies it at
//! [`ActivityEngine::end_turn`] (the `FocusManager` precedent).
//!
//! While out, the gate suppresses non-pings; a real @ping at the focused channel
//! on a reachable type earns one judgment turn per author; a real reply there
//! cuts the activity short ([`ActivityEngine::notify_reply_sent`]). Her own
//! alarms pierce any absence. The return flow generates experience text on the
//! background tier, commits the return (`finish_activity`), then best-effort:
//! writes a mechanical event-fact, persists the experience as a marked assistant
//! turn, archives long absences, promotes staged turns since departure, and wakes
//! the model only with cause (missed pings). The reserved `sleep` catalog entry
//! rides the same machinery with an engine-owned wall-clock schedule and
//! dream/maintenance passes (the passes themselves owned by subsystem 04).
//!
//! ## Rust concurrency model (DESIGN §4.4 / spec 11 Rust port notes)
//!
//! The Python engine is single-threaded (GIL + event-loop affinity). Here the
//! mutable state lives behind one [`std::sync::Mutex`] `EngineState`; the sync
//! surface (`gate` / `defer_start` / `note_*` / `should_nudge`) locks briefly,
//! and the async methods lock, extract, release across `.await`, then re-lock.
//! The three background tasks (nudge loop, return timer, sleep passes) are
//! independently-cancellable [`tokio::task::JoinHandle`]s spawned via a
//! `Weak<Self>` self-reference ([`std::sync::Arc::new_cyclic`]); `stop()` aborts
//! and awaits them (swallowing everything, like the Python cancel).
//!
//! The "never raises" hierarchy is the core contract: `end_turn`,
//! `notify_reply_sent`, the nudge-loop body, `sleep_then_return`, `run_return`'s
//! per-step guards, `set_presence`, and the passes task each catch-log-continue
//! with specific recovery — every step is a `Result` match arm, never unified by
//! `?`.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration as StdDuration;

use chrono::{DateTime, Datelike, Duration, NaiveTime, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::Tz;
use serde_json::{Value, json};
use tokio::task::JoinHandle;

use crate::activities::config::{ActivitiesConfig, ActivityType, SLEEP_TYPE_ID};
use crate::bus::envelope::{Event, payload as wrap_payload};
use crate::bus::protocols::EventBus;
use crate::bus::topics::TOPIC_DISCORD_TEXT;
use crate::history::StoreError;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{
    ActivityRecord, AppendFact, AppendTurn, FactSubject, HistoryTurn, Promotion,
};
use crate::identity::{Author, ego_canonical_key, format_turn_for_transcript};
use crate::llm::{LlmClient, Message};
use crate::log_style as ls;
use crate::processors::{DiscordTextPayload, GateAction, GateDecision};
use crate::sleep::maintenance::{
    DEFAULT_PASSES, MaintenanceContext, MaintenanceRun, SleepPromptText, create_passes, run_passes,
};
use crate::sleep::opinion_formation::OpinionPlan;
use crate::tools::start_activity::{ActivityCatalogEntry, StartActivityEngine};

// return-turn display prefix (history/RAG rendering only)
/// Display prefix on return turns (history rendering only; keying is on `mode`).
pub const RETURN_TURN_MARKER_PREFIX: &str = "[returned from ";
/// `turns.mode` tag on activity-return turns — the fact-extractor skip keys on it.
pub const ACTIVITY_RETURN_MODE: &str = "activity_return";
/// `turns.mode` tag on sleep-return (dream) turns — extractor dream-frames these.
pub const SLEEP_RETURN_MODE: &str = "sleep_return";

// boot recovery: past-due return fires at now+floor, never inline
const PAST_DUE_RETURN_FLOOR_S: i64 = 20;
const STATE_LINE_NAME_LIMIT: usize = 40;
const SCAN_LIMIT: i64 = 500;
const MAX_EXCERPTS: usize = 3;
const EXCERPT_SPAN: i64 = 2;
const VISIBLE_TAIL: i64 = 10;
const EARLY_BED_MINUTES: i64 = 60;
const SCHEDULE_OVERFLOW_GRACE_MINUTES: i64 = 15;

const NUDGE_CONTENT: &str = "[idle: the channel has been quiet for a while. If nothing needs \
    your attention, you could head out and do something — start_activity. Nobody is around: call \
    silent() with it to slip away without posting a goodbye. Staying is fine too; your call.]";

const BEDTIME_NUDGE_CONTENT: &str = "[late: your sleep window has begun. Wrap up and head to bed — \
    start_activity with 'sleep'. Nobody around: call silent() with it to slip away without posting \
    a goodnight. Stay up much longer and sleep will claim you anyway.]";

const PROVENANCE_RAIL: &str = "Write a short first-person account (2-4 sentences) of the \
    experience. Experiences are about places, things, and yourself — NEVER invent claims, \
    conversations, or encounters involving real people. Plain prose, no quotation marks, no \
    preamble.";

const DREAM_RAIL: &str = "Write a short first-person dream account (2-4 sentences) — what you \
    dreamed last night, told on waking. It is openly a dream: vivid, a little strange, woven from \
    the seed and any listed stances. NEVER present dream events as real, and never invent claims, \
    conversations, or encounters involving real people. Plain prose, no quotation marks, no \
    preamble.";

const WEEKDAY_ABBR: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// ---------------------------------------------------------------------------
// Injection seams (DESIGN §4.8)
// ---------------------------------------------------------------------------

/// The subset of `FocusManager` the engine consults.
pub trait FocusLike: Send + Sync {
    /// The current focus channel for `modality` (`"text"` → text pointer).
    fn get_focus(&self, modality: &str) -> Option<i64>;
    /// The per-channel promotion cap applied at return-time staged promotion.
    fn catch_up_limit(&self) -> usize;
}

impl FocusLike for crate::focus::FocusManager {
    fn get_focus(&self, modality: &str) -> Option<i64> {
        Self::get_focus(self, modality)
    }
    fn catch_up_limit(&self) -> usize {
        Self::catch_up_limit(self)
    }
}

/// Injected clock. `now()` drives every decision; the return timer computes its
/// delay once from `now()` at arm time, then sleeps real time.
pub trait Clock: Send + Sync {
    /// The current instant.
    fn now(&self) -> DateTime<Utc>;
}

/// A production wall-clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// Injected RNG providing an inclusive-both-ends range roll (Python `randint`).
pub trait ActivityRng: Send + Sync {
    /// A value in `[lo, hi]` (both inclusive).
    fn gen_range_inclusive(&self, lo: i64, hi: i64) -> i64;
}

/// A tiny self-contained LCG (no external `rand` dep). Deterministic enough for
/// the roll; the tests that pin an exact duration inject a high-roll stub.
pub struct LcgRng {
    state: AtomicU64,
}

impl LcgRng {
    /// Seed the generator.
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self {
            state: AtomicU64::new(seed),
        }
    }
}

impl Default for LcgRng {
    fn default() -> Self {
        Self::seeded(0x2545_F491_4F6C_DD1D)
    }
}

impl ActivityRng for LcgRng {
    fn gen_range_inclusive(&self, lo: i64, hi: i64) -> i64 {
        if lo >= hi {
            return lo;
        }
        // SplitMix64-style step.
        let x = self
            .state
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
                Some(s.wrapping_add(0x9E37_79B9_7F4A_7C15))
            })
            .unwrap_or(0)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let span = u64::try_from(hi - lo + 1).unwrap_or(1);
        lo + i64::try_from(z % span).unwrap_or(0)
    }
}

/// Presence callback (`change_presence`). Cosmetic — failures are swallowed by
/// [`ActivityEngine`]. The Python "sync or async" duality collapses to async here.
pub type PresenceCb = Arc<
    dyn Fn(String, Option<String>) -> futures::future::BoxFuture<'static, anyhow::Result<()>>
        + Send
        + Sync,
>;

/// Late-bound bot-user-id provider (`run.py` wires it before Discord login).
pub type BotUserIdFn = Arc<dyn Fn() -> Option<i64> + Send + Sync>;

/// Voice-active predicate (`bool(handle.voice_runtime)`).
pub type VoiceActiveFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// The store surface the engine needs, as a trait so the "never raises"
/// hardening tests can inject a faulting impl (Python monkeypatched the store).
#[async_trait::async_trait]
pub trait ActivityStore: Send + Sync {
    /// Insert an activity row; return its id.
    async fn create_activity(
        &self,
        familiar_id: String,
        type_id: String,
        label: String,
        started_at: DateTime<Utc>,
        planned_return_at: DateTime<Utc>,
        note: Option<String>,
    ) -> anyhow::Result<i64>;
    /// Stamp the return fields on an activity row.
    async fn finish_activity(
        &self,
        activity_id: i64,
        status: String,
        actual_return_at: DateTime<Utc>,
        experience_text: Option<String>,
    ) -> anyhow::Result<()>;
    /// Persist experience prose on an activity row.
    async fn set_activity_experience(
        &self,
        activity_id: i64,
        experience_text: String,
    ) -> anyhow::Result<()>;
    /// The active (`actual_return_at IS NULL`) row, if any.
    async fn active_activity(&self, familiar_id: String) -> anyhow::Result<Option<ActivityRecord>>;
    /// The newest row of `type_id`, if any.
    async fn latest_activity(
        &self,
        familiar_id: String,
        type_id: String,
    ) -> anyhow::Result<Option<ActivityRecord>>;
    /// Append a fact.
    async fn append_fact(&self, p: AppendFact) -> anyhow::Result<()>;
    /// Append a turn.
    async fn append_turn(&self, p: AppendTurn) -> anyhow::Result<()>;
    /// The newest turn id (globally when `channel_id` is `None`).
    async fn latest_id(
        &self,
        familiar_id: String,
        channel_id: Option<i64>,
    ) -> anyhow::Result<Option<i64>>;
    /// The newest turn id at or before `ts` (global).
    async fn latest_id_at_or_before(
        &self,
        familiar_id: String,
        ts: DateTime<Utc>,
    ) -> anyhow::Result<Option<i64>>;
    /// Set the archive watermark for every channel.
    async fn set_archive_watermark_all(
        &self,
        familiar_id: String,
        turn_id: i64,
    ) -> anyhow::Result<()>;
    /// Promote staged turns since `after_turn_id`.
    async fn promote_staged_turns_since(
        &self,
        familiar_id: String,
        after_turn_id: i64,
        catch_up_limit: Option<usize>,
    ) -> anyhow::Result<Promotion>;
    /// Recent turns for a channel (oldest first).
    async fn recent(
        &self,
        familiar_id: String,
        channel_id: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<HistoryTurn>>;
    /// Fetch turns by id.
    async fn turns_by_ids(
        &self,
        familiar_id: String,
        ids: Vec<i64>,
    ) -> anyhow::Result<Vec<HistoryTurn>>;
    /// A window of turns centred on `turn_id`.
    async fn turns_around(
        &self,
        familiar_id: String,
        channel_id: i64,
        turn_id: i64,
        before: i64,
        after: i64,
    ) -> anyhow::Result<Vec<HistoryTurn>>;
    /// The concrete store for the sleep passes, or `None` (disables the passes).
    fn maintenance_store(&self) -> Option<Arc<AsyncHistoryStore>> {
        None
    }
}

/// The production [`ActivityStore`] — a thin newtype over [`AsyncHistoryStore`]
/// so `Arc<AsyncHistoryStore>` reaches the sleep passes via
/// [`ActivityStore::maintenance_store`].
pub struct RealActivityStore(pub Arc<AsyncHistoryStore>);

fn store_err(e: &StoreError) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

#[async_trait::async_trait]
impl ActivityStore for RealActivityStore {
    async fn create_activity(
        &self,
        familiar_id: String,
        type_id: String,
        label: String,
        started_at: DateTime<Utc>,
        planned_return_at: DateTime<Utc>,
        note: Option<String>,
    ) -> anyhow::Result<i64> {
        self.0
            .create_activity(
                familiar_id,
                type_id,
                label,
                started_at,
                planned_return_at,
                note,
            )
            .await
            .map_err(|e| store_err(&e))
    }
    async fn finish_activity(
        &self,
        activity_id: i64,
        status: String,
        actual_return_at: DateTime<Utc>,
        experience_text: Option<String>,
    ) -> anyhow::Result<()> {
        self.0
            .finish_activity(activity_id, status, actual_return_at, experience_text)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn set_activity_experience(
        &self,
        activity_id: i64,
        experience_text: String,
    ) -> anyhow::Result<()> {
        self.0
            .set_activity_experience(activity_id, experience_text)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn active_activity(&self, familiar_id: String) -> anyhow::Result<Option<ActivityRecord>> {
        self.0
            .active_activity(familiar_id)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn latest_activity(
        &self,
        familiar_id: String,
        type_id: String,
    ) -> anyhow::Result<Option<ActivityRecord>> {
        self.0
            .latest_activity(familiar_id, type_id)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn append_fact(&self, p: AppendFact) -> anyhow::Result<()> {
        self.0
            .append_fact(p)
            .await
            .map(|_| ())
            .map_err(|e| store_err(&e))
    }
    async fn append_turn(&self, p: AppendTurn) -> anyhow::Result<()> {
        self.0
            .append_turn(p)
            .await
            .map(|_| ())
            .map_err(|e| store_err(&e))
    }
    async fn latest_id(
        &self,
        familiar_id: String,
        channel_id: Option<i64>,
    ) -> anyhow::Result<Option<i64>> {
        self.0
            .latest_id(familiar_id, channel_id)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn latest_id_at_or_before(
        &self,
        familiar_id: String,
        ts: DateTime<Utc>,
    ) -> anyhow::Result<Option<i64>> {
        self.0
            .latest_id_at_or_before(familiar_id, ts)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn set_archive_watermark_all(
        &self,
        familiar_id: String,
        turn_id: i64,
    ) -> anyhow::Result<()> {
        self.0
            .set_archive_watermark_all(familiar_id, turn_id)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn promote_staged_turns_since(
        &self,
        familiar_id: String,
        after_turn_id: i64,
        catch_up_limit: Option<usize>,
    ) -> anyhow::Result<Promotion> {
        self.0
            .promote_staged_turns_since(familiar_id, after_turn_id, catch_up_limit)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn recent(
        &self,
        familiar_id: String,
        channel_id: i64,
        limit: i64,
    ) -> anyhow::Result<Vec<HistoryTurn>> {
        self.0
            .recent(familiar_id, channel_id, limit, None, None)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn turns_by_ids(
        &self,
        familiar_id: String,
        ids: Vec<i64>,
    ) -> anyhow::Result<Vec<HistoryTurn>> {
        self.0
            .turns_by_ids(familiar_id, ids)
            .await
            .map_err(|e| store_err(&e))
    }
    async fn turns_around(
        &self,
        familiar_id: String,
        channel_id: i64,
        turn_id: i64,
        before: i64,
        after: i64,
    ) -> anyhow::Result<Vec<HistoryTurn>> {
        self.0
            .turns_around(familiar_id, channel_id, turn_id, before, after)
            .await
            .map_err(|e| store_err(&e))
    }
    fn maintenance_store(&self) -> Option<Arc<AsyncHistoryStore>> {
        Some(Arc::clone(&self.0))
    }
}

/// Runs the sleep consolidation + opinion passes. A seam so tests inject a fake
/// (Python monkeypatched `execute_consolidation`/`execute_opinion_formation`).
#[async_trait::async_trait]
pub trait MaintenanceRunner: Send + Sync {
    /// Run the registered passes over `ctx`, returning the threaded run.
    async fn run(&self, ctx: MaintenanceContext) -> anyhow::Result<MaintenanceRun>;
}

/// The production runner: `run_passes(create_passes(DEFAULT_PASSES, ctx))`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultMaintenanceRunner;

#[async_trait::async_trait]
impl MaintenanceRunner for DefaultMaintenanceRunner {
    async fn run(&self, ctx: MaintenanceContext) -> anyhow::Result<MaintenanceRun> {
        let passes = create_passes(&DEFAULT_PASSES, &ctx)?;
        run_passes(&passes, None).await
    }
}

// ---------------------------------------------------------------------------
// gate payload
// ---------------------------------------------------------------------------

/// The gate's typed input (the Python `dict[str, Any]` payload).
///
/// `pings_bot` is tri-state: `Some(true/false)` is authoritative; `None` falls
/// back to a raw content scan for `<@id>` mentions.
#[derive(Clone, Debug, Default)]
pub struct GatePayload {
    /// Her own alarm marker — pierces any absence.
    pub alarm: bool,
    /// Raw message content (scanned when `pings_bot` is absent).
    pub content: String,
    /// Ingest-computed ping flag (authoritative when `Some`).
    pub pings_bot: Option<bool>,
    /// Arrival channel (must equal the focused text channel for a judgment).
    pub channel_id: Option<i64>,
    /// The message author (`None`/non-Author → the shared `"someone"` slot).
    pub author: Option<Author>,
}

impl GatePayload {
    /// Adapt a bus [`DiscordTextPayload`] into a gate input.
    ///
    /// NOTE: [`DiscordTextPayload`] carries no `alarm` field (the landed struct
    /// predates this feature), so alarm-piercing does not reach the gate through
    /// the responder path yet — a shared-file request to add `alarm: bool` is
    /// filed. Synthetic wakes (`wake`) omit the ping flag → content scan.
    #[must_use]
    pub fn from_discord(p: &DiscordTextPayload) -> Self {
        Self {
            alarm: false,
            content: p.content.clone(),
            pings_bot: if p.wake { None } else { Some(p.pings_bot) },
            channel_id: Some(p.channel_id),
            author: p.author.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct StagedStart {
    activity_type: ActivityType,
    note: Option<String>,
    duration_minutes: i64,
}

#[derive(Default)]
struct EngineState {
    active: Option<ActivityRecord>,
    staged: Option<StagedStart>,
    return_task: Option<JoinHandle<()>>,
    returning: bool,
    departure_channel_id: Option<i64>,
    departure_turn_id: Option<i64>,
    missed_ping_turn_ids: HashSet<i64>,
    judged_author_keys: HashSet<String>,
    last_traffic: Option<DateTime<Utc>>,
    last_nudge: Option<DateTime<Utc>>,
    last_return: Option<DateTime<Utc>>,
    nudge_task: Option<JoinHandle<()>>,
    bedtime_nudge_occ: Option<DateTime<Utc>>,
    last_opinion_plan: Option<OpinionPlan>,
    sleep_passes_task: Option<JoinHandle<()>>,
    #[cfg(test)]
    armed_return_ats: Vec<DateTime<Utc>>,
}

/// Construction inputs for [`ActivityEngine::new`].
pub struct EngineParams {
    /// The store seam.
    pub store: Arc<dyn ActivityStore>,
    /// The activity catalog + knobs.
    pub config: ActivitiesConfig,
    /// The background-slot LLM (the only slot the engine uses).
    pub background_llm: Arc<dyn LlmClient>,
    /// The bus (synthetic wakes publish here).
    pub bus: Arc<dyn EventBus>,
    /// The focus seam.
    pub focus: Arc<dyn FocusLike>,
    /// The presence callback.
    pub presence_cb: PresenceCb,
    /// The familiar id.
    pub familiar_id: String,
    /// IANA display timezone.
    pub display_tz: String,
    /// Late-bound bot user id.
    pub bot_user_id: BotUserIdFn,
    /// The sleep wall-clock window (from `character.toml [sleep]`); `None`
    /// disarms the schedule.
    pub sleep_window: Option<(NaiveTime, NaiveTime)>,
    /// Grace minutes before the force-sleep backstop.
    pub sleep_grace_minutes: i64,
    /// Voice-active predicate.
    pub voice_active_fn: VoiceActiveFn,
    /// The injected clock.
    pub clock: Arc<dyn Clock>,
    /// The injected RNG.
    pub rng: Arc<dyn ActivityRng>,
    /// Nudge-loop period.
    pub nudge_tick: StdDuration,
    /// Display name override (defaults to `familiar_id.title()`).
    pub familiar_display_name: Option<String>,
    /// Whether the sleep consolidation/opinion passes run.
    pub sleep_passes_enabled: bool,
    /// One-shot authored first-dream file.
    pub seed_dream_path: Option<PathBuf>,
    /// Config-sourced sleep prompt text.
    pub sleep_prompts: SleepPromptText,
    /// The sleep-passes runner seam.
    pub maintenance_runner: Arc<dyn MaintenanceRunner>,
}

/// Per-familiar absence controller — DB-backed, tokio-driven.
pub struct ActivityEngine {
    store: Arc<dyn ActivityStore>,
    config: ActivitiesConfig,
    background_llm: Arc<dyn LlmClient>,
    bus: Arc<dyn EventBus>,
    focus: Arc<dyn FocusLike>,
    presence_cb: PresenceCb,
    familiar_id: String,
    tz: Tz,
    display_tz_name: String,
    display_name: String,
    sleep_window: Option<(NaiveTime, NaiveTime)>,
    sleep_grace_minutes: i64,
    sleep_passes_enabled: bool,
    seed_dream_path: Option<PathBuf>,
    sleep_prompts: SleepPromptText,
    bot_user_id_fn: BotUserIdFn,
    voice_active_fn: VoiceActiveFn,
    nudge_tick: StdDuration,
    clock: Arc<dyn Clock>,
    rng: Arc<dyn ActivityRng>,
    maintenance_runner: Arc<dyn MaintenanceRunner>,
    state: Mutex<EngineState>,
    weak_self: Weak<Self>,
}

fn author_label(author: Option<&Author>) -> String {
    author.map_or_else(|| "someone".to_owned(), Author::label)
}

const fn away_status(activity_type: Option<&ActivityType>) -> &'static str {
    match activity_type {
        Some(t) if t.reachable => "idle",
        _ => "dnd",
    }
}

const fn daypart(hour: u32) -> &'static str {
    match hour {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=21 => "evening",
        _ => "night",
    }
}

/// Python `str.capitalize()`: first char upper, the rest lowercase.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
    })
}

fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

fn schedule_message(at: &ActivityType) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(days) = &at.active_days {
        parts.push(
            days.iter()
                .map(|d| WEEKDAY_ABBR[*d as usize])
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    if let Some((start, end)) = at.active_hours {
        parts.push(format!("{}-{}", start.format("%H:%M"), end.format("%H:%M")));
    }
    format!("{} is only available {}", at.label, parts.join(", "))
}

fn seed_dream_consumed_name(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let name = path.extension().and_then(|s| s.to_str()).map_or_else(
        || format!("{stem}.consumed"),
        |ext| format!("{stem}.consumed.{ext}"),
    );
    path.with_file_name(name)
}

async fn cancel_task(task: Option<JoinHandle<()>>) {
    if let Some(task) = task {
        task.abort();
        let _ = task.await;
    }
}

impl ActivityEngine {
    /// Construct an engine (self-referential via `Arc::new_cyclic`).
    #[must_use]
    pub fn new(params: EngineParams) -> Arc<Self> {
        let tz: Tz = params.display_tz.parse().unwrap_or(chrono_tz::UTC);
        let display_name = params
            .familiar_display_name
            .unwrap_or_else(|| title_case(&params.familiar_id));
        let now = params.clock.now();
        Arc::new_cyclic(|weak| Self {
            store: params.store,
            config: params.config,
            background_llm: params.background_llm,
            bus: params.bus,
            focus: params.focus,
            presence_cb: params.presence_cb,
            familiar_id: params.familiar_id,
            tz,
            display_tz_name: params.display_tz,
            display_name,
            sleep_window: params.sleep_window,
            sleep_grace_minutes: params.sleep_grace_minutes,
            sleep_passes_enabled: params.sleep_passes_enabled,
            seed_dream_path: params.seed_dream_path,
            sleep_prompts: params.sleep_prompts,
            bot_user_id_fn: params.bot_user_id,
            voice_active_fn: params.voice_active_fn,
            nudge_tick: params.nudge_tick,
            clock: params.clock,
            rng: params.rng,
            maintenance_runner: params.maintenance_runner,
            state: Mutex::new(EngineState {
                last_traffic: Some(now),
                ..EngineState::default()
            }),
            weak_self: weak.clone(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, EngineState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn strong(&self) -> Arc<Self> {
        self.weak_self
            .upgrade()
            .expect("engine dropped while a task holds it")
    }

    fn type_for(&self, type_id: &str) -> Option<ActivityType> {
        self.config
            .catalog
            .iter()
            .find(|t| t.id == type_id)
            .cloned()
    }

    // -- lifecycle ----------------------------------------------------------

    /// Reload the active row; re-arm the return timer (past-due ⇒ `now+floor`).
    /// Arms the idle-nudge loop (idempotent — a second call returns early).
    pub async fn start(&self) {
        {
            let mut st = self.lock();
            if st.nudge_task.is_some() {
                return;
            }
            let engine = self.strong();
            st.nudge_task = Some(tokio::spawn(async move { engine.nudge_loop().await }));
        }
        let row = self
            .store
            .active_activity(self.familiar_id.clone())
            .await
            .ok()
            .flatten();
        let Some(row) = row else {
            return;
        };
        {
            self.lock().active = Some(row.clone());
        }
        let departure_channel = self.focus.get_focus("text");
        let departure_turn = self
            .store
            .latest_id_at_or_before(self.familiar_id.clone(), row.started_at)
            .await
            .ok()
            .flatten();
        {
            let mut st = self.lock();
            st.departure_channel_id = departure_channel;
            st.departure_turn_id = departure_turn;
        }
        self.set_presence(
            away_status(self.type_for(&row.type_id).as_ref()),
            Some(&row.label),
        )
        .await;
        if row.planned_return_at <= self.clock.now() {
            self.arm_return_timer(self.clock.now() + Duration::seconds(PAST_DUE_RETURN_FLOOR_S));
        } else {
            self.arm_return_timer(row.planned_return_at);
        }
    }

    /// Re-issue away presence for an in-flight activity; idle ⇒ no-op.
    pub async fn resync_presence(&self) {
        let active = { self.lock().active.clone() };
        let Some(active) = active else {
            return;
        };
        self.set_presence(
            away_status(self.type_for(&active.type_id).as_ref()),
            Some(&active.label),
        )
        .await;
    }

    /// Cancel the return timer + nudge loop + sleep passes (DB untouched).
    pub async fn stop(&self) {
        self.cancel_return_timer().await;
        let nudge = { self.lock().nudge_task.take() };
        cancel_task(nudge).await;
        let passes = { self.lock().sleep_passes_task.take() };
        cancel_task(passes).await;
    }

    async fn cancel_return_timer(&self) {
        let task = { self.lock().return_task.take() };
        cancel_task(task).await;
    }

    /// Whether the return timer is armed and pending.
    #[must_use]
    pub fn return_timer_armed(&self) -> bool {
        let st = self.lock();
        st.return_task.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Whether the idle-nudge loop is armed.
    #[must_use]
    pub fn nudge_loop_armed(&self) -> bool {
        let st = self.lock();
        st.nudge_task.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// The current activity row, or `None` when idle.
    #[must_use]
    pub fn active(&self) -> Option<ActivityRecord> {
        self.lock().active.clone()
    }

    /// The configured activity types.
    #[must_use]
    pub fn catalog(&self) -> Vec<ActivityType> {
        self.config.catalog.clone()
    }

    // -- start path (defer at tool call, apply at end_turn) -----------------

    /// Stage an activity start; returns the ack/error JSON for the model.
    pub fn defer_start(&self, type_id: &str, note: Option<&str>) -> Value {
        if (self.voice_active_fn)() {
            return json!({"error": "can't head out while in a voice channel"});
        }
        let already_out = {
            let st = self.lock();
            if st.active.is_some() || st.staged.is_some() {
                Some(
                    st.active
                        .as_ref()
                        .map_or_else(|| "another activity".to_owned(), |a| a.label.clone()),
                )
            } else {
                None
            }
        };
        if let Some(label) = already_out {
            return json!({"error": format!("already out ({label}) — finish that first")});
        }
        let Some(activity_type) = self.type_for(type_id) else {
            let valid = self
                .config
                .catalog
                .iter()
                .map(|t| t.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return json!({"error": format!("unknown activity type '{type_id}'; valid: {valid}")});
        };
        if let Some(msg) = self.schedule_violation(&activity_type) {
            return json!({ "error": msg });
        }
        let now = self.clock.now();
        let duration = if let Some(window) = self.window_for(&activity_type) {
            let (start, end) = self.window_occurrence(now, window);
            if start - now > Duration::minutes(EARLY_BED_MINUTES) {
                let local_start = start.with_timezone(&self.tz).format("%H:%M");
                return json!({"error": format!(
                    "not bedtime — the sleep window starts at {local_start}; head to bed within the hour before it"
                )});
            }
            ((end - now).num_seconds() / 60).max(1)
        } else {
            let Some((lo, mut hi)) = activity_type.duration_minutes else {
                return json!({"error": format!("activity type '{type_id}' has no duration")});
            };
            if let Some(hours) = activity_type.active_hours {
                let (_ws, win_end) = self.window_occurrence(now, hours);
                let grace = Duration::minutes(SCHEDULE_OVERFLOW_GRACE_MINUTES);
                let room = (win_end + grace - now).num_seconds() / 60;
                if room < lo {
                    return json!({"error": format!(
                        "not enough time before the {} window closes — head out earlier",
                        activity_type.label
                    )});
                }
                hi = hi.min(room);
            }
            self.rng.gen_range_inclusive(lo, hi)
        };
        let label = activity_type.label.clone();
        {
            self.lock().staged = Some(StagedStart {
                activity_type,
                note: note.map(str::to_owned),
                duration_minutes: duration,
            });
        }
        json!({
            "ack": "ok",
            "label": label,
            "duration_minutes": duration,
            "note": format!("you'll head out on the {label} after this reply, back in roughly {duration} minutes"),
        })
    }

    /// Apply a staged start (the `FocusManager` deferral pattern). Never raises
    /// into the responder turn.
    pub async fn end_turn(&self) {
        let staged = { self.lock().staged.take() };
        let Some(staged) = staged else {
            return;
        };
        let result: anyhow::Result<()> = async {
            let now = self.clock.now();
            let planned = if let Some(window) = self.window_for(&staged.activity_type) {
                self.window_occurrence(now, window).1
            } else {
                now + Duration::minutes(staged.duration_minutes)
            };
            self.begin_activity(staged.activity_type.clone(), staged.note.clone(), planned)
                .await
        }
        .await;
        if let Err(e) = result {
            tracing::error!(
                "{} deferred start failed {} {}",
                ls::tag("Activity", ls::G),
                ls::kv_styled("label", &staged.activity_type.label, ls::W, ls::LW),
                ls::kv_styled("error", &ls::trunc(&e.to_string(), 120), ls::W, ls::LW),
            );
            // Row committed ⇒ she left — make sure the timer still brings her
            // back; row missing ⇒ she never left.
            let planned = { self.lock().active.as_ref().map(|a| a.planned_return_at) };
            if let Some(planned) = planned {
                if !self.return_timer_armed() {
                    self.arm_return_timer(planned);
                }
            }
        }
    }

    async fn begin_activity(
        &self,
        activity_type: ActivityType,
        note: Option<String>,
        planned_return_at: DateTime<Utc>,
    ) -> anyhow::Result<()> {
        let now = self.clock.now();
        let activity_id = self
            .store
            .create_activity(
                self.familiar_id.clone(),
                activity_type.id.clone(),
                activity_type.label.clone(),
                now,
                planned_return_at,
                note.clone(),
            )
            .await?;
        let record = ActivityRecord {
            id: activity_id,
            familiar_id: self.familiar_id.clone(),
            type_id: activity_type.id.clone(),
            label: activity_type.label.clone(),
            started_at: now,
            planned_return_at,
            note: note.clone(),
            status: None,
            actual_return_at: None,
            experience_text: None,
        };
        {
            self.lock().active = Some(record);
        }
        if activity_type.id == SLEEP_TYPE_ID {
            self.kick_sleep_passes();
        }
        {
            let mut st = self.lock();
            st.missed_ping_turn_ids.clear();
            st.judged_author_keys.clear();
        }
        let departure_channel = self.focus.get_focus("text");
        let departure_turn = self.store.latest_id(self.familiar_id.clone(), None).await?;
        {
            let mut st = self.lock();
            st.departure_channel_id = departure_channel;
            st.departure_turn_id = departure_turn;
        }
        self.set_presence(
            away_status(Some(&activity_type)),
            Some(&activity_type.label),
        )
        .await;
        self.arm_return_timer(planned_return_at);
        Ok(())
    }

    // -- gate ---------------------------------------------------------------

    /// Decide how to handle an inbound text event while (not) absent.
    #[must_use]
    pub fn gate(&self, payload: &GatePayload) -> GateDecision {
        // Her own alarm pierces any absence.
        if payload.alarm {
            return GateDecision {
                action: GateAction::Normal,
                state_line: None,
            };
        }
        let mut st = self.lock();
        let Some(active) = st.active.clone() else {
            return GateDecision::default();
        };
        // Mid-return corpse state must not suppress the return wake.
        if st.returning {
            return GateDecision {
                action: GateAction::Normal,
                state_line: None,
            };
        }
        let is_ping = payload
            .pings_bot
            .unwrap_or_else(|| self.is_ping(&payload.content));
        if !is_ping {
            return suppress();
        }
        let Some(activity_type) = self.type_for(&active.type_id) else {
            return suppress();
        };
        if !activity_type.reachable {
            return suppress();
        }
        if payload.channel_id != self.focus.get_focus("text") {
            return suppress();
        }
        let author_key = payload
            .author
            .as_ref()
            .map_or_else(|| "someone".to_owned(), Author::canonical_key);
        if st.judged_author_keys.contains(&author_key) {
            return suppress();
        }
        st.judged_author_keys.insert(author_key);
        drop(st);
        let elapsed = ((self.clock.now() - active.started_at).num_seconds() / 60).max(0);
        let name = ls::trunc(
            &author_label(payload.author.as_ref()),
            STATE_LINE_NAME_LIMIT,
        );
        let state_line = format!(
            "You are {elapsed} min into {} — {name} pinged you. Replying means heading back; \
             silent() means you stay out. You are already out — do not call start_activity.",
            active.label
        );
        GateDecision {
            action: GateAction::Judgment,
            state_line: Some(state_line),
        }
    }

    /// A judgment turn produced a real reply ⇒ cut-short return. Never raises.
    pub async fn notify_reply_sent(&self) {
        {
            let st = self.lock();
            if st.active.is_none() || st.returning {
                return;
            }
        }
        self.cancel_return_timer().await;
        self.run_return("cut_short").await;
    }

    // -- idle nudge ---------------------------------------------------------

    /// Idle-nudge eligibility — quiet channel, gap elapsed, in hours.
    #[must_use]
    pub fn should_nudge(&self, now: DateTime<Utc>) -> bool {
        {
            let st = self.lock();
            if st.active.is_some() || st.staged.is_some() || st.returning {
                return false;
            }
            let idle = Duration::minutes(self.config.idle_nudge_minutes);
            if let Some(last_nudge) = st.last_nudge {
                if now - last_nudge < idle {
                    return false;
                }
            }
            if let Some(last_traffic) = st.last_traffic {
                if now - last_traffic < idle {
                    return false;
                }
            }
            let gap = Duration::minutes(self.config.min_gap_minutes);
            if let Some(last_return) = st.last_return {
                if now - last_return < gap {
                    return false;
                }
            }
        }
        self.in_active_hours(now)
    }

    /// Record a nudge timestamp to start the debounce window.
    pub fn mark_nudge_pending(&self) {
        self.lock().last_nudge = Some(self.clock.now());
    }

    /// Record subscribed-channel traffic; resets the quiet clock.
    pub fn note_traffic(&self) {
        self.lock().last_traffic = Some(self.clock.now());
    }

    /// Record a live-gated ping for the at-return wake.
    pub fn note_missed_ping(&self, turn_id: i64) {
        self.lock().missed_ping_turn_ids.insert(turn_id);
    }

    async fn nudge_loop(&self) {
        loop {
            tokio::time::sleep(self.nudge_tick).await;
            self.sleep_schedule_tick(self.clock.now()).await;
            if self.should_nudge(self.clock.now()) {
                self.publish_nudge().await;
            }
        }
    }

    async fn publish_nudge(&self) {
        let Some(channel_id) = self.focus.get_focus("text") else {
            return;
        };
        self.mark_nudge_pending();
        self.publish_synthetic(channel_id, NUDGE_CONTENT, "activity-nudge")
            .await;
        tracing::info!(
            "{} nudge {}",
            ls::tag("\u{1f6b6} Activity", ls::G),
            ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LW),
        );
    }

    fn in_active_hours(&self, now: DateTime<Utc>) -> bool {
        self.config
            .active_hours
            .is_none_or(|window| self.local_time_in_window(now, window))
    }

    fn local_time_in_window(&self, now: DateTime<Utc>, window: (NaiveTime, NaiveTime)) -> bool {
        let (start, end) = window;
        let local = now.with_timezone(&self.tz).time();
        if start < end {
            start <= local && local < end
        } else {
            local >= start || local < end
        }
    }

    // -- sleep window -------------------------------------------------------

    fn sleep_type(&self) -> Option<ActivityType> {
        self.sleep_window?;
        self.type_for(SLEEP_TYPE_ID)
    }

    fn window_for(&self, activity_type: &ActivityType) -> Option<(NaiveTime, NaiveTime)> {
        if activity_type.id != SLEEP_TYPE_ID {
            return None;
        }
        self.sleep_window
    }

    fn localize(&self, naive: chrono::NaiveDateTime) -> DateTime<Tz> {
        self.tz
            .from_local_datetime(&naive)
            .single()
            .or_else(|| self.tz.from_local_datetime(&naive).earliest())
            .or_else(|| self.tz.from_local_datetime(&naive).latest())
            .unwrap_or_else(|| Utc.from_utc_datetime(&naive).with_timezone(&self.tz))
    }

    #[allow(
        clippy::nonminimal_bool,
        reason = "mirrors the Python `start <= now < end or start > now` occurrence test verbatim"
    )]
    fn window_occurrence(
        &self,
        now: DateTime<Utc>,
        window: (NaiveTime, NaiveTime),
    ) -> (DateTime<Utc>, DateTime<Utc>) {
        let (win_start, win_end) = window;
        let local = now.with_timezone(&self.tz);
        let date = local.date_naive();
        let raw = (date.and_time(win_end) - date.and_time(win_start)).num_seconds();
        let length = Duration::seconds(raw.rem_euclid(86_400));
        for offset in [-1i64, 0, 1] {
            let d = date + Duration::days(offset);
            let naive_start = d.and_time(win_start);
            let start_local = self.localize(naive_start);
            // Python computes `end = start + length` on a tz-AWARE datetime,
            // which is wall-clock field arithmetic (offset re-resolved for the
            // end wall time) — NOT the absolute `DateTime<Tz> + Duration` chrono
            // would give. Add `length` to the naive value, then re-localize so a
            // DST fold between start and end shifts the offset, matching Python.
            let end_local = self.localize(naive_start + length);
            if (start_local <= local && local < end_local) || start_local > local {
                return (
                    start_local.with_timezone(&Utc),
                    end_local.with_timezone(&Utc),
                );
            }
        }
        let naive_start = date.and_time(win_start);
        let start_local = self.localize(naive_start);
        (
            start_local.with_timezone(&Utc),
            self.localize(naive_start + length).with_timezone(&Utc),
        )
    }

    async fn sleep_schedule_tick(&self, now: DateTime<Utc>) {
        let Some(entry) = self.sleep_type() else {
            return;
        };
        let Some(window) = self.sleep_window else {
            return;
        };
        {
            let st = self.lock();
            if st.active.is_some() || st.staged.is_some() || st.returning {
                return;
            }
        }
        let (start, end) = self.window_occurrence(now, window);
        if now < start {
            return;
        }
        if self.slept_this_window(&entry, start).await {
            return;
        }
        if now >= start + Duration::minutes(self.sleep_grace_minutes) {
            self.force_sleep(&entry, end).await;
        } else {
            let already = { self.lock().bedtime_nudge_occ == Some(start) };
            if !already {
                self.publish_bedtime_nudge(start).await;
            }
        }
    }

    async fn slept_this_window(
        &self,
        entry: &ActivityType,
        occurrence_start: DateTime<Utc>,
    ) -> bool {
        let row = self
            .store
            .latest_activity(self.familiar_id.clone(), entry.id.clone())
            .await
            .ok()
            .flatten();
        row.is_some_and(|r| r.started_at >= occurrence_start)
    }

    async fn force_sleep(&self, entry: &ActivityType, planned_return_at: DateTime<Utc>) {
        if let Err(e) = self
            .begin_activity(entry.clone(), None, planned_return_at)
            .await
        {
            tracing::error!(
                "{} force sleep failed {}",
                ls::tag("Activity", ls::G),
                ls::kv_styled("error", &ls::trunc(&e.to_string(), 120), ls::W, ls::LW),
            );
            return;
        }
        tracing::info!(
            "{} force sleep {}",
            ls::tag("\u{1f319} Activity", ls::G),
            ls::kv_styled("wake", &planned_return_at.to_rfc3339(), ls::W, ls::LW),
        );
    }

    async fn publish_bedtime_nudge(&self, occurrence_start: DateTime<Utc>) {
        let Some(channel_id) = self.focus.get_focus("text") else {
            return;
        };
        {
            self.lock().bedtime_nudge_occ = Some(occurrence_start);
        }
        self.publish_synthetic(channel_id, BEDTIME_NUDGE_CONTENT, "bedtime")
            .await;
        tracing::info!(
            "{} bedtime nudge {}",
            ls::tag("\u{1f319} Activity", ls::G),
            ls::kv_styled("channel", &channel_id.to_string(), ls::W, ls::LW),
        );
    }

    // -- sleep passes + dream prose -----------------------------------------

    fn kick_sleep_passes(&self) {
        {
            self.lock().last_opinion_plan = None;
        }
        let engine = self.strong();
        let task = tokio::spawn(async move { engine.run_sleep_passes().await });
        self.lock().sleep_passes_task = Some(task);
    }

    async fn run_sleep_passes(&self) {
        let active = { self.lock().active.clone() };
        if !self.sleep_passes_enabled {
            return;
        }
        let Some(maintenance_store) = self.store.maintenance_store() else {
            // No concrete store (fault-injection fake) — passes cannot run.
            return;
        };
        let ctx = MaintenanceContext::new(
            maintenance_store,
            Arc::clone(&self.background_llm),
            self.familiar_id.clone(),
            Some(self.display_name.clone()),
            self.display_tz_name.clone(),
            true,
        )
        .with_prompts(self.sleep_prompts.clone());
        let run = match self.maintenance_runner.run(ctx).await {
            Ok(run) => run,
            Err(e) => {
                tracing::error!(
                    "{} sleep passes failed {}",
                    ls::tag("Activity", ls::G),
                    ls::kv_styled("error", &ls::trunc(&e.to_string(), 120), ls::W, ls::LW),
                );
                return;
            }
        };
        {
            self.lock().last_opinion_plan = run.opinion_plan;
        }
        let Some(active) = active else {
            return;
        };
        if active.type_id != SLEEP_TYPE_ID {
            return;
        }
        {
            let st = self.lock();
            if st.active.as_ref().map(|a| a.id) != Some(active.id) {
                // A return already finished/replaced this row — late passes no-op.
                return;
            }
        }
        let persist: anyhow::Result<()> = async {
            let prose = self
                .generate_dream_prose(&active, self.type_for(&active.type_id))
                .await;
            self.append_dream_journal_fact(&active, &prose).await?;
            self.store
                .set_activity_experience(active.id, prose.clone())
                .await?;
            {
                let mut st = self.lock();
                if let Some(a) = st.active.as_mut() {
                    if a.id == active.id {
                        a.experience_text = Some(prose);
                    }
                }
            }
            Ok(())
        }
        .await;
        if let Err(e) = persist {
            tracing::error!(
                "{} dream persist failed {}",
                ls::tag("Activity", ls::G),
                ls::kv_styled("error", &ls::trunc(&e.to_string(), 120), ls::W, ls::LW),
            );
        }
    }

    // -- returning ----------------------------------------------------------

    fn arm_return_timer(&self, planned_return_at: DateTime<Utc>) {
        let engine = self.strong();
        let task = tokio::spawn(async move { engine.sleep_then_return(planned_return_at).await });
        let mut st = self.lock();
        st.return_task = Some(task);
        #[cfg(test)]
        st.armed_return_ats.push(planned_return_at);
    }

    async fn sleep_then_return(&self, planned_return_at: DateTime<Utc>) {
        let delay = (planned_return_at - self.clock.now()).num_milliseconds();
        if delay > 0 {
            tokio::time::sleep(StdDuration::from_millis(u64::try_from(delay).unwrap_or(0))).await;
        }
        self.run_return("completed").await;
    }

    async fn run_return(&self, status: &str) {
        let snapshot = {
            let mut st = self.lock();
            if st.active.is_none() || st.returning {
                return;
            }
            st.returning = true;
            let active = st.active.clone().expect("checked is_some");
            (active, st.departure_channel_id, st.departure_turn_id)
        };
        let (active, departure_channel, departure_turn) = snapshot;
        let now = self.clock.now();
        self.run_return_body(&active, status, now, departure_channel, departure_turn)
            .await;
        // finally — idempotent; exception paths land here with state cleared.
        {
            let mut st = self.lock();
            clear_absence_state(&mut st, now);
            st.returning = false;
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one cohesive best-effort return sequence"
    )]
    async fn run_return_body(
        &self,
        active: &ActivityRecord,
        status: &str,
        now: DateTime<Utc>,
        departure_channel: Option<i64>,
        departure_turn: Option<i64>,
    ) {
        let activity_type = self.type_for(&active.type_id);
        let is_sleep = active.type_id == SLEEP_TYPE_ID;
        let cut_short = status == "cut_short";
        let mut dream_already_journaled = false;
        let experience = if is_sleep {
            if let Some(exp) = active.experience_text.clone() {
                dream_already_journaled = true;
                exp
            } else {
                self.generate_dream_prose(active, activity_type.clone())
                    .await
            }
        } else {
            self.generate_experience(active, activity_type.clone(), cut_short)
                .await
        };
        // COMMIT — she is back after this line.
        if let Err(e) = self
            .store
            .finish_activity(active.id, status.to_owned(), now, Some(experience.clone()))
            .await
        {
            tracing::error!(
                "{} return commit failed {}",
                ls::tag("Activity", ls::G),
                ls::kv_styled("error", &ls::trunc(&e.to_string(), 120), ls::W, ls::LW),
            );
            return;
        }
        // (a) mechanical event-fact — no LLM
        if let Err(e) = self
            .store
            .append_fact(
                AppendFact::new(
                    self.familiar_id.clone(),
                    departure_channel,
                    self.event_fact_text(active),
                    vec![],
                )
                .valid_from(active.started_at),
            )
            .await
        {
            self.log_return_step_failed("event_fact", &e);
        }
        // (b) experience as a marked assistant turn
        if let Some(channel_id) = departure_channel {
            let content = format!("{RETURN_TURN_MARKER_PREFIX}{}] {experience}", active.label);
            let mode = if is_sleep {
                SLEEP_RETURN_MODE
            } else {
                ACTIVITY_RETURN_MODE
            };
            if let Err(e) = self
                .store
                .append_turn(
                    AppendTurn::new(self.familiar_id.clone(), channel_id, "assistant", content)
                        .mode(mode),
                )
                .await
            {
                self.log_return_step_failed("return_turn", &e);
            }
        }
        // (c) dream-journal stopgap (skipped when passes already journaled)
        if is_sleep && !dream_already_journaled {
            if let Err(e) = self.append_dream_journal_fact(active, &experience).await {
                self.log_return_step_failed("dream_journal", &e);
            }
        }
        // (d) global archive watermark for long absences
        {
            let absence = now - active.started_at;
            let threshold = Duration::minutes(self.config.archive_after_minutes);
            if absence >= threshold {
                if let Some(turn_id) = departure_turn {
                    if let Err(e) = self
                        .store
                        .set_archive_watermark_all(self.familiar_id.clone(), turn_id)
                        .await
                    {
                        self.log_return_step_failed("archive_watermark", &e);
                    }
                }
            }
        }
        // (e) staged-turn promotion
        if let Some(turn_id) = departure_turn {
            match self
                .store
                .promote_staged_turns_since(
                    self.familiar_id.clone(),
                    turn_id,
                    Some(self.focus.catch_up_limit()),
                )
                .await
            {
                Ok(promo) => {
                    if promo.consumed > 0 || promo.missed > 0 {
                        tracing::info!(
                            "{} promoted staged {} {}",
                            ls::tag("Activity", ls::G),
                            ls::kv_styled("turns", &promo.consumed.to_string(), ls::W, ls::LW),
                            ls::kv_styled("missed", &promo.missed.to_string(), ls::W, ls::LY),
                        );
                    }
                }
                Err(e) => self.log_return_step_failed("staged_promotion", &e),
            }
        }
        // (f) missed-ping collection (`collect_missed_pings` never errs here —
        // its store reads degrade to empty, matching the "reads never raise"
        // contract)
        let missed_pings = if let Some(channel_id) = departure_channel {
            self.collect_missed_pings(channel_id).await
        } else {
            Vec::new()
        };
        // C1: clear gate-consulted state BEFORE the wake publish.
        {
            let mut st = self.lock();
            clear_absence_state(&mut st, now);
        }
        if !missed_pings.is_empty() {
            if let Some(channel_id) = departure_channel {
                self.publish_wake(channel_id, &active.label, &missed_pings)
                    .await;
            }
        }
        self.set_presence("online", None).await;
        let emoji = if is_sleep { "\u{1f319}" } else { "\u{2728}" };
        tracing::info!(
            "{} returned {} {} {}",
            ls::tag(&format!("{emoji} Activity"), ls::G),
            ls::kv_styled("label", &active.label, ls::W, ls::G),
            ls::kv_styled("status", status, ls::W, ls::LW),
            ls::kv_styled(
                "missed_pings",
                &missed_pings.len().to_string(),
                ls::W,
                ls::LW
            ),
        );
    }

    #[allow(
        clippy::unused_self,
        reason = "method form keeps the return-step call sites uniform"
    )]
    fn log_return_step_failed(&self, step: &str, e: &anyhow::Error) {
        tracing::error!(
            "{} return step failed {} {}",
            ls::tag("Activity", ls::G),
            ls::kv_styled("step", step, ls::W, ls::LW),
            ls::kv_styled("error", &ls::trunc(&e.to_string(), 120), ls::W, ls::LW),
        );
    }

    async fn generate_experience(
        &self,
        active: &ActivityRecord,
        activity_type: Option<ActivityType>,
        cut_short: bool,
    ) -> String {
        let seed = activity_type
            .as_ref()
            .map_or_else(|| active.label.clone(), |t| t.seed.clone());
        let mut lines = vec![
            format!("Activity: {}", active.label),
            format!("Seed: {seed}"),
        ];
        if let Some(note) = active.note.as_deref() {
            if !note.is_empty() {
                lines.push(format!("Intent noted before leaving: {note}"));
            }
        }
        if cut_short {
            lines.push(
                "The activity was cut short — someone pinged and the return home happened early."
                    .to_owned(),
            );
        }
        let text = self.chat_text(PROVENANCE_RAIL, &lines.join("\n")).await;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            format!("Back from the {}.", active.label)
        } else {
            trimmed.to_owned()
        }
    }

    async fn generate_dream_prose(
        &self,
        active: &ActivityRecord,
        activity_type: Option<ActivityType>,
    ) -> String {
        if let Some(seeded) = self.consume_seed_dream() {
            return seeded;
        }
        let plan = { self.lock().last_opinion_plan.take() };
        let seed = activity_type
            .as_ref()
            .map_or_else(|| active.label.clone(), |t| t.seed.clone());
        let mut lines = vec![format!("Dream seed: {seed}")];
        if let Some(plan) = &plan {
            if !plan.opinions.is_empty() {
                lines.push(
                    "Stances that settled in you tonight (dream material — weave them in obliquely):".to_owned(),
                );
                lines.extend(plan.opinions.iter().map(|op| format!("- {}", op.text)));
            }
        }
        let text = self.chat_text(DREAM_RAIL, &lines.join("\n")).await;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            "Slept deep; whatever I dreamed slipped away on waking.".to_owned()
        } else {
            trimmed.to_owned()
        }
    }

    async fn chat_text(&self, system: &str, user: &str) -> String {
        match self
            .background_llm
            .chat(vec![
                Message::new("system", system),
                Message::new("user", user),
            ])
            .await
        {
            Ok(reply) => reply.content_str(),
            Err(e) => {
                tracing::warn!(
                    "{} {}",
                    ls::tag("Activity", ls::G),
                    ls::kv_styled("llm_failed", &ls::trunc(&e.to_string(), 120), ls::W, ls::LY),
                );
                String::new()
            }
        }
    }

    fn consume_seed_dream(&self) -> Option<String> {
        let path = self.seed_dream_path.as_ref()?;
        if !path.exists() {
            return None;
        }
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t.trim().to_owned(),
            Err(e) => {
                tracing::warn!(
                    "{} {}",
                    ls::tag("Activity", ls::G),
                    ls::kv_styled(
                        "seed_dream_failed",
                        &ls::trunc(&e.to_string(), 120),
                        ls::W,
                        ls::LY
                    ),
                );
                return None;
            }
        };
        if let Err(e) = std::fs::rename(path, seed_dream_consumed_name(path)) {
            tracing::warn!(
                "{} {}",
                ls::tag("Activity", ls::G),
                ls::kv_styled(
                    "seed_dream_failed",
                    &ls::trunc(&e.to_string(), 120),
                    ls::W,
                    ls::LY
                ),
            );
            return None;
        }
        tracing::info!(
            "{} seed dream consumed",
            ls::tag("\u{1f319} Activity", ls::G)
        );
        if text.is_empty() { None } else { Some(text) }
    }

    async fn append_dream_journal_fact(
        &self,
        active: &ActivityRecord,
        prose: &str,
    ) -> anyhow::Result<()> {
        let local = active.started_at.with_timezone(&self.tz);
        let subject = FactSubject {
            canonical_key: ego_canonical_key(&self.familiar_id),
            display_at_write: self.display_name.clone(),
        };
        let text = format!(
            "{} dreamed (night of {} {}): {prose}",
            self.display_name,
            local.format("%b"),
            local.day()
        );
        self.store
            .append_fact(
                AppendFact::new(self.familiar_id.clone(), None, text, vec![])
                    .subjects(vec![subject])
                    .valid_from(active.started_at),
            )
            .await
    }

    fn event_fact_text(&self, active: &ActivityRecord) -> String {
        let local = active.started_at.with_timezone(&self.tz);
        format!(
            "{} spent {} {} {} {}",
            capitalize(&self.familiar_id),
            local.format("%b"),
            local.day(),
            daypart(local.hour()),
            active.label,
        )
    }

    async fn collect_missed_pings(&self, channel_id: i64) -> Vec<HistoryTurn> {
        let scanned = self.missed_pings(channel_id).await;
        let scanned_ids: HashSet<i64> = scanned.iter().map(|t| t.id).collect();
        let noted = { std::mem::take(&mut self.lock().missed_ping_turn_ids) };
        let noted_only: Vec<i64> = noted
            .into_iter()
            .filter(|id| !scanned_ids.contains(id))
            .collect();
        if noted_only.is_empty() {
            return scanned;
        }
        let fetched = self
            .store
            .turns_by_ids(self.familiar_id.clone(), noted_only)
            .await
            .unwrap_or_default();
        let mut merged: BTreeMap<i64, HistoryTurn> = BTreeMap::new();
        for t in scanned.into_iter().chain(fetched) {
            merged.insert(t.id, t);
        }
        merged.into_values().collect()
    }

    async fn missed_pings(&self, channel_id: i64) -> Vec<HistoryTurn> {
        let departure_id = { self.lock().departure_turn_id };
        let turns = self
            .store
            .recent(self.familiar_id.clone(), channel_id, SCAN_LIMIT)
            .await
            .unwrap_or_default();
        turns
            .into_iter()
            .filter(|t| {
                departure_id.is_none_or(|id| t.id > id)
                    && t.role == "user"
                    && self.is_ping(&t.content)
            })
            .collect()
    }

    async fn publish_wake(&self, channel_id: i64, label: &str, pings: &[HistoryTurn]) {
        let content = self.wake_content(channel_id, label, pings).await;
        self.publish_synthetic(channel_id, &content, "activity-return")
            .await;
    }

    async fn publish_synthetic(&self, channel_id: i64, content: &str, turn_prefix: &str) {
        let synth_event_id = uuid::Uuid::new_v4().simple().to_string();
        let payload = DiscordTextPayload {
            familiar_id: self.familiar_id.clone(),
            channel_id,
            content: content.to_owned(),
            author: None,
            ..DiscordTextPayload::default()
        };
        let event = Event {
            event_id: synth_event_id.clone(),
            turn_id: format!("{turn_prefix}-{synth_event_id}"),
            session_id: channel_id.to_string(),
            parent_event_ids: Vec::new(),
            topic: TOPIC_DISCORD_TEXT.to_owned(),
            timestamp: self.clock.now(),
            sequence_number: 0,
            payload: wrap_payload(payload),
        };
        self.bus.publish(event).await;
    }

    async fn wake_content(&self, channel_id: i64, label: &str, pings: &[HistoryTurn]) -> String {
        let visible = self
            .store
            .recent(self.familiar_id.clone(), channel_id, VISIBLE_TAIL)
            .await
            .unwrap_or_default();
        let visible_ids: HashSet<i64> = visible.iter().map(|t| t.id).collect();
        let excerpt_ids: HashSet<i64> = pings
            .iter()
            .rev()
            .take(MAX_EXCERPTS)
            .map(|t| t.id)
            .filter(|id| !visible_ids.contains(id))
            .collect();
        let mut lines = vec![format!("[returned from {label} — missed pings while away]")];
        for ping in pings {
            let name = author_label(ping.author.as_ref());
            lines.push(format!("- {name}: {}", ls::trunc(&ping.content, 160)));
            if excerpt_ids.contains(&ping.id) {
                let anchor = if ping.channel_id == 0 {
                    channel_id
                } else {
                    ping.channel_id
                };
                let window = self
                    .store
                    .turns_around(
                        self.familiar_id.clone(),
                        anchor,
                        ping.id,
                        EXCERPT_SPAN,
                        EXCERPT_SPAN,
                    )
                    .await
                    .unwrap_or_default();
                lines.extend(window.iter().map(|t| {
                    format!(
                        "    {}",
                        format_turn_for_transcript(&t.role, t.author.as_ref(), &t.content)
                    )
                }));
            }
        }
        lines.join("\n")
    }

    fn is_ping(&self, content: &str) -> bool {
        let Some(bot_id) = (self.bot_user_id_fn)() else {
            return false;
        };
        content.contains(&format!("<@{bot_id}>")) || content.contains(&format!("<@!{bot_id}>"))
    }

    async fn set_presence(&self, status: &str, text: Option<&str>) {
        let fut = (self.presence_cb)(status.to_owned(), text.map(str::to_owned));
        if let Err(e) = fut.await {
            tracing::warn!(
                "{} presence update failed {} {}",
                ls::tag("Activity", ls::G),
                ls::kv_styled("status", status, ls::W, ls::LW),
                ls::kv_styled("error", &ls::trunc(&e.to_string(), 120), ls::W, ls::LW),
            );
        }
    }

    fn schedule_violation(&self, activity_type: &ActivityType) -> Option<String> {
        let days = activity_type.active_days.as_ref();
        let hours = activity_type.active_hours;
        if days.is_none() && hours.is_none() {
            return None;
        }
        let now = self.clock.now();
        // KNOWN LIMITATION (preserved): the weekday is taken from the calendar
        // day of `now`, so a midnight-wrapping hours window's post-midnight tail
        // is attributed to the next day's weekday. Correct for the only current
        // use (non-wrapping day+hour schedules).
        let weekday = weekday_index(now.with_timezone(&self.tz).weekday());
        let in_days = days.is_none_or(|d| d.contains(&weekday));
        let in_hours = hours.is_none_or(|h| self.local_time_in_window(now, h));
        if in_days && in_hours {
            None
        } else {
            Some(schedule_message(activity_type))
        }
    }
}

fn weekday_index(w: Weekday) -> u8 {
    u8::try_from(w.num_days_from_monday()).unwrap_or(0)
}

const fn suppress() -> GateDecision {
    GateDecision {
        action: GateAction::Suppress,
        state_line: None,
    }
}

fn clear_absence_state(st: &mut EngineState, now: DateTime<Utc>) {
    st.active = None;
    st.departure_channel_id = None;
    st.departure_turn_id = None;
    st.return_task = None;
    st.last_return = Some(now);
}

// ---------------------------------------------------------------------------
// seam-trait impls
// ---------------------------------------------------------------------------

impl StartActivityEngine for ActivityEngine {
    fn catalog(&self) -> Vec<ActivityCatalogEntry> {
        self.config
            .catalog
            .iter()
            .map(|t| ActivityCatalogEntry {
                id: t.id.clone(),
                label: t.label.clone(),
                active_days: t.active_days.as_ref().map(|d| d.iter().copied().collect()),
                active_hours: t.active_hours,
            })
            .collect()
    }

    fn is_active(&self) -> bool {
        self.lock().active.is_some()
    }

    fn defer_start(&self, type_id: &str, note: Option<&str>) -> Value {
        Self::defer_start(self, type_id, note)
    }
}

#[async_trait::async_trait]
impl crate::processors::ActivityGate for ActivityEngine {
    fn note_traffic(&self) {
        Self::note_traffic(self);
    }
    fn gate(&self, payload: &DiscordTextPayload) -> GateDecision {
        Self::gate(self, &GatePayload::from_discord(payload))
    }
    fn note_missed_ping(&self, turn_id: i64) {
        Self::note_missed_ping(self, turn_id);
    }
    async fn notify_reply_sent(&self) {
        Self::notify_reply_sent(self).await;
    }
    async fn end_turn(&self) {
        Self::end_turn(self).await;
    }
}

#[cfg(test)]
#[allow(
    clippy::default_trait_access,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "faulting-store delegation params mirror the trait's positional signature"
)]
mod tests {
    use super::*;
    use crate::bus::in_process::InProcessEventBus;
    use crate::bus::protocols::BackpressurePolicy;
    use crate::history::store::HistoryStore;
    use crate::identity::is_ego_key;
    use crate::sleep::opinion_formation::{OpinionFact, OpinionPlan};
    use chrono::TimeZone;
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration as StdDur;
    use tokio::time::timeout;

    const FAMILIAR: &str = "aria";
    const BOT_ID: i64 = 99;
    const CHANNEL: i64 = 42;

    fn dt(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }
    fn noon() -> DateTime<Utc> {
        dt(2026, 6, 12, 12, 0)
    }
    fn nt(h: u32, m: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, 0).unwrap()
    }

    // --- fakes -------------------------------------------------------------

    struct FakeClock {
        now: StdMutex<DateTime<Utc>>,
    }
    impl FakeClock {
        fn new(start: DateTime<Utc>) -> Arc<Self> {
            Arc::new(Self {
                now: StdMutex::new(start),
            })
        }
        fn advance_minutes(&self, m: i64) {
            let mut n = self.now.lock().unwrap();
            *n += Duration::minutes(m);
        }
        fn set(&self, when: DateTime<Utc>) {
            *self.now.lock().unwrap() = when;
        }
    }
    impl Clock for FakeClock {
        fn now(&self) -> DateTime<Utc> {
            *self.now.lock().unwrap()
        }
    }

    struct FakeFocus {
        channel_id: Option<i64>,
    }
    impl FocusLike for FakeFocus {
        fn get_focus(&self, modality: &str) -> Option<i64> {
            if modality == "text" {
                self.channel_id
            } else {
                None
            }
        }
        fn catch_up_limit(&self) -> usize {
            20
        }
    }

    struct PresenceRec {
        calls: StdMutex<Vec<(String, Option<String>)>>,
    }
    impl PresenceRec {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: StdMutex::new(Vec::new()),
            })
        }
        fn cb(self: &Arc<Self>) -> PresenceCb {
            let rec = Arc::clone(self);
            Arc::new(move |status, text| {
                rec.calls.lock().unwrap().push((status, text));
                Box::pin(async { Ok(()) })
            })
        }
        fn calls(&self) -> Vec<(String, Option<String>)> {
            self.calls.lock().unwrap().clone()
        }
        fn clear(&self) {
            self.calls.lock().unwrap().clear();
        }
    }

    fn boom_cb() -> PresenceCb {
        Arc::new(|_s, _t| Box::pin(async { Err(anyhow::anyhow!("ConnectionResetError")) }))
    }

    struct ScriptedLlm {
        replies: StdMutex<VecDeque<String>>,
        calls: StdMutex<Vec<Vec<Message>>>,
    }
    impl ScriptedLlm {
        fn new(replies: Vec<String>) -> Arc<Self> {
            Arc::new(Self {
                replies: StdMutex::new(replies.into_iter().collect()),
                calls: StdMutex::new(Vec::new()),
            })
        }
        fn calls(&self) -> Vec<Vec<Message>> {
            self.calls.lock().unwrap().clone()
        }
    }
    #[async_trait::async_trait]
    impl LlmClient for ScriptedLlm {
        async fn chat(&self, messages: Vec<Message>) -> anyhow::Result<Message> {
            self.calls.lock().unwrap().push(messages);
            let reply = self.replies.lock().unwrap().pop_front().unwrap_or_default();
            Ok(Message::new("assistant", reply))
        }
        async fn stream_completion(
            &self,
            _m: Vec<Message>,
            _t: Option<Vec<Value>>,
        ) -> anyhow::Result<futures::stream::BoxStream<'static, anyhow::Result<crate::llm::LlmDelta>>>
        {
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

    struct HighRollRng;
    impl ActivityRng for HighRollRng {
        fn gen_range_inclusive(&self, _lo: i64, hi: i64) -> i64 {
            hi
        }
    }

    struct FaultInjectingStore {
        inner: Arc<dyn ActivityStore>,
        faults: HashSet<&'static str>,
    }
    impl FaultInjectingStore {
        fn new(inner: Arc<dyn ActivityStore>, faults: &[&'static str]) -> Arc<Self> {
            Arc::new(Self {
                inner,
                faults: faults.iter().copied().collect(),
            })
        }
        fn guard(&self, m: &str) -> anyhow::Result<()> {
            if self.faults.contains(m) {
                Err(anyhow::anyhow!("injected fault: {m}"))
            } else {
                Ok(())
            }
        }
    }
    #[async_trait::async_trait]
    impl ActivityStore for FaultInjectingStore {
        async fn create_activity(
            &self,
            f: String,
            t: String,
            l: String,
            s: DateTime<Utc>,
            p: DateTime<Utc>,
            n: Option<String>,
        ) -> anyhow::Result<i64> {
            self.guard("create_activity")?;
            self.inner.create_activity(f, t, l, s, p, n).await
        }
        async fn finish_activity(
            &self,
            id: i64,
            s: String,
            r: DateTime<Utc>,
            e: Option<String>,
        ) -> anyhow::Result<()> {
            self.guard("finish_activity")?;
            self.inner.finish_activity(id, s, r, e).await
        }
        async fn set_activity_experience(&self, id: i64, e: String) -> anyhow::Result<()> {
            self.guard("set_activity_experience")?;
            self.inner.set_activity_experience(id, e).await
        }
        async fn active_activity(&self, f: String) -> anyhow::Result<Option<ActivityRecord>> {
            self.guard("active_activity")?;
            self.inner.active_activity(f).await
        }
        async fn latest_activity(
            &self,
            f: String,
            t: String,
        ) -> anyhow::Result<Option<ActivityRecord>> {
            self.guard("latest_activity")?;
            self.inner.latest_activity(f, t).await
        }
        async fn append_fact(&self, p: AppendFact) -> anyhow::Result<()> {
            self.guard("append_fact")?;
            self.inner.append_fact(p).await
        }
        async fn append_turn(&self, p: AppendTurn) -> anyhow::Result<()> {
            self.guard("append_turn")?;
            self.inner.append_turn(p).await
        }
        async fn latest_id(&self, f: String, c: Option<i64>) -> anyhow::Result<Option<i64>> {
            self.guard("latest_id")?;
            self.inner.latest_id(f, c).await
        }
        async fn latest_id_at_or_before(
            &self,
            f: String,
            ts: DateTime<Utc>,
        ) -> anyhow::Result<Option<i64>> {
            self.guard("latest_id_at_or_before")?;
            self.inner.latest_id_at_or_before(f, ts).await
        }
        async fn set_archive_watermark_all(&self, f: String, t: i64) -> anyhow::Result<()> {
            self.guard("set_archive_watermark_all")?;
            self.inner.set_archive_watermark_all(f, t).await
        }
        async fn promote_staged_turns_since(
            &self,
            f: String,
            a: i64,
            c: Option<usize>,
        ) -> anyhow::Result<Promotion> {
            self.guard("promote_staged_turns_since")?;
            self.inner.promote_staged_turns_since(f, a, c).await
        }
        async fn recent(&self, f: String, c: i64, l: i64) -> anyhow::Result<Vec<HistoryTurn>> {
            self.guard("recent")?;
            self.inner.recent(f, c, l).await
        }
        async fn turns_by_ids(&self, f: String, ids: Vec<i64>) -> anyhow::Result<Vec<HistoryTurn>> {
            self.guard("turns_by_ids")?;
            self.inner.turns_by_ids(f, ids).await
        }
        async fn turns_around(
            &self,
            f: String,
            c: i64,
            t: i64,
            b: i64,
            a: i64,
        ) -> anyhow::Result<Vec<HistoryTurn>> {
            self.guard("turns_around")?;
            self.inner.turns_around(f, c, t, b, a).await
        }
        fn maintenance_store(&self) -> Option<Arc<AsyncHistoryStore>> {
            None
        }
    }

    type RunHook = Box<dyn Fn() + Send + Sync>;

    struct FakeRunner {
        plan: Option<OpinionPlan>,
        fail: bool,
        calls: StdMutex<usize>,
        seen_apply: StdMutex<Option<bool>>,
        seen_tz: StdMutex<Option<String>>,
        seen_prompts: StdMutex<Option<SleepPromptText>>,
        hook: StdMutex<Option<RunHook>>,
    }
    impl FakeRunner {
        fn new(plan: Option<OpinionPlan>) -> Arc<Self> {
            Arc::new(Self {
                plan,
                fail: false,
                calls: StdMutex::new(0),
                seen_apply: StdMutex::new(None),
                seen_tz: StdMutex::new(None),
                seen_prompts: StdMutex::new(None),
                hook: StdMutex::new(None),
            })
        }
        fn failing() -> Arc<Self> {
            Arc::new(Self {
                plan: None,
                fail: true,
                calls: StdMutex::new(0),
                seen_apply: StdMutex::new(None),
                seen_tz: StdMutex::new(None),
                seen_prompts: StdMutex::new(None),
                hook: StdMutex::new(None),
            })
        }
        fn set_hook(&self, hook: RunHook) {
            *self.hook.lock().unwrap() = Some(hook);
        }
    }
    #[async_trait::async_trait]
    impl MaintenanceRunner for FakeRunner {
        async fn run(&self, ctx: MaintenanceContext) -> anyhow::Result<MaintenanceRun> {
            *self.calls.lock().unwrap() += 1;
            *self.seen_apply.lock().unwrap() = Some(ctx.apply);
            *self.seen_tz.lock().unwrap() = Some(ctx.display_tz.clone());
            *self.seen_prompts.lock().unwrap() = Some(ctx.prompts);
            if let Some(hook) = self.hook.lock().unwrap().as_ref() {
                hook();
            }
            if self.fail {
                return Err(anyhow::anyhow!("passes boom"));
            }
            Ok(MaintenanceRun {
                opinion_plan: self.plan.clone(),
                ..Default::default()
            })
        }
    }

    fn author() -> Author {
        Author::new(
            "discord",
            "1",
            Some("cor".to_owned()),
            Some("Cor".to_owned()),
        )
    }

    fn opinion_plan(texts: &[&str]) -> OpinionPlan {
        let opinions = texts
            .iter()
            .map(|t| OpinionFact {
                text: (*t).to_owned(),
                source_turn_ids: vec![1],
                valid_from_date: "2026-06-12".to_owned(),
                self_grounded: true,
                importance: 5,
            })
            .collect();
        OpinionPlan::new(FAMILIAR, opinions, Vec::new(), Vec::new(), 5)
    }

    // --- catalog builders --------------------------------------------------

    fn base_catalog() -> Vec<ActivityType> {
        vec![
            ActivityType {
                id: "walk".into(),
                label: "creek walk".into(),
                duration_minutes: Some((20, 40)),
                reachable: true,
                seed: "A walk along the creek behind the house.".into(),
                ..Default::default()
            },
            ActivityType {
                id: "hatbox".into(),
                label: "hatbox tending".into(),
                duration_minutes: Some((10, 20)),
                reachable: false,
                seed: "Tending the hatbox.".into(),
                ..Default::default()
            },
        ]
    }

    fn config() -> ActivitiesConfig {
        ActivitiesConfig {
            catalog: base_catalog(),
            archive_after_minutes: 45,
            idle_nudge_minutes: 20,
            min_gap_minutes: 90,
            active_hours: Some((nt(10, 0), nt(23, 0))),
        }
    }

    fn scheduled_entry() -> ActivityType {
        ActivityType {
            id: "errands".into(),
            label: "errands run".into(),
            duration_minutes: Some((20, 40)),
            reachable: true,
            seed: "Out running errands.".into(),
            active_days: Some([0, 1, 2, 3, 4].into_iter().collect()),
            active_hours: Some((nt(9, 0), nt(17, 0))),
            ..Default::default()
        }
    }

    fn scheduled_config() -> ActivitiesConfig {
        let mut c = config();
        c.catalog.push(scheduled_entry());
        c
    }

    fn clamp_config() -> ActivitiesConfig {
        let mut c = config();
        c.catalog.push(ActivityType {
            id: "weekday_rounds".into(),
            label: "weekday rounds".into(),
            duration_minutes: Some((90, 180)),
            reachable: true,
            seed: "Walking the weekday rounds.".into(),
            active_days: Some([0, 1, 2, 3, 4].into_iter().collect()),
            active_hours: Some((nt(9, 0), nt(17, 0))),
            ..Default::default()
        });
        c.catalog.push(ActivityType {
            id: "daysonly".into(),
            label: "weekday wander".into(),
            duration_minutes: Some((90, 180)),
            reachable: true,
            seed: "A weekday wander.".into(),
            active_days: Some([0, 1, 2, 3, 4].into_iter().collect()),
            ..Default::default()
        });
        c.catalog.push(ActivityType {
            id: "nightwatch".into(),
            label: "night watch".into(),
            duration_minutes: Some((90, 300)),
            reachable: true,
            seed: "The night watch.".into(),
            active_days: Some(std::iter::once(1).collect()),
            active_hours: Some((nt(22, 0), nt(2, 0))),
            ..Default::default()
        });
        c
    }

    fn sleep_entry() -> ActivityType {
        ActivityType {
            id: "sleep".into(),
            label: "asleep".into(),
            duration_minutes: None,
            reachable: false,
            seed: "The night's dream, retold on waking.".into(),
            ..Default::default()
        }
    }

    fn sleep_config() -> ActivitiesConfig {
        let mut c = config();
        c.catalog.push(sleep_entry());
        c
    }

    // --- fixture -----------------------------------------------------------

    /// Test store wrapper that keeps the backing `TempDir` alive for exactly as
    /// long as the engine (which owns the `Arc<dyn ActivityStore>`) — so inline
    /// `Fx::new(..).build()` temporaries don't delete the SQLite dir, and the
    /// dir IS cleaned up when the engine drops at test end (no leak).
    struct KeepDirStore {
        inner: Arc<dyn ActivityStore>,
        _dir: Arc<tempfile::TempDir>,
    }

    #[async_trait::async_trait]
    impl ActivityStore for KeepDirStore {
        async fn create_activity(
            &self,
            f: String,
            t: String,
            l: String,
            s: DateTime<Utc>,
            p: DateTime<Utc>,
            n: Option<String>,
        ) -> anyhow::Result<i64> {
            self.inner.create_activity(f, t, l, s, p, n).await
        }
        async fn finish_activity(
            &self,
            id: i64,
            s: String,
            r: DateTime<Utc>,
            e: Option<String>,
        ) -> anyhow::Result<()> {
            self.inner.finish_activity(id, s, r, e).await
        }
        async fn set_activity_experience(&self, id: i64, e: String) -> anyhow::Result<()> {
            self.inner.set_activity_experience(id, e).await
        }
        async fn active_activity(&self, f: String) -> anyhow::Result<Option<ActivityRecord>> {
            self.inner.active_activity(f).await
        }
        async fn latest_activity(
            &self,
            f: String,
            t: String,
        ) -> anyhow::Result<Option<ActivityRecord>> {
            self.inner.latest_activity(f, t).await
        }
        async fn append_fact(&self, p: AppendFact) -> anyhow::Result<()> {
            self.inner.append_fact(p).await
        }
        async fn append_turn(&self, p: AppendTurn) -> anyhow::Result<()> {
            self.inner.append_turn(p).await
        }
        async fn latest_id(&self, f: String, c: Option<i64>) -> anyhow::Result<Option<i64>> {
            self.inner.latest_id(f, c).await
        }
        async fn latest_id_at_or_before(
            &self,
            f: String,
            ts: DateTime<Utc>,
        ) -> anyhow::Result<Option<i64>> {
            self.inner.latest_id_at_or_before(f, ts).await
        }
        async fn set_archive_watermark_all(&self, f: String, t: i64) -> anyhow::Result<()> {
            self.inner.set_archive_watermark_all(f, t).await
        }
        async fn promote_staged_turns_since(
            &self,
            f: String,
            a: i64,
            c: Option<usize>,
        ) -> anyhow::Result<Promotion> {
            self.inner.promote_staged_turns_since(f, a, c).await
        }
        async fn recent(&self, f: String, c: i64, l: i64) -> anyhow::Result<Vec<HistoryTurn>> {
            self.inner.recent(f, c, l).await
        }
        async fn turns_by_ids(&self, f: String, ids: Vec<i64>) -> anyhow::Result<Vec<HistoryTurn>> {
            self.inner.turns_by_ids(f, ids).await
        }
        async fn turns_around(
            &self,
            f: String,
            c: i64,
            t: i64,
            b: i64,
            a: i64,
        ) -> anyhow::Result<Vec<HistoryTurn>> {
            self.inner.turns_around(f, c, t, b, a).await
        }
        fn maintenance_store(&self) -> Option<Arc<AsyncHistoryStore>> {
            self.inner.maintenance_store()
        }
    }

    struct Fx {
        clock: Arc<FakeClock>,
        real: Arc<AsyncHistoryStore>,
        dir: Arc<tempfile::TempDir>,
        bus: Arc<InProcessEventBus>,
        config: ActivitiesConfig,
        presence: Arc<PresenceRec>,
        presence_cb: Option<PresenceCb>,
        focused: Option<i64>,
        llm: Arc<ScriptedLlm>,
        voice_active: bool,
        bot_id: BotUserIdFn,
        nudge_tick: StdDur,
        familiar_id: String,
        sleep_passes_enabled: bool,
        seed_dream_path: Option<PathBuf>,
        sleep_prompts: SleepPromptText,
        sleep_window: Option<(NaiveTime, NaiveTime)>,
        sleep_grace: i64,
        display_tz: String,
        rng: Arc<dyn ActivityRng>,
        runner: Arc<dyn MaintenanceRunner>,
        store_override: Option<Arc<dyn ActivityStore>>,
    }

    impl Fx {
        fn new(clock: Arc<FakeClock>) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let store = HistoryStore::open(dir.path().join("history.db")).unwrap();
            let real = Arc::new(AsyncHistoryStore::new(store));
            let bus = Arc::new(InProcessEventBus::new());
            Self {
                clock,
                real,
                dir: Arc::new(dir),
                bus,
                config: config(),
                presence: PresenceRec::new(),
                presence_cb: None,
                focused: Some(CHANNEL),
                llm: ScriptedLlm::new(vec![
                    "I walked along the creek and watched dragonflies.".to_owned(),
                ]),
                voice_active: false,
                bot_id: Arc::new(|| Some(BOT_ID)),
                nudge_tick: StdDur::from_secs_f64(60.0),
                familiar_id: FAMILIAR.to_owned(),
                sleep_passes_enabled: false,
                seed_dream_path: None,
                sleep_prompts: SleepPromptText::default(),
                sleep_window: Some((nt(0, 0), nt(8, 0))),
                sleep_grace: 30,
                display_tz: "UTC".to_owned(),
                rng: Arc::new(LcgRng::seeded(7)),
                runner: Arc::new(DefaultMaintenanceRunner),
                store_override: None,
            }
        }

        fn experience(mut self, exp: &str) -> Self {
            self.llm = ScriptedLlm::new(vec![exp.to_owned()]);
            self
        }

        fn build(&self) -> Arc<ActivityEngine> {
            let inner_store = self
                .store_override
                .clone()
                .unwrap_or_else(|| Arc::new(RealActivityStore(Arc::clone(&self.real))));
            let store: Arc<dyn ActivityStore> = Arc::new(KeepDirStore {
                inner: inner_store,
                _dir: Arc::clone(&self.dir),
            });
            let presence_cb = self
                .presence_cb
                .clone()
                .unwrap_or_else(|| self.presence.cb());
            let voice = self.voice_active;
            ActivityEngine::new(EngineParams {
                store,
                config: self.config.clone(),
                background_llm: Arc::clone(&self.llm) as Arc<dyn LlmClient>,
                bus: Arc::clone(&self.bus) as Arc<dyn EventBus>,
                focus: Arc::new(FakeFocus {
                    channel_id: self.focused,
                }),
                presence_cb,
                familiar_id: self.familiar_id.clone(),
                display_tz: self.display_tz.clone(),
                bot_user_id: Arc::clone(&self.bot_id),
                sleep_window: self.sleep_window,
                sleep_grace_minutes: self.sleep_grace,
                voice_active_fn: Arc::new(move || voice),
                clock: Arc::clone(&self.clock) as Arc<dyn Clock>,
                rng: Arc::clone(&self.rng),
                nudge_tick: self.nudge_tick,
                familiar_display_name: None,
                sleep_passes_enabled: self.sleep_passes_enabled,
                seed_dream_path: self.seed_dream_path.clone(),
                sleep_prompts: self.sleep_prompts.clone(),
                maintenance_runner: Arc::clone(&self.runner),
            })
        }
    }

    async fn start_activity(
        engine: &Arc<ActivityEngine>,
        type_id: &str,
        note: Option<&str>,
    ) -> Value {
        let ack = engine.defer_start(type_id, note);
        engine.end_turn().await;
        ack
    }

    fn subscribe_text(bus: &Arc<InProcessEventBus>) -> crate::bus::in_process::Subscription {
        bus.subscribe(&[TOPIC_DISCORD_TEXT], BackpressurePolicy::Unbounded, 0)
    }

    fn wake_payload(ev: &Event) -> DiscordTextPayload {
        ev.payload
            .downcast_ref::<DiscordTextPayload>()
            .expect("discord text payload")
            .clone()
    }

    async fn recv_wake(
        sub: &mut crate::bus::in_process::Subscription,
    ) -> Option<std::sync::Arc<Event>> {
        // Pure hang-guard: every caller `.expect`s `Some`, so the bound only
        // exists to fail a genuinely-stuck test instead of hanging CI forever.
        // Keep it generous — the runner does several `spawn_blocking` DB
        // round-trips before publishing, which can outlast a tight 1s deadline
        // on a loaded current-thread test runtime (the `c1_*` flake).
        timeout(StdDur::from_secs(30), sub.recv())
            .await
            .ok()
            .flatten()
    }

    async fn sleep_and_wake(engine: &Arc<ActivityEngine>, clock: &Arc<FakeClock>) {
        engine.sleep_schedule_tick(clock.now()).await;
        let active = engine.active().expect("active after sleep tick");
        engine.cancel_return_timer().await;
        clock.set(active.planned_return_at);
        engine.run_return("completed").await;
    }

    fn gp(content: &str) -> GatePayload {
        gp_ch(content, CHANNEL)
    }
    fn gp_ch(content: &str, ch: i64) -> GatePayload {
        GatePayload {
            alarm: false,
            content: content.to_owned(),
            pings_bot: None,
            channel_id: Some(ch),
            author: Some(author()),
        }
    }

    // --- TestDeferStart ----------------------------------------------------

    #[tokio::test]
    async fn defer_unknown_type_returns_error() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        assert!(engine.defer_start("scuba", None).get("error").is_some());
    }

    #[tokio::test]
    async fn defer_ack_carries_label_and_rolled_duration() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        let ack = engine.defer_start("walk", Some("want fresh air"));
        assert_eq!(ack["ack"], "ok");
        assert_eq!(ack["label"], "creek walk");
        let d = ack["duration_minutes"].as_i64().unwrap();
        assert!((20..=40).contains(&d), "{d}");
    }

    #[tokio::test]
    async fn defer_does_not_start_until_end_turn() {
        let fx = Fx::new(FakeClock::new(noon()));
        let engine = fx.build();
        engine.defer_start("walk", None);
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        assert!(fx.presence.calls().is_empty());
    }

    #[tokio::test]
    async fn defer_refuses_while_staged_or_active() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        assert_eq!(engine.defer_start("walk", None)["ack"], "ok");
        assert!(engine.defer_start("hatbox", None).get("error").is_some());
        engine.end_turn().await;
        assert!(engine.defer_start("hatbox", None).get("error").is_some());
        engine.stop().await;
    }

    #[tokio::test]
    async fn defer_refuses_while_voice_subscribed() {
        let mut fx = Fx::new(FakeClock::new(noon()));
        fx.voice_active = true;
        let engine = fx.build();
        let r = engine.defer_start("walk", None);
        let msg = r["error"].as_str().unwrap();
        assert!(msg.contains("voice"), "{msg}");
    }

    // --- TestScheduleGate --------------------------------------------------

    #[tokio::test]
    async fn schedule_rejected_outside_active_days() {
        let mut fx = Fx::new(FakeClock::new(dt(2026, 6, 13, 14, 0))); // Saturday
        fx.config = scheduled_config();
        let engine = fx.build();
        let r = engine.defer_start("errands", None);
        let msg = r["error"].as_str().unwrap();
        assert!(msg.contains("errands run"), "{msg}");
        assert!(msg.contains("Mon Tue Wed Thu Fri"), "{msg}");
        assert!(msg.contains("09:00-17:00"), "{msg}");
        assert!(engine.active().is_none());
        assert!(engine.lock().staged.is_none());
    }

    #[tokio::test]
    async fn schedule_rejected_outside_active_hours() {
        let mut fx = Fx::new(FakeClock::new(dt(2026, 6, 16, 20, 0))); // Tue 20:00
        fx.config = scheduled_config();
        let engine = fx.build();
        let r = engine.defer_start("errands", None);
        assert!(r["error"].as_str().unwrap().contains("errands run"));
        assert!(engine.lock().staged.is_none());
    }

    #[tokio::test]
    async fn schedule_staged_when_inside() {
        let mut fx = Fx::new(FakeClock::new(dt(2026, 6, 16, 14, 0)));
        fx.config = scheduled_config();
        let engine = fx.build();
        assert_eq!(engine.defer_start("errands", None)["ack"], "ok");
        assert!(engine.lock().staged.is_some());
    }

    #[tokio::test]
    async fn schedule_hour_start_inclusive() {
        let mut fx = Fx::new(FakeClock::new(dt(2026, 6, 16, 9, 0)));
        fx.config = scheduled_config();
        let engine = fx.build();
        assert_eq!(engine.defer_start("errands", None)["ack"], "ok");
        assert!(engine.lock().staged.is_some());
    }

    #[tokio::test]
    async fn schedule_hour_end_exclusive() {
        let mut fx = Fx::new(FakeClock::new(dt(2026, 6, 16, 17, 0)));
        fx.config = scheduled_config();
        let engine = fx.build();
        assert!(engine.defer_start("errands", None).get("error").is_some());
        assert!(engine.lock().staged.is_none());
    }

    #[tokio::test]
    async fn schedule_evaluated_in_display_tz() {
        let mut fx = Fx::new(FakeClock::new(dt(2026, 6, 14, 23, 30))); // Sun 23:30 UTC = Mon 09:30 Sydney
        fx.config = scheduled_config();
        fx.display_tz = "Australia/Sydney".to_owned();
        let engine = fx.build();
        assert_eq!(engine.defer_start("errands", None)["ack"], "ok");
        assert!(engine.lock().staged.is_some());
    }

    #[tokio::test]
    async fn schedule_unscheduled_entry_unaffected() {
        let mut fx = Fx::new(FakeClock::new(dt(2026, 6, 13, 14, 0)));
        fx.config = scheduled_config();
        let engine = fx.build();
        assert_eq!(engine.defer_start("walk", None)["ack"], "ok");
        assert!(engine.lock().staged.is_some());
    }

    // --- TestScheduleDurationClamp -----------------------------------------

    fn clamp_engine(clock: DateTime<Utc>) -> Arc<ActivityEngine> {
        let mut fx = Fx::new(FakeClock::new(clock));
        fx.config = clamp_config();
        fx.rng = Arc::new(HighRollRng);
        fx.build()
    }

    #[tokio::test]
    async fn clamp_high_bound_near_window_end() {
        let engine = clamp_engine(dt(2026, 6, 16, 15, 0));
        let ack = engine.defer_start("weekday_rounds", None);
        assert_eq!(ack["duration_minutes"].as_i64(), Some(135));
    }

    #[tokio::test]
    async fn clamp_boundary_room_equals_lo_stages() {
        let engine = clamp_engine(dt(2026, 6, 16, 15, 45));
        assert_eq!(
            engine.defer_start("weekday_rounds", None)["duration_minutes"].as_i64(),
            Some(90)
        );
    }

    #[tokio::test]
    async fn clamp_rejects_when_room_below_lo() {
        let engine = clamp_engine(dt(2026, 6, 16, 16, 0));
        let r = engine.defer_start("weekday_rounds", None);
        let msg = r["error"].as_str().unwrap();
        assert!(msg.contains("weekday rounds"), "{msg}");
        assert!(msg.contains("not enough time"), "{msg}");
        assert!(engine.lock().staged.is_none());
    }

    #[tokio::test]
    async fn clamp_no_clamp_when_ample_room() {
        let engine = clamp_engine(dt(2026, 6, 16, 9, 30));
        assert_eq!(
            engine.defer_start("weekday_rounds", None)["duration_minutes"].as_i64(),
            Some(180)
        );
    }

    #[tokio::test]
    async fn clamp_days_only_entry_never_clamped() {
        let engine = clamp_engine(dt(2026, 6, 16, 16, 0));
        assert_eq!(
            engine.defer_start("daysonly", None)["duration_minutes"].as_i64(),
            Some(180)
        );
    }

    #[tokio::test]
    async fn clamp_wrap_window_room_against_next_day_end() {
        let engine = clamp_engine(dt(2026, 6, 16, 23, 0));
        assert_eq!(
            engine.defer_start("nightwatch", None)["duration_minutes"].as_i64(),
            Some(195)
        );
    }

    // --- TestEndTurnAppliesStart ------------------------------------------

    #[tokio::test]
    async fn end_turn_creates_row_sets_presence_arms_timer() {
        let fx = Fx::new(FakeClock::new(noon()));
        let engine = fx.build();
        let ack = start_activity(&engine, "walk", Some("fresh air")).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "walk");
        assert_eq!(row.note.as_deref(), Some("fresh air"));
        let expected =
            fx.clock.now() + Duration::minutes(ack["duration_minutes"].as_i64().unwrap());
        assert_eq!(row.planned_return_at, expected);
        assert_eq!(
            fx.presence.calls(),
            vec![("idle".to_owned(), Some("creek walk".to_owned()))]
        );
        assert!(engine.return_timer_armed());
        engine.stop().await;
    }

    #[tokio::test]
    async fn end_turn_unreachable_sets_dnd_presence() {
        let fx = Fx::new(FakeClock::new(noon()));
        let engine = fx.build();
        start_activity(&engine, "hatbox", None).await;
        assert_eq!(
            fx.presence.calls(),
            vec![("dnd".to_owned(), Some("hatbox tending".to_owned()))]
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn end_turn_without_staged_is_noop() {
        let fx = Fx::new(FakeClock::new(noon()));
        let engine = fx.build();
        engine.end_turn().await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        assert!(fx.presence.calls().is_empty());
    }

    // --- TestGate ----------------------------------------------------------

    #[tokio::test]
    async fn gate_normal_when_idle() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        assert_eq!(engine.gate(&gp("hello")).action, GateAction::Normal);
    }

    #[tokio::test]
    async fn gate_suppress_non_ping_while_active() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        assert_eq!(
            engine.gate(&gp("anyone around?")).action,
            GateAction::Suppress
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_judgment_on_ping_when_reachable() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(20);
        let d = engine.gate(&gp(&format!("hey <@{BOT_ID}> you there?")));
        assert_eq!(d.action, GateAction::Judgment);
        let line = d.state_line.unwrap();
        assert!(line.contains("creek walk"), "{line}");
        assert!(line.contains("20 min"), "{line}");
        assert!(line.contains("Cor"), "{line}");
        assert!(line.contains("silent()"), "{line}");
        assert!(line.contains("do not call start_activity"), "{line}");
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_nickname_mention_form_counts() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        assert_eq!(
            engine.gate(&gp(&format!("<@!{BOT_ID}> ping"))).action,
            GateAction::Judgment
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_bare_name_does_not_count() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        assert_eq!(
            engine.gate(&gp("aria where are you")).action,
            GateAction::Suppress
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_suppress_ping_when_unreachable() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "hatbox", None).await;
        assert_eq!(
            engine.gate(&gp(&format!("<@{BOT_ID}> hello?"))).action,
            GateAction::Suppress
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_pings_bot_flag_true_counts_without_mention() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        let mut p = gp("you still there?");
        p.pings_bot = Some(true);
        let d = engine.gate(&p);
        assert_eq!(d.action, GateAction::Judgment);
        assert!(d.state_line.is_some());
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_pings_bot_flag_false_overrides_content() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        let mut p = gp(&format!("<@{BOT_ID}> hello?"));
        p.pings_bot = Some(false);
        assert_eq!(engine.gate(&p).action, GateAction::Suppress);
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_missing_flag_falls_back_to_content_scan() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        let p = gp(&format!("<@{BOT_ID}> hello?"));
        assert!(p.pings_bot.is_none());
        assert_eq!(engine.gate(&p).action, GateAction::Judgment);
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_alarm_payload_normal_while_out_reachable() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        let mut p = gp("[alarm fired: check the tea]");
        p.author = None;
        p.alarm = true;
        assert_eq!(engine.gate(&p).action, GateAction::Normal);
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_alarm_payload_normal_while_out_unreachable() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "hatbox", None).await;
        let mut p = gp("[alarm fired: check the tea]");
        p.author = None;
        p.alarm = true;
        assert_eq!(engine.gate(&p).action, GateAction::Normal);
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_unfocused_ping_suppressed() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        assert_eq!(
            engine
                .gate(&gp_ch(&format!("<@{BOT_ID}> hey"), CHANNEL + 1))
                .action,
            GateAction::Suppress
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_ping_with_no_focus_suppressed() {
        let mut fx = Fx::new(FakeClock::new(noon()));
        fx.focused = None;
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        assert_eq!(
            engine.gate(&gp(&format!("<@{BOT_ID}> hey"))).action,
            GateAction::Suppress
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_second_ping_same_author_suppressed() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        let p = gp(&format!("<@{BOT_ID}> you there?"));
        assert_eq!(engine.gate(&p).action, GateAction::Judgment);
        assert_eq!(engine.gate(&p).action, GateAction::Suppress);
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_second_ping_different_author_judged() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        assert_eq!(
            engine.gate(&gp(&format!("<@{BOT_ID}> hi"))).action,
            GateAction::Judgment
        );
        let mut other = gp(&format!("<@{BOT_ID}> hi"));
        other.author = Some(Author::new(
            "discord",
            "2",
            Some("mia".to_owned()),
            Some("Mia".to_owned()),
        ));
        assert_eq!(engine.gate(&other).action, GateAction::Judgment);
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_judgment_latch_clears_at_next_activity_start() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        start_activity(&engine, "walk", None).await;
        let p = gp(&format!("<@{BOT_ID}> you there?"));
        assert_eq!(engine.gate(&p).action, GateAction::Judgment);
        clock.advance_minutes(5);
        engine.notify_reply_sent().await;
        start_activity(&engine, "walk", None).await;
        assert_eq!(engine.gate(&p).action, GateAction::Judgment);
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_state_line_truncates_long_author_name() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        let long_name: String = "N".repeat(80);
        let mut p = gp(&format!("<@{BOT_ID}> hey"));
        p.author = Some(Author::new(
            "discord",
            "3",
            Some("n".to_owned()),
            Some(long_name.clone()),
        ));
        let d = engine.gate(&p);
        assert_eq!(d.action, GateAction::Judgment);
        let line = d.state_line.unwrap();
        assert!(!line.contains(&long_name), "{line}");
        assert!(line.contains(&"N".repeat(40)), "{line}");
        engine.stop().await;
    }

    #[tokio::test]
    async fn gate_normal_while_returning() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        start_activity(&engine, "walk", None).await;
        engine.lock().returning = true;
        assert_eq!(engine.gate(&gp("hello")).action, GateAction::Normal);
        engine.lock().returning = false;
        engine.stop().await;
    }

    // --- TestReturnFlow ----------------------------------------------------

    #[tokio::test]
    async fn return_cut_short_finishes_row_and_restores_presence() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        assert_eq!(
            fx.presence.calls().last(),
            Some(&("online".to_owned(), None))
        );
        assert!(!engine.return_timer_armed());
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_presence_failure_never_breaks_turn_flow() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.presence_cb = Some(boom_cb());
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        assert!(engine.active().is_some());
        assert!(engine.return_timer_armed());
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_cut_short_keeps_nudge_loop_armed() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        engine.start().await;
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        assert!(engine.nudge_loop_armed());
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_experience_turn_marked_and_persisted() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone()).experience("The creek was high after the rain.");
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert_eq!(marked.len(), 1);
        assert_eq!(marked[0].role, "assistant");
        assert!(marked[0].content.starts_with("[returned from creek walk]"));
        assert!(
            marked[0]
                .content
                .contains("The creek was high after the rain.")
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_turn_carries_activity_return_mode() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert_eq!(marked.len(), 1);
        assert_eq!(marked[0].mode.as_deref(), Some(ACTIVITY_RETURN_MODE));
        assert_eq!(marked[0].mode.as_deref(), Some("activity_return"));
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_event_fact_written_mechanically() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        let facts = fx
            .real
            .sync()
            .recent_facts(FAMILIAR, 5, false, None)
            .unwrap();
        assert_eq!(facts.len(), 1);
        let text = &facts[0].text;
        assert!(text.contains("creek walk"), "{text}");
        assert!(text.contains("spent"), "{text}");
        assert!(!text.contains("went on a"), "{text}");
        assert!(text.contains("Jun 12"), "{text}");
        assert!(text.contains("afternoon"), "{text}");
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_long_absence_sets_archive_watermark_at_departure() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        let departure = fx
            .real
            .sync()
            .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "see you").author(author()))
            .unwrap();
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(60);
        engine.notify_reply_sent().await;
        let mark = fx
            .real
            .sync()
            .get_archive_watermark(FAMILIAR, CHANNEL)
            .unwrap();
        assert_eq!(mark, Some(departure.id));
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_long_absence_archives_all_channels() {
        let clock = FakeClock::new(noon());
        let other = CHANNEL + 1;
        let fx = Fx::new(clock.clone());
        let other_turn = fx
            .real
            .sync()
            .append_turn(
                AppendTurn::new(FAMILIAR, other, "user", "art link chatter").author(author()),
            )
            .unwrap();
        let departure = fx
            .real
            .sync()
            .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "see you").author(author()))
            .unwrap();
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(90);
        engine.notify_reply_sent().await;
        assert_eq!(
            fx.real
                .sync()
                .get_archive_watermark(FAMILIAR, other)
                .unwrap(),
            Some(departure.id)
        );
        let window = fx
            .real
            .sync()
            .recent_cross_channel(FAMILIAR, 50, true)
            .unwrap();
        let ids: HashSet<i64> = window.iter().map(|t| t.id).collect();
        assert!(!ids.contains(&other_turn.id));
        assert!(!ids.contains(&departure.id));
        assert!(
            window
                .iter()
                .any(|t| t.mode.as_deref() == Some("activity_return"))
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_short_absence_leaves_no_watermark() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.real
            .sync()
            .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "see you").author(author()))
            .unwrap();
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        assert_eq!(
            fx.real
                .sync()
                .get_archive_watermark(FAMILIAR, CHANNEL)
                .unwrap(),
            None
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn return_missed_ping_publishes_wake_event() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.bus.start().await;
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        fx.real
            .sync()
            .append_turn(
                AppendTurn::new(
                    FAMILIAR,
                    CHANNEL,
                    "user",
                    format!("<@{BOT_ID}> where did you go?"),
                )
                .author(author()),
            )
            .unwrap();
        fx.real
            .sync()
            .append_turn(
                AppendTurn::new(FAMILIAR, CHANNEL, "user", "probably out again").author(author()),
            )
            .unwrap();
        clock.advance_minutes(10);
        let mut sub = subscribe_text(&fx.bus);
        engine.notify_reply_sent().await;
        let ev = recv_wake(&mut sub).await.expect("wake event");
        assert_eq!(ev.topic, TOPIC_DISCORD_TEXT);
        let p = wake_payload(&ev);
        assert_eq!(p.channel_id, CHANNEL);
        assert!(p.author.is_none());
        assert!(p.content.contains("creek walk"), "{}", p.content);
        assert!(p.content.contains("where did you go?"), "{}", p.content);
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn return_no_pings_no_wake_event() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.bus.start().await;
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        fx.real
            .sync()
            .append_turn(
                AppendTurn::new(FAMILIAR, CHANNEL, "user", "just chatting").author(author()),
            )
            .unwrap();
        clock.advance_minutes(10);
        let mut sub = subscribe_text(&fx.bus);
        engine.notify_reply_sent().await;
        let got = timeout(StdDur::from_millis(200), sub.recv()).await;
        assert!(got.is_err(), "unexpected wake");
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn return_past_due_reload_arms_floor_timer_not_inline() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.real
            .sync()
            .create_activity(
                FAMILIAR,
                "walk",
                "creek walk",
                clock.now() - Duration::minutes(30),
                clock.now() - Duration::minutes(5),
                None,
            )
            .unwrap();
        let engine = fx.build();
        engine.start().await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_some());
        assert!(engine.return_timer_armed());
        assert_eq!(
            engine.lock().armed_return_ats,
            vec![clock.now() + Duration::seconds(20)]
        );
        assert_eq!(
            fx.presence.calls(),
            vec![("idle".to_owned(), Some("creek walk".to_owned()))]
        );
        assert_eq!(engine.gate(&gp("hello")).action, GateAction::Suppress);
        engine.stop().await;
    }

    // --- TestRestart -------------------------------------------------------

    #[tokio::test]
    async fn restart_rearms_future_return() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.real
            .sync()
            .create_activity(
                FAMILIAR,
                "walk",
                "creek walk",
                clock.now() - Duration::minutes(5),
                clock.now() + Duration::minutes(25),
                None,
            )
            .unwrap();
        let engine = fx.build();
        engine.start().await;
        assert!(engine.return_timer_armed());
        assert_eq!(engine.gate(&gp("hello")).action, GateAction::Suppress);
        engine.stop().await;
    }

    #[tokio::test]
    async fn restart_restores_idle_presence_for_reachable_row() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.real
            .sync()
            .create_activity(
                FAMILIAR,
                "walk",
                "creek walk",
                clock.now() - Duration::minutes(5),
                clock.now() + Duration::minutes(25),
                None,
            )
            .unwrap();
        let engine = fx.build();
        engine.start().await;
        assert_eq!(
            fx.presence.calls(),
            vec![("idle".to_owned(), Some("creek walk".to_owned()))]
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn restart_restores_dnd_presence_for_unreachable_row() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.real
            .sync()
            .create_activity(
                FAMILIAR,
                "hatbox",
                "hatbox tending",
                clock.now() - Duration::minutes(5),
                clock.now() + Duration::minutes(15),
                None,
            )
            .unwrap();
        let engine = fx.build();
        engine.start().await;
        assert_eq!(
            fx.presence.calls(),
            vec![("dnd".to_owned(), Some("hatbox tending".to_owned()))]
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn restart_resync_presence_reissues_away() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.real
            .sync()
            .create_activity(
                FAMILIAR,
                "hatbox",
                "hatbox tending",
                clock.now() - Duration::minutes(5),
                clock.now() + Duration::minutes(15),
                None,
            )
            .unwrap();
        let engine = fx.build();
        engine.start().await;
        fx.presence.clear();
        engine.resync_presence().await;
        assert_eq!(
            fx.presence.calls(),
            vec![("dnd".to_owned(), Some("hatbox tending".to_owned()))]
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn restart_resync_presence_noop_when_idle() {
        let fx = Fx::new(FakeClock::new(noon()));
        let engine = fx.build();
        engine.start().await;
        engine.resync_presence().await;
        assert!(fx.presence.calls().is_empty());
        engine.stop().await;
    }

    #[tokio::test]
    async fn restart_stop_cancels_timer() {
        let fx = Fx::new(FakeClock::new(noon()));
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        assert!(engine.return_timer_armed());
        engine.stop().await;
        assert!(!engine.return_timer_armed());
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_some());
    }

    // --- TestShouldNudge ---------------------------------------------------

    #[tokio::test]
    async fn should_nudge_eligible_when_quiet_and_inside_hours() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        clock.advance_minutes(30);
        assert!(engine.should_nudge(clock.now()));
    }

    #[tokio::test]
    async fn should_nudge_not_eligible_while_active() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(30);
        assert!(!engine.should_nudge(clock.now()));
        engine.stop().await;
    }

    #[tokio::test]
    async fn should_nudge_not_eligible_when_recently_active() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        clock.advance_minutes(30);
        engine.note_traffic();
        clock.advance_minutes(5);
        assert!(!engine.should_nudge(clock.now()));
    }

    #[tokio::test]
    async fn should_nudge_min_gap_since_last_activity() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        clock.advance_minutes(30);
        assert!(!engine.should_nudge(clock.now()));
        clock.advance_minutes(70);
        assert!(engine.should_nudge(clock.now()));
        engine.stop().await;
    }

    #[tokio::test]
    async fn should_nudge_outside_active_hours() {
        let clock = FakeClock::new(dt(2026, 6, 12, 3, 0));
        let engine = Fx::new(clock.clone()).build();
        clock.advance_minutes(30);
        assert!(!engine.should_nudge(clock.now()));
    }

    #[tokio::test]
    async fn should_nudge_wrapped_active_hours() {
        let clock = FakeClock::new(dt(2026, 6, 12, 23, 30));
        let mut fx = Fx::new(clock.clone());
        fx.config = {
            let mut c = config();
            c.active_hours = Some((nt(22, 0), nt(2, 0)));
            c
        };
        let engine = fx.build();
        clock.advance_minutes(30);
        assert!(engine.should_nudge(clock.now()));
    }

    #[tokio::test]
    async fn should_nudge_debounce_after_mark_nudge_pending() {
        let clock = FakeClock::new(noon());
        let engine = Fx::new(clock.clone()).build();
        clock.advance_minutes(30);
        assert!(engine.should_nudge(clock.now()));
        engine.mark_nudge_pending();
        assert!(!engine.should_nudge(clock.now()));
    }

    // --- TestNoteMissedPing (engine-only) ----------------------------------

    #[tokio::test]
    async fn note_reply_ping_without_mention_reaches_wake() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.bus.start().await;
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        let turn = fx
            .real
            .sync()
            .append_turn(
                AppendTurn::new(FAMILIAR, CHANNEL, "user", "you still there?").author(author()),
            )
            .unwrap();
        engine.note_missed_ping(turn.id);
        clock.advance_minutes(10);
        let mut sub = subscribe_text(&fx.bus);
        engine.notify_reply_sent().await;
        let ev = recv_wake(&mut sub).await.expect("wake");
        assert!(wake_payload(&ev).content.contains("you still there?"));
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn note_live_note_dedupes_with_content_scan() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.bus.start().await;
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        let turn = fx
            .real
            .sync()
            .append_turn(
                AppendTurn::new(
                    FAMILIAR,
                    CHANNEL,
                    "user",
                    format!("<@{BOT_ID}> where did you go?"),
                )
                .author(author()),
            )
            .unwrap();
        engine.note_missed_ping(turn.id);
        engine.note_missed_ping(turn.id);
        clock.advance_minutes(10);
        let mut sub = subscribe_text(&fx.bus);
        engine.notify_reply_sent().await;
        let ev = recv_wake(&mut sub).await.expect("wake");
        assert_eq!(
            wake_payload(&ev)
                .content
                .matches("where did you go?")
                .count(),
            1
        );
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn note_cleared_at_activity_start() {
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.bus.start().await;
        let engine = fx.build();
        let turn = fx
            .real
            .sync()
            .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "old ping").author(author()))
            .unwrap();
        engine.note_missed_ping(turn.id);
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        let mut sub = subscribe_text(&fx.bus);
        engine.notify_reply_sent().await;
        assert!(timeout(StdDur::from_millis(200), sub.recv()).await.is_err());
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    // --- TestBotUserIdProvider ---------------------------------------------

    #[tokio::test]
    async fn bot_user_id_callable_resolves_at_gate_time() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        let holder = Arc::new(StdMutex::new(None::<i64>));
        let h = Arc::clone(&holder);
        fx.bot_id = Arc::new(move || *h.lock().unwrap());
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        let p = gp(&format!("<@{BOT_ID}> hey"));
        assert_eq!(engine.gate(&p).action, GateAction::Suppress);
        *holder.lock().unwrap() = Some(BOT_ID);
        assert_eq!(engine.gate(&p).action, GateAction::Judgment);
        engine.stop().await;
    }

    // --- TestNudgeLoop -----------------------------------------------------

    #[tokio::test]
    async fn nudge_loop_publishes_wake_into_focused_channel_when_quiet() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.nudge_tick = StdDur::from_secs_f64(0.01);
        fx.bus.start().await;
        let engine = fx.build();
        clock.advance_minutes(30);
        let mut sub = subscribe_text(&fx.bus);
        engine.start().await;
        let ev = timeout(StdDur::from_secs(2), sub.recv())
            .await
            .ok()
            .flatten()
            .expect("nudge");
        let p = wake_payload(&ev);
        assert_eq!(p.channel_id, CHANNEL);
        assert_eq!(p.familiar_id, FAMILIAR);
        assert!(p.author.is_none());
        assert!(p.content.contains("quiet"), "{}", p.content);
        assert!(p.content.contains("start_activity"), "{}", p.content);
        assert!(!engine.should_nudge(clock.now()));
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn nudge_loop_no_nudge_while_traffic_fresh() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.nudge_tick = StdDur::from_secs_f64(0.01);
        fx.bus.start().await;
        let engine = fx.build();
        let mut sub = subscribe_text(&fx.bus);
        engine.start().await;
        assert!(timeout(StdDur::from_millis(200), sub.recv()).await.is_err());
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn nudge_loop_start_arms_and_stop_cancels() {
        let engine = Fx::new(FakeClock::new(noon())).build();
        assert!(!engine.nudge_loop_armed());
        engine.start().await;
        assert!(engine.nudge_loop_armed());
        engine.stop().await;
        assert!(!engine.nudge_loop_armed());
    }

    // --- TestEndTurnHardening ----------------------------------------------

    fn faulting(fx: &Fx, faults: &[&'static str]) -> Arc<dyn ActivityStore> {
        FaultInjectingStore::new(Arc::new(RealActivityStore(Arc::clone(&fx.real))), faults)
    }

    #[tokio::test]
    async fn hardening_end_turn_never_raises_when_create_fails() {
        let mut fx = Fx::new(FakeClock::new(noon()));
        fx.store_override = Some(faulting(&fx, &["create_activity"]));
        let engine = fx.build();
        engine.defer_start("walk", None);
        engine.end_turn().await;
        assert!(engine.active().is_none());
        assert!(!engine.return_timer_armed());
        assert_eq!(engine.defer_start("walk", None)["ack"], "ok");
        engine.stop().await;
    }

    #[tokio::test]
    async fn hardening_end_turn_partial_failure_still_arms_timer() {
        let mut fx = Fx::new(FakeClock::new(noon()));
        fx.store_override = Some(faulting(&fx, &["latest_id"]));
        let engine = fx.build();
        engine.defer_start("walk", None);
        engine.end_turn().await;
        assert!(engine.active().is_some());
        assert!(engine.return_timer_armed());
        engine.stop().await;
    }

    // --- TestReturnHardening -----------------------------------------------

    #[tokio::test]
    async fn hardening_return_commits_even_when_return_turn_write_fails() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.store_override = Some(faulting(&fx, &["append_turn"]));
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        assert!(engine.active().is_none());
        engine.stop().await;
    }

    #[tokio::test]
    async fn hardening_event_fact_failure_does_not_kill_remaining_steps() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.bus.start().await;
        fx.store_override = Some(faulting(&fx, &["append_fact"]));
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        fx.real
            .sync()
            .append_turn(
                AppendTurn::new(
                    FAMILIAR,
                    CHANNEL,
                    "user",
                    format!("<@{BOT_ID}> where did you go?"),
                )
                .author(author()),
            )
            .unwrap();
        let mut sub = subscribe_text(&fx.bus);
        engine.notify_reply_sent().await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        assert!(
            turns
                .iter()
                .any(|t| t.content.starts_with("[returned from"))
        );
        let ev = recv_wake(&mut sub).await.expect("wake");
        assert!(wake_payload(&ev).content.contains("where did you go?"));
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn hardening_finish_activity_failure_still_leaves_her_back() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.store_override = Some(faulting(&fx, &["finish_activity"]));
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        assert!(engine.active().is_none());
        assert_eq!(engine.gate(&gp("hello")).action, GateAction::Normal);
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_some());
        engine.stop().await;
    }

    #[tokio::test]
    async fn hardening_wake_publish_normal_still_restores_presence() {
        // NOTE: `EventBus::publish` is infallible in the Rust port, so the
        // Python "bus.publish boom" injection is structurally impossible; this
        // verifies the observable outcome (presence restored, she is back).
        let clock = FakeClock::new(noon());
        let fx = Fx::new(clock.clone());
        fx.bus.start().await;
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        fx.real
            .sync()
            .append_turn(
                AppendTurn::new(FAMILIAR, CHANNEL, "user", format!("<@{BOT_ID}> hello?"))
                    .author(author()),
            )
            .unwrap();
        engine.notify_reply_sent().await;
        assert_eq!(
            fx.presence.calls().last(),
            Some(&("online".to_owned(), None))
        );
        assert!(engine.active().is_none());
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    // --- TestStagedPromotionAtReturn ---------------------------------------

    #[tokio::test]
    async fn staged_absence_chatter_promoted_at_return() {
        let clock = FakeClock::new(noon());
        let other = CHANNEL + 1;
        let fx = Fx::new(clock.clone());
        let before = fx
            .real
            .sync()
            .append_turn(
                AppendTurn::new(FAMILIAR, other, "user", "old unread")
                    .author(author())
                    .consumed(false),
            )
            .unwrap();
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        let during = fx
            .real
            .sync()
            .append_turn(
                AppendTurn::new(FAMILIAR, other, "user", "chatter while she was out")
                    .author(author())
                    .consumed(false),
            )
            .unwrap();
        clock.advance_minutes(10);
        engine.notify_reply_sent().await;
        let window = fx
            .real
            .sync()
            .recent_cross_channel(FAMILIAR, 50, false)
            .unwrap();
        let ids: HashSet<i64> = window.iter().map(|t| t.id).collect();
        assert!(ids.contains(&during.id));
        assert!(!ids.contains(&before.id));
        engine.stop().await;
    }

    // --- C1 (engine-level; the responder-in-loop integration is skipped) ---

    #[tokio::test]
    async fn c1_state_cleared_before_wake_publish() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.bus.start().await;
        // Presence cb that blocks on "online" until released — the wake is
        // published (and state cleared) BEFORE the online flip.
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate2 = Arc::clone(&gate);
        fx.presence_cb = Some(Arc::new(move |status, _text| {
            let gate = Arc::clone(&gate2);
            Box::pin(async move {
                if status == "online" {
                    gate.notified().await;
                }
                Ok(())
            })
        }));
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(10);
        fx.real
            .sync()
            .append_turn(
                AppendTurn::new(
                    FAMILIAR,
                    CHANNEL,
                    "user",
                    format!("<@{BOT_ID}> where did you go?"),
                )
                .author(author()),
            )
            .unwrap();
        clock.advance_minutes(10);
        let mut sub = subscribe_text(&fx.bus);
        let e2 = Arc::clone(&engine);
        let runner = tokio::spawn(async move { e2.notify_reply_sent().await });
        let ev = recv_wake(&mut sub)
            .await
            .expect("wake published mid-return");
        // Mid-return: state already cleared, so the wake gates NORMAL.
        assert!(engine.active().is_none());
        assert_eq!(
            engine.gate(&wake_gate_payload(&ev)).action,
            GateAction::Normal
        );
        gate.notify_one();
        runner.await.unwrap();
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    fn wake_gate_payload(ev: &Event) -> GatePayload {
        GatePayload::from_discord(&wake_payload(ev))
    }

    // --- TestSleepSchedule -------------------------------------------------

    fn night(hour: u32, minute: u32, day: u32) -> Arc<FakeClock> {
        FakeClock::new(dt(2026, 6, day, hour, minute))
    }

    #[tokio::test]
    async fn sleep_defer_returns_at_window_end() {
        let clock = night(0, 10, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        let ack = engine.defer_start("sleep", None);
        assert_eq!(ack["ack"], "ok");
        assert_eq!(ack["duration_minutes"].as_i64(), Some(470));
        engine.end_turn().await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.planned_return_at, dt(2026, 6, 13, 8, 0));
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_arms_from_character_config_window() {
        let clock = night(0, 31, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "sleep");
        assert_eq!(row.planned_return_at, dt(2026, 6, 13, 8, 0));
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_no_window_disarms_schedule() {
        let clock = night(0, 31, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_window = None;
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_bedtime_nudge_published_once_per_occurrence() {
        let clock = night(0, 5, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.bus.start().await;
        let engine = fx.build();
        let mut sub = subscribe_text(&fx.bus);
        engine.sleep_schedule_tick(clock.now()).await;
        let ev = recv_wake(&mut sub).await.expect("bedtime nudge");
        let p = wake_payload(&ev);
        assert!(p.content.contains("sleep"), "{}", p.content);
        assert!(p.content.contains("start_activity"), "{}", p.content);
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        clock.advance_minutes(5);
        engine.sleep_schedule_tick(clock.now()).await;
        assert!(timeout(StdDur::from_millis(200), sub.recv()).await.is_err());
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn sleep_force_after_grace() {
        let clock = night(0, 31, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "sleep");
        assert_eq!(row.planned_return_at, dt(2026, 6, 13, 8, 0));
        assert_eq!(
            fx.presence.calls(),
            vec![("dnd".to_owned(), Some("asleep".to_owned()))]
        );
        assert!(engine.return_timer_armed());
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_fixed_wake_regardless_of_start() {
        let clock = night(3, 0, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.planned_return_at, dt(2026, 6, 13, 8, 0));
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_wake_uses_wall_clock_across_dst_spring_forward() {
        // US spring-forward: 2026-03-08, clocks jump 02:00->03:00 EST->EDT.
        // Window 00:00-08:00 in America/New_York: start = 00:00 EST (05:00 UTC),
        // end = 08:00 EDT (12:00 UTC) — a 7h absolute span. Python does the
        // wall-clock `start + length` and wakes at wall 08:00 (12:00 UTC);
        // absolute `start + 8h` would (wrongly) wake at 13:00 UTC / 09:00 wall.
        // now = 05:31 UTC = 00:31 EST, inside the window and 31 min past the
        // 00:00 start (> 30 min grace) so force-sleep fires this tick.
        let clock = FakeClock::new(dt(2026, 3, 8, 5, 31));
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.display_tz = "America/New_York".to_owned();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "sleep");
        assert_eq!(row.planned_return_at, dt(2026, 3, 8, 12, 0));
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_nothing_happens_outside_window() {
        let clock = night(12, 0, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.bus.start().await;
        let engine = fx.build();
        let mut sub = subscribe_text(&fx.bus);
        engine.sleep_schedule_tick(clock.now()).await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        assert!(timeout(StdDur::from_millis(200), sub.recv()).await.is_err());
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn sleep_already_slept_this_window_no_refire() {
        let clock = night(1, 0, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.bus.start().await;
        let slept = fx
            .real
            .sync()
            .create_activity(
                FAMILIAR,
                "sleep",
                "asleep",
                dt(2026, 6, 13, 0, 5),
                dt(2026, 6, 13, 8, 0),
                None,
            )
            .unwrap();
        fx.real
            .sync()
            .finish_activity(slept, "completed", dt(2026, 6, 13, 0, 50), None)
            .unwrap();
        let engine = fx.build();
        let mut sub = subscribe_text(&fx.bus);
        engine.sleep_schedule_tick(clock.now()).await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        assert!(timeout(StdDur::from_millis(200), sub.recv()).await.is_err());
        engine.stop().await;
        fx.bus.shutdown().await;
    }

    #[tokio::test]
    async fn sleep_prior_night_does_not_block_tonight() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let old = fx
            .real
            .sync()
            .create_activity(
                FAMILIAR,
                "sleep",
                "asleep",
                dt(2026, 6, 12, 0, 10),
                dt(2026, 6, 12, 8, 0),
                None,
            )
            .unwrap();
        fx.real
            .sync()
            .finish_activity(old, "completed", dt(2026, 6, 12, 8, 0), None)
            .unwrap();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "sleep");
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_backstop_waits_while_out_on_other_activity() {
        let clock = night(0, 0, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        clock.advance_minutes(45);
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "walk");
        engine.notify_reply_sent().await;
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "sleep");
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_wrapped_window_evening_side() {
        let clock = night(23, 45, 12);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_window = Some((nt(23, 0), nt(7, 0)));
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.planned_return_at, dt(2026, 6, 13, 7, 0));
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_wrapped_window_morning_side() {
        let clock = night(1, 0, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_window = Some((nt(23, 0), nt(7, 0)));
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.planned_return_at, dt(2026, 6, 13, 7, 0));
        engine.stop().await;
    }

    #[tokio::test]
    async fn sleep_boot_mid_window_first_tick_force_sleeps() {
        let clock = night(1, 0, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.nudge_tick = StdDur::from_secs_f64(0.01);
        let engine = fx.build();
        engine.start().await;
        let mut slept = false;
        for _ in 0..200 {
            if fx.real.sync().active_activity(FAMILIAR).unwrap().is_some() {
                slept = true;
                break;
            }
            tokio::time::sleep(StdDur::from_millis(10)).await;
        }
        assert!(slept, "force sleep never fired");
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.type_id, "sleep");
        engine.stop().await;
    }

    // --- TestSleepLifecyclePasses ------------------------------------------

    #[tokio::test]
    async fn passes_consolidation_then_opinion_applied_on_departure() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_passes_enabled = true;
        let runner = FakeRunner::new(Some(opinion_plan(&["Rain is best heard from indoors."])));
        fx.runner = Arc::clone(&runner) as Arc<dyn MaintenanceRunner>;
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let task = engine.lock().sleep_passes_task.take().expect("passes task");
        task.await.unwrap();
        assert_eq!(*runner.seen_apply.lock().unwrap(), Some(true));
        assert_eq!(runner.seen_tz.lock().unwrap().as_deref(), Some("UTC"));
        assert!(fx.llm.calls().iter().any(|c| {
            c.iter()
                .any(|m| m.content_str().contains("Rain is best heard from indoors."))
        }));
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert!(row.experience_text.is_some());
        engine.stop().await;
    }

    #[tokio::test]
    async fn passes_configured_sleep_prompts_thread_into_passes() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_passes_enabled = true;
        fx.sleep_prompts = SleepPromptText::from_config(
            "CFG consolidation",
            "CFG stance {self_name}",
            "CFG synthesis {self_name}",
        );
        let runner = FakeRunner::new(Some(opinion_plan(&[])));
        fx.runner = Arc::clone(&runner) as Arc<dyn MaintenanceRunner>;
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let task = engine.lock().sleep_passes_task.take().expect("passes task");
        task.await.unwrap();
        let prompts = runner.seen_prompts.lock().unwrap().clone().unwrap();
        assert_eq!(prompts.consolidation_system, "CFG consolidation");
        assert_eq!(prompts.stance_system, "CFG stance {self_name}");
        assert_eq!(prompts.synthesis_system, "CFG synthesis {self_name}");
        engine.stop().await;
    }

    #[tokio::test]
    async fn passes_not_fired_for_non_sleep_activity() {
        let clock = FakeClock::new(noon());
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_passes_enabled = true;
        let engine = fx.build();
        start_activity(&engine, "walk", None).await;
        assert!(engine.lock().sleep_passes_task.is_none());
        engine.stop().await;
    }

    #[tokio::test]
    async fn passes_skipped_when_sleep_passes_disabled() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_passes_enabled = false;
        let runner = FakeRunner::new(None);
        fx.runner = Arc::clone(&runner) as Arc<dyn MaintenanceRunner>;
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let task = engine.lock().sleep_passes_task.take().expect("passes task");
        task.await.unwrap();
        assert_eq!(*runner.calls.lock().unwrap(), 0);
        engine.stop().await;
    }

    #[tokio::test]
    async fn passes_failure_logged_keeps_none_never_blocks_return() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.sleep_passes_enabled = true;
        fx.runner = FakeRunner::failing();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let task = engine.lock().sleep_passes_task.take().expect("passes task");
        task.await.unwrap();
        assert!(engine.lock().last_opinion_plan.is_none());
        engine.cancel_return_timer().await;
        clock.set(dt(2026, 6, 13, 8, 0));
        engine.run_return("completed").await;
        assert!(fx.real.sync().active_activity(FAMILIAR).unwrap().is_none());
        engine.stop().await;
    }

    #[tokio::test]
    async fn passes_persist_dream_prose_and_journal() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone()).experience("A dream of rain on glass.");
        fx.config = sleep_config();
        fx.sleep_passes_enabled = true;
        fx.runner = FakeRunner::new(Some(opinion_plan(&["Rain is best heard from indoors."])));
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        assert!(engine.active().is_some());
        let task = engine.lock().sleep_passes_task.take().expect("passes task");
        task.await.unwrap();
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(
            row.experience_text.as_deref(),
            Some("A dream of rain on glass.")
        );
        assert_eq!(
            engine.active().unwrap().experience_text.as_deref(),
            Some("A dream of rain on glass.")
        );
        let facts = fx
            .real
            .sync()
            .recent_facts(FAMILIAR, 10, false, None)
            .unwrap();
        let dream: Vec<_> = facts
            .iter()
            .filter(|f| f.text.contains("dreamed"))
            .collect();
        assert_eq!(dream.len(), 1);
        assert!(dream[0].text.contains("A dream of rain on glass."));
        assert!(
            dream[0]
                .subjects
                .iter()
                .all(|s| is_ego_key(&s.canonical_key))
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn passes_persist_no_ops_when_return_beat_passes() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone()).experience("A dream of rain on glass.");
        fx.config = sleep_config();
        fx.sleep_passes_enabled = true;
        let runner = FakeRunner::new(Some(opinion_plan(&["Rain is best heard from indoors."])));
        fx.runner = Arc::clone(&runner) as Arc<dyn MaintenanceRunner>;
        let engine = fx.build();
        // The opinion "pass" clears _active mid-run (as a finishing return would).
        let weak = Arc::downgrade(&engine);
        runner.set_hook(Box::new(move || {
            if let Some(e) = weak.upgrade() {
                e.lock().active = None;
            }
        }));
        engine.sleep_schedule_tick(clock.now()).await;
        let task = engine.lock().sleep_passes_task.take().expect("passes task");
        task.await.unwrap();
        // The active row exists in the DB (never finished) but the late passes
        // must not have written experience_text nor a dream-journal fact.
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert!(row.experience_text.is_none());
        let facts = fx
            .real
            .sync()
            .recent_facts(FAMILIAR, 10, false, None)
            .unwrap();
        assert!(facts.iter().all(|f| !f.text.contains("dreamed")));
        engine.stop().await;
    }

    // --- TestDreamReturn ---------------------------------------------------

    #[tokio::test]
    async fn dream_return_turn_uses_sleep_return_mode() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone()).experience("I dreamed of spools.");
        fx.config = sleep_config();
        let engine = fx.build();
        sleep_and_wake(&engine, &clock).await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert_eq!(marked.len(), 1);
        assert_eq!(
            marked[0].content,
            "[returned from asleep] I dreamed of spools."
        );
        assert_eq!(marked[0].mode.as_deref(), Some(SLEEP_RETURN_MODE));
        assert_eq!(marked[0].mode.as_deref(), Some("sleep_return"));
        engine.stop().await;
    }

    #[tokio::test]
    async fn dream_prompt_carries_seed_and_minted_opinions() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let active = engine.active().expect("active");
        engine.lock().last_opinion_plan = Some(opinion_plan(&["Rain is best heard from indoors."]));
        engine.cancel_return_timer().await;
        clock.set(active.planned_return_at);
        engine.run_return("completed").await;
        let calls = fx.llm.calls();
        let last = calls.last().expect("dream prose reached the LLM");
        assert!(last[0].content_str().to_lowercase().contains("dream"));
        assert!(
            last[1]
                .content_str()
                .contains("The night's dream, retold on waking.")
        );
        assert!(
            last[1]
                .content_str()
                .contains("Rain is best heard from indoors.")
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn dream_prose_seed_only_when_no_plan() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        assert!(engine.lock().last_opinion_plan.is_none());
        sleep_and_wake(&engine, &clock).await;
        let calls = fx.llm.calls();
        assert!(
            calls.last().unwrap()[1]
                .content_str()
                .contains("The night's dream, retold on waking.")
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn dream_journal_self_fact_minted() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone()).experience("I dreamed of spools.");
        fx.config = sleep_config();
        let engine = fx.build();
        sleep_and_wake(&engine, &clock).await;
        let facts = fx
            .real
            .sync()
            .recent_facts(FAMILIAR, 10, false, None)
            .unwrap();
        let journal: Vec<_> = facts
            .iter()
            .filter(|f| f.text.contains("dreamed"))
            .collect();
        assert_eq!(journal.len(), 1);
        assert!(journal[0].text.contains("I dreamed of spools."));
        assert!(
            journal[0]
                .subjects
                .iter()
                .any(|s| is_ego_key(&s.canonical_key))
        );
        assert!(
            facts
                .iter()
                .any(|f| f.text.contains("spent") && f.text.contains("asleep"))
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn dream_llm_failure_degrades_to_stock_line() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone()).experience("");
        fx.config = sleep_config();
        let engine = fx.build();
        sleep_and_wake(&engine, &clock).await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from asleep]"))
            .collect();
        assert_eq!(marked.len(), 1);
        assert!(marked[0].content.len() > "[returned from asleep] ".len());
        engine.stop().await;
    }

    #[tokio::test]
    async fn dream_return_reuses_persisted_prose_no_regen() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let active = engine.active().expect("active");
        engine.lock().active.as_mut().unwrap().experience_text =
            Some("Persisted dream.".to_owned());
        fx.real
            .sync()
            .set_activity_experience(active.id, "Persisted dream.")
            .unwrap();
        engine.cancel_return_timer().await;
        clock.set(active.planned_return_at);
        engine.run_return("completed").await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert_eq!(marked[0].content, "[returned from asleep] Persisted dream.");
        assert!(fx.llm.calls().is_empty());
        let facts = fx
            .real
            .sync()
            .recent_facts(FAMILIAR, 10, false, None)
            .unwrap();
        assert!(facts.iter().all(|f| !f.text.contains("dreamed")));
        engine.stop().await;
    }

    #[tokio::test]
    async fn dream_restart_reload_reuses_persisted_prose() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let active = engine.active().expect("active");
        fx.real
            .sync()
            .set_activity_experience(active.id, "Survived the restart.")
            .unwrap();
        engine.stop().await;
        // Fresh engine reloads the active row WITH prose.
        let mut fx2 = Fx::new(clock.clone());
        fx2.config = sleep_config();
        fx2.real = Arc::clone(&fx.real);
        let engine2 = fx2.build();
        engine2.start().await;
        let reloaded = engine2.active().expect("reloaded");
        assert_eq!(
            reloaded.experience_text.as_deref(),
            Some("Survived the restart.")
        );
        engine2.cancel_return_timer().await;
        clock.set(reloaded.planned_return_at);
        engine2.run_return("completed").await;
        let turns = fx2
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert_eq!(
            marked[0].content,
            "[returned from asleep] Survived the restart."
        );
        assert!(fx2.llm.calls().is_empty());
        engine2.stop().await;
    }

    #[tokio::test]
    async fn dream_fallback_generates_and_journals_once_when_unpersisted() {
        let clock = night(0, 45, 13);
        let mut fx = Fx::new(clock.clone()).experience("Fallback dream.");
        fx.config = sleep_config();
        let engine = fx.build();
        engine.sleep_schedule_tick(clock.now()).await;
        let active = engine.active().expect("active");
        assert!(active.experience_text.is_none());
        engine.cancel_return_timer().await;
        clock.set(active.planned_return_at);
        engine.run_return("completed").await;
        assert!(!fx.llm.calls().is_empty());
        let facts = fx
            .real
            .sync()
            .recent_facts(FAMILIAR, 10, false, None)
            .unwrap();
        let journal: Vec<_> = facts
            .iter()
            .filter(|f| f.text.contains("dreamed"))
            .collect();
        assert_eq!(journal.len(), 1);
        assert!(journal[0].text.contains("Fallback dream."));
        engine.stop().await;
    }

    // --- TestSeedDreamConsumable -------------------------------------------

    #[tokio::test]
    async fn seed_dream_used_verbatim_then_renamed() {
        let clock = night(0, 45, 13);
        let dir = tempfile::tempdir().unwrap();
        let seed_path = dir.path().join("seed_dream.md");
        std::fs::write(&seed_path, "The Spools, hand-authored.").unwrap();
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        fx.seed_dream_path = Some(seed_path.clone());
        let engine = fx.build();
        sleep_and_wake(&engine, &clock).await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert_eq!(
            marked[0].content,
            "[returned from asleep] The Spools, hand-authored."
        );
        assert!(fx.llm.calls().is_empty());
        assert!(!seed_path.exists());
        let consumed = dir.path().join("seed_dream.consumed.md");
        assert!(consumed.exists());
        assert_eq!(
            std::fs::read_to_string(&consumed).unwrap(),
            "The Spools, hand-authored."
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn seed_dream_second_night_generates() {
        let clock = night(0, 45, 13);
        let dir = tempfile::tempdir().unwrap();
        let seed_path = dir.path().join("seed_dream.md");
        std::fs::write(&seed_path, "The Spools.").unwrap();
        let mut fx = Fx::new(clock.clone()).experience("A generated second dream.");
        fx.config = sleep_config();
        fx.seed_dream_path = Some(seed_path);
        let engine = fx.build();
        sleep_and_wake(&engine, &clock).await;
        clock.set(dt(2026, 6, 14, 0, 45));
        sleep_and_wake(&engine, &clock).await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert_eq!(marked.len(), 2);
        assert!(
            marked
                .last()
                .unwrap()
                .content
                .contains("A generated second dream.")
        );
        engine.stop().await;
    }

    #[tokio::test]
    async fn seed_dream_missing_file_generates() {
        let clock = night(0, 45, 13);
        let dir = tempfile::tempdir().unwrap();
        let mut fx = Fx::new(clock.clone()).experience("A generated dream.");
        fx.config = sleep_config();
        fx.seed_dream_path = Some(dir.path().join("seed_dream.md"));
        let engine = fx.build();
        sleep_and_wake(&engine, &clock).await;
        let turns = fx
            .real
            .sync()
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap();
        let marked: Vec<_> = turns
            .iter()
            .filter(|t| t.content.starts_with("[returned from"))
            .collect();
        assert!(marked[0].content.contains("A generated dream."));
        engine.stop().await;
    }

    // --- TestEarlyBedGuard -------------------------------------------------

    #[tokio::test]
    async fn early_bed_midday_sleep_refused() {
        let clock = night(12, 0, 13);
        let mut fx = Fx::new(clock);
        fx.config = sleep_config();
        let engine = fx.build();
        let r = engine.defer_start("sleep", None);
        assert!(r["error"].as_str().unwrap().contains("window"));
    }

    #[tokio::test]
    async fn early_bed_within_hour_before_window_allowed() {
        let clock = night(23, 30, 12);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        assert_eq!(engine.defer_start("sleep", None)["ack"], "ok");
        engine.end_turn().await;
        let row = fx.real.sync().active_activity(FAMILIAR).unwrap().unwrap();
        assert_eq!(row.planned_return_at, dt(2026, 6, 13, 8, 0));
        engine.stop().await;
    }

    #[tokio::test]
    async fn early_bed_inside_window_allowed() {
        let clock = night(2, 0, 13);
        let mut fx = Fx::new(clock.clone());
        fx.config = sleep_config();
        let engine = fx.build();
        assert_eq!(engine.defer_start("sleep", None)["ack"], "ok");
        engine.end_turn().await;
        engine.stop().await;
    }
}
