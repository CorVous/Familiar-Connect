//! Opinion-formation pass — form the familiar's stances (subsystem 04; Python
//! `sleep/opinion_formation.py`).
//!
//! Sleep function #2. Reads the conversation log up to the sleep watermark's turn
//! axis, bucketed per calendar day in the familiar's `display_tz`, and forms its
//! OPINIONS — stored as ordinary facts routed to the `ego:` subject, grounded in
//! the turns that demonstrate them via `source_turn_ids`.
//!
//! Two-pass, mirroring consolidation's LLM-proposes/code-decides shape:
//! 1. per-day stance-moments — each carries that day's cited turn-ids;
//! 2. ONE synthesis over all stance-moments — code enforces every formed
//!    opinion's grounding ⊆ the union of its input stance-moments' ids, so the
//!    synthesis can never invent grounding.

use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

use chrono::{DateTime, Utc};
use chrono_tz::Tz;
use serde_json::Value;

use super::{SleepError, len_i64, normalize_fact_text, py_int_list_repr, py_str_field};
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{FactSubject, HistoryTurn, SleepWatermark};
use crate::identity::ego_canonical_key;
use crate::llm::{LlmClient, Message};
use crate::prompt_fill::fill_placeholders;
use crate::structured_output::coerce_positive_int_list;
use crate::structured_request::{
    DEFAULT_MAX_RETRIES, Field, Schema, render_contract, request_structured,
};
use crate::support::time::parse_iso;

/// Declared but unused in the Python source — no per-day turn cap is
/// implemented. Retained as a documented constant for parity; **not** consulted.
pub const DEFAULT_TURNS_MAX_PER_DAY: i64 = 600;
/// Default cap on the number of opinions one synthesis may record.
pub const DEFAULT_OPINION_CAP: i64 = 60;

/// Stage-1 (per-day) reply shape: `{"candidates": [{text, turn_ids}, ...]}`.
static STANCE_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    Schema::object_container(
        vec![
            Field::new("text", "\"<stance>\""),
            Field::new("turn_ids", "[<id>...]"),
        ],
        "candidates",
    )
    .with_empty_note("Empty list when nothing stands out.")
});

/// Stage-2 (synthesis) reply shape:
/// `{"opinions": [{text, source_turn_ids, importance, reason}, ...]}`.
static SYNTHESIS_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    Schema::object_container(
        vec![
            Field::new("text", "\"<their stance>\""),
            Field::new("source_turn_ids", "[<id>...]"),
            Field::new("importance", "<1-10>"),
            Field::new("reason", "\"<why>\""),
        ],
        "opinions",
    )
});

/// One calendar day of turns (local to the familiar's `display_tz`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DayBatch {
    /// `YYYY-MM-DD` in `display_tz`.
    pub date: String,
    /// The day's turns, id-ordered (chronological).
    pub turns: Vec<HistoryTurn>,
}

impl DayBatch {
    /// Every turn id in this day — the grounding pool.
    #[must_use]
    pub fn turn_ids(&self) -> HashSet<i64> {
        self.turns.iter().map(|t| t.id).collect()
    }

    /// Turns the familiar authored (`role == "assistant"`) — its own acts.
    #[must_use]
    pub fn self_turn_ids(&self) -> HashSet<i64> {
        self.turns
            .iter()
            .filter(|t| t.role == "assistant")
            .map(|t| t.id)
            .collect()
    }
}

/// The log slice one opinion pass reasons over, bucketed by local day.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpinionWindow {
    /// Familiar this window belongs to.
    pub familiar_id: String,
    /// Day batches, oldest-first.
    pub days: Vec<DayBatch>,
    /// The prior sleep watermark, if any.
    pub prior_watermark: Option<SleepWatermark>,
    /// True (uncapped) highest turn id — the turn-axis watermark target.
    pub max_turn_id: i64,
}

/// A stance from one day, grounded in that day's turn-ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StanceMoment {
    /// The stance text.
    pub text: String,
    /// The day it belongs to (`YYYY-MM-DD`).
    pub date: String,
    /// Cited turn ids (⊆ the day's ids).
    pub turn_ids: Vec<i64>,
}

/// An accepted opinion, ready to record as an `ego:` fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpinionFact {
    /// The opinion text.
    pub text: String,
    /// Cited turn ids.
    pub source_turn_ids: Vec<i64>,
    /// `YYYY-MM-DD`, earliest grounding day.
    pub valid_from_date: String,
    /// Whether ≥1 source turn the familiar authored.
    pub self_grounded: bool,
    /// LLM-rated 1-10 durability/centrality.
    pub importance: i64,
}

/// A synthesized opinion a grounding rail refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedOpinion {
    /// The raw LLM proposal object.
    pub payload: Value,
    /// The rail name that refused it.
    pub rail: String,
    /// Human-readable detail.
    pub detail: String,
}

/// Validated outcome of one opinion pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpinionPlan {
    /// Familiar this plan belongs to.
    pub familiar_id: String,
    /// Accepted opinions.
    pub opinions: Vec<OpinionFact>,
    /// Rail-blocked proposals.
    pub rejected: Vec<RejectedOpinion>,
    /// `no_self_authored: {text}` flags (accepted but the room's stance, not hers).
    pub flags: Vec<String>,
    /// Turn-axis watermark the apply step advances to.
    pub new_last_turn_id: i64,
    /// How many day batches the window considered.
    pub days_considered: usize,
    /// How many stance-moments fed the synthesis.
    pub stance_moments_considered: usize,
    /// Non-fatal notes.
    pub notes: Vec<String>,
}

impl OpinionPlan {
    /// A plan with the counts zeroed and no notes (the Python default-arg
    /// constructor; [`validate_opinions`] fills the counts directly).
    #[must_use]
    pub fn new(
        familiar_id: impl Into<String>,
        opinions: Vec<OpinionFact>,
        rejected: Vec<RejectedOpinion>,
        flags: Vec<String>,
        new_last_turn_id: i64,
    ) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            opinions,
            rejected,
            flags,
            new_last_turn_id,
            days_considered: 0,
            stance_moments_considered: 0,
            notes: Vec::new(),
        }
    }
}

/// Outcome of recording an opinion plan's opinions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpinionApplyReport {
    /// `(opinion text, fact id)` per recorded opinion.
    pub recorded: Vec<(String, i64)>,
    /// The advanced `last_turn_id`.
    pub watermark: i64,
}

/// Group turns into calendar-day batches in `tz_name`, oldest first.
///
/// Turn timestamps are UTC-aware; the bucket key is the LOCAL date. Within a day
/// turns stay id-ordered (chronological).
///
/// # Errors
/// [`SleepError::InvalidTimezone`] when `tz_name` is not an IANA zone — Python's
/// `ZoneInfo(tz_name)` raises here too; the engine guard catches it.
pub fn bucket_by_day(turns: &[HistoryTurn], tz_name: &str) -> Result<Vec<DayBatch>, SleepError> {
    let tz: Tz = tz_name
        .parse()
        .map_err(|_| SleepError::InvalidTimezone(tz_name.to_owned()))?;
    let mut sorted: Vec<&HistoryTurn> = turns.iter().collect();
    sorted.sort_by_key(|t| t.id);
    let mut by_date: BTreeMap<String, Vec<HistoryTurn>> = BTreeMap::new();
    for t in sorted {
        let key = t
            .timestamp
            .with_timezone(&tz)
            .format("%Y-%m-%d")
            .to_string();
        by_date.entry(key).or_default().push(t.clone());
    }
    Ok(by_date
        .into_iter()
        .map(|(date, turns)| DayBatch { date, turns })
        .collect())
}

/// Collect turns since the turn-axis watermark, bucketed by day.
///
/// # Errors
/// Propagates store faults and [`SleepError::InvalidTimezone`].
pub async fn gather_days(
    store: &AsyncHistoryStore,
    familiar_id: &str,
    display_tz: &str,
) -> anyhow::Result<OpinionWindow> {
    let prior = store.get_sleep_watermark(familiar_id.to_owned()).await?;
    let min_turn = prior.map_or(0, |p| p.last_turn_id);
    let max_turn_id = store.latest_fts_id(familiar_id.to_owned()).await?;
    let turns = store
        .turns_in_id_range(familiar_id.to_owned(), min_turn, max_turn_id, None)
        .await?;
    let days = bucket_by_day(&turns, display_tz)?;
    Ok(OpinionWindow {
        familiar_id: familiar_id.to_owned(),
        days,
        prior_watermark: prior,
        max_turn_id,
    })
}

/// Render a turn for the prompt, marking the familiar's own turns.
///
/// Only assistant-role turns are the familiar's; a user posting under its
/// display name renders with an explicit disambiguator so the model never reads
/// it as the familiar's own act.
#[must_use]
pub fn render_turn(t: &HistoryTurn, self_name: &str) -> String {
    if t.role == "assistant" {
        return format!("[{}] {self_name} (you): {}", t.id, t.content);
    }
    let mut label = t
        .author
        .as_ref()
        .map_or_else(|| t.role.clone(), crate::identity::Author::label);
    if label == self_name {
        label = format!("{label} (a user, not you)");
    }
    format!("[{}] {label}: {}", t.id, t.content)
}

/// 1-10 importance from model output; clamp range, default 5. Never a rejection
/// reason — a bad/missing value degrades to neutral 5.
#[must_use]
pub fn coerce_importance(raw: Option<&Value>) -> i64 {
    match raw {
        // JSON integers clamp to [1, 10]. A JSON `true`/`false` is `Value::Bool`
        // (distinct from `Number`), a float has no `as_i64`, and absent/other
        // values all fall to the wildcard — every non-integer degrades to 5,
        // matching Python's `isinstance(bool) → 5` / non-int → 5 rules.
        Some(Value::Number(n)) => n.as_i64().map_or(5, |v| v.clamp(1, 10)),
        Some(Value::String(s)) => {
            let t = s.trim();
            // Python: raw.strip().lstrip("-").isdigit() then int(raw.strip()).
            let body = t.strip_prefix('-').unwrap_or(t);
            if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
                t.parse::<i64>().map_or(5, |v| v.clamp(1, 10))
            } else {
                5
            }
        }
        _ => 5,
    }
}

/// Build the per-day stance prompt.
///
/// `system` is the config-sourced static instruction (`{self_name}` filled here,
/// crash-safe); the known-bits deny block and the JSON reply shape are assembled
/// in code and appended.
#[must_use]
pub fn build_stance_prompt(
    day: &DayBatch,
    self_name: &str,
    denylist: &[String],
    system: &str,
) -> Vec<Message> {
    let deny = if denylist.is_empty() {
        String::new()
    } else {
        let body = denylist
            .iter()
            .map(|d| format!("- {d}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "\n\nKNOWN BITS / NOISE (already judged not-real — {self_name} may have a TAKE on \
             these, e.g. finding a running joke tedious, but must never treat them as true \
             events):\n{body}"
        )
    };
    let instruction = fill_placeholders(system, &[("self_name", self_name)]);
    let body = day
        .turns
        .iter()
        .map(|t| render_turn(t, self_name))
        .collect::<Vec<_>>()
        .join("\n");
    let contract = render_contract(&STANCE_SCHEMA);
    vec![
        Message::new("system", format!("{instruction}{deny}\n\n{contract}")),
        Message::new("user", format!("Day {}:\n{body}", day.date)),
    ]
}

/// Stage 1: pull stance-moments for one day, grounding ⊆ the day.
///
/// # Errors
/// Propagates LLM transport faults.
pub async fn extract_stance_moments(
    llm: &dyn LlmClient,
    day: &DayBatch,
    self_name: &str,
    denylist: &[String],
    system: &str,
) -> anyhow::Result<Vec<StanceMoment>> {
    let messages = build_stance_prompt(day, self_name, denylist, system);
    let result = request_structured(llm, &messages, &STANCE_SCHEMA, DEFAULT_MAX_RETRIES).await?;
    let Some(obj) = result.value.as_ref().and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let Some(raw) = obj.get("candidates").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let day_ids = day.turn_ids();
    let mut out: Vec<StanceMoment> = Vec::new();
    for item in raw {
        if !item.is_object() {
            continue;
        }
        let text = py_str_field(item, "text").trim().to_owned();
        if text.is_empty() {
            continue;
        }
        // code-enforce: keep only ids that really belong to this day.
        let ids: Vec<i64> = coerce_positive_int_list(item.get("turn_ids").unwrap_or(&Value::Null))
            .into_iter()
            .filter(|i| day_ids.contains(i))
            .collect();
        if ids.is_empty() {
            continue;
        }
        out.push(StanceMoment {
            text,
            date: day.date.clone(),
            turn_ids: ids,
        });
    }
    Ok(out)
}

/// Build the synthesis prompt.
///
/// `system` is the config-sourced static instruction (`{self_name}` filled here,
/// crash-safe); the prior-dossier block and the JSON reply shape are appended in
/// code.
#[must_use]
pub fn build_synthesis_prompt(
    stance_moments: &[StanceMoment],
    self_name: &str,
    prior_self_dossier: Option<&str>,
    system: &str,
) -> Vec<Message> {
    let prior = match prior_self_dossier {
        Some(d) if !d.is_empty() => format!(
            "\n\n{self_name}'s existing self-understanding (refine/extend, do not blindly \
             repeat):\n{d}"
        ),
        _ => String::new(),
    };
    let instruction = fill_placeholders(system, &[("self_name", self_name)]);
    let lines = stance_moments
        .iter()
        .map(|c| {
            format!(
                "- ({}) ids={}: {}",
                c.date,
                py_int_list_repr(&c.turn_ids),
                c.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let body = format!("Stance-moments:\n{lines}");
    let contract = render_contract(&SYNTHESIS_SCHEMA);
    vec![
        Message::new("system", format!("{instruction}{prior}\n\n{contract}")),
        Message::new("user", body),
    ]
}

/// Stage 2: one call → raw opinion dicts (validated separately). An empty
/// stance-moment list short-circuits to `[]` WITHOUT an LLM call.
///
/// # Errors
/// Propagates LLM transport faults.
pub async fn synthesize(
    llm: &dyn LlmClient,
    stance_moments: &[StanceMoment],
    self_name: &str,
    prior_self_dossier: Option<&str>,
    system: &str,
) -> anyhow::Result<Vec<Value>> {
    if stance_moments.is_empty() {
        return Ok(Vec::new());
    }
    let messages = build_synthesis_prompt(stance_moments, self_name, prior_self_dossier, system);
    let result = request_structured(llm, &messages, &SYNTHESIS_SCHEMA, DEFAULT_MAX_RETRIES).await?;
    let Some(obj) = result.value.as_ref().and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let Some(raw) = obj.get("opinions").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    Ok(raw
        .iter()
        .filter(|item| item.is_object())
        .cloned()
        .collect())
}

/// Filter synthesized opinions through the grounding rails. Pure, no I/O.
///
/// Rails per raw item in order: `empty_text` → `ungrounded` (empty or any id
/// outside the union of ALL stance-moments' ids) → `duplicate` (normalized-text
/// dedup) → `cap` (counts accepted opinions). An opinion grounded in no turn the
/// familiar authored is accepted but FLAGGED. `valid_from` = earliest grounding
/// day.
///
/// # Errors
/// [`SleepError::EmptyOpinionGroundingDays`] when an accepted opinion's ids are
/// all absent from the window's turn→day map (Python's `min(...)` raises
/// `ValueError` here). Unreachable via [`plan_opinions`], where
/// `grounding_union ⊆ turn_day` by construction.
pub fn validate_opinions(
    raw: &[Value],
    stance_moments: &[StanceMoment],
    window: &OpinionWindow,
    cap: i64,
    notes: Vec<String>,
) -> Result<OpinionPlan, SleepError> {
    let grounding_union: HashSet<i64> = stance_moments
        .iter()
        .flat_map(|c| c.turn_ids.iter().copied())
        .collect();
    let mut turn_day: BTreeMap<i64, String> = BTreeMap::new();
    let mut self_ids: HashSet<i64> = HashSet::new();
    for d in &window.days {
        for t in &d.turns {
            turn_day.insert(t.id, d.date.clone());
            if t.role == "assistant" {
                self_ids.insert(t.id);
            }
        }
    }

    let mut opinions: Vec<OpinionFact> = Vec::new();
    let mut rejected: Vec<RejectedOpinion> = Vec::new();
    let mut flags: Vec<String> = Vec::new();
    let mut seen_norm: HashSet<String> = HashSet::new();

    for payload in raw {
        let text = py_str_field(payload, "text").trim().to_owned();
        if text.is_empty() {
            rejected.push(RejectedOpinion {
                payload: payload.clone(),
                rail: "empty_text".to_owned(),
                detail: "blank text".to_owned(),
            });
            continue;
        }
        let ids = coerce_positive_int_list(payload.get("source_turn_ids").unwrap_or(&Value::Null));
        let bad: Vec<i64> = ids
            .iter()
            .copied()
            .filter(|i| !grounding_union.contains(i))
            .collect();
        if ids.is_empty() || !bad.is_empty() {
            rejected.push(RejectedOpinion {
                payload: payload.clone(),
                rail: "ungrounded".to_owned(),
                detail: format!(
                    "ids={} bad={}",
                    py_int_list_repr(&ids),
                    py_int_list_repr(&bad)
                ),
            });
            continue;
        }
        let norm = normalize_fact_text(&text);
        if seen_norm.contains(&norm) {
            rejected.push(RejectedOpinion {
                payload: payload.clone(),
                rail: "duplicate".to_owned(),
                detail: text.clone(),
            });
            continue;
        }
        if len_i64(opinions.len()) >= cap {
            rejected.push(RejectedOpinion {
                payload: payload.clone(),
                rail: "cap".to_owned(),
                detail: format!("cap={cap}"),
            });
            continue;
        }
        seen_norm.insert(norm);
        let self_grounded = ids.iter().any(|i| self_ids.contains(i));
        if !self_grounded {
            flags.push(format!("no_self_authored: {text}"));
        }
        // earliest grounding day (honest timeline); ids are all in grounding_union
        // ⊆ turn_day, so `min` is always present via plan_opinions. An absent
        // day map (inconsistent window/stance pairing) mirrors Python's `min()`
        // ValueError rather than silently degrading to an empty date.
        let valid_from = ids
            .iter()
            .filter_map(|i| turn_day.get(i))
            .min()
            .cloned()
            .ok_or(SleepError::EmptyOpinionGroundingDays)?;
        opinions.push(OpinionFact {
            text,
            source_turn_ids: ids,
            valid_from_date: valid_from,
            self_grounded,
            importance: coerce_importance(payload.get("importance")),
        });
    }

    Ok(OpinionPlan {
        familiar_id: window.familiar_id.clone(),
        opinions,
        rejected,
        flags,
        new_last_turn_id: window.max_turn_id,
        days_considered: window.days.len(),
        stance_moments_considered: stance_moments.len(),
        notes,
    })
}

/// Gather days → per-day stance-moments → one synthesis → validate.
///
/// Days are processed sequentially in date order (LLM budget/ordering parity).
///
/// # Errors
/// Propagates store + LLM transport faults.
#[allow(clippy::too_many_arguments)]
pub async fn plan_opinions(
    store: &AsyncHistoryStore,
    llm: &dyn LlmClient,
    familiar_id: &str,
    display_tz: &str,
    self_name: &str,
    denylist: &[String],
    prior_self_dossier: Option<&str>,
    cap: i64,
    stance_system: &str,
    synthesis_system: &str,
) -> anyhow::Result<OpinionPlan> {
    let window = gather_days(store, familiar_id, display_tz).await?;
    let mut stance_moments: Vec<StanceMoment> = Vec::new();
    for day in &window.days {
        let moments = extract_stance_moments(llm, day, self_name, denylist, stance_system).await?;
        stance_moments.extend(moments);
    }
    let raw = synthesize(
        llm,
        &stance_moments,
        self_name,
        prior_self_dossier,
        synthesis_system,
    )
    .await?;
    let plan = validate_opinions(&raw, &stance_moments, &window, cap, Vec::new())?;
    tracing::info!(
        target: "familiar_connect.sleep.opinion_formation",
        "opinion plan familiar={familiar_id} days={} stance_moments={} opinions={} rejected={} flags={}",
        plan.days_considered,
        plan.stance_moments_considered,
        plan.opinions.len(),
        plan.rejected.len(),
        plan.flags.len(),
    );
    Ok(plan)
}

/// Record opinions as `ego:` facts; advance the turn watermark.
///
/// Each opinion becomes one global (channel-less) fact with the single ego
/// subject, `valid_from = <date>T00:00:00+00:00` (midnight UTC of the earliest
/// grounding day), and the coerced importance. Then advances **only**
/// `last_turn_id`.
///
/// # Errors
/// Propagates store faults.
pub async fn apply_opinions(
    store: &AsyncHistoryStore,
    plan: &OpinionPlan,
    familiar_display_name: Option<&str>,
) -> anyhow::Result<OpinionApplyReport> {
    let fam = &plan.familiar_id;
    let subj = FactSubject {
        canonical_key: ego_canonical_key(fam),
        display_at_write: familiar_display_name.map_or_else(|| fam.clone(), ToOwned::to_owned),
    };
    let mut recorded: Vec<(String, i64)> = Vec::new();
    for op in &plan.opinions {
        // valid_from = midnight UTC of the source day.
        let valid_from: DateTime<Utc> =
            parse_iso(&format!("{}T00:00:00+00:00", op.valid_from_date)).ok_or_else(|| {
                anyhow::anyhow!("invalid valid_from_date: {}", op.valid_from_date)
            })?;
        // `dedup` stays at its `true` default, matching Python's `append_fact`.
        let builder = crate::history::store::AppendFact::new(
            fam.clone(),
            None, // opinions are global stances, not channel-bound
            op.text.clone(),
            op.source_turn_ids.clone(),
        )
        .subjects(vec![subj.clone()])
        .valid_from(valid_from)
        .importance(op.importance);
        let fact = store.append_fact(builder).await?;
        recorded.push((op.text.clone(), fact.id));
    }
    // opinion-formation owns the TURN axis only — leaves the fact axis intact.
    store
        .advance_sleep_watermark(fam.clone(), None, Some(plan.new_last_turn_id))
        .await?;
    tracing::info!(
        target: "familiar_connect.sleep.opinion_formation",
        "opinions applied familiar={fam} recorded={} turn_watermark={}",
        recorded.len(),
        plan.new_last_turn_id,
    );
    Ok(OpinionApplyReport {
        recorded,
        watermark: plan.new_last_turn_id,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DayBatch, OpinionWindow, StanceMoment, bucket_by_day, build_stance_prompt,
        build_synthesis_prompt, coerce_importance, extract_stance_moments, render_turn,
        validate_opinions,
    };
    use crate::history::store::HistoryTurn;
    use crate::identity::Author;
    use crate::llm::{LlmClient, LlmDelta, Message};
    use async_trait::async_trait;
    use chrono::{DateTime, TimeZone, Utc};
    use futures::stream::BoxStream;
    use serde_json::{Value, json};
    use std::sync::Mutex;

    /// One-shot scripted LLM: returns `reply` on the first `chat`, then empty.
    struct OneShotLlm {
        reply: Mutex<Option<String>>,
    }

    impl OneShotLlm {
        fn new(reply: &str) -> Self {
            Self {
                reply: Mutex::new(Some(reply.to_owned())),
            }
        }
    }

    #[async_trait]
    impl LlmClient for OneShotLlm {
        async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
            let content = self.reply.lock().unwrap().take().unwrap_or_default();
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

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }

    fn turn(tid: i64, when: DateTime<Utc>, role: &str, content: &str) -> HistoryTurn {
        HistoryTurn {
            id: tid,
            timestamp: when,
            role: role.to_owned(),
            author: None,
            content: content.to_owned(),
            channel_id: 1,
            mode: None,
            platform_message_id: None,
            reply_to_message_id: None,
            guild_id: None,
            arrived_at: None,
            consumed_at: None,
            pings_bot: false,
        }
    }

    fn user_turn(tid: i64, display: &str) -> HistoryTurn {
        HistoryTurn {
            id: tid,
            timestamp: at(2026, 6, 12, 0, 0),
            role: "user".to_owned(),
            author: Some(Author::new(
                "discord",
                tid.to_string(),
                Some("u".to_owned()),
                Some(display.to_owned()),
            )),
            content: "hi".to_owned(),
            channel_id: 1,
            mode: None,
            platform_message_id: None,
            reply_to_message_id: None,
            guild_id: None,
            arrived_at: None,
            consumed_at: None,
            pings_bot: false,
        }
    }

    fn cand(text: &str, date: &str, ids: Vec<i64>) -> StanceMoment {
        StanceMoment {
            text: text.to_owned(),
            date: date.to_owned(),
            turn_ids: ids,
        }
    }

    fn day(date: &str, turns: Vec<HistoryTurn>) -> DayBatch {
        DayBatch {
            date: date.to_owned(),
            turns,
        }
    }

    fn window(days: Vec<DayBatch>) -> OpinionWindow {
        let max_id = days
            .iter()
            .flat_map(|d| d.turns.iter().map(|t| t.id))
            .max()
            .unwrap_or(0);
        OpinionWindow {
            familiar_id: "fam".to_owned(),
            days,
            prior_watermark: None,
            max_turn_id: max_id,
        }
    }

    // --- render_turn ---------------------------------------------------------

    #[test]
    fn self_turn_marked_you() {
        let r = render_turn(
            &turn(1, at(2026, 6, 12, 0, 0), "assistant", "x"),
            "Sapphire",
        );
        assert!(r.contains("(you)"));
        assert!(r.contains("Sapphire"));
    }

    #[test]
    fn same_named_user_disambiguated() {
        let r = render_turn(&user_turn(2, "Sapphire"), "Sapphire");
        assert!(!r.contains("(you)"));
        assert!(r.to_lowercase().contains("not you"));
    }

    #[test]
    fn other_user_plain() {
        let r = render_turn(&user_turn(3, "Aria"), "Sapphire");
        assert!(r.contains("Aria"));
        assert!(!r.contains("(you)"));
        assert!(!r.to_lowercase().contains("not you"));
    }

    // --- bucket_by_day -------------------------------------------------------

    #[test]
    fn buckets_by_local_calendar_day() {
        // 02:00Z = 2026-06-11 19:00 PT ; 10:00Z = 2026-06-12 03:00 PT
        let turns = vec![
            turn(1, at(2026, 6, 12, 2, 0), "assistant", "x"),
            turn(2, at(2026, 6, 12, 10, 0), "assistant", "x"),
        ];
        let days = bucket_by_day(&turns, "America/Los_Angeles").unwrap();
        assert_eq!(
            days.iter().map(|d| d.date.clone()).collect::<Vec<_>>(),
            vec!["2026-06-11".to_owned(), "2026-06-12".to_owned(),]
        );
        assert_eq!(days[0].turn_ids(), std::collections::HashSet::from([1]));
        assert_eq!(days[1].turn_ids(), std::collections::HashSet::from([2]));
    }

    #[test]
    fn days_ordered_oldest_first() {
        let turns = vec![
            turn(2, at(2026, 6, 13, 12, 0), "assistant", "x"),
            turn(1, at(2026, 6, 12, 12, 0), "assistant", "x"),
        ];
        let days = bucket_by_day(&turns, "UTC").unwrap();
        assert_eq!(
            days.iter().map(|d| d.date.clone()).collect::<Vec<_>>(),
            vec!["2026-06-12".to_owned(), "2026-06-13".to_owned(),]
        );
    }

    #[test]
    fn self_turn_ids_are_assistant_role() {
        let turns = vec![
            turn(1, at(2026, 6, 12, 12, 0), "assistant", "x"),
            turn(2, at(2026, 6, 12, 12, 1), "user", "x"),
        ];
        let d = &bucket_by_day(&turns, "UTC").unwrap()[0];
        assert_eq!(d.self_turn_ids(), std::collections::HashSet::from([1]));
        assert_eq!(d.turn_ids(), std::collections::HashSet::from([1, 2]));
    }

    #[test]
    fn bucket_empty_is_empty() {
        assert_eq!(bucket_by_day(&[], "UTC").unwrap(), Vec::<DayBatch>::new());
    }

    // --- prompts -------------------------------------------------------------

    #[test]
    fn synthesis_configured_system_formats_self_name() {
        let msgs = build_synthesis_prompt(
            &[cand("likes lo-fi", "2026-06-12", vec![1])],
            "Sapphire",
            None,
            "settle {self_name}'s views",
        );
        assert!(msgs[0].content_str().contains("settle Sapphire's views"));
        assert!(!msgs[0].content_str().contains("{self_name}"));
    }

    #[test]
    fn synthesis_stray_brace_degrades() {
        let msgs = build_synthesis_prompt(
            &[cand("likes lo-fi", "2026-06-12", vec![1])],
            "Sapphire",
            None,
            "settle {self_name}'s views {money} a { brace",
        );
        assert!(
            msgs[0]
                .content_str()
                .contains("settle Sapphire's views {money} a { brace")
        );
    }

    #[test]
    fn stance_configured_system_formats_self_name() {
        let d = day(
            "2026-06-12",
            vec![turn(1, at(2026, 6, 12, 12, 0), "assistant", "x")],
        );
        let msgs = build_stance_prompt(&d, "Sapphire", &[], "stances for {self_name}");
        assert!(msgs[0].content_str().contains("stances for Sapphire"));
        assert!(!msgs[0].content_str().contains("{self_name}"));
    }

    #[test]
    fn stance_stray_brace_degrades() {
        let d = day(
            "2026-06-12",
            vec![turn(1, at(2026, 6, 12, 12, 0), "assistant", "x")],
        );
        let msgs = build_stance_prompt(
            &d,
            "Sapphire",
            &[],
            "stances for {self_name} {mood} a { brace",
        );
        assert!(
            msgs[0]
                .content_str()
                .contains("stances for Sapphire {mood} a { brace")
        );
    }

    // --- extract_stance_moments (grounding ⊆ the day) ------------------------

    #[tokio::test]
    async fn extract_keeps_only_in_day_turn_ids() {
        let d = day(
            "2026-06-12",
            vec![
                turn(1, at(2026, 6, 12, 12, 0), "assistant", "x"),
                turn(2, at(2026, 6, 12, 12, 1), "user", "x"),
            ],
        );
        // LLM cites a real id (1) and a hallucinated out-of-day id (999).
        let reply = json!({"candidates": [{"text": "She liked the joke.", "turn_ids": [1, 999]}]})
            .to_string();
        let llm = OneShotLlm::new(&reply);
        let cands = extract_stance_moments(&llm, &d, "Sapphire", &[], "")
            .await
            .unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].turn_ids, vec![1]); // 999 dropped
    }

    #[tokio::test]
    async fn extract_drops_stance_moment_with_no_valid_ids() {
        let d = day(
            "2026-06-12",
            vec![turn(1, at(2026, 6, 12, 12, 0), "assistant", "x")],
        );
        let reply = json!({"candidates": [{"text": "ungrounded", "turn_ids": [999]}]}).to_string();
        let llm = OneShotLlm::new(&reply);
        let cands = extract_stance_moments(&llm, &d, "Sapphire", &[], "")
            .await
            .unwrap();
        assert!(cands.is_empty());
    }

    // --- validate_opinions ---------------------------------------------------

    fn val_window() -> OpinionWindow {
        window(vec![
            day(
                "2026-06-12",
                vec![
                    turn(1, at(2026, 6, 12, 12, 0), "assistant", "x"),
                    turn(2, at(2026, 6, 12, 12, 1), "user", "x"),
                ],
            ),
            day(
                "2026-06-20",
                vec![turn(5, at(2026, 6, 20, 12, 0), "assistant", "x")],
            ),
        ])
    }

    fn validate(raw: &[Value], cands: &[StanceMoment], cap: i64) -> super::OpinionPlan {
        validate_opinions(raw, cands, &val_window(), cap, Vec::new()).unwrap()
    }

    #[test]
    fn accepts_grounded_and_sets_valid_from_earliest() {
        let cands = vec![
            cand("likes lo-fi", "2026-06-12", vec![1]),
            cand("again", "2026-06-20", vec![5]),
        ];
        let raw = vec![
            json!({"text": "Sapphire loves lo-fi.", "source_turn_ids": [1, 5], "reason": "x"}),
        ];
        let plan = validate(&raw, &cands, 50);
        assert_eq!(plan.opinions.len(), 1);
        let op = &plan.opinions[0];
        assert_eq!(
            op.source_turn_ids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            std::collections::HashSet::from([1, 5])
        );
        assert_eq!(op.valid_from_date, "2026-06-12");
    }

    #[test]
    fn rejects_grounding_outside_candidate_union() {
        let cands = vec![cand("likes lo-fi", "2026-06-12", vec![1])];
        let raw = vec![json!({"text": "x", "source_turn_ids": [1, 5], "reason": "x"})];
        let plan = validate(&raw, &cands, 50);
        assert!(plan.opinions.is_empty());
        assert_eq!(plan.rejected[0].rail, "ungrounded");
    }

    #[test]
    fn rejects_zero_grounding() {
        let cands = vec![cand("likes lo-fi", "2026-06-12", vec![1])];
        let raw = vec![json!({"text": "x", "source_turn_ids": [], "reason": "x"})];
        let plan = validate(&raw, &cands, 50);
        assert!(plan.opinions.is_empty());
        assert_eq!(plan.rejected[0].rail, "ungrounded");
    }

    #[test]
    fn flags_no_self_authored_grounding() {
        // turn 2 is a user turn; an opinion grounded only there isn't HER act.
        let cands = vec![cand("room said", "2026-06-12", vec![2])];
        let raw = vec![
            json!({"text": "Sapphire thinks tea is nice.", "source_turn_ids": [2], "reason": "x"}),
        ];
        let plan = validate(&raw, &cands, 50);
        assert_eq!(plan.opinions.len(), 1);
        assert!(!plan.opinions[0].self_grounded);
        assert!(plan.flags.iter().any(|f| f.contains("no_self_authored")));
    }

    #[test]
    fn dedups_restatements() {
        let cands = vec![
            cand("a", "2026-06-12", vec![1]),
            cand("b", "2026-06-20", vec![5]),
        ];
        let raw = vec![
            json!({"text": "Sapphire loves lo-fi.", "source_turn_ids": [1], "reason": "x"}),
            json!({"text": "Sapphire loves lo-fi!", "source_turn_ids": [5], "reason": "x"}),
        ];
        let plan = validate(&raw, &cands, 50);
        assert_eq!(plan.opinions.len(), 1);
        assert!(plan.rejected.iter().any(|r| r.rail == "duplicate"));
    }

    #[test]
    fn cap_limits_accepted() {
        let cands: Vec<StanceMoment> = (0..5)
            .map(|i| cand(&format!("c{i}"), "2026-06-12", vec![1]))
            .collect();
        let raw: Vec<Value> = (0..5)
            .map(|i| json!({"text": format!("op {i}"), "source_turn_ids": [1], "reason": "x"}))
            .collect();
        let plan = validate(&raw, &cands, 3);
        assert_eq!(plan.opinions.len(), 3);
        assert_eq!(plan.rejected.iter().filter(|r| r.rail == "cap").count(), 2);
    }

    #[test]
    fn errors_when_accepted_grounding_absent_from_window_days() {
        // grounding_union carries id 99 (so the ungrounded rail passes), but 99
        // is in no window turn, so the earliest-grounding-day `min` is empty.
        // Python's `min(... for i in ids if i in turn_day)` raises ValueError;
        // the port mirrors the raise instead of degrading to an empty date.
        // Unreachable via plan_opinions (grounding_union ⊆ turn_day there).
        let cands = vec![cand("orphan stance", "2026-06-12", vec![99])];
        let raw =
            vec![json!({"text": "Sapphire likes tea.", "source_turn_ids": [99], "reason": "x"})];
        let err = validate_opinions(&raw, &cands, &val_window(), 50, Vec::new()).unwrap_err();
        assert!(matches!(err, super::SleepError::EmptyOpinionGroundingDays));
    }

    // --- importance matrix ---------------------------------------------------

    fn importance_of(raw_val: Value) -> i64 {
        let cands = vec![cand("likes lo-fi", "2026-06-12", vec![1])];
        let mut obj = serde_json::Map::new();
        obj.insert("text".to_owned(), json!("Sapphire loves lo-fi."));
        obj.insert("source_turn_ids".to_owned(), json!([1]));
        obj.insert("reason".to_owned(), json!("x"));
        if !raw_val.is_null() {
            obj.insert("importance".to_owned(), raw_val);
        }
        let plan = validate(&[Value::Object(obj)], &cands, 50);
        assert_eq!(plan.opinions.len(), 1);
        plan.opinions[0].importance
    }

    #[test]
    fn importance_matrix() {
        assert_eq!(importance_of(json!(8)), 8);
        assert_eq!(importance_of(json!(99)), 10);
        assert_eq!(importance_of(json!(0)), 1);
        assert_eq!(importance_of(Value::Null), 5); // absent → default 5
        assert_eq!(importance_of(json!("very")), 5);
    }

    #[test]
    fn coerce_importance_units() {
        assert_eq!(coerce_importance(Some(&json!(true))), 5); // bool → 5
        assert_eq!(coerce_importance(Some(&json!(-5))), 1); // clamp low
        assert_eq!(coerce_importance(Some(&json!("7"))), 7); // digit-string
        assert_eq!(coerce_importance(Some(&json!(2.5))), 5); // float → 5
        assert_eq!(coerce_importance(None), 5);
    }
}
