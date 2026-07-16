//! FactExtractor rich-note worker (subsystem 07; Python `processors/fact_extractor.py`).
//!
//! Distils atomic facts from new turns via a cheap background LLM pass; stores
//! them in `facts` with `source_turn_ids` pointing back to `turns`. Advances
//! `memory_writer_watermark` once a batch is processed — even on malformed LLM
//! output — so the worker never loops forever on the same batch.
//!
//! The batch gate is exact: `turns_since_watermark(limit=batch_size)` must
//! return a full `batch_size` before any LLM call fires. `activity_return` turns
//! are filtered out of the LLM batch (self-generated fiction the activity engine
//! already recorded); `sleep_return` (dream) turns stay in the batch but are
//! forced under the self subject and dream-framed by a code rail. The self key
//! (`ego:<id>`) is reserved in the manifest so the model can file the familiar's
//! own narrative there, while self-*capability* statements are dropped by a
//! post-filter regardless of how they were tagged.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use chrono::{DateTime, NaiveDate, Utc};
use regex::Regex;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::diagnostics::spans::timed_async;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{AppendFact, FactSubject, HistoryTurn};
use crate::identity::{ego_canonical_key, is_ego_key};
use crate::llm::{LlmClient, Message};
use crate::log_style as ls;
use crate::prompt_fill::fill_placeholders;
use crate::structured_request::{
    DEFAULT_MAX_RETRIES, Field, Schema, render_contract, request_structured,
};
use crate::support::time::parse_iso;

/// Log/task label + registry name (`rich_note`) for this projector.
const NAME: &str = "fact-extractor";

// Turn-mode tags owned by subsystem 11 (`activities::engine`). That module is
// still a stub in this tree, so the two constants the extractor keys on are
// duplicated here as string literals; a shared-file request tracks re-exporting
// them from `activities` once it lands. Skip logic keys on `turns.mode`, never
// on the display-only `"[returned from "` content prefix.
const ACTIVITY_RETURN_MODE: &str = "activity_return";
const SLEEP_RETURN_MODE: &str = "sleep_return";

/// Reply-shape contract for fact extraction (root=array).
static FACT_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    Schema::array(vec![
        Field::new("text", "\"<one sentence>\"").with_desc("one sentence"),
        Field::new("source_turn_ids", "[<id>...]")
            .with_desc("list of turn ids the fact was distilled from"),
        Field::new("subject_keys", "[<key>...]")
            .with_desc(
                "list of canonical keys from the Participants block, \
                 identifying which people the fact is about. Use this \
                 whenever the fact mentions someone by name and you can \
                 match that name to a participant. Leave it out or empty \
                 if you can't tell.",
            )
            .optional(),
        Field::new("valid_from", "\"<ISO-8601 timestamp>\"")
            .with_desc(
                "only set when the speaker explicitly anchors the fact to \
                 a different moment than 'now' (e.g., 'as of last June', \
                 'back in 2019'). Otherwise omit; the source turn's \
                 timestamp is used.",
            )
            .optional(),
        Field::new("valid_to", "\"<ISO-8601 timestamp>\"")
            .with_desc(
                "world-time end only. Set this when the speaker explicitly \
                 anchors the end of the fact in real time (e.g., 'until \
                 last June', 'ended in 2019', 'no longer lives there as of \
                 March'). Do NOT use valid_to to mark a fact as outdated, \
                 replaced, or superseded by something newer in this \
                 conversation — supersession is tracked separately. When \
                 in doubt, omit.",
            )
            .optional(),
        Field::new("importance", "<1-10>")
            .with_desc(
                "integer 1-10 — how much this fact should influence future \
                 replies. 1 = throwaway aside, 5 = ordinary detail, 10 = \
                 safety-critical / identity-defining (allergies, names, \
                 long-standing preferences, life events). Omit when unsure.",
            )
            .optional(),
    ])
});

/// Prefix pattern marking first-person / generic self-capability "facts".
///
/// `re.match` in Python (anchored at start); the leading `^\s*` keeps the
/// same anchoring here. Note `can(?:not|'t|\s+not)?` makes the suffix optional
/// — a bare "I can …" matches too, faithfully to the Python source.
static SELF_CAPABILITY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:i\s+(?:can(?:not|'t|\s+not)?|do(?:n't|\s+not)|am\s+(?:not|unable))|i'm\s+(?:not|unable)|i\s+have\s+no\b|as\s+(?:an?\s+)?(?:ai|assistant|language\s+model|llm)\b|the\s+(?:assistant|ai|familiar|model|bot)\b)",
    )
    .expect("valid self-capability regex")
});

/// Inability tail following a third-person self-name. Narrow + word-bounded on
/// purpose: copula/dynamic negation ("is not fond"), positive ability ("can
/// sing"), and word-prefix collisions ("cancelled" ⊃ "can") are narrative and
/// must NOT match.
const NAME_CAPABILITY_TAIL: &str = r"\s+(?:cannot\b|can't\b|can\s+not\b|is\s+unable\b|has\s+no\b)";

/// Heuristic prefix-match for self-capability "facts". `name_re` (optional) is
/// the pre-compiled display-name inability matcher.
fn is_self_capability(text: &str, name_re: Option<&Regex>) -> bool {
    if SELF_CAPABILITY_RE.is_match(text) {
        return true;
    }
    name_re.is_some_and(|re| re.is_match(text))
}

/// Distils facts from new turns; forever loop via [`FactExtractor::run`].
pub struct FactExtractor {
    store: Arc<AsyncHistoryStore>,
    llm: Arc<dyn LlmClient>,
    familiar_id: String,
    familiar_display_name: Option<String>,
    dream_extraction_clause: String,
    batch_size: i64,
    participants_max: i64,
    tick_interval: Duration,
}

impl FactExtractor {
    /// Construct with the required handles; knobs default per spec
    /// (`batch_size = 10`, `tick_interval_s = 15.0`, `participants_max = 30`,
    /// no dream clause) and are set via the consuming builders below.
    #[must_use]
    pub fn new(
        store: Arc<AsyncHistoryStore>,
        llm: Arc<dyn LlmClient>,
        familiar_id: impl Into<String>,
    ) -> Self {
        Self {
            store,
            llm,
            familiar_id: familiar_id.into(),
            familiar_display_name: None,
            dream_extraction_clause: String::new(),
            batch_size: 10,
            participants_max: 30,
            tick_interval: Duration::from_secs_f64(15.0),
        }
    }

    /// Display name for the reserved self-subject (title-cased id when unset).
    #[must_use]
    pub fn familiar_display_name(mut self, name: impl Into<String>) -> Self {
        self.familiar_display_name = Some(name.into());
        self
    }

    /// Config-sourced static dream-framing clause (phrasing only; the code rail
    /// enforces self-keying/framing regardless).
    #[must_use]
    pub fn dream_extraction_clause(mut self, clause: impl Into<String>) -> Self {
        self.dream_extraction_clause = clause.into();
        self
    }

    /// Trigger threshold AND per-tick cap (clamped to `>= 1`).
    #[must_use]
    pub const fn batch_size(mut self, batch_size: i64) -> Self {
        self.batch_size = if batch_size < 1 { 1 } else { batch_size };
        self
    }

    /// Manifest hard cap (clamped to `>= 1`).
    #[must_use]
    pub const fn participants_max(mut self, participants_max: i64) -> Self {
        self.participants_max = if participants_max < 1 {
            1
        } else {
            participants_max
        };
        self
    }

    /// Idle-loop interval in seconds.
    #[must_use]
    pub fn tick_interval_s(mut self, secs: f64) -> Self {
        self.tick_interval = Duration::from_secs_f64(secs);
        self
    }

    /// The projector's log/task label.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        NAME
    }

    /// Resolved self-subject display name (explicit, else title-cased id).
    ///
    /// Mirrors Python `familiar_display_name or familiar_id.title()`: the `or`
    /// falls back for an EMPTY explicit name as well as an absent one, so an
    /// explicit `""` still title-cases the id (and never feeds the empty-name
    /// inability regex, which would over-drop).
    fn display_name(&self) -> String {
        resolve_display_name(self.familiar_display_name.as_deref(), &self.familiar_id)
    }

    /// Display-name inability matcher (recompiled per tick from the resolved
    /// display name).
    fn self_name_capability_re(&self) -> Regex {
        let escaped = regex::escape(&self.display_name());
        Regex::new(&format!("(?i)^\\s*{escaped}{NAME_CAPABILITY_TAIL}"))
            .expect("valid display-name capability regex")
    }

    /// Forever loop; tick on interval. Cancel the token to stop.
    pub async fn run(&self, cancel: CancellationToken) {
        loop {
            if cancel.is_cancelled() {
                break;
            }
            if let Err(exc) = self.tick().await {
                tracing::warn!(
                    target: "familiar_connect.processors.fact_extractor",
                    "{} {}",
                    ls::tag("FactExtractor", ls::R),
                    ls::kv_styled("tick_error", &format!("{exc:?}"), ls::W, ls::R),
                );
            }
            tokio::select! {
                () = cancel.cancelled() => break,
                () = tokio::time::sleep(self.tick_interval) => {}
            }
        }
    }

    /// Process one batch of unprocessed turns if enough accumulated.
    pub async fn tick(&self) -> anyhow::Result<()> {
        timed_async("facts.tick", async move { self.tick_inner().await }).await
    }

    #[allow(
        clippy::too_many_lines,
        reason = "faithful 1:1 transliteration of the Python tick body"
    )]
    async fn tick_inner(&self) -> anyhow::Result<()> {
        let new_turns = self
            .store
            .turns_since_watermark(self.familiar_id.clone(), self.batch_size)
            .await?;
        if i64::try_from(new_turns.len()).unwrap_or(i64::MAX) < self.batch_size {
            return Ok(());
        }

        // Activity-return turns never enter extraction (self-generated fiction;
        // the activity engine already recorded a mechanical event-fact). Keyed
        // on `turns.mode`. Sleep-return (dream) turns DO enter — under the
        // dream-framing rail below.
        let batch: Vec<&HistoryTurn> = new_turns
            .iter()
            .filter(|t| t.mode.as_deref() != Some(ACTIVITY_RETURN_MODE))
            .collect();
        let skipped_return = new_turns.len() - batch.len();
        let dream_ids: HashSet<i64> = batch
            .iter()
            .filter(|t| t.mode.as_deref() == Some(SLEEP_RETURN_MODE))
            .map(|t| t.id)
            .collect();

        let self_key = ego_canonical_key(&self.familiar_id);
        let self_name = self.display_name();
        let mut facts: Vec<FactItem> = Vec::new();
        let mut participants: Vec<(String, String)> = Vec::new();
        if !batch.is_empty() {
            participants = build_participants(
                self.store.as_ref(),
                &batch,
                &self.familiar_id,
                self.participants_max,
            )
            .await?;
            // Reserve the self-subject so the model can tag the familiar's own
            // narrative; validated like any other manifest key downstream.
            upsert(&mut participants, self_key.clone(), self_name.clone());
            let prompt = build_extract_prompt(
                &batch,
                &participants,
                Some(&self_key),
                Some(&self_name),
                &dream_ids,
                &self.dream_extraction_clause,
            );
            let result = request_structured(
                self.llm.as_ref(),
                &prompt,
                &FACT_SCHEMA,
                DEFAULT_MAX_RETRIES,
            )
            .await?;
            facts = normalize_fact_items(result.value.as_ref());
        }
        let valid_ids: HashSet<i64> = batch.iter().map(|t| t.id).collect();
        let channel_ids: HashMap<i64, i64> = batch.iter().map(|t| (t.id, t.channel_id)).collect();
        let ts_by_id: HashMap<i64, DateTime<Utc>> =
            batch.iter().map(|t| (t.id, t.timestamp)).collect();

        let name_re = self.self_name_capability_re();
        let mut dropped_self_cap = 0_i64;
        for fact in &facts {
            let mut source_ids: Vec<i64> = fact
                .source_turn_ids
                .iter()
                .copied()
                .filter(|i| valid_ids.contains(i))
                .collect();
            if source_ids.is_empty() {
                // Fall back to the whole batch minus dream turns (real facts
                // about people stay person-attributable); all batch ids if
                // every turn was a dream. Python `set` order is unspecified —
                // sort ascending for determinism (membership is the contract).
                let mut fallback: Vec<i64> = valid_ids
                    .iter()
                    .copied()
                    .filter(|i| !dream_ids.contains(i))
                    .collect();
                if fallback.is_empty() {
                    fallback = valid_ids.iter().copied().collect();
                }
                fallback.sort_unstable();
                source_ids = fallback;
            }
            let channel_id = channel_ids.get(&source_ids[0]).copied();
            let mut text = py_str(&fact.text).trim().to_string();
            if text.is_empty() {
                continue;
            }
            if is_self_capability(&text, Some(&name_re)) {
                dropped_self_cap += 1;
                tracing::debug!(
                    target: "familiar_connect.processors.fact_extractor",
                    "{} {} {}",
                    ls::tag("Facts", ls::Y),
                    ls::kv_styled("drop", "self_capability", ls::W, ls::LY),
                    ls::kv_styled("text", &ls::trunc(&text, 120), ls::W, ls::LW),
                );
                continue;
            }
            let mut subjects = resolve_subjects(&fact.subject_keys, &participants);
            if source_ids.iter().any(|i| dream_ids.contains(i)) {
                // Claim-discipline rail (code, not prompt): dream-grounded facts
                // land under self ONLY, dream-framed — never under a person.
                subjects = vec![FactSubject {
                    canonical_key: self_key.clone(),
                    display_at_write: self_name.clone(),
                }];
                if !text.to_lowercase().contains("dream") {
                    text = format!("{self_name} dreamed that {text}");
                }
            }
            let valid_from = parse_iso_dt(fact.valid_from.as_ref())
                .or_else(|| ts_by_id.get(&source_ids[0]).copied());
            let valid_to = parse_iso_dt(fact.valid_to.as_ref());
            let importance = parse_importance(fact.importance.as_ref());

            let mut append = AppendFact::new(
                self.familiar_id.clone(),
                channel_id,
                text.clone(),
                source_ids.clone(),
            )
            .subjects(subjects.clone());
            if let Some(vf) = valid_from {
                append = append.valid_from(vf);
            }
            if let Some(vt) = valid_to {
                append = append.valid_to(vt);
            }
            if let Some(imp) = importance {
                append = append.importance(imp);
            }
            self.store.append_fact(append).await?;

            // Mirror resolved subjects into `turn_mentions` (bridges bare-text
            // name references into the same index Discord pings populate). Skip
            // the self key — it is always-injected by the read path, so
            // mirroring it only pollutes the index and burns a max_people slot.
            let keys: Vec<String> = subjects
                .iter()
                .filter(|s| !is_ego_key(&s.canonical_key))
                .map(|s| s.canonical_key.clone())
                .collect();
            if !keys.is_empty() {
                for tid in &source_ids {
                    self.store.record_mentions(*tid, keys.clone()).await?;
                }
            }
        }

        // Always advance the watermark — even on empty/bad output — to prevent
        // loops. Uses the last of the ORIGINAL batch (filtered return turns
        // advance past too).
        let last_id = new_turns[new_turns.len() - 1].id;
        self.store
            .put_writer_watermark(self.familiar_id.clone(), last_id)
            .await?;
        tracing::info!(
            target: "familiar_connect.processors.fact_extractor",
            "{} {} {} {} {} {}",
            ls::tag("Facts", ls::LC),
            ls::kv_styled("batch_size", &new_turns.len().to_string(), ls::W, ls::LY),
            ls::kv_styled("extracted", &facts.len().to_string(), ls::W, ls::LC),
            ls::kv_styled("dropped_self_cap", &dropped_self_cap.to_string(), ls::W, ls::LY),
            ls::kv_styled("skipped_return", &skipped_return.to_string(), ls::W, ls::LY),
            ls::kv_styled("watermark", &last_id.to_string(), ls::W, ls::LW),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Prompt + parsing helpers (private)
// ---------------------------------------------------------------------------

/// Insert-or-update `(key, value)` into an ordered participant list, preserving
/// the first-occurrence position (Python dict assignment semantics).
fn upsert(out: &mut Vec<(String, String)>, key: String, value: String) {
    if let Some(entry) = out.iter_mut().find(|(k, _)| *k == key) {
        entry.1 = value;
    } else {
        out.push((key, value));
    }
}

/// Map `canonical_key` → current display name for fact resolution, batch-first.
///
/// (1) Every batch turn's author, resolved via `resolve_label` with the turn's
/// `guild_id`; (2) widened per batch channel with `recent_distinct_authors`,
/// skipping keys already present, hard-capped at `max_total` TOTAL. Insertion
/// order is meaningful (batch authors first, then widened) and preserved.
async fn build_participants(
    store: &AsyncHistoryStore,
    turns: &[&HistoryTurn],
    familiar_id: &str,
    max_total: i64,
) -> anyhow::Result<Vec<(String, String)>> {
    let cap = usize::try_from(max_total.max(0)).unwrap_or(usize::MAX);
    let mut out: Vec<(String, String)> = Vec::new();
    // Batch authors first, tracking each channel's last-seen guild id.
    let mut channel_guilds: Vec<(i64, Option<i64>)> = Vec::new();
    for t in turns {
        if let Some(entry) = channel_guilds.iter_mut().find(|(c, _)| *c == t.channel_id) {
            entry.1 = t.guild_id;
        } else {
            channel_guilds.push((t.channel_id, t.guild_id));
        }
        if let Some(author) = &t.author {
            let key = author.canonical_key();
            let label = store
                .resolve_label(key.clone(), t.guild_id, Some(familiar_id.to_string()))
                .await?;
            upsert(&mut out, key, label);
        }
    }
    // Widen with recent channel participants, capped at max_total TOTAL.
    for (channel_id, guild_id) in &channel_guilds {
        if out.len() >= cap {
            break;
        }
        let authors = store
            .recent_distinct_authors(familiar_id.to_string(), *channel_id, max_total)
            .await?;
        for author in &authors {
            let key = author.canonical_key();
            if out.iter().any(|(k, _)| *k == key) {
                continue;
            }
            if out.len() >= cap {
                break;
            }
            let label = store
                .resolve_label(key.clone(), *guild_id, Some(familiar_id.to_string()))
                .await?;
            out.push((key, label));
        }
    }
    Ok(out)
}

/// Validate LLM-emitted `subject_keys` against the manifest. Unknown keys are
/// dropped silently; duplicates deduped (first occurrence); `display_at_write`
/// comes from the manifest, never LLM echo.
fn resolve_subjects(raw: &[String], participants: &[(String, String)]) -> Vec<FactSubject> {
    let mut out: Vec<FactSubject> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for key in raw {
        if seen.contains(key.as_str()) {
            continue;
        }
        if let Some((_, display)) = participants.iter().find(|(k, _)| k == key) {
            seen.insert(key.as_str());
            out.push(FactSubject {
                canonical_key: key.clone(),
                display_at_write: display.clone(),
            });
        }
    }
    out
}

#[allow(
    clippy::too_many_lines,
    reason = "the persona/guidance prose blocks are ported verbatim from Python"
)]
fn build_extract_prompt(
    turns: &[&HistoryTurn],
    participants: &[(String, String)],
    self_key: Option<&str>,
    self_name: Option<&str>,
    dream_turn_ids: &HashSet<i64>,
    dream_clause_template: &str,
) -> Vec<Message> {
    let mut dream_clause = String::new();
    if !dream_turn_ids.is_empty() {
        let mut ids_vec: Vec<i64> = dream_turn_ids.iter().copied().collect();
        ids_vec.sort_unstable();
        let ids = ids_vec
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let filled = fill_placeholders(
            dream_clause_template,
            &[
                ("self_name", self_name.unwrap_or("")),
                ("self_key", self_key.unwrap_or("")),
                ("ids", ids.as_str()),
            ],
        );
        if !filled.is_empty() {
            dream_clause = format!("\n\n{filled}");
        }
    }
    let mut self_clause = String::new();
    if let (Some(self_key), Some(self_name)) = (self_key, self_name) {
        if !self_key.is_empty() && !self_name.is_empty() {
            self_clause = format!(
                "\n\nYou ({self_name}) are a participant too, keyed \
                 ``{self_key}``. Record YOUR OWN narrative under that key: \
                 the bits/performances you ran, choices you made, and your \
                 relational stances or feelings toward people ('{self_name} \
                 performed a teasing bit', '{self_name} chose to disengage', \
                 '{self_name} privately felt proud'). Tag those with \
                 ``{self_key}`` — never file them under the person the bit \
                 was about. This is the ONLY exception to 'not about you', \
                 and it covers narrative/choices/feelings ONLY. Self-\
                 CAPABILITY or limitation statements ('I cannot remember \
                 names', 'as an AI…') are still NOT facts — drop them \
                 entirely; they belong in the system prompt."
            );
        }
    }
    let intro = "Extract a short list of atomic facts about the people and \
        events in the chat turns below — observations about the \
        world, not about you.";
    let guidance = "Distinguish events from assertions:\n\
        - What a speaker asserts about ANOTHER person — their body, \
        health, medication, preferences, relationships, or state of \
        mind — is a claim, not a fact. Record it attributed ('X \
        claims ...', 'X says ...') with the speaker named, or skip \
        it. Record it as flat fact only when the person it concerns \
        confirms it in these turns.\n\
        - In-character bits, running jokes, roleplay, and fictional \
        narration (violence, transformations, creatures, magic done \
        to people) are fiction. Never record fictional events as \
        real ones ('Y was hit with a crowbar'). If a bit recurs \
        enough to matter, record the BIT as the fact ('running joke \
        in which ...', 'X and Y's shared fiction that ...'), \
        including who plays along and who refuses.\n\
        - A speaker describing themselves (their history, tastes, \
        abilities) may be recorded as their own account; \
        extraordinary self-claims stay attributed ('X describes \
        herself as ...').\n\
        - Identity ties to a Participant's canonical_key, never to a \
        name a speaker adopts in play. A member impersonating another, \
        claiming to BE them, or borrowing their name in a bit is \
        roleplay: record it as a bit if it matters ('X joked they were \
        Y'), never as an identity fact, and NEVER merge two \
        Participants into one. Impersonated or confused, they stay \
        distinct people with distinct keys.\n\
        - World trivia, game lore, or general knowledge a speaker \
        happens to mention is not a fact ABOUT that speaker. Skip it or \
        leave it subjectless; never attach trivia to them as a \
        personal fact.\n\n\
        Skip small talk and transient feelings. If nothing useful, \
        reply with []. Do NOT emit self-capability statements about \
        yourself, the assistant, or your own limitations (e.g., 'I \
        cannot remember names', 'the assistant has no internet \
        access', 'as an AI, I…'). Those belong in the system \
        prompt, not the facts store — they expire the moment a \
        capability changes.";
    let header = format!(
        "{intro}{self_clause}{dream_clause}\n\n{}\n\n{guidance}",
        render_contract(&FACT_SCHEMA)
    );
    let mut lines: Vec<String> = Vec::new();
    if !participants.is_empty() {
        lines.push("Participants (canonical_key — current display name):".to_string());
        for (key, display) in participants {
            lines.push(format!("- {key} — {display}"));
        }
        lines.push(String::new());
    }
    lines.push("Turns (id prefixed):".to_string());
    for t in turns {
        let who = t
            .author
            .as_ref()
            .map_or_else(|| t.role.clone(), crate::identity::Author::label);
        lines.push(format!("- id={} [{who}] {}", t.id, t.content));
    }
    vec![
        Message::new("system", header),
        Message::new("user", lines.join("\n")),
    ]
}

/// One normalized fact item — the raw JSON reshaped into the worker's shape.
struct FactItem {
    text: Value,
    source_turn_ids: Vec<i64>,
    subject_keys: Vec<String>,
    valid_from: Option<Value>,
    valid_to: Option<Value>,
    importance: Option<Value>,
}

/// Normalize parsed fact items into the worker's shape; a non-array (including
/// `None` on a fumbled reply) degrades to `[]`.
fn normalize_fact_items(parsed: Option<&Value>) -> Vec<FactItem> {
    let Some(Value::Array(items)) = parsed else {
        return Vec::new();
    };
    let mut out: Vec<FactItem> = Vec::new();
    for item in items {
        let Value::Object(map) = item else {
            continue;
        };
        out.push(FactItem {
            text: map
                .get("text")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new())),
            source_turn_ids: normalize_source_ids(map.get("source_turn_ids")),
            subject_keys: normalize_subject_keys(map.get("subject_keys")),
            valid_from: map.get("valid_from").cloned(),
            valid_to: map.get("valid_to").cloned(),
            importance: map.get("importance").cloned(),
        });
    }
    out
}

/// Coerce `source_turn_ids` to ints: JSON integers, and integer-valued digit
/// strings (a leading `-` tolerated). Floats, bools, and non-digit strings are
/// dropped; non-array input → `[]`.
fn normalize_source_ids(raw: Option<&Value>) -> Vec<i64> {
    let Some(Value::Array(arr)) = raw else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|x| match x {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => {
                let stripped = s.trim_start_matches('-');
                if !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit()) {
                    s.parse::<i64>().ok()
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect()
}

/// Keep only string `subject_keys`; non-array input → `[]`.
fn normalize_subject_keys(raw: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(arr)) = raw else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|s| s.as_str().map(String::from))
        .collect()
}

/// Coerce LLM-emitted importance to int; `None` for missing / non-numeric.
/// Out-of-range values pass through verbatim (the store clamps to `[1, 10]`).
/// Bools reject; floats truncate toward zero; numeric strings parse.
#[allow(clippy::cast_possible_truncation)]
fn parse_importance(raw: Option<&Value>) -> Option<i64> {
    match raw? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                return None;
            }
            s.parse::<i64>()
                .ok()
                .or_else(|| s.parse::<f64>().ok().map(|f| f.trunc() as i64))
        }
        // Null, bool, arrays, and objects are all non-numeric → None.
        _ => None,
    }
}

/// Permissive ISO-8601 → UTC datetime; `None` for non-strings / bad input.
/// Accepts date-only (`2024-01-15`) via a midnight-UTC fallback; naive values
/// are assumed UTC.
fn parse_iso_dt(raw: Option<&Value>) -> Option<DateTime<Utc>> {
    let s = raw?.as_str()?.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(dt) = parse_iso(s) {
        return Some(dt);
    }
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|ndt| ndt.and_utc())
}

/// Resolve the self-subject display name from an optional explicit override,
/// falling back to the title-cased id. Mirrors Python `explicit or id.title()`:
/// an empty explicit string is falsy and falls back too.
fn resolve_display_name(explicit: Option<&str>, familiar_id: &str) -> String {
    match explicit {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => title_case(familiar_id),
    }
}

/// ASCII title-case matching Python `str.title()`.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for ch in s.chars() {
        if ch.is_alphabetic() {
            if prev_alpha {
                out.extend(ch.to_lowercase());
            } else {
                out.extend(ch.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(ch);
            prev_alpha = false;
        }
    }
    out
}

/// Python `str(value)` for the JSON `text` field (identity on strings).
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        // Numbers, arrays, and objects all render via serde's `Display`.
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NAME_CAPABILITY_TAIL, is_self_capability, normalize_fact_items, parse_importance,
        parse_iso_dt, resolve_display_name, resolve_subjects, title_case,
    };
    use chrono::{TimeZone, Utc};
    use regex::Regex;
    use serde_json::json;

    fn name_re(display: &str) -> Regex {
        Regex::new(&format!(
            "(?i)^\\s*{}{NAME_CAPABILITY_TAIL}",
            regex::escape(display)
        ))
        .unwrap()
    }

    #[test]
    fn self_capability_first_person_forms_match() {
        for t in [
            "I cannot remember the names of people.",
            "The assistant does not have access to the internet.",
            "As an AI, I have no personal preferences.",
            "I'm not able to browse.",
            "I have no memory of that.",
        ] {
            assert!(is_self_capability(t, None), "should match: {t}");
        }
    }

    #[test]
    fn ordinary_world_facts_survive() {
        assert!(!is_self_capability("Aria likes strawberries.", None));
        assert!(!is_self_capability("Boris works nights.", None));
    }

    #[test]
    fn named_inability_drops_but_narrative_survives() {
        let re = name_re("Sapphire");
        // Genuine inability drops.
        for t in [
            "Sapphire cannot remember names.",
            "Sapphire has no internet access.",
        ] {
            assert!(is_self_capability(t, Some(&re)), "should drop: {t}");
        }
        // Word-prefix collisions, copula/dynamic negation, and positive ability
        // are narrative and must survive.
        for t in [
            "Sapphire cancelled the movie night.",
            "Sapphire candidly admitted she was wrong.",
            "Sapphire is not fond of KaillaDame.",
            "Sapphire doesn't trust easily.",
            "Sapphire can sing surprisingly well.",
            "Sapphire chose to walk away.",
        ] {
            assert!(!is_self_capability(t, Some(&re)), "should keep: {t}");
        }
    }

    #[test]
    fn normalize_rejects_non_array_and_non_dict_items() {
        assert!(normalize_fact_items(None).is_empty());
        assert!(normalize_fact_items(Some(&json!({"not": "an array"}))).is_empty());
        let items = normalize_fact_items(Some(&json!([
            {"text": "a", "source_turn_ids": [1, "3", 4.5, true, "-5"]},
            "skip",
            {"text": "b", "subject_keys": ["k1", 7, "k2"]},
        ])));
        assert_eq!(items.len(), 2);
        // Ints + digit-strings kept (incl. negative); float/bool dropped.
        assert_eq!(items[0].source_turn_ids, vec![1, 3, -5]);
        assert_eq!(
            items[1].subject_keys,
            vec!["k1".to_string(), "k2".to_string()]
        );
    }

    #[test]
    fn parse_importance_covers_every_branch() {
        assert_eq!(parse_importance(None), None);
        assert_eq!(parse_importance(Some(&json!(null))), None);
        assert_eq!(parse_importance(Some(&json!(true))), None);
        assert_eq!(parse_importance(Some(&json!(9))), Some(9));
        assert_eq!(parse_importance(Some(&json!(99))), Some(99));
        assert_eq!(parse_importance(Some(&json!(-3))), Some(-3));
        assert_eq!(parse_importance(Some(&json!(9.7))), Some(9));
        assert_eq!(parse_importance(Some(&json!("7"))), Some(7));
        assert_eq!(parse_importance(Some(&json!("7.9"))), Some(7));
        assert_eq!(parse_importance(Some(&json!("very important"))), None);
        assert_eq!(parse_importance(Some(&json!("  "))), None);
    }

    #[test]
    fn parse_iso_dt_accepts_full_and_date_only() {
        let dt = parse_iso_dt(Some(&json!("2024-01-15T00:00:00+00:00"))).unwrap();
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap());
        let date_only = parse_iso_dt(Some(&json!("2024-01-15"))).unwrap();
        assert_eq!(
            date_only,
            Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap()
        );
        assert_eq!(parse_iso_dt(Some(&json!("garbage"))), None);
        assert_eq!(parse_iso_dt(Some(&json!(5))), None);
        assert_eq!(parse_iso_dt(None), None);
    }

    #[test]
    fn resolve_subjects_soft_validates_and_dedups() {
        let manifest = vec![
            ("discord:111".to_string(), "Cass".to_string()),
            ("ego:fam".to_string(), "Sapphire".to_string()),
        ];
        let subs = resolve_subjects(
            &[
                "discord:111".to_string(),
                "discord:unknown".to_string(),
                "discord:111".to_string(),
            ],
            &manifest,
        );
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].canonical_key, "discord:111");
        assert_eq!(subs[0].display_at_write, "Cass");
    }

    #[test]
    fn title_case_matches_python() {
        assert_eq!(title_case("fam"), "Fam");
        assert_eq!(title_case("my-fam"), "My-Fam");
    }

    #[test]
    fn resolve_display_name_falls_back_for_empty_and_none() {
        // Python `familiar_display_name or familiar_id.title()`: a non-empty
        // explicit name wins; an empty string (falsy) or None title-cases the id.
        assert_eq!(resolve_display_name(Some("Sapphire"), "fam"), "Sapphire");
        assert_eq!(resolve_display_name(Some(""), "my-fam"), "My-Fam");
        assert_eq!(resolve_display_name(None, "my-fam"), "My-Fam");
    }
}
