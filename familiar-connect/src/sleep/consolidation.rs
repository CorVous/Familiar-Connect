//! Memory-consolidation pass — propose + validate fact consolidations
//! (subsystem 04; Python `sleep/consolidation.py`).
//!
//! Runs on sleep-activity departure. The LLM sees the whole window at once —
//! unlike small-batch extraction it spots day-level patterns (a claim asserted
//! 9× by one speaker, denied by the subject every time = a bit, not a fact).
//!
//! Two plan-level action verbs, both resolving to the store's single
//! `supersede` (never DELETE): `retire` (drop junk/noise, no replacement) and
//! `rewrite` (merge near-dups / re-attribute a misfiled claim into one
//! consolidated fact). Every LLM proposal is validated against rails *in code*
//! ([`validate`]) — the model advises, does not decide. The pass produces a
//! [`ConsolidationPlan`] and never touches the DB itself;
//! [`super::apply::apply_consolidation`] executes an accepted plan.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use super::{
    len_i64, normalize_fact_text, py_str_field, py_str_list_repr, py_tuple_repr, subject_key_set,
};
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{Fact, HistoryTurn, SleepWatermark};
use crate::identity::ego_canonical_key;
use crate::llm::{LlmClient, Message};
use crate::structured_output::{Expect, coerce_json, coerce_positive_int_list, coerce_str_list};

/// Default cap on facts considered by one pass (newest-kept).
pub const DEFAULT_FACTS_MAX: i64 = 500;
/// Default cap on turns considered by one pass (newest-kept tail).
pub const DEFAULT_TURNS_MAX: i64 = 400;
/// Default cap on the number of facts one pass may retire/rewrite.
pub const DEFAULT_RETIRE_CAP: i64 = 50;

/// The slice of memory one consolidation pass reasons over.
///
/// `facts` are current (non-superseded) facts, capped to `facts_max` newest,
/// prompted oldest-first. `turns` are conversation since the prior sleep
/// watermark, capped to `turns_max` newest. `max_fact_id` / `max_turn_id` are
/// the true (uncapped) high-water marks the watermark advances to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationWindow {
    /// Familiar this window belongs to.
    pub familiar_id: String,
    /// Current facts, oldest-first, capped to `facts_max`.
    pub facts: Vec<Fact>,
    /// Recent turns since the prior watermark, newest-`turns_max` kept.
    pub turns: Vec<HistoryTurn>,
    /// The prior sleep watermark, if any.
    pub prior_watermark: Option<SleepWatermark>,
    /// True (uncapped) highest fact id — the fact-axis watermark target.
    pub max_fact_id: i64,
    /// True (uncapped) highest turn id — the turn-axis watermark target.
    pub max_turn_id: i64,
    /// How many current facts the cap dropped.
    pub facts_truncated: usize,
    /// How many in-window turns the cap dropped.
    pub turns_truncated: usize,
}

/// Accepted: retire `fact_ids` with no replacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetireAction {
    /// Facts to supersede (no replacement).
    pub fact_ids: Vec<i64>,
    /// The LLM's stated reason (trimmed).
    pub reason: String,
}

/// Accepted: supersede `old_fact_ids` with one new consolidated fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteAction {
    /// Facts merged into the replacement.
    pub old_fact_ids: Vec<i64>,
    /// The consolidated replacement text.
    pub new_text: String,
    /// Subject keys carried onto the minted fact.
    pub subject_keys: Vec<String>,
    /// The LLM's stated reason (trimmed).
    pub reason: String,
}

/// A proposed action a rail refused. `payload` is the raw LLM proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedAction {
    /// `"retire"` or `"rewrite"`.
    pub kind: String,
    /// The raw LLM proposal object.
    pub payload: Value,
    /// The rail name that refused it.
    pub rail: String,
    /// Human-readable detail (logged, truncated to 80).
    pub detail: String,
}

/// Validated outcome of one pass — accepted actions + rejections.
///
/// Pure description; applying is [`super::apply::apply_consolidation`]'s job.
/// `new_last_fact_id` / `new_last_turn_id` are the watermark the apply step
/// advances to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationPlan {
    /// Familiar this plan belongs to.
    pub familiar_id: String,
    /// Accepted retire actions.
    pub retire: Vec<RetireAction>,
    /// Accepted rewrite actions.
    pub rewrite: Vec<RewriteAction>,
    /// Rail-blocked proposals (the audit trail).
    pub rejected: Vec<RejectedAction>,
    /// Fact-axis watermark the apply step advances to.
    pub new_last_fact_id: i64,
    /// Turn-axis watermark carried through (opinion pass owns writing it).
    pub new_last_turn_id: i64,
    /// How many facts the window considered.
    pub facts_considered: usize,
    /// How many current facts the cap dropped.
    pub facts_truncated: usize,
    /// How many turns the window considered.
    pub turns_considered: usize,
    /// How many in-window turns the cap dropped.
    pub turns_truncated: usize,
    /// Non-fatal notes (e.g. a garbage-reply flag).
    pub notes: Vec<String>,
}

impl ConsolidationPlan {
    /// A plan with the counts zeroed and no notes (the Python default-arg
    /// constructor; [`validate`] fills the counts directly).
    #[must_use]
    pub fn new(
        familiar_id: impl Into<String>,
        retire: Vec<RetireAction>,
        rewrite: Vec<RewriteAction>,
        rejected: Vec<RejectedAction>,
        new_last_fact_id: i64,
        new_last_turn_id: i64,
    ) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            retire,
            rewrite,
            rejected,
            new_last_fact_id,
            new_last_turn_id,
            facts_considered: 0,
            facts_truncated: 0,
            turns_considered: 0,
            turns_truncated: 0,
            notes: Vec::new(),
        }
    }

    /// Facts retired + superseded if this plan is applied.
    #[must_use]
    pub fn mutated_count(&self) -> usize {
        let r: usize = self.retire.iter().map(|a| a.fact_ids.len()).sum();
        let w: usize = self.rewrite.iter().map(|a| a.old_fact_ids.len()).sum();
        r + w
    }
}

/// Collect current facts + recent turns into a consolidation window.
///
/// The fact window is intentionally NOT id-bounded by the watermark —
/// consolidation reasons over the whole live fact base every night; only the
/// turn axis is windowed.
///
/// # Errors
/// Propagates store faults.
pub async fn gather_window(
    store: &AsyncHistoryStore,
    familiar_id: &str,
    facts_max: i64,
    turns_max: i64,
) -> anyhow::Result<ConsolidationWindow> {
    let prior = store.get_sleep_watermark(familiar_id.to_owned()).await?;
    let max_fact_id = store.latest_fact_id(familiar_id.to_owned()).await?;
    let max_turn_id = store.latest_fts_id(familiar_id.to_owned()).await?;

    // all current facts, newest-first; keep newest facts_max, prompt them
    // oldest-first for stable ordering.
    let all_current = store
        .recent_facts(familiar_id.to_owned(), 100_000, false, None)
        .await?;
    let take = usize::try_from(facts_max).unwrap_or(0);
    let facts_truncated = all_current.len().saturating_sub(take);
    let mut kept: Vec<Fact> = all_current.into_iter().take(take).collect();
    kept.reverse();

    // turns since prior sleep (re-attribution context), bounded to newest.
    let min_turn = prior.map_or(0, |p| p.last_turn_id);
    let all_turns = store
        .turns_in_id_range(familiar_id.to_owned(), min_turn, max_turn_id, None)
        .await?;
    let kept_turns: Vec<HistoryTurn> = if turns_max > 0 {
        let n = usize::try_from(turns_max).unwrap_or(0);
        let start = all_turns.len().saturating_sub(n);
        all_turns[start..].to_vec()
    } else {
        Vec::new()
    };
    let turns_truncated = all_turns.len().saturating_sub(kept_turns.len());

    Ok(ConsolidationWindow {
        familiar_id: familiar_id.to_owned(),
        facts: kept,
        turns: kept_turns,
        prior_watermark: prior,
        max_fact_id,
        max_turn_id,
        facts_truncated,
        turns_truncated,
    })
}

/// Render the window into the consolidation LLM prompt.
///
/// `system` is the config-sourced static instruction, relayed **verbatim** (may
/// be empty). The dynamic window data (facts, turns, `self_key`) is assembled
/// here regardless.
#[must_use]
pub fn build_prompt(window: &ConsolidationWindow, self_key: &str, system: &str) -> Vec<Message> {
    let mut lines: Vec<String> = vec![
        format!("Self subject key (the character): {self_key}"),
        String::new(),
        "Facts (current):".to_owned(),
    ];
    for f in &window.facts {
        let subj = if f.subjects.is_empty() {
            "\u{2014}".to_owned()
        } else {
            f.subjects
                .iter()
                .map(|s| s.canonical_key.clone())
                .collect::<Vec<_>>()
                .join(", ")
        };
        lines.push(format!("- id={} subjects=[{subj}]: {}", f.id, f.text));
    }
    if !window.turns.is_empty() {
        lines.push(String::new());
        lines.push("Recent conversation (for attribution context):".to_owned());
        for t in &window.turns {
            let who = t
                .author
                .as_ref()
                .map_or_else(|| t.role.clone(), crate::identity::Author::label);
            lines.push(format!("- {who}: {}", t.content));
        }
    }
    vec![
        Message::new("system", system),
        Message::new("user", lines.join("\n")),
    ]
}

/// Extract `(retire, rewrite)` raw proposal lists from an LLM reply.
///
/// Permissive: bad JSON, missing keys, non-list values all degrade to empty
/// lists rather than raising. Non-object items are dropped.
#[must_use]
pub fn parse_actions(reply: &str) -> (Vec<Value>, Vec<Value>) {
    let result = coerce_json(reply, Expect::Object);
    // Python `coerce_json(...).value or {}` then `isinstance(dict)`: only a
    // (possibly empty) object proceeds — everything else degrades to ([], []).
    let Some(obj) = result.value.as_ref().and_then(Value::as_object) else {
        return (Vec::new(), Vec::new());
    };
    let dicts = |key: &str| -> Vec<Value> {
        obj.get(key)
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter(|item| item.is_object())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    };
    (dicts("retire"), dicts("rewrite"))
}

/// Report whether a non-empty reply yielded no parseable plan object.
///
/// A silent (empty/whitespace) reply is a clean empty plan (`false`); a
/// non-empty reply that fails to parse to a JSON object is a fumble (`true`).
#[must_use]
pub fn reply_parse_failed(reply: &str) -> bool {
    if reply.trim().is_empty() {
        return false;
    }
    let result = coerce_json(reply, Expect::Object);
    !(result.parsed_ok && result.value.as_ref().is_some_and(Value::is_object))
}

/// Filter proposals through the safety rails; build the plan. Pure, no I/O.
///
/// Processing order: all retire proposals in input order, THEN all rewrite
/// proposals in input order — an earlier action reserves ids and cap budget from
/// later ones.
#[must_use]
#[allow(clippy::too_many_lines)] // the two rail loops read best kept together (parity with Python)
pub fn validate(
    window: &ConsolidationWindow,
    retire_raw: &[Value],
    rewrite_raw: &[Value],
    self_key: &str,
    cap: i64,
    notes: Vec<String>,
) -> ConsolidationPlan {
    let by_id: HashMap<i64, &Fact> = window.facts.iter().map(|f| (f.id, f)).collect();
    let mut retire: Vec<RetireAction> = Vec::new();
    let mut rewrite: Vec<RewriteAction> = Vec::new();
    let mut rejected: Vec<RejectedAction> = Vec::new();
    let mut claimed: HashSet<i64> = HashSet::new();
    let mut mutated: i64 = 0;

    // Rails (a)/(b) from behavior 13: unknown id / self-subject / duplicate.
    let check_ids = |ids: &[i64], claimed: &HashSet<i64>| -> Option<&'static str> {
        for &fid in ids {
            let Some(f) = by_id.get(&fid) else {
                return Some("unknown_id");
            };
            // consolidation never adjudicates feelings — leave self: facts alone.
            if f.subjects.iter().any(|s| s.canonical_key == self_key) {
                return Some("self_subject");
            }
        }
        if ids.iter().any(|fid| claimed.contains(fid)) {
            return Some("duplicate_target");
        }
        None
    };

    for payload in retire_raw {
        let ids = coerce_positive_int_list(payload.get("fact_ids").unwrap_or(&Value::Null));
        if ids.is_empty() {
            rejected.push(RejectedAction {
                kind: "retire".to_owned(),
                payload: payload.clone(),
                rail: "no_ids".to_owned(),
                detail: "no fact_ids".to_owned(),
            });
            continue;
        }
        if let Some(rail) = check_ids(&ids, &claimed) {
            rejected.push(RejectedAction {
                kind: "retire".to_owned(),
                payload: payload.clone(),
                rail: rail.to_owned(),
                detail: py_tuple_repr(&ids),
            });
            continue;
        }
        if mutated + len_i64(ids.len()) > cap {
            rejected.push(RejectedAction {
                kind: "retire".to_owned(),
                payload: payload.clone(),
                rail: "cap".to_owned(),
                detail: format!("cap={cap} reached"),
            });
            continue;
        }
        let reason = py_str_field(payload, "reason").trim().to_owned();
        for &id in &ids {
            claimed.insert(id);
        }
        mutated += len_i64(ids.len());
        retire.push(RetireAction {
            fact_ids: ids,
            reason,
        });
    }

    for payload in rewrite_raw {
        let ids = coerce_positive_int_list(payload.get("old_fact_ids").unwrap_or(&Value::Null));
        if ids.is_empty() {
            rejected.push(RejectedAction {
                kind: "rewrite".to_owned(),
                payload: payload.clone(),
                rail: "no_ids".to_owned(),
                detail: "no old_fact_ids".to_owned(),
            });
            continue;
        }
        if let Some(rail) = check_ids(&ids, &claimed) {
            rejected.push(RejectedAction {
                kind: "rewrite".to_owned(),
                payload: payload.clone(),
                rail: rail.to_owned(),
                detail: py_tuple_repr(&ids),
            });
            continue;
        }
        let new_text = py_str_field(payload, "new_text").trim().to_owned();
        if new_text.is_empty() {
            rejected.push(RejectedAction {
                kind: "rewrite".to_owned(),
                payload: payload.clone(),
                rail: "empty_text".to_owned(),
                detail: "blank new_text".to_owned(),
            });
            continue;
        }
        let keys = coerce_str_list(payload.get("subject_keys").unwrap_or(&Value::Null));
        let mut source_keys: HashSet<String> = HashSet::new();
        for &fid in &ids {
            if let Some(f) = by_id.get(&fid) {
                for s in &f.subjects {
                    source_keys.insert(s.canonical_key.clone());
                }
            }
        }
        // subject_lost: dropping all subjects when sources had some silently
        // orphans the fact — reject.
        if !source_keys.is_empty() && keys.is_empty() {
            let mut sorted: Vec<String> = source_keys.into_iter().collect();
            sorted.sort();
            rejected.push(RejectedAction {
                kind: "rewrite".to_owned(),
                payload: payload.clone(),
                rail: "subject_lost".to_owned(),
                detail: py_str_list_repr(&sorted),
            });
            continue;
        }
        let introduced: Vec<String> = keys
            .iter()
            .filter(|k| !source_keys.contains(*k) && k.as_str() != self_key)
            .cloned()
            .collect();
        if !introduced.is_empty() {
            rejected.push(RejectedAction {
                kind: "rewrite".to_owned(),
                payload: payload.clone(),
                rail: "subject_introduced".to_owned(),
                detail: py_str_list_repr(&introduced),
            });
            continue;
        }
        // no-op: single source fact, identical normalized text + subject set.
        if ids.len() == 1 {
            if let Some(only) = by_id.get(&ids[0]) {
                let same_text = normalize_fact_text(&only.text) == normalize_fact_text(&new_text);
                let same_subj =
                    subject_key_set(&only.subjects) == keys.iter().cloned().collect::<HashSet<_>>();
                if same_text && same_subj {
                    rejected.push(RejectedAction {
                        kind: "rewrite".to_owned(),
                        payload: payload.clone(),
                        rail: "noop".to_owned(),
                        detail: "restatement".to_owned(),
                    });
                    continue;
                }
            }
        }
        if mutated + len_i64(ids.len()) > cap {
            rejected.push(RejectedAction {
                kind: "rewrite".to_owned(),
                payload: payload.clone(),
                rail: "cap".to_owned(),
                detail: format!("cap={cap} reached"),
            });
            continue;
        }
        let reason = py_str_field(payload, "reason").trim().to_owned();
        for &id in &ids {
            claimed.insert(id);
        }
        mutated += len_i64(ids.len());
        rewrite.push(RewriteAction {
            old_fact_ids: ids,
            new_text,
            subject_keys: keys,
            reason,
        });
    }

    // `mutated` is a running budget; its final value is intentionally unused.
    let _ = mutated;

    ConsolidationPlan {
        familiar_id: window.familiar_id.clone(),
        retire,
        rewrite,
        rejected,
        new_last_fact_id: window.max_fact_id,
        new_last_turn_id: window.max_turn_id,
        facts_considered: window.facts.len(),
        facts_truncated: window.facts_truncated,
        turns_considered: window.turns.len(),
        turns_truncated: window.turns_truncated,
        notes,
    }
}

/// Gather → prompt → LLM → parse → validate. Touches no fact rows.
///
/// Makes ONE raw `llm.chat` call (no `request_structured` retry loop) — the
/// permissive [`parse_actions`] degrades garbage to an empty plan.
///
/// # Errors
/// Propagates store + LLM transport faults.
pub async fn plan_consolidation(
    store: &AsyncHistoryStore,
    llm: &dyn LlmClient,
    familiar_id: &str,
    facts_max: i64,
    turns_max: i64,
    cap: i64,
    system: &str,
) -> anyhow::Result<ConsolidationPlan> {
    let self_key = ego_canonical_key(familiar_id);
    let window = gather_window(store, familiar_id, facts_max, turns_max).await?;
    let prompt = build_prompt(&window, &self_key, system);
    let reply = llm.chat(prompt).await?;
    let content = reply.content_str();
    let (retire_raw, rewrite_raw) = parse_actions(&content);
    let mut notes: Vec<String> = Vec::new();
    if reply_parse_failed(&content) {
        notes.push("llm reply did not parse to a plan object — treated as empty".to_owned());
        tracing::warn!(
            target: "familiar_connect.sleep.consolidation",
            "sleep-consolidation parse failure familiar={familiar_id}"
        );
    }
    let plan = validate(&window, &retire_raw, &rewrite_raw, &self_key, cap, notes);
    tracing::info!(
        target: "familiar_connect.sleep.consolidation",
        "sleep-consolidation plan familiar={familiar_id} retire={} rewrite={} rejected={} facts={}(+{} trunc)",
        plan.retire.len(),
        plan.rewrite.len(),
        plan.rejected.len(),
        plan.facts_considered,
        plan.facts_truncated,
    );
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::{
        ConsolidationWindow, RejectedAction, build_prompt, parse_actions, reply_parse_failed,
        validate,
    };
    use crate::history::store::{Fact, FactSubject};
    use crate::identity::ego_canonical_key;
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    fn fact(fid: i64, text: &str, subjects: Vec<FactSubject>) -> Fact {
        Fact {
            id: fid,
            familiar_id: "fam".to_owned(),
            channel_id: Some(1),
            text: text.to_owned(),
            source_turn_ids: vec![fid],
            created_at: Utc.with_ymd_and_hms(2026, 6, 12, 0, 0, 0).unwrap(),
            superseded_at: None,
            superseded_by: None,
            subjects,
            valid_from: None,
            valid_to: None,
            importance: None,
        }
    }

    fn window(facts: Vec<Fact>) -> ConsolidationWindow {
        let max_fid = facts.iter().map(|f| f.id).max().unwrap_or(0);
        ConsolidationWindow {
            familiar_id: "fam".to_owned(),
            facts,
            turns: Vec::new(),
            prior_watermark: None,
            max_fact_id: max_fid,
            max_turn_id: 0,
            facts_truncated: 0,
            turns_truncated: 0,
        }
    }

    fn aria() -> Vec<FactSubject> {
        vec![FactSubject {
            canonical_key: "discord:A".to_owned(),
            display_at_write: "Aria".to_owned(),
        }]
    }

    fn self_subjects() -> Vec<FactSubject> {
        vec![FactSubject {
            canonical_key: ego_canonical_key("fam"),
            display_at_write: "Sapphire".to_owned(),
        }]
    }

    // --- retire rails --------------------------------------------------------

    #[test]
    fn accepts_valid_retire() {
        let win = window(vec![
            fact(1, "noise", vec![]),
            fact(2, "Aria likes tea.", aria()),
        ]);
        let plan = validate(
            &win,
            &[json!({"fact_ids": [1], "reason": "junk"})],
            &[],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert_eq!(plan.retire.len(), 1);
        assert_eq!(plan.retire[0].fact_ids, vec![1]);
        assert_eq!(plan.retire[0].reason, "junk");
        assert!(plan.rejected.is_empty());
    }

    #[test]
    fn rejects_unknown_id() {
        let win = window(vec![fact(1, "x", vec![])]);
        let plan = validate(
            &win,
            &[json!({"fact_ids": [999], "reason": "x"})],
            &[],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert!(plan.retire.is_empty());
        assert_eq!(plan.rejected[0].rail, "unknown_id");
    }

    #[test]
    fn cap_defers_excess() {
        let facts: Vec<Fact> = (1..=5).map(|i| fact(i, &format!("f{i}"), vec![])).collect();
        let win = window(facts);
        let retire_raw: Vec<_> = (1..=5)
            .map(|i| json!({"fact_ids": [i], "reason": "j"}))
            .collect();
        let plan = validate(
            &win,
            &retire_raw,
            &[],
            &ego_canonical_key("fam"),
            3,
            Vec::new(),
        );
        assert_eq!(plan.retire.len(), 3);
        assert_eq!(plan.rejected.iter().filter(|r| r.rail == "cap").count(), 2);
    }

    #[test]
    fn double_target_rejected() {
        let win = window(vec![fact(1, "x", vec![])]);
        let plan = validate(
            &win,
            &[
                json!({"fact_ids": [1], "reason": "a"}),
                json!({"fact_ids": [1], "reason": "b"}),
            ],
            &[],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert_eq!(plan.retire.len(), 1);
        assert!(plan.rejected.iter().any(|r| r.rail == "duplicate_target"));
    }

    // --- self-subject rail ---------------------------------------------------

    #[test]
    fn rejects_retire_of_self_fact() {
        let win = window(vec![fact(1, "Sapphire loves lo-fi.", self_subjects())]);
        let plan = validate(
            &win,
            &[json!({"fact_ids": [1], "reason": "dup"})],
            &[],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert!(plan.retire.is_empty());
        assert_eq!(plan.rejected[0].rail, "self_subject");
    }

    #[test]
    fn rejects_rewrite_touching_self_fact() {
        let win = window(vec![
            fact(1, "Sapphire loves lo-fi.", self_subjects()),
            fact(2, "Sapphire really loves lo-fi.", self_subjects()),
        ]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1, 2],
                "new_text": "Sapphire loves lo-fi.",
                "subject_keys": [ego_canonical_key("fam")],
                "reason": "merge",
            })],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert!(plan.rewrite.is_empty());
        assert_eq!(plan.rejected[0].rail, "self_subject");
    }

    #[test]
    fn self_rail_does_not_block_ordinary_facts() {
        let win = window(vec![fact(1, "noise", aria())]);
        let plan = validate(
            &win,
            &[json!({"fact_ids": [1], "reason": "junk"})],
            &[],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert_eq!(plan.retire.len(), 1);
    }

    // --- rewrite rails -------------------------------------------------------

    #[test]
    fn accepts_merge() {
        let win = window(vec![
            fact(1, "Aria likes berries.", aria()),
            fact(2, "Aria really likes berries.", aria()),
        ]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1, 2],
                "new_text": "Aria likes berries.",
                "subject_keys": ["discord:A"],
                "reason": "merged dups",
            })],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert_eq!(plan.rewrite.len(), 1);
        assert_eq!(plan.rewrite[0].old_fact_ids, vec![1, 2]);
        assert_eq!(plan.rewrite[0].subject_keys, vec!["discord:A".to_owned()]);
    }

    #[test]
    fn rejects_introduced_subject() {
        let win = window(vec![fact(1, "Aria said a thing.", aria())]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1],
                "new_text": "Boris did a thing.",
                "subject_keys": ["discord:B"],
                "reason": "x",
            })],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert!(plan.rewrite.is_empty());
        assert_eq!(plan.rejected[0].rail, "subject_introduced");
    }

    #[test]
    fn allows_self_subject_reattribution() {
        let self_key = ego_canonical_key("fam");
        let win = window(vec![fact(1, "Cor wears no pants.", aria())]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1],
                "new_text": "Sapphire ran a no-pants bit about Cor.",
                "subject_keys": [self_key],
                "reason": "bit misfiled under Cor",
            })],
            &self_key,
            50,
            Vec::new(),
        );
        assert_eq!(plan.rewrite.len(), 1);
        assert_eq!(plan.rewrite[0].subject_keys, vec![self_key]);
    }

    #[test]
    fn rejects_noop_rewrite() {
        let win = window(vec![fact(1, "Aria likes tea.", aria())]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1],
                "new_text": "Aria likes tea.",
                "subject_keys": ["discord:A"],
                "reason": "x",
            })],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert!(plan.rewrite.is_empty());
        assert_eq!(plan.rejected[0].rail, "noop");
    }

    #[test]
    fn rejects_empty_new_text() {
        let win = window(vec![fact(1, "Aria likes tea.", aria())]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1],
                "new_text": "   ",
                "subject_keys": ["discord:A"],
                "reason": "x",
            })],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert!(plan.rewrite.is_empty());
        assert_eq!(plan.rejected[0].rail, "empty_text");
    }

    #[test]
    fn rejects_empty_keys_when_source_has_subjects() {
        let win = window(vec![fact(1, "Aria likes tea.", aria())]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1],
                "new_text": "Aria enjoys tea.",
                "subject_keys": [],
                "reason": "x",
            })],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert!(plan.rewrite.is_empty());
        assert_eq!(plan.rejected[0].rail, "subject_lost");
    }

    #[test]
    fn allows_empty_keys_when_source_subjectless() {
        let win = window(vec![fact(1, "The weather is nice.", vec![])]);
        let plan = validate(
            &win,
            &[],
            &[json!({
                "old_fact_ids": [1],
                "new_text": "The weather turned grim.",
                "subject_keys": [],
                "reason": "x",
            })],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        assert_eq!(plan.rewrite.len(), 1);
    }

    // --- build_prompt --------------------------------------------------------

    #[test]
    fn system_text_is_caller_supplied() {
        let win = window(vec![fact(1, "noise", vec![])]);
        let msgs = build_prompt(&win, "ego:fam", "MY OWN INSTRUCTIONS");
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content_str(), "MY OWN INSTRUCTIONS");
    }

    #[test]
    fn build_prompt_renders_facts_and_em_dash_for_subjectless() {
        let win = window(vec![
            fact(1, "noise", vec![]),
            fact(2, "Aria likes tea.", aria()),
        ]);
        let msgs = build_prompt(&win, "ego:fam", "");
        let user = msgs[1].content_str();
        assert!(user.contains("Self subject key (the character): ego:fam"));
        assert!(user.contains("- id=1 subjects=[\u{2014}]: noise"));
        assert!(user.contains("- id=2 subjects=[discord:A]: Aria likes tea."));
    }

    // --- parse_actions / reply_parse_failed ----------------------------------

    #[test]
    fn parse_actions_extracts_dicts_and_drops_non_dicts() {
        let reply = json!({
            "retire": [{"fact_ids": [1]}, "junk", 5],
            "rewrite": [{"old_fact_ids": [2]}],
        })
        .to_string();
        let (retire, rewrite) = parse_actions(&reply);
        assert_eq!(retire.len(), 1);
        assert_eq!(rewrite.len(), 1);
    }

    #[test]
    fn parse_actions_degrades_on_garbage() {
        assert_eq!(parse_actions("not json at all"), (Vec::new(), Vec::new()));
        // a bare array is not an object → degrades.
        assert_eq!(parse_actions("[1, 2, 3]"), (Vec::new(), Vec::new()));
    }

    #[test]
    fn reply_parse_failed_distinguishes_silent_from_fumble() {
        assert!(!reply_parse_failed(""));
        assert!(!reply_parse_failed("   \n\t "));
        assert!(reply_parse_failed("not json at all"));
        assert!(!reply_parse_failed(r#"{"retire": [], "rewrite": []}"#));
    }

    #[test]
    fn rejected_action_records_payload() {
        let win = window(vec![fact(1, "x", vec![])]);
        let plan = validate(
            &win,
            &[json!({"fact_ids": [999], "reason": "phantom"})],
            &[],
            &ego_canonical_key("fam"),
            50,
            Vec::new(),
        );
        let RejectedAction {
            payload, detail, ..
        } = &plan.rejected[0];
        assert_eq!(payload["reason"], json!("phantom"));
        assert_eq!(detail, "(999,)");
    }
}
