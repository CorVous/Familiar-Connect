//! Apply a validated [`ConsolidationPlan`] to the DB (subsystem 04; Python
//! `sleep/apply.py`).
//!
//! Separate from planning so a dry run can compute the plan without touching the
//! DB. Apply is supersede-only: retires drop facts with no replacement, rewrites
//! append one consolidated fact then supersede the old ones by it. The sleep
//! watermark advances once, on the **fact axis only** (the turn axis belongs to
//! the opinion pass), regardless of how many actions ran.

use std::collections::HashMap;

use super::consolidation::{ConsolidationPlan, RewriteAction};
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{Fact, FactDraft, FactSubject, NewFact};
use crate::identity::is_ego_key;

/// Outcome of applying a plan.
///
/// `skipped` records actions a concurrent writer invalidated between plan and
/// apply — each `(kind, fact_id, reason)`. Per-action skip-and-record makes
/// apply non-fatal and idempotent: a partial or already-applied plan re-runs
/// cleanly instead of crashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyReport {
    /// Obsolete ids actually retired this call.
    pub retired_fact_ids: Vec<i64>,
    /// `(old_ids, new_id)` for each rewrite that minted a replacement.
    pub rewritten: Vec<(Vec<i64>, i64)>,
    /// `(last_fact_id, last_turn_id)` — the turn element is informational only
    /// (this pass writes only the fact axis).
    pub watermark: (i64, i64),
    /// `(kind, fact_id, reason)` for rows a concurrent writer invalidated.
    pub skipped: Vec<(String, i64, String)>,
}

/// Build the `FactSubject` list for a rewrite's chosen keys.
///
/// `display_at_write` resolves from the matching source-fact subject when
/// present (first-seen wins, iterating `old_fact_ids` order then subject order);
/// for an ego key it falls back to `familiar_display_name` or the key's local
/// part; otherwise the key's local part.
///
/// # Errors
/// Raises when a source id is absent from `by_id` — Python indexes `by_id[fid]`,
/// raising `KeyError`; supersede-only removal keeps every plan-time id fetchable
/// by exact id at apply time, so this is unreachable in the pipeline.
fn subjects_for_rewrite(
    action: &RewriteAction,
    by_id: &HashMap<i64, Fact>,
    familiar_display_name: Option<&str>,
) -> anyhow::Result<Vec<FactSubject>> {
    let mut display_by_key: HashMap<String, String> = HashMap::new();
    for fid in &action.old_fact_ids {
        let f = by_id
            .get(fid)
            .ok_or_else(|| anyhow::anyhow!("rewrite source fact {fid} missing from snapshot"))?;
        for s in &f.subjects {
            display_by_key
                .entry(s.canonical_key.clone())
                .or_insert_with(|| s.display_at_write.clone());
        }
    }
    let mut out: Vec<FactSubject> = Vec::new();
    for key in &action.subject_keys {
        let display = display_by_key.get(key).map_or_else(
            || {
                if is_ego_key(key) {
                    familiar_display_name.map_or_else(|| local_part(key), ToOwned::to_owned)
                } else {
                    local_part(key)
                }
            },
            Clone::clone,
        );
        out.push(FactSubject {
            canonical_key: key.clone(),
            display_at_write: display,
        });
    }
    Ok(out)
}

/// Python `key.split(":", 1)[-1]`: everything after the first `:`, or the whole
/// string when there is no colon.
fn local_part(key: &str) -> String {
    key.split_once(':')
        .map_or_else(|| key.to_owned(), |(_, rest)| rest.to_owned())
}

/// Execute a plan's accepted actions; advance the sleep watermark (fact axis).
///
/// This is the ONLY consolidation mutator — planning is dry-run-safe by
/// construction. Apply order: snapshot all rewrite source facts up front (exact
/// id fetch, includes superseded), execute all retires, then all rewrites, then
/// advance the fact-axis watermark once (even when everything skipped).
///
/// # Errors
/// Propagates store faults.
pub async fn apply_consolidation(
    store: &AsyncHistoryStore,
    plan: &ConsolidationPlan,
    familiar_display_name: Option<&str>,
) -> anyhow::Result<ApplyReport> {
    let fam = &plan.familiar_id;

    // snapshot rewrite sources up front (exact id fetch, includes superseded) to
    // resolve subject displays + the merge channel even as rows get superseded.
    let mut needed: Vec<i64> = Vec::new();
    for a in &plan.rewrite {
        for &id in &a.old_fact_ids {
            if !needed.contains(&id) {
                needed.push(id);
            }
        }
    }
    let by_id: HashMap<i64, Fact> = store
        .facts_by_ids(fam.clone(), needed)
        .await?
        .into_iter()
        .map(|f| (f.id, f))
        .collect();

    let mut skipped: Vec<(String, i64, String)> = Vec::new();
    let mut retired: Vec<i64> = Vec::new();
    for action in &plan.retire {
        // the store retires each id, recording (not raising on) any a concurrent
        // writer already retired/superseded since plan time.
        let result = store
            .supersede(fam.clone(), action.fact_ids.clone(), NewFact::Retire)
            .await?;
        retired.extend(result.superseded);
        skipped.extend(
            result
                .skipped
                .into_iter()
                .map(|(fid, reason)| ("retire".to_owned(), fid, reason)),
        );
    }

    let mut rewritten: Vec<(Vec<i64>, i64)> = Vec::new();
    for action in &plan.rewrite {
        // the store owns the merge: it pre-flights every source, unions their
        // provenance, and mints only if all are current — declining the whole
        // merge otherwise. Subjects stay caller-prepared; channel follows the
        // first obsolete row. A source id absent from the snapshot raises
        // (Python indexes `by_id[...]`, a KeyError) rather than degrading —
        // unreachable in the pipeline, where supersede is the only removal path.
        let channel_id = by_id
            .get(&action.old_fact_ids[0])
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rewrite source fact {} missing from snapshot",
                    action.old_fact_ids[0]
                )
            })?
            .channel_id;
        let draft = FactDraft {
            channel_id,
            text: action.new_text.clone(),
            subjects: subjects_for_rewrite(action, &by_id, familiar_display_name)?,
        };
        let result = store
            .supersede(
                fam.clone(),
                action.old_fact_ids.clone(),
                NewFact::Merge(draft),
            )
            .await?;
        match result.minted {
            Some(minted) => rewritten.push((action.old_fact_ids.clone(), minted.id)),
            None => {
                // a source raced between plan + apply — the store declined the
                // whole merge; record every skip and continue.
                skipped.extend(
                    result
                        .skipped
                        .into_iter()
                        .map(|(fid, reason)| ("rewrite".to_owned(), fid, reason)),
                );
            }
        }
    }

    // consolidation owns the FACT axis only; the turn axis belongs to the
    // opinion pass. Advancing just last_fact_id leaves its progress untouched.
    store
        .advance_sleep_watermark(fam.clone(), Some(plan.new_last_fact_id), None)
        .await?;
    tracing::info!(
        target: "familiar_connect.sleep.apply",
        "sleep-consolidation applied familiar={fam} retired={} rewritten={} skipped={} fact_watermark={}",
        retired.len(),
        rewritten.len(),
        skipped.len(),
        plan.new_last_fact_id,
    );
    Ok(ApplyReport {
        retired_fact_ids: retired,
        rewritten,
        watermark: (plan.new_last_fact_id, plan.new_last_turn_id),
        skipped,
    })
}
