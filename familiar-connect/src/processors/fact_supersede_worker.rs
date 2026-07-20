//! FactSupersedeWorker retirement (subsystem 07; Python `processors/fact_supersede_worker.py`).
//!
//! For each newly-appended fact, ask the background LLM whether it retires any
//! prior current facts about the same subject — if so, call
//! [`AsyncHistoryStore::supersede`] (existing-id form) to repoint them at the
//! new fact. This is the *system-time* half of the fact lifecycle; `valid_to`
//! (world-time) is left to the extractor and to a speaker who anchors a
//! real-world end.
//!
//! The watermark (`last_seen_fact_id`) is **in-memory only** — it starts at 0
//! and is primed to `latest_fact_id` at run start so a fresh deploy never burns
//! LLM calls re-evaluating historical facts. State is lost on restart and
//! re-primed to "latest", so facts appended while the process was down are never
//! supersede-evaluated (accepted: this bookkeeping is best-effort).

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::diagnostics::spans::timed_async;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{Fact, NewFact};
use crate::llm::{LlmClient, Message};
use crate::log_style as ls;
use crate::structured_output::coerce_positive_int_list;
use crate::structured_request::{
    DEFAULT_MAX_RETRIES, Field, Schema, render_contract, request_structured,
};

/// Log/task label + registry name for this projector.
const NAME: &str = "fact-supersede-worker";

/// Reply-shape contract for "which priors does this new fact retire".
static SUPERSEDE_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    Schema::object(vec![Field::new("superseded_ids", "[<id>...]")])
        .with_empty_note("Empty list when nothing is retired.")
        .with_constraints(vec![
            "Only include ids from the Prior facts list below — do not invent ids.".to_string(),
        ])
});

/// Retires prior facts replaced by newer ones about the same subject.
pub struct FactSupersedeWorker {
    store: Arc<AsyncHistoryStore>,
    llm: Arc<dyn LlmClient>,
    familiar_id: String,
    batch_size: i64,
    priors_max: usize,
    tick_interval: Duration,
    last_seen_fact_id: AtomicI64,
}

impl FactSupersedeWorker {
    /// Construct with the required handles; knobs default per spec
    /// (`batch_size = 5`, `tick_interval_s = 60.0`, `priors_max = 20`).
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
            batch_size: 5,
            priors_max: 20,
            tick_interval: Duration::from_secs_f64(60.0),
            last_seen_fact_id: AtomicI64::new(0),
        }
    }

    /// Max new facts per tick (clamped to `>= 1`).
    #[must_use]
    pub const fn batch_size(mut self, batch_size: i64) -> Self {
        self.batch_size = if batch_size < 1 { 1 } else { batch_size };
        self
    }

    /// Cap on prior facts shown to the LLM per subject (clamped to `>= 1`).
    #[must_use]
    pub fn priors_max(mut self, priors_max: i64) -> Self {
        self.priors_max = usize::try_from(priors_max.max(1)).unwrap_or(usize::MAX);
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

    /// Skip the historical backlog; start from the latest current fact id.
    ///
    /// Called by the projector factory at process start (and directly by tests
    /// to assert the no-new-facts path) so a fresh deploy doesn't burn LLM calls
    /// re-evaluating every old fact on the first tick. SYNC — reads
    /// `store.sync().latest_fact_id`, mirroring the Python `.sync` escape hatch.
    ///
    /// # Errors
    /// Propagates a [`crate::history::StoreError`] if the underlying read fails.
    pub fn prime_watermark(&self) -> Result<(), crate::history::StoreError> {
        let latest = self.store.sync().latest_fact_id(&self.familiar_id)?;
        self.last_seen_fact_id.store(latest, Ordering::SeqCst);
        Ok(())
    }

    /// Forever loop; primes the watermark, then ticks on interval. Cancel to stop.
    pub async fn run(&self, cancel: CancellationToken) {
        if let Err(exc) = self.prime_watermark() {
            tracing::warn!(
                target: "familiar_connect.processors.fact_supersede_worker",
                "{} {}",
                ls::tag("Supersede", ls::R),
                ls::kv_styled("prime_error", &format!("{exc:?}"), ls::W, ls::R),
            );
        }
        loop {
            if cancel.is_cancelled() {
                break;
            }
            if let Err(exc) = self.tick().await {
                tracing::warn!(
                    target: "familiar_connect.processors.fact_supersede_worker",
                    "{} {}",
                    ls::tag("Supersede", ls::R),
                    ls::kv_styled("tick_error", &format!("{exc:?}"), ls::W, ls::R),
                );
            }
            tokio::select! {
                () = cancel.cancelled() => break,
                () = tokio::time::sleep(self.tick_interval) => {}
            }
        }
    }

    /// Evaluate up to `batch_size` new current facts; return the count retired.
    pub async fn tick(&self) -> anyhow::Result<i64> {
        timed_async("fact_supersede.tick", async move {
            let candidates = self
                .store
                .recent_facts(self.familiar_id.clone(), self.batch_size, false, None)
                .await?;
            let watermark = self.last_seen_fact_id.load(Ordering::SeqCst);
            let mut new: Vec<Fact> = candidates
                .iter()
                .filter(|f| f.id > watermark)
                .cloned()
                .collect();
            if new.is_empty() {
                return Ok(0);
            }
            // Oldest-new first so cascading retirements settle deterministically.
            new.sort_by_key(|f| f.id);
            let mut retired: i64 = 0;
            for f_new in &new {
                retired += self.evaluate(f_new).await?;
            }
            // Advance the watermark to the highest id seen this tick — even on
            // garbage LLM output — so a consistently-unparseable fact can't loop.
            if let Some(max_id) = candidates.iter().map(|f| f.id).max() {
                self.last_seen_fact_id.store(max_id, Ordering::SeqCst);
            }
            if retired > 0 {
                tracing::info!(
                    target: "familiar_connect.processors.fact_supersede_worker",
                    "{} {} {} {}",
                    ls::tag("Supersede", ls::LM),
                    ls::kv_styled("evaluated", &new.len().to_string(), ls::W, ls::LW),
                    ls::kv_styled("retired", &retired.to_string(), ls::W, ls::LM),
                    ls::kv_styled(
                        "watermark",
                        &self.last_seen_fact_id.load(Ordering::SeqCst).to_string(),
                        ls::W,
                        ls::LW,
                    ),
                );
            }
            Ok(retired)
        })
        .await
    }

    /// Ask the LLM which priors `f_new` retires; supersede them; return the count.
    async fn evaluate(&self, f_new: &Fact) -> anyhow::Result<i64> {
        if f_new.subjects.is_empty() {
            return Ok(0);
        }
        let mut retired: i64 = 0;
        let mut seen_priors: HashSet<i64> = HashSet::new();
        for subject in &f_new.subjects {
            let priors = self
                .store
                .facts_for_subject(
                    self.familiar_id.clone(),
                    subject.canonical_key.clone(),
                    0,
                    false,
                    None,
                )
                .await?;
            // Exclude f_new itself, dedupe across subjects, cap to the last
            // `priors_max` (most recent).
            let filtered: Vec<&Fact> = priors
                .iter()
                .filter(|p| p.id != f_new.id && !seen_priors.contains(&p.id))
                .collect();
            let start = filtered.len().saturating_sub(self.priors_max);
            let unique = &filtered[start..];
            if unique.is_empty() {
                continue;
            }
            for p in unique {
                seen_priors.insert(p.id);
            }
            let valid: HashSet<i64> = unique.iter().map(|p| p.id).collect();

            let prompt = build_supersede_prompt(f_new, unique);
            let result = request_structured(
                self.llm.as_ref(),
                &prompt,
                &SUPERSEDE_SCHEMA,
                DEFAULT_MAX_RETRIES,
            )
            .await?;
            let ids = superseded_ids(result.value.as_ref(), &valid);
            if ids.is_empty() {
                continue;
            }
            // Existing-id form: repoint each old row at f_new (mints nothing).
            // Per-id skip-and-record — a prior already retired by an earlier
            // subject lands in `skipped`, not an exception.
            let outcome = self
                .store
                .supersede(self.familiar_id.clone(), ids, NewFact::Repoint(f_new.id))
                .await?;
            retired += i64::try_from(outcome.superseded.len()).unwrap_or(i64::MAX);
        }
        Ok(retired)
    }
}

/// LLM prompt: which priors does `f_new` replace?
fn build_supersede_prompt(f_new: &Fact, priors: &[&Fact]) -> Vec<Message> {
    let persona = "You decide whether a new fact retires earlier facts about the \
        same person. A fact is *retired* when the new one contradicts \
        or directly replaces it (e.g., 'Alice loves hiking' is retired \
        by \"Alice now hates hiking\"). A fact is NOT retired just \
        because it's older or differently worded — facts about \
        independent topics coexist.";
    let header = format!("{persona}\n\n{}", render_contract(&SUPERSEDE_SCHEMA));
    let mut lines: Vec<String> = vec![
        format!("New fact (id={}): {}", f_new.id, f_new.text),
        String::new(),
        "Prior facts:".to_string(),
    ];
    lines.extend(priors.iter().map(|p| format!("- id={}: {}", p.id, p.text)));
    vec![
        Message::new("system", header),
        Message::new("user", lines.join("\n")),
    ]
}

/// Distinct prior ids the model marked superseded, filtered to `valid`.
///
/// A non-object, missing key, or non-list value all degrade to `[]`.
fn superseded_ids(value: Option<&Value>, valid: &HashSet<i64>) -> Vec<i64> {
    let Some(Value::Object(map)) = value else {
        return Vec::new();
    };
    let raw = map.get("superseded_ids").cloned().unwrap_or(Value::Null);
    coerce_positive_int_list(&raw)
        .into_iter()
        .filter(|i| valid.contains(i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::superseded_ids;
    use serde_json::json;
    use std::collections::HashSet;

    fn valid(ids: &[i64]) -> HashSet<i64> {
        ids.iter().copied().collect()
    }

    #[test]
    fn non_object_degrades_to_empty() {
        assert!(superseded_ids(Some(&json!([1, 2])), &valid(&[1, 2])).is_empty());
        assert!(superseded_ids(Some(&json!("x")), &valid(&[1])).is_empty());
        assert!(superseded_ids(None, &valid(&[1])).is_empty());
    }

    #[test]
    fn missing_key_degrades_to_empty() {
        assert!(superseded_ids(Some(&json!({"other": [1]})), &valid(&[1])).is_empty());
    }

    #[test]
    fn filters_to_valid_and_drops_hallucinated() {
        // 9999 is not in the shown prior set; the new fact's own id would be
        // filtered the same way.
        assert_eq!(
            superseded_ids(
                Some(&json!({"superseded_ids": [3, 9999, 4]})),
                &valid(&[3, 4])
            ),
            vec![3, 4]
        );
    }

    #[test]
    fn coercion_rejects_bools_and_dedups() {
        // JSON booleans never coerce to 1; duplicates collapse.
        assert_eq!(
            superseded_ids(
                Some(&json!({"superseded_ids": [true, 3, "3", "4", 4]})),
                &valid(&[3, 4])
            ),
            vec![3, 4]
        );
    }
}
