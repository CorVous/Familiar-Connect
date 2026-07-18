//! PeopleDossierWorker per-person dossiers (subsystem 07; Python `processors/people_dossier_worker.py`).
//!
//! For each `canonical_key` appearing as a subject on at least one current fact,
//! maintains a compounded summary in `people_dossiers`. Refreshes when the
//! subject's max `facts.id` exceeds the dossier's `last_fact_id` watermark —
//! same compounding shape as [`SummaryWorker`](super::summary_worker) so dossier
//! cost stays bounded (only new facts ride each refresh).
//!
//! The self-dossier (`ego:<id>`) is special: it is always-injected on the read
//! path, so low-importance "texture" facts are filtered out of it, kept facts
//! are ordered importance-descending, and the writer is steered to preserve
//! settled opinions/stances rather than blanket-drop feelings.

use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use tokio_util::sync::CancellationToken;

use crate::diagnostics::spans::timed_async;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::Fact;
use crate::identity::is_ego_key;
use crate::llm::{LlmClient, Message};
use crate::log_style as ls;

/// Log/task label + registry name for this projector.
const NAME: &str = "people-dossier-worker";

/// Self-dossier importance floor: the dream pass writes momentary "texture"
/// opinions at low importance (2-3); they stay in the DB (RAG-recallable) but
/// must not flood the always-injected self-record. NULL-importance (legacy /
/// extractor) facts are kept — only explicitly-low ones are filtered.
const SELF_DOSSIER_MIN_IMPORTANCE: i64 = 5;

/// The self prompt annotates facts `(importance N)`; strip any the writer echoes
/// back so metadata never lands in the always-injected self-record.
static IMPORTANCE_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(importance \d+\)\s*").expect("valid importance-tag regex"));

/// Facts that should shape a dossier.
///
/// The self-dossier drops low-importance texture, then orders kept facts
/// importance-descending (NULL ranks at the keep-threshold; a stable sort
/// preserves recency within a tier) so the writer sees the most central stances
/// first. Other subjects (and their NULL-importance facts) pass through unchanged.
fn dossier_facts(facts: Vec<Fact>, is_self: bool) -> Vec<Fact> {
    if !is_self {
        return facts;
    }
    let mut kept: Vec<Fact> = facts
        .into_iter()
        .filter(|f| {
            f.importance
                .is_none_or(|i| i >= SELF_DOSSIER_MIN_IMPORTANCE)
        })
        .collect();
    // Stable sort, importance-descending; NULL ranks at the keep-threshold band.
    kept.sort_by(|a, b| {
        let ka = a.importance.unwrap_or(SELF_DOSSIER_MIN_IMPORTANCE);
        let kb = b.importance.unwrap_or(SELF_DOSSIER_MIN_IMPORTANCE);
        kb.cmp(&ka)
    });
    kept
}

/// Rebuilds per-person dossiers off the facts watermark.
pub struct PeopleDossierWorker {
    store: Arc<AsyncHistoryStore>,
    llm: Arc<dyn LlmClient>,
    familiar_id: String,
    familiar_display_name: Option<String>,
    tick_interval: Duration,
}

impl PeopleDossierWorker {
    /// Construct with the required handles; `familiar_display_name` defaults to
    /// the title-cased `familiar_id`, `tick_interval_s` to 20.0.
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
            tick_interval: Duration::from_secs_f64(20.0),
        }
    }

    /// Label for the reserved self-subject.
    #[must_use]
    pub fn familiar_display_name(mut self, name: impl Into<String>) -> Self {
        self.familiar_display_name = Some(name.into());
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

    /// The self-subject display name (explicit, else the title-cased id).
    ///
    /// Mirrors Python `familiar_display_name or familiar_id.title()`: the `or`
    /// falls back for an EMPTY explicit name as well as an absent one.
    fn display_name(&self) -> String {
        resolve_display_name(self.familiar_display_name.as_deref(), &self.familiar_id)
    }

    /// Forever loop; tick on interval. Cancel the token to stop.
    pub async fn run(&self, cancel: CancellationToken) {
        loop {
            if cancel.is_cancelled() {
                break;
            }
            if let Err(exc) = self.tick().await {
                tracing::warn!(
                    target: "familiar_connect.processors.people_dossier_worker",
                    "{} {}",
                    ls::tag("PeopleDossier", ls::R),
                    ls::kv_styled("tick_error", &format!("{exc:?}"), ls::W, ls::R),
                );
            }
            tokio::select! {
                () = cancel.cancelled() => break,
                () = tokio::time::sleep(self.tick_interval) => {}
            }
        }
    }

    /// Refresh any dossier whose subject has new facts.
    pub async fn tick(&self) -> anyhow::Result<()> {
        timed_async("people_dossier.tick", async move {
            let latest_per_subject = self
                .store
                .subjects_with_facts(self.familiar_id.clone())
                .await?;
            for (canonical_key, latest_fact_id) in latest_per_subject {
                self.maybe_refresh(&canonical_key, latest_fact_id).await?;
            }
            Ok(())
        })
        .await
    }

    async fn maybe_refresh(&self, canonical_key: &str, latest_fact_id: i64) -> anyhow::Result<()> {
        let prior = self
            .store
            .get_people_dossier(self.familiar_id.clone(), canonical_key.to_string())
            .await?;
        let prior_wm = prior.as_ref().map_or(0, |p| p.last_fact_id);
        if latest_fact_id <= prior_wm {
            return Ok(());
        }

        let new_facts = self
            .store
            .facts_for_subject(
                self.familiar_id.clone(),
                canonical_key.to_string(),
                prior_wm,
                false,
                None,
            )
            .await?;
        if new_facts.is_empty() {
            return Ok(());
        }

        let is_self = is_ego_key(canonical_key);
        let facts = dossier_facts(new_facts, is_self);
        if facts.is_empty() {
            // Window held only low-importance texture — advance the watermark
            // past it (keeping the prior text) so we don't re-filter every tick.
            // CAS on the read-time watermark so a concurrent supersede's delete
            // (cache-invalidation) is not undone by resurrecting the row (#130).
            if let Some(p) = &prior {
                let landed = self
                    .store
                    .put_people_dossier_if_current(
                        self.familiar_id.clone(),
                        canonical_key.to_string(),
                        Some(prior_wm),
                        latest_fact_id,
                        p.dossier_text.clone(),
                    )
                    .await?;
                if !landed {
                    log_raced_supersede(canonical_key, "texture_watermark");
                }
            }
            return Ok(());
        }

        // The self-subject resolves to the familiar's display name — the store
        // has no account row for `ego:<id>`, so resolve_label would fall through
        // to the raw id tail.
        let display_label = if is_self {
            self.display_name()
        } else {
            self.store
                .resolve_label(
                    canonical_key.to_string(),
                    None,
                    Some(self.familiar_id.clone()),
                )
                .await?
        };
        let prompt = build_dossier_prompt(
            &display_label,
            prior.as_ref().map(|p| p.dossier_text.as_str()),
            &facts,
            is_self,
        );
        let reply = self.llm.chat(prompt).await?;
        let mut text = reply.content_str().trim().to_string();
        if is_self {
            // Defensive: the writer may echo the "(importance N)" annotations.
            text = IMPORTANCE_TAG.replace_all(&text, "").trim().to_string();
        }
        if text.is_empty() {
            // Don't overwrite a real dossier with an empty reply.
            return Ok(());
        }
        // CAS on the read-time watermark: if a supersede deleted the row while
        // the LLM ran, drop this write rather than resurrect an invalidated
        // dossier — the next tick sees prior=None and rebuilds cleanly (#130).
        let landed = self
            .store
            .put_people_dossier_if_current(
                self.familiar_id.clone(),
                canonical_key.to_string(),
                prior.as_ref().map(|p| p.last_fact_id),
                latest_fact_id,
                text.clone(),
            )
            .await?;
        if !landed {
            log_raced_supersede(canonical_key, "rebuild");
            return Ok(());
        }
        tracing::info!(
            target: "familiar_connect.processors.people_dossier_worker",
            "{} {} {} {} {}",
            ls::tag("PeopleDossier", ls::LC),
            ls::kv_styled("subject", canonical_key, ls::W, ls::LY),
            ls::kv_styled("display", &display_label, ls::W, ls::LW),
            ls::kv_styled("watermark", &latest_fact_id.to_string(), ls::W, ls::LC),
            ls::kv_styled("chars", &text.chars().count().to_string(), ls::W, ls::LW),
        );
        Ok(())
    }
}

/// A dossier write was dropped because a concurrent supersede invalidated
/// (deleted) the row between our read and this write (#130). Logged at debug —
/// the worker self-heals on the next tick (`prior=None` → clean rebuild).
fn log_raced_supersede(canonical_key: &str, stage: &str) {
    tracing::debug!(
        target: "familiar_connect.processors.people_dossier_worker",
        "{} {} {} {}",
        ls::tag("PeopleDossier", ls::LC),
        ls::kv_styled("skip", "raced_supersede", ls::W, ls::LY),
        ls::kv_styled("subject", canonical_key, ls::W, ls::LY),
        ls::kv_styled("stage", stage, ls::W, ls::LW),
    );
}

/// Compounding prompt for one subject. Self and non-self differ in header
/// (opinion-preserving vs transient-dropping) and body (importance-annotated vs
/// plain).
fn build_dossier_prompt(
    display_name: &str,
    prior_dossier: Option<&str>,
    new_facts: &[Fact],
    is_self: bool,
) -> Vec<Message> {
    let header = if is_self {
        // The self-record is the substrate for consistently-forming opinions
        // (feeds the sleep cycle) — keep settled feelings/stances, shed only
        // momentary reactions. Do NOT blanket-drop feelings.
        format!(
            "You maintain {display_name}'s evolving self-record — who she \
             is becoming — in 3-5 sentences. Preserve her settled opinions, \
             stances, and feelings about people and things (the views she \
             holds consistently), plus concrete choices and commitments. \
             Drop only momentary, in-the-moment reactions and filler. \
             Reconcile contradictions in favour of newer evidence. Facts \
             carry an importance score (higher = more central/durable to \
             who she is); weight higher-importance stances more heavily, \
             and since the record is only 3-5 sentences, when space is \
             tight favour durable high-importance stances over lower ones \
             (never invent). Reply with the updated self-record text only."
        )
    } else {
        format!(
            "You maintain a short, retrieval-friendly dossier about one \
             person ({display_name}) — 3-5 sentences. Preserve concrete \
             details, names, places, commitments. Drop transient feelings \
             and conversational filler. Reconcile contradictions in favour \
             of newer evidence. Reply with the updated dossier text only."
        )
    };
    let mut body_lines: Vec<String> = Vec::new();
    match prior_dossier {
        Some(prior) if !prior.is_empty() => {
            body_lines.push(format!("Previous dossier:\n{prior}"));
            body_lines.push("\nNew facts:".to_string());
        }
        _ => body_lines.push("Facts:".to_string()),
    }
    if is_self {
        // Annotate scored facts so the writer can weight them; NULL untagged.
        for f in new_facts {
            match f.importance {
                Some(i) => body_lines.push(format!("- (importance {i}) {}", f.text)),
                None => body_lines.push(format!("- {}", f.text)),
            }
        }
    } else {
        for f in new_facts {
            body_lines.push(format!("- {}", f.text));
        }
    }
    vec![
        Message::new("system", header),
        Message::new("user", body_lines.join("\n")),
    ]
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

/// ASCII title-case matching Python `str.title()`: the first alphabetic char of
/// each maximal alphabetic run is upper-cased, the rest lower-cased; every
/// non-alphabetic char passes through and resets the "start of word" flag.
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

#[cfg(test)]
mod tests {
    use super::{SELF_DOSSIER_MIN_IMPORTANCE, dossier_facts, resolve_display_name, title_case};
    use crate::history::store::Fact;
    use chrono::Utc;

    fn fact(id: i64, text: &str, importance: Option<i64>) -> Fact {
        Fact {
            id,
            familiar_id: "fam".to_string(),
            channel_id: Some(1),
            text: text.to_string(),
            source_turn_ids: vec![1],
            created_at: Utc::now(),
            superseded_at: None,
            superseded_by: None,
            subjects: Vec::new(),
            valid_from: None,
            valid_to: None,
            importance,
        }
    }

    #[test]
    fn non_self_passes_through_unchanged() {
        let facts = vec![fact(1, "a", Some(2)), fact(2, "b", None)];
        let out = dossier_facts(facts.clone(), false);
        assert_eq!(out, facts);
    }

    #[test]
    fn self_drops_low_importance_keeps_null() {
        let facts = vec![
            fact(1, "durable", Some(9)),
            fact(2, "texture", Some(2)),
            fact(3, "legacy", None),
        ];
        let out = dossier_facts(facts, true);
        let texts: Vec<&str> = out.iter().map(|f| f.text.as_str()).collect();
        assert!(texts.contains(&"durable"));
        assert!(texts.contains(&"legacy"));
        assert!(!texts.contains(&"texture"));
    }

    #[test]
    fn self_orders_importance_desc_null_in_keep_band() {
        // Seeded [5, 9, 7, None] must render 9, 7, then the 5-band (5 then NULL,
        // in insertion order — stable sort).
        let facts = vec![
            fact(1, "five", Some(5)),
            fact(2, "nine", Some(9)),
            fact(3, "seven", Some(7)),
            fact(4, "null", None),
        ];
        let out = dossier_facts(facts, true);
        let texts: Vec<&str> = out.iter().map(|f| f.text.as_str()).collect();
        assert_eq!(texts, vec!["nine", "seven", "five", "null"]);
    }

    #[test]
    fn keep_threshold_is_five() {
        assert_eq!(SELF_DOSSIER_MIN_IMPORTANCE, 5);
        // A fact exactly at the threshold is kept.
        let out = dossier_facts(vec![fact(1, "edge", Some(5))], true);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn title_case_matches_python_str_title() {
        assert_eq!(title_case("fam"), "Fam");
        assert_eq!(title_case("my-fam"), "My-Fam");
        assert_eq!(title_case("myFam"), "Myfam");
        assert_eq!(title_case("fam2go"), "Fam2Go");
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
