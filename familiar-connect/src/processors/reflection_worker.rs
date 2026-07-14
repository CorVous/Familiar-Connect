//! ReflectionWorker higher-order reflections (subsystem 07; Python `processors/reflection_worker.py`).
//!
//! Compounds higher-order syntheses over recent turns + facts. Fires when at
//! least `turns_threshold` new turns have accumulated past the
//! `reflection_watermark`, asks the LLM "what high-level questions do recent
//! events raise?", and persists each answer as one `reflections` row with
//! `cited_turn_ids` / `cited_fact_ids` provenance.
//!
//! Two guardrails prevent runaway token spend: the per-tick window is capped at
//! `max_turns_per_tick` (older turns skipped, not deferred), and the watermark
//! advances at the END of every tick — even on `[]`, malformed JSON, all-items-
//! filtered, or a mid-tick transport error. Without the always-advance, a no-op
//! tick would re-send the same ever-growing window and balloon into a
//! 100k-token prompt.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::diagnostics::spans::timed_async;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{Fact, HistoryTurn};
use crate::llm::{LlmClient, Message};
use crate::log_style as ls;
use crate::structured_request::{
    DEFAULT_MAX_RETRIES, Field, Schema, render_contract, request_structured,
};

/// Log/task label + registry name for this projector.
const NAME: &str = "reflection-worker";

/// Writes higher-order reflections off the turns + facts watermark.
pub struct ReflectionWorker {
    store: Arc<AsyncHistoryStore>,
    llm: Arc<dyn LlmClient>,
    familiar_id: String,
    turns_threshold: i64,
    max_per_tick: i64,
    max_turns_per_tick: usize,
    recent_facts_limit: i64,
    tick_interval: Duration,
}

impl ReflectionWorker {
    /// Construct with the required handles; knobs default per spec
    /// (`turns_threshold = 20`, `max_reflections_per_tick = 3`,
    /// `max_turns_per_tick = 50`, `recent_facts_limit = 20`,
    /// `tick_interval_s = 60.0`).
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
            turns_threshold: 20,
            max_per_tick: 3,
            max_turns_per_tick: 50,
            recent_facts_limit: 20,
            tick_interval: Duration::from_secs_f64(60.0),
        }
    }

    /// New turns required to fire (clamped to `>= 1`).
    #[must_use]
    pub const fn turns_threshold(mut self, threshold: i64) -> Self {
        self.turns_threshold = if threshold < 1 { 1 } else { threshold };
        self
    }

    /// Cap on reflections requested per tick (clamped to `>= 1`).
    #[must_use]
    pub const fn max_reflections_per_tick(mut self, cap: i64) -> Self {
        self.max_per_tick = if cap < 1 { 1 } else { cap };
        self
    }

    /// Cap on turns entering the prompt per tick (clamped to `>= 1`).
    #[must_use]
    pub fn max_turns_per_tick(mut self, cap: i64) -> Self {
        self.max_turns_per_tick = usize::try_from(cap.max(1)).unwrap_or(usize::MAX);
        self
    }

    /// Context-fact limit (clamped to `>= 0`).
    #[must_use]
    pub const fn recent_facts_limit(mut self, limit: i64) -> Self {
        self.recent_facts_limit = if limit < 0 { 0 } else { limit };
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

    /// Forever loop; tick on interval. Cancel the token to stop.
    pub async fn run(&self, cancel: CancellationToken) {
        loop {
            if cancel.is_cancelled() {
                break;
            }
            if let Err(exc) = self.tick().await {
                tracing::warn!(
                    target: "familiar_connect.processors.reflection_worker",
                    "{} {}",
                    ls::tag("Reflection", ls::R),
                    ls::kv_styled("tick_error", &format!("{exc:?}"), ls::W, ls::R),
                );
            }
            tokio::select! {
                () = cancel.cancelled() => break,
                () = tokio::time::sleep(self.tick_interval) => {}
            }
        }
    }

    /// One pass; write reflections if enough new turns accumulated.
    ///
    /// The watermark ALWAYS advances to `latest_turn` — mirroring the Python
    /// `try/finally` — so an empty/malformed/all-filtered reply, or a mid-tick
    /// transport error, cannot pin the worker to an ever-growing window.
    pub async fn tick(&self) -> anyhow::Result<()> {
        timed_async("reflection.tick", async move {
            let latest_turn = match self.store.latest_id(self.familiar_id.clone(), None).await? {
                Some(t) if t > 0 => t,
                _ => return Ok(()),
            };
            let (prior_turn_wm, _prior_fact_wm) = self
                .store
                .latest_reflection_watermarks(self.familiar_id.clone())
                .await?;
            if latest_turn - prior_turn_wm < self.turns_threshold {
                return Ok(());
            }
            let latest_fact = self.store.latest_fact_id(self.familiar_id.clone()).await?;

            // Run the body, then ALWAYS advance the watermark (the `finally`).
            let result = self.do_tick(prior_turn_wm, latest_turn, latest_fact).await;
            self.store
                .set_reflection_watermark(self.familiar_id.clone(), latest_turn, latest_fact)
                .await?;
            result
        })
        .await
    }

    async fn do_tick(
        &self,
        prior_turn_wm: i64,
        latest_turn: i64,
        latest_fact: i64,
    ) -> anyhow::Result<()> {
        let mut new_turns = self
            .store
            .turns_in_id_range(self.familiar_id.clone(), prior_turn_wm, latest_turn, None)
            .await?;
        if new_turns.is_empty() {
            return Ok(());
        }
        // Cap the window — keep the most-recent tail; older turns are skipped,
        // not deferred (runaway-cost guardrail).
        if new_turns.len() > self.max_turns_per_tick {
            let start = new_turns.len() - self.max_turns_per_tick;
            new_turns = new_turns.split_off(start);
        }

        let recent_facts = self
            .store
            .recent_facts(
                self.familiar_id.clone(),
                self.recent_facts_limit,
                false,
                None,
            )
            .await?;

        let schema = reflection_schema(self.max_per_tick);
        let prompt = build_reflection_prompt(&new_turns, &recent_facts, &schema);
        let result =
            request_structured(self.llm.as_ref(), &prompt, &schema, DEFAULT_MAX_RETRIES).await?;
        let items = normalize_reflection_items(result.value.as_ref());

        let valid_turn_ids: HashSet<i64> = new_turns.iter().map(|t| t.id).collect();
        let valid_fact_ids: HashSet<i64> = recent_facts.iter().map(|f| f.id).collect();
        // A reflection may legitimately cite older facts surfaced via a dossier;
        // widen to all known facts so we don't drop those. But when the recent-
        // facts context was empty, the valid fact set is empty (no query).
        let all_known_fact_ids: HashSet<i64> = if valid_fact_ids.is_empty() {
            valid_fact_ids.clone()
        } else {
            self.store.all_fact_ids(self.familiar_id.clone()).await?
        };

        // Per-channel scoping: pick the most-frequent channel, else None.
        let channel_id = dominant_channel(&new_turns);

        let mut written: i64 = 0;
        for item in &items {
            let text = item.text.trim();
            if text.is_empty() {
                continue;
            }
            let cited_turns: Vec<i64> = item
                .cited_turn_ids
                .iter()
                .copied()
                .filter(|i| valid_turn_ids.contains(i))
                .collect();
            let cited_facts: Vec<i64> = item
                .cited_fact_ids
                .iter()
                .copied()
                .filter(|i| all_known_fact_ids.contains(i))
                .collect();
            // Require at least one valid citation; an uncited reflection is a
            // free-floating opinion, not synthesis.
            if cited_turns.is_empty() && cited_facts.is_empty() {
                continue;
            }
            self.store
                .append_reflection(
                    self.familiar_id.clone(),
                    channel_id,
                    text.to_string(),
                    cited_turns,
                    cited_facts,
                    latest_turn,
                    latest_fact,
                )
                .await?;
            written += 1;
        }

        tracing::info!(
            target: "familiar_connect.processors.reflection_worker",
            "{} {} {} {} {}",
            ls::tag("Reflection", ls::LC),
            ls::kv_styled("new_turns", &new_turns.len().to_string(), ls::W, ls::LY),
            ls::kv_styled("written", &written.to_string(), ls::W, ls::LC),
            ls::kv_styled("turn_wm", &latest_turn.to_string(), ls::W, ls::LW),
            ls::kv_styled("fact_wm", &latest_fact.to_string(), ls::W, ls::LW),
        );
        Ok(())
    }
}

/// Most-frequent channel id across `turns`; `None` for a cross-channel batch
/// with no majority winner. Ties go to the first-encountered channel (Python
/// `max` over the insertion-ordered count dict).
fn dominant_channel(turns: &[HistoryTurn]) -> Option<i64> {
    let mut order: Vec<i64> = Vec::new();
    let mut counts: HashMap<i64, i64> = HashMap::new();
    for t in turns {
        if !counts.contains_key(&t.channel_id) {
            order.push(t.channel_id);
        }
        *counts.entry(t.channel_id).or_insert(0) += 1;
    }
    if order.len() == 1 {
        return Some(order[0]);
    }
    let first = *order.first()?;
    let mut best = first;
    let mut best_count = counts[&first];
    for &ch in &order[1..] {
        if counts[&ch] > best_count {
            best = ch;
            best_count = counts[&ch];
        }
    }
    Some(best)
}

/// Reply-shape contract for a reflection batch, capped at `max_reflections`.
fn reflection_schema(max_reflections: i64) -> Schema {
    Schema::array(vec![
        Field::new("text", "\"<one or two sentences>\"").with_desc("one or two sentences"),
        Field::new("cited_turn_ids", "[<id>...]").with_desc(
            "turn ids the reflection draws from; pick the most \
             representative, not all of them",
        ),
        Field::new("cited_fact_ids", "[<id>...]")
            .with_desc("fact ids if the reflection leans on stored facts; may be empty")
            .optional(),
    ])
    .with_constraints(vec![
        format!("Reply with at most {max_reflections} items."),
        "Cite at least one turn id or fact id per reflection.".to_string(),
    ])
    .with_empty_note("If nothing of substance is happening, reply with [].")
}

fn build_reflection_prompt(
    new_turns: &[HistoryTurn],
    recent_facts: &[Fact],
    schema: &Schema,
) -> Vec<Message> {
    let persona = "You write short, high-level reflections over recent chat \
        history — patterns, recurring tensions, open questions, \
        themes the participants keep circling back to. Skip blow-by-\
        blow recaps; that's what summaries are for. Each reflection \
        is one or two sentences.";
    let header = format!("{persona}\n\n{}", render_contract(schema));
    let mut lines: Vec<String> = vec!["Recent turns (id prefixed):".to_string()];
    for t in new_turns {
        // Python: `who = t.author.display_name if t.author is not None else
        // t.role`, then f-string-rendered. An author present with a `None`
        // display name renders the literal `"None"` (not empty, not `role`).
        let who = t.author.as_ref().map_or_else(
            || t.role.clone(),
            |a| a.display_name.clone().unwrap_or_else(|| "None".to_string()),
        );
        lines.push(format!("- id={} [{who}] {}", t.id, t.content));
    }
    if !recent_facts.is_empty() {
        lines.push(String::new());
        lines.push("Recent facts (id prefixed):".to_string());
        for f in recent_facts {
            lines.push(format!("- id={} {}", f.id, f.text));
        }
    }
    vec![
        Message::new("system", header),
        Message::new("user", lines.join("\n")),
    ]
}

/// One normalized reflection item. Citations accept `int` only (no string/bool
/// coercion — stricter than the extractor).
struct ReflectionItem {
    text: String,
    cited_turn_ids: Vec<i64>,
    cited_fact_ids: Vec<i64>,
}

/// Normalize parsed reflection items; bad / non-array input → `[]`.
fn normalize_reflection_items(parsed: Option<&Value>) -> Vec<ReflectionItem> {
    let Some(Value::Array(items)) = parsed else {
        return Vec::new();
    };
    let mut out: Vec<ReflectionItem> = Vec::new();
    for item in items {
        let Value::Object(map) = item else {
            continue;
        };
        let text = map.get("text").map_or_else(String::new, py_str);
        let cited_turn_ids = int_list(map.get("cited_turn_ids"));
        let cited_fact_ids = int_list(map.get("cited_fact_ids"));
        out.push(ReflectionItem {
            text,
            cited_turn_ids,
            cited_fact_ids,
        });
    }
    out
}

/// Integer-only citation list: keeps JSON integers, drops strings, floats,
/// bools (JSON `true`/`false` are distinct variants — the Python `True == 1`
/// hazard cannot occur here), and non-array input.
fn int_list(raw: Option<&Value>) -> Vec<i64> {
    let Some(Value::Array(arr)) = raw else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|v| match v {
            Value::Number(n) => n.as_i64(),
            _ => None,
        })
        .collect()
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
        build_reflection_prompt, dominant_channel, int_list, normalize_reflection_items,
        reflection_schema,
    };
    use crate::history::store::{Fact, HistoryTurn};
    use crate::identity::Author;
    use chrono::Utc;
    use serde_json::json;

    fn turn(id: i64, channel_id: i64) -> HistoryTurn {
        HistoryTurn {
            id,
            timestamp: Utc::now(),
            role: "user".to_string(),
            author: None,
            content: format!("m{id}"),
            channel_id,
            mode: None,
            platform_message_id: None,
            reply_to_message_id: None,
            guild_id: None,
            arrived_at: None,
            consumed_at: None,
            pings_bot: false,
        }
    }

    #[test]
    fn dominant_single_channel() {
        assert_eq!(dominant_channel(&[turn(1, 7), turn(2, 7)]), Some(7));
    }

    #[test]
    fn dominant_empty_is_none() {
        assert_eq!(dominant_channel(&[]), None);
    }

    #[test]
    fn dominant_most_frequent_first_encountered_on_tie() {
        // channel 3 appears first; tie (2 each) → first-encountered wins.
        let turns = [turn(1, 3), turn(2, 5), turn(3, 3), turn(4, 5)];
        assert_eq!(dominant_channel(&turns), Some(3));
        // A clear majority wins regardless of order.
        let turns = [turn(1, 5), turn(2, 3), turn(3, 3)];
        assert_eq!(dominant_channel(&turns), Some(3));
    }

    #[test]
    fn int_list_keeps_ints_drops_everything_else() {
        assert_eq!(int_list(Some(&json!([1, "2", 3.5, true, 4]))), vec![1, 4]);
        assert_eq!(int_list(Some(&json!("not a list"))), Vec::<i64>::new());
        assert_eq!(int_list(None), Vec::<i64>::new());
    }

    #[test]
    fn normalize_skips_non_objects_and_non_array_root() {
        assert!(normalize_reflection_items(Some(&json!("nope"))).is_empty());
        assert!(normalize_reflection_items(None).is_empty());
        let items = normalize_reflection_items(Some(&json!([
            {"text": "a", "cited_turn_ids": [1], "cited_fact_ids": []},
            "skip me",
            {"text": "b", "cited_turn_ids": [2, "x"]},
        ])));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "a");
        assert_eq!(items[0].cited_turn_ids, vec![1]);
        assert_eq!(items[1].cited_turn_ids, vec![2]);
        assert!(items[1].cited_fact_ids.is_empty());
    }

    #[test]
    fn reflection_who_renders_role_display_name_and_literal_none() {
        // turn() sets role="user", content="m{id}", author=None.
        let mut named = turn(1, 7);
        named.author = Some(Author::new(
            "discord",
            "1",
            Some("cor".to_string()),
            Some("Cass".to_string()),
        ));
        let mut none_dn = turn(2, 7);
        none_dn.author = Some(Author::new("discord", "2", Some("kd".to_string()), None));
        let role_only = turn(3, 7); // author None → role
        let turns = [named, none_dn, role_only];
        let facts: Vec<Fact> = Vec::new();
        let schema = reflection_schema(3);
        let body = build_reflection_prompt(&turns, &facts, &schema)[1].content_str();
        // display_name present → the name; author present + display_name None →
        // literal "None" (Python f-string parity); author None → role.
        assert!(body.contains("[Cass] m1"), "{body}");
        assert!(body.contains("[None] m2"), "{body}");
        assert!(body.contains("[user] m3"), "{body}");
    }
}
