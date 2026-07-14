//! SummaryWorker rolling focus-stream summary (subsystem 07; Python `processors/summary_worker.py`).
//!
//! Regenerates the focus-stream rolling summary from the raw `turns` table —
//! the consumed cross-channel stream the familiar actually attended to, stored
//! in `summaries` under [`FOCUS_STREAM_CHANNEL_ID`]. When `turns_threshold` new
//! consumed turns accumulate past the composite `(consumed_at, id)` watermark,
//! the worker compounds the prior summary plus the new turns via one `chat`
//! call. Watermarking on `consumed_at` (not `id`) catches late-promoted staged
//! turns; the first run is bounded by `backfill_cap`.

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::diagnostics::spans::timed_async;
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::{FOCUS_STREAM_CHANNEL_ID, HistoryTurn};
use crate::llm::{LlmClient, Message};
use crate::log_style as ls;
use crate::support::time::iso_utc;

/// Log/task label + registry name for this projector.
const NAME: &str = "summary-worker";

/// Rebuilds the focus-stream rolling summary off SQLite watermarks.
pub struct SummaryWorker {
    store: Arc<AsyncHistoryStore>,
    llm: Arc<dyn LlmClient>,
    familiar_id: String,
    turns_threshold: i64,
    backfill_cap: i64,
    tick_interval: Duration,
}

impl SummaryWorker {
    /// Construct with the required handles; knobs default per spec
    /// (`turns_threshold = 10`, `backfill_cap = 200`, `tick_interval_s = 5.0`).
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
            turns_threshold: 10,
            backfill_cap: 200,
            tick_interval: Duration::from_secs_f64(5.0),
        }
    }

    /// New consumed turns required to fire (clamped to `>= 1`).
    #[must_use]
    pub const fn turns_threshold(mut self, threshold: i64) -> Self {
        self.turns_threshold = if threshold < 1 { 1 } else { threshold };
        self
    }

    /// First-run per-tick turn cap (clamped to `>= 1`). NOT threaded by the 06
    /// factory — it stays the constructor default in production.
    #[must_use]
    pub const fn backfill_cap(mut self, cap: i64) -> Self {
        self.backfill_cap = if cap < 1 { 1 } else { cap };
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
                    target: "familiar_connect.processors.summary_worker",
                    "{} {}",
                    ls::tag("SummaryWorker", ls::R),
                    ls::kv_styled("tick_error", &format!("{exc:?}"), ls::W, ls::R),
                );
            }
            tokio::select! {
                () = cancel.cancelled() => break,
                () = tokio::time::sleep(self.tick_interval) => {}
            }
        }
    }

    /// Refresh the focus-stream summary.
    pub async fn tick(&self) -> anyhow::Result<()> {
        timed_async(
            "summary.tick",
            async move { self.refresh_focus_stream().await },
        )
        .await
    }

    async fn refresh_focus_stream(&self) -> anyhow::Result<()> {
        let prior = self
            .store
            .get_summary(self.familiar_id.clone(), FOCUS_STREAM_CHANNEL_ID)
            .await?;
        let (after_consumed_at, after_id) = prior.as_ref().map_or_else(
            || (String::new(), 0),
            |p| {
                (
                    p.last_consumed_at.clone().unwrap_or_default(),
                    p.last_summarised_id,
                )
            },
        );

        let new_turns = self
            .store
            .consumed_turns_after(
                self.familiar_id.clone(),
                after_consumed_at,
                after_id,
                self.turns_threshold.max(self.backfill_cap),
            )
            .await?;
        if i64::try_from(new_turns.len()).unwrap_or(i64::MAX) < self.turns_threshold {
            return Ok(());
        }

        let prompt =
            build_rolling_prompt(prior.as_ref().map(|p| p.summary_text.as_str()), &new_turns);
        let reply = self.llm.chat(prompt).await?;
        let text = reply.content_str().trim().to_string();
        if text.is_empty() {
            return Ok(());
        }
        let last = &new_turns[new_turns.len() - 1];
        let last_consumed = last.consumed_at.map(iso_utc);
        self.store
            .put_summary(
                self.familiar_id.clone(),
                last.id,
                text.clone(),
                FOCUS_STREAM_CHANNEL_ID,
                last_consumed,
            )
            .await?;
        tracing::info!(
            target: "familiar_connect.processors.summary_worker",
            "{} {} {} {}",
            ls::tag("Summary", ls::LC),
            ls::kv_styled("focus_stream", &new_turns.len().to_string(), ls::W, ls::LY),
            ls::kv_styled("watermark", &last.id.to_string(), ls::W, ls::LC),
            ls::kv_styled("chars", &text.chars().count().to_string(), ls::W, ls::LW),
        );
        Ok(())
    }
}

/// Compounding prompt: prior summary (when present) plus the new turns.
fn build_rolling_prompt(prior_summary: Option<&str>, new_turns: &[HistoryTurn]) -> Vec<Message> {
    let header = "You produce concise, retrieval-friendly summaries of a familiar's \
        attended conversation across channels. 3-5 sentences. Preserve \
        proper nouns, commitments, and open questions. Omit small talk.";
    let mut body_lines: Vec<String> = Vec::new();
    match prior_summary {
        Some(prior) if !prior.is_empty() => {
            body_lines.push(format!("Previous summary:\n{prior}"));
            body_lines.push("\nNew turns:".to_string());
        }
        _ => body_lines.push("Turns:".to_string()),
    }
    for t in new_turns {
        // Python: `who = t.author.display_name if t.author is not None else
        // t.role`, then f-string-rendered. An author present with a `None`
        // display name renders the literal `"None"` (not empty, not `role`).
        let who = t.author.as_ref().map_or_else(
            || t.role.clone(),
            |a| a.display_name.clone().unwrap_or_else(|| "None".to_string()),
        );
        body_lines.push(format!("- [#{} {who}] {}", t.channel_id, t.content));
    }
    vec![
        Message::new("system", header),
        Message::new("user", body_lines.join("\n")),
    ]
}

#[cfg(test)]
mod tests {
    use super::build_rolling_prompt;
    use crate::history::store::HistoryTurn;
    use crate::identity::Author;
    use chrono::Utc;

    fn turn(role: &str, author: Option<Author>, content: &str) -> HistoryTurn {
        HistoryTurn {
            id: 1,
            timestamp: Utc::now(),
            role: role.to_string(),
            author,
            content: content.to_string(),
            channel_id: 7,
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
    fn who_renders_role_display_name_and_literal_none() {
        let turns = vec![
            turn("assistant", None, "no author"),
            turn(
                "user",
                Some(Author::new("discord", "1", Some("cor".to_string()), None)),
                "author no display name",
            ),
            turn(
                "user",
                Some(Author::new(
                    "discord",
                    "2",
                    Some("kd".to_string()),
                    Some("Cass".to_string()),
                )),
                "author with display name",
            ),
        ];
        let body = build_rolling_prompt(None, &turns)[1].content_str();
        // author=None → role; author present + display_name None → the literal
        // "None" (Python f-string parity); display_name present → the name.
        assert!(body.contains("[#7 assistant] no author"), "{body}");
        assert!(body.contains("[#7 None] author no display name"), "{body}");
        assert!(
            body.contains("[#7 Cass] author with display name"),
            "{body}"
        );
    }
}
