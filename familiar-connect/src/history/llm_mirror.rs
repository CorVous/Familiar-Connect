//! History-backed [`LlmCallSink`] — the `llm_calls` table writer (subsystem 03).
//!
//! Closes the loop between the capture seam in `crate::llm` and the store: one
//! mirrored row per LLM call, carrying the assembled prompt, the message array,
//! the response, tool calls + results, and the same timings/tokens/status the
//! `[LLM call]` log line reports.
//!
//! Two things it deliberately does NOT do:
//!
//! - **Block.** [`mirror`](HistoryLlmMirror::mirror) returns as soon as it has
//!   handed the row to a detached task. Prompts run ~15 KB; a synchronous write
//!   would land on the turn's latency path, which is exactly what the operator
//!   is already fighting.
//! - **Fail.** A store error, a closed DB, or a missing runtime is logged and
//!   swallowed. The conversation outranks the telemetry.
//!
//! Retention rides the same call: [`HistoryStore::append_llm_call`] prunes to
//! the newest `max_rows` for the familiar inside the insert's transaction.

use std::sync::Arc;

use crate::diagnostics::llm_mirror::{LlmCallRecord, LlmCallSink};
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::AppendLlmCall;
use crate::log_style as ls;

/// Writes mirrored calls into one familiar's `llm_calls` table.
pub struct HistoryLlmMirror {
    store: Arc<AsyncHistoryStore>,
    familiar_id: String,
    max_rows: i64,
}

impl HistoryLlmMirror {
    /// `max_rows` caps retained rows per familiar; `<= 0` prunes nothing.
    #[must_use]
    pub fn new(
        store: Arc<AsyncHistoryStore>,
        familiar_id: impl Into<String>,
        max_rows: i64,
    ) -> Self {
        Self {
            store,
            familiar_id: familiar_id.into(),
            max_rows,
        }
    }

    /// The row this sink would write for `record`.
    #[must_use]
    pub fn row(&self, record: LlmCallRecord) -> AppendLlmCall {
        AppendLlmCall {
            familiar_id: self.familiar_id.clone(),
            turn_id: record.turn_id,
            turn_scope: record.turn_scope,
            channel_id: record.channel_id,
            slot: record.slot,
            model: record.model,
            provider: record.provider,
            status: record.status,
            system_prompt: record.system_prompt,
            messages_json: record.messages_json,
            response_text: record.response_text,
            tool_calls_json: record.tool_calls_json,
            tool_results_json: record.tool_results_json,
            ttfb_ms: record.ttfb_ms,
            ttft_ms: record.ttft_ms,
            total_ms: record.total_ms,
            est_in_tokens: record.est_in_tokens,
            in_tokens: record.in_tokens,
            out_tokens: record.out_tokens,
            cached: record.cached,
        }
    }

    /// Write one row inline, logging and swallowing any store fault.
    ///
    /// The awaited half of [`mirror`](Self::mirror); tests drive it directly so
    /// the write is observable without racing a detached task.
    pub async fn write(&self, record: LlmCallRecord) {
        let row = self.row(record);
        if let Err(e) = self.store.append_llm_call(row, self.max_rows).await {
            log_drop(&format!("{e}"));
        }
    }
}

impl LlmCallSink for HistoryLlmMirror {
    fn mirror(&self, record: LlmCallRecord) {
        let row = self.row(record);
        let store = Arc::clone(&self.store);
        let max_rows = self.max_rows;
        // Off the latency path: the turn never awaits this. Outside a runtime
        // (unit tests, teardown) there is nowhere to defer to — drop the row
        // rather than block the caller.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            log_drop("no_runtime");
            return;
        };
        handle.spawn(async move {
            if let Err(e) = store.append_llm_call(row, max_rows).await {
                log_drop(&format!("{e}"));
            }
        });
    }
}

/// One WARN per dropped mirror row; never raises.
fn log_drop(reason: &str) {
    tracing::warn!(
        target: "familiar_connect.history.llm_mirror",
        "{} {}",
        ls::tag("LLM mirror", ls::LY),
        ls::kv_styled("dropped", reason, ls::W, ls::LY),
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::HistoryLlmMirror;
    use crate::diagnostics::llm_mirror::{LlmCallRecord, LlmCallSink};
    use crate::history::async_store::AsyncHistoryStore;
    use crate::history::store::HistoryStore;

    fn store() -> Arc<AsyncHistoryStore> {
        Arc::new(AsyncHistoryStore::new(
            HistoryStore::open(":memory:").expect("store"),
        ))
    }

    fn record() -> LlmCallRecord {
        LlmCallRecord {
            model: "anthropic/claude-haiku-4.5".to_owned(),
            status: "ok".to_owned(),
            system_prompt: "In the call: Ada, Bel".to_owned(),
            messages_json: "[]".to_owned(),
            response_text: "Umu.".to_owned(),
            ..LlmCallRecord::default()
        }
    }

    #[tokio::test]
    async fn write_persists_the_row() {
        let store = store();
        let mirror = HistoryLlmMirror::new(Arc::clone(&store), "aria", 10);
        mirror.write(record()).await;
        let rows = store
            .recent_llm_calls("aria".to_owned(), None, 10)
            .await
            .expect("read");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].system_prompt, "In the call: Ada, Bel");
    }

    #[tokio::test]
    async fn a_closed_store_is_swallowed() {
        let store = store();
        let mirror = HistoryLlmMirror::new(Arc::clone(&store), "aria", 10);
        store.close();
        // No panic, no propagated error: the turn is unaffected.
        mirror.write(record()).await;
        mirror.mirror(record());
    }

    #[test]
    fn mirror_without_a_runtime_drops_rather_than_blocks() {
        let store = store();
        let mirror = HistoryLlmMirror::new(store, "aria", 10);
        mirror.mirror(record());
    }
}
