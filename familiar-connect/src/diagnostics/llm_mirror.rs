//! LLM call/response mirror seam (subsystem 01).
//!
//! `CallMetrics::emit` (`crate::llm`) writes one `[LLM call]` INFO line per
//! request, truncated and ephemeral. This module carries the *whole* call —
//! assembled system prompt, message array, response, tool calls and results,
//! alongside the same timings/tokens/status — to a pluggable sink so it can be
//! mirrored durably (subsystem 03 provides the history-backed sink).
//!
//! Layer 0: this module knows nothing about storage. It owns the record shape,
//! the [`LlmCallSink`] trait, the process-wide sink slot, and the per-turn
//! [`CallContext`] the responders scope around a turn so a mirrored row can name
//! its turn and channel.
//!
//! Two invariants, both load-bearing:
//!
//! - **Mirroring never fails a turn.** [`mirror_call`] swallows a missing sink,
//!   a poisoned lock, and any sink fault. Conversation beats telemetry.
//! - **Mirroring is never on the latency path.** A sink must return promptly;
//!   the history sink defers the write to a detached task.

use std::sync::{Arc, Mutex, PoisonError};

/// One complete LLM request/response pair.
///
/// Every field is filled from the same call at the same seam — nothing here is
/// correlated after the fact. Optional fields stay `None` when the call never
/// produced them (a failed call reports no usage; a background worker has no
/// turn).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LlmCallRecord {
    /// `turns.id` of the turn this call served; `None` off the turn path.
    pub turn_id: Option<i64>,
    /// Bus turn-scope id — the `turn=` value in log lines.
    pub turn_scope: Option<String>,
    pub channel_id: Option<i64>,
    /// `fast` / `prose` / `background`; `None` when the client named no slot.
    pub slot: Option<String>,
    pub model: String,
    pub provider: Option<String>,
    /// Open vocabulary (`ok` / `error` / `cancelled` / `silent` / …).
    pub status: String,
    /// Every system-role message, joined — the assembled prompt, greppable.
    pub system_prompt: String,
    /// The full message array as sent, JSON.
    pub messages_json: String,
    /// Assistant text, concatenated across deltas.
    pub response_text: String,
    /// Tool calls the model requested, JSON array; `None` when it made none.
    pub tool_calls_json: Option<String>,
    /// Results of those calls, JSON array; attached by the agentic loop.
    pub tool_results_json: Option<String>,
    pub ttfb_ms: Option<i64>,
    pub ttft_ms: Option<i64>,
    pub total_ms: Option<i64>,
    pub est_in_tokens: Option<i64>,
    pub in_tokens: Option<i64>,
    pub out_tokens: Option<i64>,
    pub cached: Option<i64>,
}

/// Where mirrored calls go.
///
/// Implementations must return promptly and must not propagate faults — the
/// call site is a turn in flight.
pub trait LlmCallSink: Send + Sync {
    /// Accept one record. Fire-and-forget.
    fn mirror(&self, record: LlmCallRecord);
}

/// Turn identity for calls made inside a scoped future.
///
/// Set by the responders (subsystem 06) around a turn's LLM work; read at
/// request-construction time so a stream polled or dropped elsewhere still
/// carries the right turn.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallContext {
    /// `turns.id` of the anchoring turn.
    pub turn_id: Option<i64>,
    /// Bus turn-scope id.
    pub turn_scope: Option<String>,
    pub channel_id: Option<i64>,
}

impl CallContext {
    /// Context naming a turn row, its bus scope, and its channel.
    #[must_use]
    pub fn new(turn_id: Option<i64>, turn_scope: impl Into<String>, channel_id: i64) -> Self {
        Self {
            turn_id,
            turn_scope: Some(turn_scope.into()),
            channel_id: Some(channel_id),
        }
    }
}

tokio::task_local! {
    static CALL_CONTEXT: CallContext;
}

/// Run `fut` with `ctx` visible to every LLM call it makes.
pub async fn with_call_context<F: Future>(ctx: CallContext, fut: F) -> F::Output {
    CALL_CONTEXT.scope(ctx, fut).await
}

/// The enclosing [`CallContext`], or an empty one outside any scope.
#[must_use]
pub fn current_call_context() -> CallContext {
    CALL_CONTEXT.try_with(Clone::clone).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// process-wide sink — mirrors get_span_collector
// ---------------------------------------------------------------------------

static SINK: Mutex<Option<Arc<dyn LlmCallSink>>> = Mutex::new(None);

/// Install the process-wide sink, replacing any previous one.
pub fn set_llm_call_sink(sink: Arc<dyn LlmCallSink>) {
    *SINK.lock().unwrap_or_else(PoisonError::into_inner) = Some(sink);
}

/// The installed sink, or `None` when mirroring is off.
#[must_use]
pub fn get_llm_call_sink() -> Option<Arc<dyn LlmCallSink>> {
    SINK.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// Clear the sink — tests, and the `0` (disabled) retention setting.
pub fn reset_llm_call_sink() {
    *SINK.lock().unwrap_or_else(PoisonError::into_inner) = None;
}

/// Hand one record to the installed sink.
///
/// No sink, a poisoned slot, or a panicking sink all resolve to "nothing was
/// mirrored" — never to a failed turn.
pub fn mirror_call(record: LlmCallRecord) {
    let Some(sink) = get_llm_call_sink() else {
        return;
    };
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink.mirror(record))).is_err() {
        tracing::warn!(
            target: "familiar_connect.diagnostics.llm_mirror",
            "{}",
            crate::log_style::kv("mirror_sink_panicked", "true"),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{
        CallContext, LlmCallRecord, LlmCallSink, current_call_context, get_llm_call_sink,
        mirror_call, reset_llm_call_sink, set_llm_call_sink, with_call_context,
    };
    use crate::diagnostics::testutil::singleton_guard;

    #[derive(Default)]
    struct Collecting(Mutex<Vec<LlmCallRecord>>);

    impl LlmCallSink for Collecting {
        fn mirror(&self, record: LlmCallRecord) {
            self.0.lock().expect("sink mutex").push(record);
        }
    }

    struct Exploding;

    impl LlmCallSink for Exploding {
        fn mirror(&self, _record: LlmCallRecord) {
            panic!("sink is broken");
        }
    }

    fn record(model: &str) -> LlmCallRecord {
        LlmCallRecord {
            model: model.to_owned(),
            status: "ok".to_owned(),
            ..LlmCallRecord::default()
        }
    }

    #[test]
    fn absent_sink_drops_the_record() {
        let _guard = singleton_guard();
        reset_llm_call_sink();
        assert!(get_llm_call_sink().is_none());
        mirror_call(record("a/b"));
    }

    #[test]
    fn installed_sink_receives_the_record() {
        let _guard = singleton_guard();
        reset_llm_call_sink();
        let sink = Arc::new(Collecting::default());
        set_llm_call_sink(Arc::clone(&sink) as Arc<dyn LlmCallSink>);
        mirror_call(record("a/b"));
        let got = {
            let guard = sink.0.lock().expect("sink mutex");
            guard.clone()
        };
        reset_llm_call_sink();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].model, "a/b");
    }

    #[test]
    fn a_panicking_sink_is_swallowed() {
        let _guard = singleton_guard();
        reset_llm_call_sink();
        set_llm_call_sink(Arc::new(Exploding));
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        mirror_call(record("a/b"));
        std::panic::set_hook(prev);
        reset_llm_call_sink();
    }

    #[tokio::test]
    async fn call_context_is_empty_outside_a_scope() {
        assert_eq!(current_call_context(), CallContext::default());
    }

    #[tokio::test]
    async fn call_context_reaches_nested_awaits() {
        let ctx = CallContext::new(Some(7), "turn-1", 42);
        let seen = with_call_context(ctx.clone(), async {
            tokio::task::yield_now().await;
            current_call_context()
        })
        .await;
        assert_eq!(seen, ctx);
        assert_eq!(seen.turn_id, Some(7));
        assert_eq!(seen.channel_id, Some(42));
    }
}
