//! Integration tests for the LLM call mirror (`llm_calls`) — subsystems 01 + 03.
//!
//! Pins the four contracts the mirror exists for: a call round-trips with its
//! prompt/response/metadata intact, a failing mirror write never reaches the
//! turn, retention prunes exactly at the cap, and a row joins to `turns` by
//! `turn_id`. No network: every test drives the store and the sink directly.

use std::sync::Arc;

use familiar_connect::diagnostics::llm_mirror::{
    CallContext, LlmCallRecord, LlmCallSink, current_call_context, mirror_call,
    reset_llm_call_sink, set_llm_call_sink, with_call_context,
};
use familiar_connect::history::async_store::AsyncHistoryStore;
use familiar_connect::history::llm_mirror::HistoryLlmMirror;
use familiar_connect::history::store::{AppendLlmCall, AppendTurn, HistoryStore};

fn store() -> HistoryStore {
    HistoryStore::open(":memory:").expect("store opens")
}

fn call(familiar_id: &str) -> AppendLlmCall {
    AppendLlmCall {
        familiar_id: familiar_id.to_owned(),
        turn_scope: Some("turn-9".to_owned()),
        channel_id: Some(77),
        slot: Some("fast".to_owned()),
        model: "anthropic/claude-haiku-4.5".to_owned(),
        provider: Some("anthropic".to_owned()),
        status: "ok".to_owned(),
        system_prompt: "You are Aria.\n\nIn the call: Ada, Bel".to_owned(),
        messages_json: r#"[{"role":"system","content":"In the call: Ada, Bel"}]"#.to_owned(),
        response_text: "Ada and Bel are here.".to_owned(),
        tool_calls_json: Some(r#"[{"id":"c1","function":{"name":"read_channel"}}]"#.to_owned()),
        tool_results_json: Some(r#"[{"tool_call_id":"c1","content":"[]"}]"#.to_owned()),
        ttfb_ms: Some(410),
        ttft_ms: Some(455),
        total_ms: Some(910),
        est_in_tokens: Some(2000),
        in_tokens: Some(2100),
        out_tokens: Some(80),
        cached: Some(1890),
        ..AppendLlmCall::default()
    }
}

// --- capture fidelity ------------------------------------------------------

#[test]
fn a_call_round_trips_with_prompt_response_and_metadata() {
    let store = store();
    store.append_llm_call(call("aria"), 100).expect("append");
    let rows = store.recent_llm_calls("aria", None, 10).expect("read back");
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.system_prompt, "You are Aria.\n\nIn the call: Ada, Bel");
    assert_eq!(
        row.messages_json,
        r#"[{"role":"system","content":"In the call: Ada, Bel"}]"#
    );
    assert_eq!(row.response_text, "Ada and Bel are here.");
    assert!(
        row.tool_calls_json
            .as_deref()
            .expect("tool calls")
            .contains("read_channel")
    );
    assert!(
        row.tool_results_json
            .as_deref()
            .expect("tool results")
            .contains("c1")
    );
    assert_eq!(row.slot.as_deref(), Some("fast"));
    assert_eq!(row.provider.as_deref(), Some("anthropic"));
    assert_eq!(row.status, "ok");
    assert_eq!(row.ttfb_ms, Some(410));
    assert_eq!(row.ttft_ms, Some(455));
    assert_eq!(row.total_ms, Some(910));
    assert_eq!(row.est_in_tokens, Some(2000));
    assert_eq!(row.in_tokens, Some(2100));
    assert_eq!(row.out_tokens, Some(80));
    assert_eq!(row.cached, Some(1890));
    assert_eq!(row.channel_id, Some(77));
    assert_eq!(row.turn_scope.as_deref(), Some("turn-9"));
}

#[test]
fn status_vocabulary_is_open() {
    let store = store();
    for status in ["ok", "error", "cancelled", "silent", "suppressed", "weird"] {
        let mut row = call("aria");
        row.status = status.to_owned();
        store.append_llm_call(row, 100).expect("append");
    }
    let rows = store.recent_llm_calls("aria", None, 10).expect("read");
    let seen: Vec<&str> = rows.iter().map(|r| r.status.as_str()).collect();
    assert!(seen.contains(&"suppressed"));
    assert!(seen.contains(&"weird"));
}

#[test]
fn created_at_is_lexicographically_ordered() {
    // `iso_utc` fixed-width stamps: string order == chronological order.
    let store = store();
    for _ in 0..3 {
        store.append_llm_call(call("aria"), 100).expect("append");
    }
    let rows = store.recent_llm_calls("aria", None, 10).expect("read");
    let stamps: Vec<String> = rows
        .iter()
        .map(|r| familiar_connect::support::time::iso_utc(r.created_at))
        .collect();
    let mut sorted = stamps.clone();
    sorted.sort();
    sorted.reverse();
    // Newest first from the query; the stamps agree.
    assert_eq!(stamps, sorted);
}

// --- joins to turns --------------------------------------------------------

#[test]
fn a_row_joins_to_turns_by_turn_id() {
    let store = store();
    let turn = store
        .append_turn(AppendTurn::new("aria", 77, "user", "who is in the call?"))
        .expect("turn");
    let mut anchored = call("aria");
    anchored.turn_id = Some(turn.id);
    store.append_llm_call(anchored, 100).expect("append");
    // An unanchored call from the same familiar must not answer the turn query.
    store.append_llm_call(call("aria"), 100).expect("append");

    let rows = store.llm_calls_for_turn("aria", turn.id).expect("join");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].turn_id, Some(turn.id));
    // The join carries the turn's own text through.
    assert_eq!(rows[0].turn_content.as_deref(), Some("who is in the call?"));
}

#[test]
fn a_turn_id_naming_no_turn_yields_no_join() {
    let store = store();
    let mut dangling = call("aria");
    dangling.turn_id = Some(9999);
    store.append_llm_call(dangling, 100).expect("append");
    assert!(
        store
            .llm_calls_for_turn("aria", 9999)
            .expect("join")
            .is_empty()
    );
}

#[test]
fn the_join_is_scoped_to_one_familiar() {
    let store = store();
    let turn = store
        .append_turn(AppendTurn::new("aria", 77, "user", "hi"))
        .expect("turn");
    let mut other = call("bel");
    other.turn_id = Some(turn.id);
    store.append_llm_call(other, 100).expect("append");
    assert!(
        store
            .llm_calls_for_turn("bel", turn.id)
            .expect("join")
            .is_empty()
    );
}

// --- retention -------------------------------------------------------------

#[test]
fn retention_prunes_exactly_at_the_boundary() {
    let store = store();
    let cap = 3;
    for _ in 0..cap {
        store.append_llm_call(call("aria"), cap).expect("append");
    }
    assert_eq!(store.count_llm_calls("aria").expect("count"), cap);
    let oldest = store.recent_llm_calls("aria", None, 10).expect("read");
    let oldest_id = oldest.last().expect("row").id;

    // One past the cap evicts exactly one row — the oldest.
    store.append_llm_call(call("aria"), cap).expect("append");
    assert_eq!(store.count_llm_calls("aria").expect("count"), cap);
    let kept = store.recent_llm_calls("aria", None, 10).expect("read");
    assert!(kept.iter().all(|r| r.id != oldest_id));
}

#[test]
fn retention_is_scoped_per_familiar() {
    let store = store();
    store.append_llm_call(call("bel"), 1).expect("append");
    store.append_llm_call(call("aria"), 1).expect("append");
    store.append_llm_call(call("aria"), 1).expect("append");
    assert_eq!(store.count_llm_calls("aria").expect("count"), 1);
    assert_eq!(store.count_llm_calls("bel").expect("count"), 1);
}

#[test]
fn a_zero_cap_prunes_nothing() {
    // The `0` off-switch is handled at wiring time; the store treats it as
    // "no pruning" so a directly-driven write is never silently discarded.
    let store = store();
    for _ in 0..4 {
        store.append_llm_call(call("aria"), 0).expect("append");
    }
    assert_eq!(store.count_llm_calls("aria").expect("count"), 4);
}

// --- never fails or blocks a turn ------------------------------------------

/// A sink that always faults, standing in for a broken DB.
struct FaultingSink;

impl LlmCallSink for FaultingSink {
    fn mirror(&self, _record: LlmCallRecord) {
        panic!("mirror write exploded");
    }
}

/// Stands in for a turn: mirrors mid-flight, then finishes and returns a reply.
/// If mirroring could raise, this never returns.
fn turn_that_mirrors() -> &'static str {
    mirror_call(LlmCallRecord::default());
    "reply"
}

/// One test, not three: the sink is a process-wide singleton and these arms
/// would otherwise race each other inside the same test binary.
#[tokio::test]
async fn a_failing_mirror_write_never_reaches_the_turn() {
    // No sink at all.
    reset_llm_call_sink();
    assert_eq!(turn_that_mirrors(), "reply");

    // A sink that panics.
    set_llm_call_sink(Arc::new(FaultingSink));
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let through_a_panicking_sink = turn_that_mirrors();
    std::panic::set_hook(prev);
    assert_eq!(through_a_panicking_sink, "reply");

    // A sink whose store is closed underneath it.
    let store = Arc::new(AsyncHistoryStore::new(store()));
    let mirror = HistoryLlmMirror::new(Arc::clone(&store), "aria", 10);
    store.close();
    set_llm_call_sink(Arc::new(mirror));
    assert_eq!(turn_that_mirrors(), "reply");
    reset_llm_call_sink();

    // And nothing was written, which is the point: the row is expendable.
    assert!(
        store.sync().count_llm_calls("aria").is_err(),
        "a closed store answers nothing"
    );
}

// --- turn context ----------------------------------------------------------

#[tokio::test]
async fn the_scoped_turn_context_names_the_row() {
    let store = Arc::new(AsyncHistoryStore::new(store()));
    let turn = store
        .append_turn(AppendTurn::new("aria", 77, "user", "who is here?"))
        .await
        .expect("turn");
    let mirror = HistoryLlmMirror::new(Arc::clone(&store), "aria", 10);

    with_call_context(CallContext::new(Some(turn.id), "turn-9", 77), async {
        let ctx = current_call_context();
        mirror
            .write(LlmCallRecord {
                turn_id: ctx.turn_id,
                turn_scope: ctx.turn_scope,
                channel_id: ctx.channel_id,
                model: "anthropic/claude-haiku-4.5".to_owned(),
                status: "ok".to_owned(),
                system_prompt: "In the call: Ada, Bel".to_owned(),
                messages_json: "[]".to_owned(),
                response_text: "Umu.".to_owned(),
                ..LlmCallRecord::default()
            })
            .await;
    })
    .await;

    let rows = store
        .llm_calls_for_turn("aria".to_owned(), turn.id)
        .await
        .expect("join");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].turn_scope.as_deref(), Some("turn-9"));
    assert_eq!(rows[0].channel_id, Some(77));
    assert_eq!(rows[0].system_prompt, "In the call: Ada, Bel");
}
