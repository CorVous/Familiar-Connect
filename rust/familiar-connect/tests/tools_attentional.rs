//! Ported from Python `tests/test_attentional_tools.py` — the `shift_focus`,
//! `read_channel`, and `start_activity` handlers, driven through the narrow
//! `FocusControl` / `ChannelReadStore` / `StartActivityEngine` seams with
//! scripted doubles (replacing the Python `MagicMock`s).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{NaiveTime, TimeZone, Utc};
use serde_json::{Value, json};

use familiar_connect::history::StoreError;
use familiar_connect::history::store::HistoryTurn;
use familiar_connect::tools::read_channel::read_channel_handler;
use familiar_connect::tools::registry::{ChannelReadStore, FocusControl, ToolContext, ToolOutput};
use familiar_connect::tools::shift_focus::shift_focus_handler;
use familiar_connect::tools::silent::SILENT_RESULT;
use familiar_connect::tools::start_activity::{
    ActivityCatalogEntry, StartActivityEngine, build_start_activity_tool,
};

// ---------------------------------------------------------------------------
// Doubles
// ---------------------------------------------------------------------------

struct MockFocus {
    shift_calls: Mutex<Vec<i64>>,
    focus_text: Option<i64>,
    subscribed: Vec<i64>,
    is_subscribed: bool,
    catch_up: usize,
}

impl MockFocus {
    fn new() -> Self {
        Self {
            shift_calls: Mutex::new(Vec::new()),
            focus_text: Some(99),
            subscribed: vec![55, 99],
            is_subscribed: true,
            catch_up: 20,
        }
    }
}

#[async_trait]
impl FocusControl for MockFocus {
    fn is_subscribed(&self, _channel_id: i64) -> bool {
        self.is_subscribed
    }
    fn subscribed_channels(&self) -> Vec<i64> {
        self.subscribed.clone()
    }
    fn channel_label(&self, channel_id: i64) -> String {
        format!("#{channel_id}")
    }
    fn get_focus(&self, _modality: &str) -> Option<i64> {
        self.focus_text
    }
    fn catch_up_limit(&self) -> usize {
        self.catch_up
    }
    async fn shift_now(&self, channel_id: i64) {
        self.shift_calls.lock().unwrap().push(channel_id);
    }
}

#[derive(Clone)]
struct RecentCall {
    channel_id: i64,
    limit: i64,
    before_id: Option<i64>,
}

#[derive(Clone)]
struct AroundCall {
    channel_id: i64,
    turn_id: i64,
    before: i64,
    after: i64,
}

struct MockStore {
    recent_return: Vec<HistoryTurn>,
    around_return: Vec<HistoryTurn>,
    recent_calls: Mutex<Vec<RecentCall>>,
    around_calls: Mutex<Vec<AroundCall>>,
}

impl MockStore {
    const fn new() -> Self {
        Self {
            recent_return: Vec::new(),
            around_return: Vec::new(),
            recent_calls: Mutex::new(Vec::new()),
            around_calls: Mutex::new(Vec::new()),
        }
    }
    fn with_recent(mut self, turns: Vec<HistoryTurn>) -> Self {
        self.recent_return = turns;
        self
    }
    fn with_around(mut self, turns: Vec<HistoryTurn>) -> Self {
        self.around_return = turns;
        self
    }
}

#[async_trait]
impl ChannelReadStore for MockStore {
    async fn recent(
        &self,
        _familiar_id: &str,
        channel_id: i64,
        limit: i64,
        before_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.recent_calls.lock().unwrap().push(RecentCall {
            channel_id,
            limit,
            before_id,
        });
        Ok(self.recent_return.clone())
    }
    async fn turns_around(
        &self,
        _familiar_id: &str,
        channel_id: i64,
        turn_id: i64,
        before: i64,
        after: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.around_calls.lock().unwrap().push(AroundCall {
            channel_id,
            turn_id,
            before,
            after,
        });
        Ok(self.around_return.clone())
    }
}

fn history_turn(id: i64, role: &str, content: &str, channel_id: i64) -> HistoryTurn {
    HistoryTurn {
        id,
        timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
            + chrono::Duration::seconds(id),
        role: role.to_owned(),
        author: None,
        content: content.to_owned(),
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

fn text(out: ToolOutput) -> String {
    match out {
        ToolOutput::Text(s) => s,
        ToolOutput::Image(_) => panic!("expected text output"),
    }
}

fn parse(out: ToolOutput) -> Value {
    serde_json::from_str(&text(out)).unwrap()
}

fn ctx_with(
    fm: Option<Arc<dyn FocusControl>>,
    store: Option<Arc<dyn ChannelReadStore>>,
) -> ToolContext {
    let mut ctx = ToolContext::new("fam-1", 42, "text", "turn-1");
    if let Some(fm) = fm {
        ctx = ctx.with_focus_manager(fm);
    }
    if let Some(store) = store {
        ctx = ctx.with_store(store);
    }
    ctx
}

// ---------------------------------------------------------------------------
// shift_focus
// ---------------------------------------------------------------------------

#[tokio::test]
async fn shift_focus_calls_shift_now() {
    let fm = Arc::new(MockFocus::new());
    let ctx = ctx_with(Some(fm.clone()), None);
    shift_focus_handler(&json!({"channel_id": 55}), &ctx)
        .await
        .unwrap();
    assert_eq!(*fm.shift_calls.lock().unwrap(), vec![55]);
}

#[tokio::test]
async fn shift_focus_returns_ok_json_without_store() {
    let fm = Arc::new(MockFocus::new());
    let ctx = ctx_with(Some(fm), None);
    let out = shift_focus_handler(&json!({"channel_id": 55}), &ctx)
        .await
        .unwrap();
    assert_eq!(parse(out), json!({"ok": true, "channel_id": 55}));
}

#[tokio::test]
async fn shift_focus_returns_messages_from_target_channel() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new().with_recent(vec![
        history_turn(1, "user", "first", 55),
        history_turn(2, "user", "second", 55),
    ]));
    let ctx = ctx_with(Some(fm), Some(store));
    let parsed = parse(
        shift_focus_handler(&json!({"channel_id": 55}), &ctx)
            .await
            .unwrap(),
    );
    assert_eq!(parsed["ok"], true);
    assert_eq!(parsed["channel_id"], 55);
    let contents: Vec<&str> = parsed["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["content"].as_str().unwrap())
        .collect();
    assert_eq!(contents, ["first", "second"]);
}

#[tokio::test]
async fn shift_focus_fetches_target_channel_not_current_focus() {
    let fm = Arc::new(MockFocus::new()); // current focus is 99
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    shift_focus_handler(&json!({"channel_id": 55}), &ctx)
        .await
        .unwrap();
    assert_eq!(store.recent_calls.lock().unwrap()[0].channel_id, 55);
}

#[tokio::test]
async fn shift_focus_preview_limit_matches_catch_up_window() {
    let mut fm = MockFocus::new();
    fm.catch_up = 7;
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(Arc::new(fm)), Some(store.clone()));
    shift_focus_handler(&json!({"channel_id": 55}), &ctx)
        .await
        .unwrap();
    assert_eq!(store.recent_calls.lock().unwrap()[0].limit, 7);
}

#[tokio::test]
async fn shift_focus_empty_channel_returns_empty_messages() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store));
    let parsed = parse(
        shift_focus_handler(&json!({"channel_id": 55}), &ctx)
            .await
            .unwrap(),
    );
    assert_eq!(parsed["messages"], json!([]));
}

#[tokio::test]
async fn shift_focus_error_when_no_focus_manager() {
    let ctx = ctx_with(None, None);
    let parsed = parse(
        shift_focus_handler(&json!({"channel_id": 55}), &ctx)
            .await
            .unwrap(),
    );
    assert!(parsed.get("error").is_some());
}

#[tokio::test]
async fn shift_focus_error_when_missing_channel_id() {
    let fm = Arc::new(MockFocus::new());
    let ctx = ctx_with(Some(fm), None);
    let parsed = parse(shift_focus_handler(&json!({}), &ctx).await.unwrap());
    assert!(parsed.get("error").is_some());
}

#[tokio::test]
async fn shift_focus_rejects_unsubscribed_channel() {
    let mut fm = MockFocus::new();
    fm.is_subscribed = false;
    let fm = Arc::new(fm);
    let ctx = ctx_with(Some(fm.clone()), None);
    let parsed = parse(
        shift_focus_handler(&json!({"channel_id": 12345}), &ctx)
            .await
            .unwrap(),
    );
    assert!(parsed.get("error").is_some());
    assert!(fm.shift_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn shift_focus_unsubscribed_error_lists_available_channels() {
    let mut fm = MockFocus::new();
    fm.is_subscribed = false;
    fm.subscribed = vec![55, 99];
    let ctx = ctx_with(Some(Arc::new(fm)), None);
    let parsed = parse(
        shift_focus_handler(&json!({"channel_id": 12345}), &ctx)
            .await
            .unwrap(),
    );
    let ids: Vec<i64> = parsed["available_channels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["channel_id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, [55, 99]);
}

// ---------------------------------------------------------------------------
// read_channel
// ---------------------------------------------------------------------------

#[tokio::test]
async fn read_channel_returns_recent_turns_as_json() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new().with_recent(vec![
        history_turn(1, "user", "hello", 99),
        history_turn(2, "assistant", "hi there", 99),
    ]));
    let ctx = ctx_with(Some(fm), Some(store));
    let parsed = parse(
        read_channel_handler(&json!({"limit": 20}), &ctx)
            .await
            .unwrap(),
    );
    let arr = parsed.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["role"], "user");
    assert_eq!(arr[0]["content"], "hello");
    assert_eq!(arr[1]["role"], "assistant");
}

#[tokio::test]
async fn read_channel_passes_limit_to_store() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    read_channel_handler(&json!({"limit": 10}), &ctx)
        .await
        .unwrap();
    assert_eq!(store.recent_calls.lock().unwrap()[0].limit, 10);
}

#[tokio::test]
async fn read_channel_clamps_limit_to_50() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    read_channel_handler(&json!({"limit": 100}), &ctx)
        .await
        .unwrap();
    assert_eq!(store.recent_calls.lock().unwrap()[0].limit, 50);
}

#[tokio::test]
async fn read_channel_error_when_no_store() {
    let fm = Arc::new(MockFocus::new());
    let ctx = ctx_with(Some(fm), None);
    let parsed = parse(read_channel_handler(&json!({}), &ctx).await.unwrap());
    assert!(parsed.get("error").is_some());
}

#[tokio::test]
async fn read_channel_error_when_no_focus_manager() {
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(None, Some(store));
    let parsed = parse(read_channel_handler(&json!({}), &ctx).await.unwrap());
    assert!(parsed.get("error").is_some());
}

#[tokio::test]
async fn read_channel_error_when_no_text_focus() {
    let mut fm = MockFocus::new();
    fm.focus_text = None;
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(Arc::new(fm)), Some(store));
    let parsed = parse(read_channel_handler(&json!({}), &ctx).await.unwrap());
    assert!(parsed.get("error").is_some());
}

#[tokio::test]
async fn read_channel_passes_before_id_to_store() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    read_channel_handler(&json!({"limit": 10, "before_id": 7}), &ctx)
        .await
        .unwrap();
    assert_eq!(store.recent_calls.lock().unwrap()[0].before_id, Some(7));
}

#[tokio::test]
async fn read_channel_before_id_defaults_none() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    read_channel_handler(&json!({}), &ctx).await.unwrap();
    assert_eq!(store.recent_calls.lock().unwrap()[0].before_id, None);
}

#[tokio::test]
async fn read_channel_around_id_calls_turns_around() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new().with_around(vec![
        history_turn(9, "user", "before", 99),
        history_turn(10, "assistant", "anchor", 99),
        history_turn(11, "user", "after", 99),
    ]));
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    let parsed = parse(
        read_channel_handler(&json!({"around_id": 10, "limit": 20}), &ctx)
            .await
            .unwrap(),
    );
    let call = store.around_calls.lock().unwrap()[0].clone();
    assert_eq!(
        (call.channel_id, call.turn_id, call.before, call.after),
        (99, 10, 10, 10)
    );
    assert!(store.recent_calls.lock().unwrap().is_empty());
    let ids: Vec<i64> = parsed
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["id"].as_i64().unwrap())
        .collect();
    assert_eq!(ids, [9, 10, 11]);
    assert_eq!(parsed[1]["content"], "anchor");
}

#[tokio::test]
async fn read_channel_around_id_split_respects_max_limit() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    read_channel_handler(&json!({"around_id": 10, "limit": 100}), &ctx)
        .await
        .unwrap();
    let call = store.around_calls.lock().unwrap()[0].clone();
    assert_eq!((call.before, call.after), (25, 25));
}

#[tokio::test]
async fn read_channel_around_id_small_limit_keeps_anchor_window() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    read_channel_handler(&json!({"around_id": 10, "limit": 1}), &ctx)
        .await
        .unwrap();
    let call = store.around_calls.lock().unwrap()[0].clone();
    assert!(call.before >= 1 && call.after >= 1);
}

#[tokio::test]
async fn read_channel_before_id_and_around_id_mutually_exclusive() {
    let fm = Arc::new(MockFocus::new());
    let store = Arc::new(MockStore::new());
    let ctx = ctx_with(Some(fm), Some(store.clone()));
    let parsed = parse(
        read_channel_handler(&json!({"before_id": 5, "around_id": 10}), &ctx)
            .await
            .unwrap(),
    );
    assert!(parsed.get("error").is_some());
    assert!(store.recent_calls.lock().unwrap().is_empty());
    assert!(store.around_calls.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// start_activity
// ---------------------------------------------------------------------------

struct FakeEngine {
    catalog: Vec<ActivityCatalogEntry>,
    calls: Mutex<Vec<(String, Option<String>)>>,
    active: bool,
    result: Value,
}

impl FakeEngine {
    fn new() -> Self {
        Self {
            catalog: vec![
                ActivityCatalogEntry {
                    id: "creek_walk".into(),
                    label: "a creek walk".into(),
                    active_days: None,
                    active_hours: None,
                },
                ActivityCatalogEntry {
                    id: "hatbox".into(),
                    label: "tending the hatbox".into(),
                    active_days: None,
                    active_hours: None,
                },
            ],
            calls: Mutex::new(Vec::new()),
            active: false,
            result: json!({"ack": "ok", "label": "a creek walk", "duration_minutes": 30}),
        }
    }
}

impl StartActivityEngine for FakeEngine {
    fn catalog(&self) -> Vec<ActivityCatalogEntry> {
        self.catalog.clone()
    }
    fn is_active(&self) -> bool {
        self.active
    }
    fn defer_start(&self, type_id: &str, note: Option<&str>) -> Value {
        self.calls
            .lock()
            .unwrap()
            .push((type_id.to_owned(), note.map(ToOwned::to_owned)));
        self.result.clone()
    }
}

async fn call_start_activity(engine: Arc<FakeEngine>, args: Value) -> ToolOutput {
    let tool = build_start_activity_tool(engine);
    let ctx = ToolContext::new("fam-1", 42, "text", "turn-1");
    tool.handler.call(args, &ctx).await.unwrap()
}

#[test]
fn start_activity_tool_name_and_schema() {
    let tool = build_start_activity_tool(Arc::new(FakeEngine::new()));
    assert_eq!(tool.name, "start_activity");
    let props = &tool.parameters["properties"];
    assert_eq!(props["activity"]["type"], "string");
    assert_eq!(props["activity"]["enum"], json!(["creek_walk", "hatbox"]));
    assert_eq!(props["note"]["type"], "string");
    assert_eq!(tool.parameters["required"], json!(["activity"]));
    let desc = props["activity"]["description"].as_str().unwrap();
    assert!(desc.contains("a creek walk"));
    assert!(desc.contains("tending the hatbox"));
    assert!(tool.description.len() <= 450);
    assert!(tool.description.contains("in-character goodbye"));
}

#[tokio::test]
async fn start_activity_handler_calls_defer_start_with_note() {
    let engine = Arc::new(FakeEngine::new());
    let out = call_start_activity(
        engine.clone(),
        json!({"activity": "creek_walk", "note": "want to see the herons"}),
    )
    .await;
    assert_eq!(
        *engine.calls.lock().unwrap(),
        vec![(
            "creek_walk".to_owned(),
            Some("want to see the herons".to_owned())
        )]
    );
    assert_eq!(
        parse(out),
        json!({"ack": "ok", "label": "a creek walk", "duration_minutes": 30})
    );
}

#[tokio::test]
async fn start_activity_handler_note_defaults_none() {
    let engine = Arc::new(FakeEngine::new());
    call_start_activity(engine.clone(), json!({"activity": "hatbox"})).await;
    assert_eq!(
        *engine.calls.lock().unwrap(),
        vec![("hatbox".to_owned(), None)]
    );
}

#[tokio::test]
async fn start_activity_handler_passes_engine_error_through() {
    let mut engine = FakeEngine::new();
    engine.result = json!({"error": "already out"});
    let out = call_start_activity(Arc::new(engine), json!({"activity": "hatbox"})).await;
    assert_eq!(parse(out), json!({"error": "already out"}));
}

#[tokio::test]
async fn start_activity_handler_missing_activity_returns_error() {
    let engine = Arc::new(FakeEngine::new());
    let out = call_start_activity(engine.clone(), json!({})).await;
    assert!(parse(out).get("error").is_some());
    assert!(engine.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn start_activity_handler_non_string_note_returns_error() {
    let engine = Arc::new(FakeEngine::new());
    let out = call_start_activity(engine.clone(), json!({"activity": "hatbox", "note": 5})).await;
    assert!(parse(out).get("error").is_some());
    assert!(engine.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn start_activity_already_out_is_silent() {
    let mut engine = FakeEngine::new();
    engine.active = true;
    let engine = Arc::new(engine);
    let out = call_start_activity(engine.clone(), json!({"activity": "creek_walk"})).await;
    assert_eq!(out, ToolOutput::Text(SILENT_RESULT.to_owned()));
    assert!(engine.calls.lock().unwrap().is_empty());
}

#[test]
fn start_activity_scheduled_entry_appends_availability_window() {
    let mut engine = FakeEngine::new();
    engine.catalog = vec![
        ActivityCatalogEntry {
            id: "weekday_rounds".into(),
            label: "weekday rounds".into(),
            active_days: Some(vec![0, 1, 2, 3, 4]),
            active_hours: Some((
                NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(17, 0, 0).unwrap(),
            )),
        },
        ActivityCatalogEntry {
            id: "creek_walk".into(),
            label: "a creek walk".into(),
            active_days: None,
            active_hours: None,
        },
    ];
    let tool = build_start_activity_tool(Arc::new(engine));
    let desc = tool.parameters["properties"]["activity"]["description"]
        .as_str()
        .unwrap()
        .to_owned();
    let weekday_seg = desc
        .split("; ")
        .find(|s| s.contains("weekday_rounds"))
        .unwrap();
    let creek_seg = desc.split("; ").find(|s| s.contains("creek_walk")).unwrap();
    assert!(weekday_seg.contains("Mon Tue Wed Thu Fri"));
    assert!(!weekday_seg.contains("Sun"));
    assert!(weekday_seg.contains("09:00-17:00"));
    assert!(!creek_seg.contains('['));
    assert!(!creek_seg.contains(':'));
    assert_eq!(
        tool.parameters["properties"]["activity"]["enum"],
        json!(["weekday_rounds", "creek_walk"])
    );
    assert!(desc.contains("'weekday_rounds' = weekday rounds"));
    assert!(desc.contains("'creek_walk' = a creek walk"));
}
