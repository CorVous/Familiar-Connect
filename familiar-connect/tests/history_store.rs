//! Integration tests for `history::store` — turns, reads, summaries, watermarks,
//! attentional stream, activities. Ports `test_history_store.py` and
//! `test_attentional_store.py` (FTS-independent parts).

use chrono::{TimeZone, Utc};
use familiar_connect::history::db::Value;
use familiar_connect::history::{AppendTurn, Author, ChannelUnread, HistoryStore, HistoryTurn};
use familiar_connect::support::time::iso_utc;

const CHANNEL: i64 = 200;
const FAMILIAR: &str = "aria";

fn mem() -> HistoryStore {
    HistoryStore::open(":memory:").unwrap()
}

fn author(user_id: &str, username: &str, display: &str) -> Author {
    Author::new(
        "discord",
        user_id,
        Some(username.to_owned()),
        Some(display.to_owned()),
    )
}

fn alice() -> Author {
    author("1", "alice", "Alice")
}

fn seed(store: &HistoryStore, n: usize) -> Vec<HistoryTurn> {
    let mut out = Vec::new();
    for i in 0..n {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        let mut p = AppendTurn::new(FAMILIAR, CHANNEL, role, format!("turn {i}"));
        if role == "user" {
            p = p.author(alice());
        }
        out.push(store.append_turn(p).unwrap());
    }
    out
}

// --- construction ---------------------------------------------------------

#[test]
fn creates_database_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    assert!(!path.exists());
    let store = HistoryStore::open(&path).unwrap();
    assert!(path.exists());
    store.close();
}

#[test]
fn creates_intermediate_directories() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("data/familiars/aria/history.db");
    let store = HistoryStore::open(&nested).unwrap();
    assert!(nested.exists());
    store.close();
}

#[test]
fn in_memory_database_for_tests() {
    let store = mem();
    store
        .append_turn(AppendTurn::new("x", 1, "user", "hello"))
        .unwrap();
    assert_eq!(store.count("x", Some(1)).unwrap(), 1);
}

// --- append_turn ----------------------------------------------------------

#[test]
fn append_returns_history_turn_with_id() {
    let store = mem();
    let turn = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "hello").author(alice()))
        .unwrap();
    assert!(turn.id > 0);
    assert_eq!(turn.role, "user");
    assert_eq!(turn.content, "hello");
    assert_eq!(turn.author, Some(alice()));
}

#[test]
fn author_round_trips_through_select() {
    let store = mem();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "hello").author(alice()))
        .unwrap();
    let turns = store.recent(FAMILIAR, CHANNEL, 1, None, None).unwrap();
    let a = turns[0].author.as_ref().unwrap();
    assert_eq!(a.canonical_key(), "discord:1");
    assert_eq!(a.display_name.as_deref(), Some("Alice"));
    assert_eq!(a.username.as_deref(), Some("alice"));
}

#[test]
fn assistant_turn_has_no_author() {
    let store = mem();
    let turn = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "assistant", "hi back"))
        .unwrap();
    assert!(turn.author.is_none());
}

#[test]
fn ids_are_monotonically_increasing() {
    let store = mem();
    let ids: Vec<i64> = seed(&store, 5).iter().map(|t| t.id).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);
    let unique: std::collections::HashSet<i64> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len());
}

#[test]
fn persistent_across_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    let store = HistoryStore::open(&path).unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "persisted"))
        .unwrap();
    store.close();
    drop(store);

    let reopened = HistoryStore::open(&path).unwrap();
    let turns = reopened.recent(FAMILIAR, CHANNEL, 10, None, None).unwrap();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].content, "persisted");
}

// --- recent ---------------------------------------------------------------

#[test]
fn recent_empty_returns_empty() {
    let store = mem();
    assert!(
        store
            .recent(FAMILIAR, CHANNEL, 10, None, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn recent_chronological_oldest_first() {
    let store = mem();
    seed(&store, 5);
    let turns = store.recent(FAMILIAR, CHANNEL, 10, None, None).unwrap();
    let contents: Vec<&str> = turns.iter().map(|t| t.content.as_str()).collect();
    assert_eq!(contents, ["turn 0", "turn 1", "turn 2", "turn 3", "turn 4"]);
}

#[test]
fn recent_limit_returns_latest_n() {
    let store = mem();
    seed(&store, 5);
    let turns = store.recent(FAMILIAR, CHANNEL, 3, None, None).unwrap();
    let contents: Vec<&str> = turns.iter().map(|t| t.content.as_str()).collect();
    assert_eq!(contents, ["turn 2", "turn 3", "turn 4"]);
}

#[test]
fn recent_isolated_per_channel() {
    let store = mem();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 200, "user", "ch1"))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 300, "user", "ch2"))
        .unwrap();
    let ch1 = store.recent(FAMILIAR, 200, 10, None, None).unwrap();
    let ch2 = store.recent(FAMILIAR, 300, 10, None, None).unwrap();
    assert_eq!(
        ch1.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["ch1"]
    );
    assert_eq!(
        ch2.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["ch2"]
    );
}

#[test]
fn recent_isolated_per_familiar() {
    let store = mem();
    store
        .append_turn(AppendTurn::new("aria", CHANNEL, "user", "for-aria"))
        .unwrap();
    store
        .append_turn(AppendTurn::new("bob", CHANNEL, "user", "for-bob"))
        .unwrap();
    let aria = store.recent("aria", CHANNEL, 10, None, None).unwrap();
    assert_eq!(
        aria.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["for-aria"]
    );
}

#[test]
fn recent_before_id_filters_at_or_above() {
    let store = mem();
    let turns = seed(&store, 5);
    let got = store
        .recent(FAMILIAR, CHANNEL, 10, None, Some(turns[3].id))
        .unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["turn 0", "turn 1", "turn 2"]
    );
}

#[test]
fn recent_before_id_combines_with_limit() {
    let store = mem();
    let turns = seed(&store, 5);
    let got = store
        .recent(FAMILIAR, CHANNEL, 2, None, Some(turns[4].id))
        .unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["turn 2", "turn 3"]
    );
}

#[test]
fn recent_filters_by_mode_and_combines() {
    let store = mem();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "rp early").mode("full_rp"))
        .unwrap();
    let cut = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "voice").mode("imitate_voice"))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "rp late").mode("full_rp"))
        .unwrap();
    let rp = store
        .recent(FAMILIAR, CHANNEL, 10, Some("full_rp"), None)
        .unwrap();
    assert_eq!(
        rp.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["rp early", "rp late"]
    );
    let got = store
        .recent(FAMILIAR, CHANNEL, 10, Some("full_rp"), Some(cut.id))
        .unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["rp early"]
    );
}

// --- recent_distinct_authors ---------------------------------------------

#[test]
fn distinct_authors_most_recent_first() {
    let store = mem();
    let bob = author("2", "bob", "Bob");
    let carol = author("3", "carol", "Carol");
    for a in [&alice(), &bob, &alice(), &carol] {
        store
            .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "x").author(a.clone()))
            .unwrap();
    }
    let authors = store.recent_distinct_authors(FAMILIAR, CHANNEL, 5).unwrap();
    let keys: Vec<String> = authors.iter().map(Author::canonical_key).collect();
    assert_eq!(keys, ["discord:3", "discord:1", "discord:2"]);
}

#[test]
fn distinct_authors_respects_limit_and_skips_assistant() {
    let store = mem();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "x").author(alice()))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "assistant", "hi back"))
        .unwrap();
    let authors = store.recent_distinct_authors(FAMILIAR, CHANNEL, 5).unwrap();
    assert_eq!(
        authors
            .iter()
            .map(Author::canonical_key)
            .collect::<Vec<_>>(),
        ["discord:1"]
    );
    assert!(
        store
            .recent_distinct_authors(FAMILIAR, CHANNEL, 0)
            .unwrap()
            .is_empty()
    );
}

// --- older_than / latest_id / count --------------------------------------

#[test]
fn older_than_returns_at_or_below() {
    let store = mem();
    let turns = seed(&store, 5);
    let older = store
        .older_than(FAMILIAR, turns[2].id, None, 10_000)
        .unwrap();
    assert_eq!(
        older.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["turn 0", "turn 1", "turn 2"]
    );
    assert!(
        store
            .older_than(FAMILIAR, 0, None, 10_000)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn older_than_scoped_to_channel() {
    let store = mem();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 100, "user", "ch100"))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 200, "user", "ch200"))
        .unwrap();
    let older = store.older_than(FAMILIAR, 999, Some(100), 10_000).unwrap();
    assert_eq!(
        older.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["ch100"]
    );
}

#[test]
fn latest_id_global_and_per_channel() {
    let store = mem();
    assert!(store.latest_id(FAMILIAR, None).unwrap().is_none());
    store
        .append_turn(AppendTurn::new(FAMILIAR, 100, "user", "a"))
        .unwrap();
    let last = store
        .append_turn(AppendTurn::new(FAMILIAR, 200, "user", "b"))
        .unwrap();
    assert_eq!(store.latest_id(FAMILIAR, None).unwrap(), Some(last.id));
    assert!(store.latest_id(FAMILIAR, Some(100)).unwrap().unwrap() < last.id);
}

#[test]
fn count_scoping() {
    let store = mem();
    assert_eq!(store.count(FAMILIAR, Some(CHANNEL)).unwrap(), 0);
    store
        .append_turn(AppendTurn::new(FAMILIAR, 200, "user", "a"))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 300, "user", "b"))
        .unwrap();
    assert_eq!(store.count(FAMILIAR, None).unwrap(), 2);
    assert_eq!(store.count(FAMILIAR, Some(200)).unwrap(), 1);
}

// --- summaries ------------------------------------------------------------

#[test]
fn summary_round_trip_and_overwrite() {
    let store = mem();
    assert!(store.get_summary(FAMILIAR, 0).unwrap().is_none());
    store
        .put_summary(FAMILIAR, 42, "they argued about ska", 0, None)
        .unwrap();
    let entry = store.get_summary(FAMILIAR, 0).unwrap().unwrap();
    assert_eq!(entry.last_summarised_id, 42);
    assert_eq!(entry.summary_text, "they argued about ska");
    assert!(entry.last_consumed_at.is_none());
    store.put_summary(FAMILIAR, 15, "new", 0, None).unwrap();
    assert_eq!(
        store
            .get_summary(FAMILIAR, 0)
            .unwrap()
            .unwrap()
            .last_summarised_id,
        15
    );
}

#[test]
fn summary_round_trips_last_consumed_at() {
    let store = mem();
    store
        .put_summary(FAMILIAR, 7, "x", 0, Some("2026-06-13T10:00:00+00:00"))
        .unwrap();
    let entry = store.get_summary(FAMILIAR, 0).unwrap().unwrap();
    assert_eq!(
        entry.last_consumed_at.as_deref(),
        Some("2026-06-13T10:00:00+00:00")
    );
}

#[test]
fn summary_scoped_per_channel() {
    let store = mem();
    store
        .put_summary(FAMILIAR, 10, "channel 100 summary", 100, None)
        .unwrap();
    store
        .put_summary(FAMILIAR, 20, "channel 200 summary", 200, None)
        .unwrap();
    assert_eq!(
        store
            .get_summary(FAMILIAR, 100)
            .unwrap()
            .unwrap()
            .summary_text,
        "channel 100 summary"
    );
    assert_eq!(
        store
            .get_summary(FAMILIAR, 200)
            .unwrap()
            .unwrap()
            .summary_text,
        "channel 200 summary"
    );
    assert!(store.get_summary(FAMILIAR, 999).unwrap().is_none());
}

// --- consumed_turns_after -------------------------------------------------

#[test]
fn consumed_turns_after_empty_cursor_and_limit() {
    let store = mem();
    seed(&store, 5);
    let out = store.consumed_turns_after(FAMILIAR, "", 0, 10).unwrap();
    assert_eq!(
        out.iter().map(|t| t.id).collect::<Vec<_>>(),
        [1, 2, 3, 4, 5]
    );
    let out3 = store.consumed_turns_after(FAMILIAR, "", 0, 3).unwrap();
    assert_eq!(out3.iter().map(|t| t.id).collect::<Vec<_>>(), [1, 2, 3]);
}

#[test]
fn consumed_turns_after_cursor_advances_to_empty() {
    let store = mem();
    let turns = seed(&store, 4);
    let last = &turns[3];
    let cursor = iso_utc(last.consumed_at.unwrap());
    let out = store
        .consumed_turns_after(FAMILIAR, &cursor, last.id, 10)
        .unwrap();
    assert!(out.is_empty());
}

#[test]
fn consumed_turns_after_excludes_staged() {
    let store = mem();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "staged").consumed(false))
        .unwrap();
    assert!(
        store
            .consumed_turns_after(FAMILIAR, "", 0, 10)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn consumed_turns_after_includes_late_promoted_low_id() {
    let store = mem();
    let old = Utc.with_ymd_and_hms(2026, 6, 13, 9, 0, 0).unwrap();
    // id=1: staged in a dormant channel with an old arrived_at.
    store
        .append_turn(
            AppendTurn::new(FAMILIAR, 300, "user", "dormant-channel msg")
                .consumed(false)
                .arrived_at(old),
        )
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "b"))
        .unwrap();
    let c = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "assistant", "c"))
        .unwrap();
    let cursor = iso_utc(c.consumed_at.unwrap());
    let promoted = store.promote_staged_turns(FAMILIAR, 300, None).unwrap();
    assert_eq!(promoted.consumed, 1);
    let out = store
        .consumed_turns_after(FAMILIAR, &cursor, c.id, 10)
        .unwrap();
    assert_eq!(out.iter().map(|t| t.id).collect::<Vec<_>>(), [1]);
}

// --- migration backfill scope + ego migration -----------------------------

#[test]
fn staged_turn_stays_staged_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    let store = HistoryStore::open(&path).unwrap();
    let turn = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "staged").consumed(false))
        .unwrap();
    assert!(turn.consumed_at.is_none());
    store.close();
    drop(store);

    let reopened = HistoryStore::open(&path).unwrap();
    let consumed = reopened
        .conn()
        .query_scalar_string(
            "SELECT consumed_at FROM turns WHERE id = ?",
            vec![Value::Integer(turn.id)],
        )
        .unwrap();
    assert!(
        consumed.is_none(),
        "staged turn must not be re-promoted by backfill"
    );
}

#[test]
fn ego_key_migration_rewrites_self_keys_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    let store = HistoryStore::open(&path).unwrap();
    store
        .conn()
        .execute(
            "INSERT INTO facts (familiar_id, channel_id, text, source_turn_ids, created_at, \
             subjects_json, valid_from) VALUES (?, ?, ?, ?, ?, ?, ?)",
            vec![
                Value::Text(FAMILIAR.to_owned()),
                Value::Integer(CHANNEL),
                Value::Text("Aria felt proud of the bit.".to_owned()),
                Value::Text("[]".to_owned()),
                Value::Text("2026-06-19T00:00:00+00:00".to_owned()),
                Value::Text(
                    "[{\"canonical_key\": \"self:aria\", \"display_at_write\": \"Aria\"}]"
                        .to_owned(),
                ),
                Value::Text("2026-06-19T00:00:00+00:00".to_owned()),
            ],
        )
        .unwrap();
    store
        .conn()
        .execute(
            "INSERT INTO people_dossiers (familiar_id, canonical_key, last_fact_id, dossier_text, \
             created_at) VALUES (?, ?, ?, ?, ?)",
            vec![
                Value::Text(FAMILIAR.to_owned()),
                Value::Text("self:aria".to_owned()),
                Value::Integer(1),
                Value::Text("Aria's self-record.".to_owned()),
                Value::Text("2026-06-19T00:00:00+00:00".to_owned()),
            ],
        )
        .unwrap();
    store.close();
    drop(store);

    let reopened = HistoryStore::open(&path).unwrap();
    let subjects = reopened
        .conn()
        .query_scalar_string(
            "SELECT subjects_json FROM facts WHERE familiar_id = ?",
            vec![Value::Text(FAMILIAR.to_owned())],
        )
        .unwrap()
        .unwrap();
    let dossier_key = reopened
        .conn()
        .query_scalar_string(
            "SELECT canonical_key FROM people_dossiers WHERE familiar_id = ?",
            vec![Value::Text(FAMILIAR.to_owned())],
        )
        .unwrap()
        .unwrap();
    assert!(subjects.contains("ego:aria"));
    assert!(!subjects.contains("self:aria"));
    assert_eq!(dossier_key, "ego:aria");

    // Idempotent: a second reopen leaves ego: intact.
    reopened.close();
    drop(reopened);
    let again = HistoryStore::open(&path).unwrap();
    let key = again
        .conn()
        .query_scalar_string(
            "SELECT canonical_key FROM people_dossiers WHERE familiar_id = ?",
            vec![Value::Text(FAMILIAR.to_owned())],
        )
        .unwrap()
        .unwrap();
    assert_eq!(key, "ego:aria");
}

#[test]
fn legacy_db_gains_pings_bot_column_with_default_zero() {
    use familiar_connect::history::db::Db;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.db");
    // Hand-build a legacy `turns` table lacking arrived_at/consumed_at/missed_at/pings_bot.
    let legacy = Db::open(&path).unwrap();
    legacy
        .execute_batch(
            "CREATE TABLE turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                familiar_id TEXT NOT NULL, channel_id INTEGER NOT NULL, guild_id INTEGER,
                role TEXT NOT NULL, author_platform TEXT, author_user_id TEXT,
                author_username TEXT, author_display_name TEXT, content TEXT NOT NULL,
                timestamp TEXT NOT NULL, mode TEXT, platform_message_id TEXT,
                reply_to_message_id TEXT, tool_calls_json TEXT, tool_call_id TEXT
            );
            INSERT INTO turns (familiar_id, channel_id, role, content, timestamp)
            VALUES ('aria', 200, 'user', 'legacy ping', '2026-06-13T09:00:00+00:00');",
        )
        .unwrap();
    legacy.close();
    drop(legacy);

    let store = HistoryStore::open(&path).unwrap();
    let pings = store
        .conn()
        .query_scalar_i64("SELECT pings_bot FROM turns WHERE id = 1", vec![])
        .unwrap();
    assert_eq!(pings, Some(0));
    let reloaded = &store.recent("aria", 200, 1, None, None).unwrap()[0];
    assert!(!reloaded.pings_bot);
    store.close();
    drop(store);

    // Second open must not error (idempotent ALTER).
    let again = HistoryStore::open(&path).unwrap();
    again.close();
}

// --- mode / pings_bot columns ---------------------------------------------

#[test]
fn append_turn_persists_mode_or_null() {
    let store = mem();
    let with_mode = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "hello").mode("full_rp"))
        .unwrap();
    let m = store
        .conn()
        .query_scalar_string(
            "SELECT mode FROM turns WHERE id = ?",
            vec![Value::Integer(with_mode.id)],
        )
        .unwrap();
    assert_eq!(m.as_deref(), Some("full_rp"));
    let no_mode = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "legacy"))
        .unwrap();
    let m2 = store
        .conn()
        .query_scalar_string(
            "SELECT mode FROM turns WHERE id = ?",
            vec![Value::Integer(no_mode.id)],
        )
        .unwrap();
    assert!(m2.is_none());
}

#[test]
fn pings_bot_persists_and_round_trips() {
    let store = mem();
    let turn = store
        .append_turn(
            AppendTurn::new(FAMILIAR, CHANNEL, "user", "hey @bot")
                .author(alice())
                .pings_bot(true),
        )
        .unwrap();
    assert!(turn.pings_bot);
    let raw = store
        .conn()
        .query_scalar_i64(
            "SELECT pings_bot FROM turns WHERE id = ?",
            vec![Value::Integer(turn.id)],
        )
        .unwrap();
    assert_eq!(raw, Some(1));
    let reloaded = &store.recent(FAMILIAR, CHANNEL, 1, None, None).unwrap()[0];
    assert!(reloaded.pings_bot);

    let plain = store
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "hello"))
        .unwrap();
    assert!(!plain.pings_bot);
}

// --- distinct_other_channels ---------------------------------------------

#[test]
fn distinct_other_channels_carries_mode() {
    let store = mem();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 100, "user", "a").mode("full_rp"))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 200, "user", "b").mode("text_conversation_rp"))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 300, "user", "c").mode("imitate_voice"))
        .unwrap();
    let others = store.distinct_other_channels(FAMILIAR, 100).unwrap();
    let ids: std::collections::HashSet<i64> = others.iter().map(|o| o.channel_id).collect();
    assert_eq!(ids, [200, 300].into_iter().collect());
    let modes: std::collections::HashMap<i64, Option<String>> = others
        .iter()
        .map(|o| (o.channel_id, o.mode.clone()))
        .collect();
    assert_eq!(modes[&200].as_deref(), Some("text_conversation_rp"));
    assert_eq!(modes[&300].as_deref(), Some("imitate_voice"));

    let store2 = mem();
    store2
        .append_turn(AppendTurn::new(FAMILIAR, 100, "user", "only").mode("full_rp"))
        .unwrap();
    assert!(
        store2
            .distinct_other_channels(FAMILIAR, 100)
            .unwrap()
            .is_empty()
    );
}

// --- memory-writer watermark ---------------------------------------------

#[test]
fn writer_watermark_round_trip_and_turns_since() {
    let store = mem();
    assert!(store.get_writer_watermark(FAMILIAR).unwrap().is_none());
    let turns = seed(&store, 10);
    store.put_writer_watermark(FAMILIAR, turns[4].id).unwrap();
    assert_eq!(
        store
            .get_writer_watermark(FAMILIAR)
            .unwrap()
            .unwrap()
            .last_written_id,
        turns[4].id
    );
    let since = store.turns_since_watermark(FAMILIAR, 10_000).unwrap();
    assert_eq!(since.len(), 5);
    assert_eq!(since[0].id, turns[5].id);
    assert_eq!(since[4].id, turns[9].id);

    let store2 = mem();
    let t2 = seed(&store2, 6);
    let all = store2.turns_since_watermark(FAMILIAR, 10_000).unwrap();
    assert_eq!(all.len(), 6);
    assert_eq!(all[0].id, t2[0].id);
}

// --- recent_cross_channel respect_archive --------------------------------

#[test]
fn cross_channel_respect_archive_hides_at_or_below() {
    let store = mem();
    let turns = seed(&store, 5);
    store
        .set_archive_watermark(FAMILIAR, CHANNEL, turns[2].id)
        .unwrap();
    let got = store.recent_cross_channel(FAMILIAR, 10, true).unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["turn 3", "turn 4"]
    );
    // default ignores watermark
    assert_eq!(
        store
            .recent_cross_channel(FAMILIAR, 10, false)
            .unwrap()
            .len(),
        5
    );
}

#[test]
fn cross_channel_window_shrinks_not_backfills() {
    let store = mem();
    let turns = seed(&store, 6);
    store
        .set_archive_watermark(FAMILIAR, CHANNEL, turns[3].id)
        .unwrap();
    let got = store.recent_cross_channel(FAMILIAR, 4, true).unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["turn 4", "turn 5"]
    );
}

#[test]
fn cross_channel_archive_scoped_per_channel() {
    let store = mem();
    let a1 = store
        .append_turn(AppendTurn::new(FAMILIAR, 100, "user", "a before").author(alice()))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 101, "user", "b before").author(alice()))
        .unwrap();
    store.set_archive_watermark(FAMILIAR, 100, a1.id).unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 100, "user", "a after").author(alice()))
        .unwrap();
    store
        .append_turn(AppendTurn::new(FAMILIAR, 101, "user", "b after").author(alice()))
        .unwrap();
    let got = store.recent_cross_channel(FAMILIAR, 10, true).unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["b before", "a after", "b after"]
    );
}

// --- promote_staged_turns_since ------------------------------------------

fn stage(store: &HistoryStore, channel_id: i64, content: &str) -> HistoryTurn {
    store
        .append_turn(
            AppendTurn::new(FAMILIAR, channel_id, "user", content)
                .author(alice())
                .consumed(false),
        )
        .unwrap()
}

#[test]
fn promote_since_across_channels() {
    let store = mem();
    let departure = store
        .append_turn(AppendTurn::new(FAMILIAR, 100, "user", "before"))
        .unwrap();
    stage(&store, 100, "a during");
    stage(&store, 101, "b during");
    let promo = store
        .promote_staged_turns_since(FAMILIAR, departure.id, None)
        .unwrap();
    assert_eq!(promo.consumed, 2);
    assert_eq!(promo.missed, 0);
    assert_eq!(store.count_staged(FAMILIAR, 100).unwrap(), 0);
    assert_eq!(store.count_staged(FAMILIAR, 101).unwrap(), 0);
}

#[test]
fn promote_since_leaves_at_or_below_untouched() {
    let store = mem();
    let pre = stage(&store, 100, "never attended");
    let during = stage(&store, 101, "during");
    let promo = store
        .promote_staged_turns_since(FAMILIAR, pre.id, None)
        .unwrap();
    assert_eq!(promo.consumed, 1);
    assert_eq!(store.count_staged(FAMILIAR, 100).unwrap(), 1);
    assert_eq!(store.count_staged(FAMILIAR, 101).unwrap(), 0);
    let promo2 = store
        .promote_staged_turns_since(FAMILIAR, during.id, None)
        .unwrap();
    assert_eq!(promo2.consumed, 0);
}

#[test]
fn promote_since_caps_per_channel_and_catches_pings() {
    let store = mem();
    let departure = store
        .append_turn(AppendTurn::new(FAMILIAR, 100, "assistant", "departure"))
        .unwrap();
    for i in 0..4 {
        stage(&store, 100, &format!("a{i}"));
        stage(&store, 101, &format!("b{i}"));
    }
    let promo = store
        .promote_staged_turns_since(FAMILIAR, departure.id, Some(2))
        .unwrap();
    assert_eq!(promo.consumed, 4);
    assert_eq!(promo.missed, 4);

    let store2 = mem();
    let dep = store2
        .append_turn(AppendTurn::new(FAMILIAR, 100, "assistant", "departure"))
        .unwrap();
    let ping = store2
        .append_turn(
            AppendTurn::new(FAMILIAR, 100, "user", "<@bot> you there?")
                .consumed(false)
                .pings_bot(true),
        )
        .unwrap();
    for i in 0..3 {
        stage(&store2, 100, &format!("chatter{i}"));
    }
    let promo = store2
        .promote_staged_turns_since(FAMILIAR, dep.id, Some(1))
        .unwrap();
    assert_eq!(promo.consumed, 2);
    assert_eq!(promo.missed, 2);
    let got = store2.recent_cross_channel(FAMILIAR, 10, false).unwrap();
    assert!(got.iter().any(|t| t.id == ping.id));
}

// --- turns_around ---------------------------------------------------------

#[test]
fn turns_around_centred_and_clipped() {
    let store = mem();
    let turns = seed(&store, 11);
    let got = store
        .turns_around(FAMILIAR, CHANNEL, turns[5].id, 2, 2)
        .unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["turn 3", "turn 4", "turn 5", "turn 6", "turn 7"]
    );
    let store2 = mem();
    let t2 = seed(&store2, 3);
    let clipped = store2
        .turns_around(FAMILIAR, CHANNEL, t2[0].id, 5, 5)
        .unwrap();
    assert_eq!(
        clipped
            .iter()
            .map(|t| t.content.clone())
            .collect::<Vec<_>>(),
        ["turn 0", "turn 1", "turn 2"]
    );
}

#[test]
fn turns_around_defaults_and_partition() {
    let store = mem();
    let turns = seed(&store, 20);
    let got = store
        .turns_around(FAMILIAR, CHANNEL, turns[10].id, 5, 5)
        .unwrap();
    let want: Vec<i64> = turns[5..16].iter().map(|t| t.id).collect();
    assert_eq!(got.iter().map(|t| t.id).collect::<Vec<_>>(), want);

    let store2 = mem();
    let anchor = store2
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "here"))
        .unwrap();
    store2
        .append_turn(AppendTurn::new(FAMILIAR, 999, "user", "elsewhere"))
        .unwrap();
    store2
        .append_turn(AppendTurn::new(FAMILIAR, CHANNEL, "user", "here too"))
        .unwrap();
    let got = store2
        .turns_around(FAMILIAR, CHANNEL, anchor.id, 5, 5)
        .unwrap();
    assert_eq!(
        got.iter().map(|t| t.content.clone()).collect::<Vec<_>>(),
        ["here", "here too"]
    );
}

// --- archive watermark CRUD ----------------------------------------------

#[test]
fn archive_watermark_crud() {
    let store = mem();
    assert!(
        store
            .get_archive_watermark(FAMILIAR, CHANNEL)
            .unwrap()
            .is_none()
    );
    store.set_archive_watermark(FAMILIAR, CHANNEL, 42).unwrap();
    assert_eq!(
        store.get_archive_watermark(FAMILIAR, CHANNEL).unwrap(),
        Some(42)
    );
    store.set_archive_watermark(FAMILIAR, CHANNEL, 50).unwrap();
    assert_eq!(
        store.get_archive_watermark(FAMILIAR, CHANNEL).unwrap(),
        Some(50)
    );
    assert!(
        store
            .get_archive_watermark(FAMILIAR, 999)
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .get_archive_watermark("bob", CHANNEL)
            .unwrap()
            .is_none()
    );
}

// --- activities -----------------------------------------------------------

#[test]
fn activities_lifecycle() {
    let store = mem();
    let t0 = Utc.with_ymd_and_hms(2026, 6, 12, 10, 0, 0).unwrap();
    let t1 = Utc.with_ymd_and_hms(2026, 6, 12, 10, 30, 0).unwrap();
    assert!(store.active_activity(FAMILIAR).unwrap().is_none());
    let id = store
        .create_activity(FAMILIAR, "walk", "on a walk", t0, t1, Some("creek"))
        .unwrap();
    assert!(id > 0);
    let rec = store.active_activity(FAMILIAR).unwrap().unwrap();
    assert_eq!(rec.id, id);
    assert_eq!(rec.type_id, "walk");
    assert_eq!(rec.started_at, t0);
    assert_eq!(rec.planned_return_at, t1);
    assert_eq!(rec.note.as_deref(), Some("creek"));
    assert!(rec.status.is_none());
    assert!(rec.actual_return_at.is_none());

    store
        .finish_activity(id, "completed", t1, Some("saw a heron"))
        .unwrap();
    assert!(store.active_activity(FAMILIAR).unwrap().is_none());

    // bad status rejected
    let id2 = store
        .create_activity(FAMILIAR, "walk", "x", t0, t1, None)
        .unwrap();
    let err = store.finish_activity(id2, "abandoned", t1, None);
    assert!(err.is_err());
}

#[test]
fn activities_finish_persists_and_experience() {
    let store = mem();
    let t0 = Utc.with_ymd_and_hms(2026, 6, 12, 10, 0, 0).unwrap();
    let t1 = t0 + chrono::Duration::minutes(30);
    let id = store
        .create_activity(FAMILIAR, "walk", "on a walk", t0, t1, Some("creek"))
        .unwrap();
    store
        .finish_activity(id, "cut_short", t1, Some("got pinged"))
        .unwrap();
    let status = store
        .conn()
        .query_scalar_string(
            "SELECT status FROM activities WHERE id = ?",
            vec![Value::Integer(id)],
        )
        .unwrap();
    assert_eq!(status.as_deref(), Some("cut_short"));
    let ret = store
        .conn()
        .query_scalar_string(
            "SELECT actual_return_at FROM activities WHERE id = ?",
            vec![Value::Integer(id)],
        )
        .unwrap();
    assert_eq!(ret.as_deref(), Some(iso_utc(t1).as_str()));

    let store2 = mem();
    let a = store2
        .create_activity(FAMILIAR, "walk", "on a walk", t0, t1, None)
        .unwrap();
    store2
        .set_activity_experience(a, "a dream of rain")
        .unwrap();
    assert_eq!(
        store2
            .active_activity(FAMILIAR)
            .unwrap()
            .unwrap()
            .experience_text
            .as_deref(),
        Some("a dream of rain")
    );
}

#[test]
fn activities_append_only_and_latest() {
    let store = mem();
    let t0 = Utc.with_ymd_and_hms(2026, 6, 12, 10, 0, 0).unwrap();
    let t1 = t0 + chrono::Duration::minutes(30);
    let first = store
        .create_activity(FAMILIAR, "walk", "on a walk", t0, t1, None)
        .unwrap();
    store.finish_activity(first, "completed", t1, None).unwrap();
    let second = store
        .create_activity(FAMILIAR, "walk", "on a walk", t0, t1, None)
        .unwrap();
    assert_eq!(store.active_activity(FAMILIAR).unwrap().unwrap().id, second);
    let n = store
        .conn()
        .query_scalar_i64(
            "SELECT COUNT(*) AS n FROM activities WHERE familiar_id = ?",
            vec![Value::Text(FAMILIAR.to_owned())],
        )
        .unwrap();
    assert_eq!(n, Some(2));
    let latest = store.latest_activity(FAMILIAR, "walk").unwrap().unwrap();
    assert_eq!(latest.id, second);
    assert!(store.latest_activity(FAMILIAR, "sleep").unwrap().is_none());

    let store2 = mem();
    let only = store2
        .create_activity(FAMILIAR, "walk", "x", t0, t1, None)
        .unwrap();
    store2.finish_activity(only, "completed", t1, None).unwrap();
    let rec = store2.latest_activity(FAMILIAR, "walk").unwrap().unwrap();
    assert_eq!(rec.id, only);
    assert_eq!(rec.status.as_deref(), Some("completed"));
}

// --- attentional: staged/promote/focus/digest ----------------------------

#[test]
fn append_new_fields() {
    let store = mem();
    let consumed = store
        .append_turn(AppendTurn::new("fam", 1, "user", "hi"))
        .unwrap();
    assert!(consumed.consumed_at.is_some());
    assert!(consumed.arrived_at.is_some());
    let staged = store
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").consumed(false))
        .unwrap();
    assert!(staged.consumed_at.is_none());
    let ts = Utc.with_ymd_and_hms(2025, 1, 2, 12, 0, 0).unwrap();
    let with_arr = store
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").arrived_at(ts))
        .unwrap();
    assert_eq!(iso_utc(with_arr.arrived_at.unwrap()), iso_utc(ts));
    assert_eq!(
        iso_utc(with_arr.consumed_at.unwrap()),
        iso_utc(with_arr.arrived_at.unwrap())
    );
}

#[test]
fn stage_turn_convenience() {
    let store = mem();
    let turn = store
        .stage_turn(AppendTurn::new("fam", 1, "user", "staged"))
        .unwrap();
    assert!(turn.consumed_at.is_none());
    assert!(turn.arrived_at.is_some());
}

#[test]
fn count_staged_scoping() {
    let store = mem();
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "c"))
        .unwrap();
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "s").consumed(false))
        .unwrap();
    store
        .append_turn(AppendTurn::new("fam", 2, "user", "s").consumed(false))
        .unwrap();
    assert_eq!(store.count_staged("fam", 1).unwrap(), 1);
    assert_eq!(store.count_staged("fam", 2).unwrap(), 1);
}

#[test]
fn promote_catch_up_window_and_missed_terminality() {
    let store = mem();
    let staged: Vec<HistoryTurn> = (0..5)
        .map(|_| {
            store
                .append_turn(AppendTurn::new("fam", 1, "user", "x").consumed(false))
                .unwrap()
        })
        .collect();
    let promo = store.promote_staged_turns("fam", 1, Some(2)).unwrap();
    assert_eq!(promo.consumed, 2);
    assert_eq!(promo.missed, 3);
    // missed excluded from every read path
    assert_eq!(store.count_staged("fam", 1).unwrap(), 0);
    assert!(store.staged_channels("fam").unwrap().is_empty());
    let cross: std::collections::HashSet<i64> = store
        .recent_cross_channel("fam", 10, false)
        .unwrap()
        .iter()
        .map(|t| t.id)
        .collect();
    assert_eq!(cross, [staged[4].id, staged[3].id].into_iter().collect());
    // missed_at set for the older three
    for t in &staged[..3] {
        let m = store
            .conn()
            .query_scalar_string(
                "SELECT missed_at FROM turns WHERE id = ?",
                vec![Value::Integer(t.id)],
            )
            .unwrap();
        assert!(m.is_some());
    }
}

#[test]
fn promote_pings_always_caught() {
    let store = mem();
    let ping = store
        .append_turn(
            AppendTurn::new("fam", 1, "user", "ping")
                .consumed(false)
                .pings_bot(true),
        )
        .unwrap();
    for _ in 0..4 {
        store
            .append_turn(AppendTurn::new("fam", 1, "user", "chatter").consumed(false))
            .unwrap();
    }
    let promo = store.promote_staged_turns("fam", 1, Some(2)).unwrap();
    assert_eq!(promo.consumed, 3);
    assert_eq!(promo.missed, 2);
    let m = store
        .conn()
        .query_scalar_string(
            "SELECT missed_at FROM turns WHERE id = ?",
            vec![Value::Integer(ping.id)],
        )
        .unwrap();
    assert!(m.is_none());
}

#[test]
fn staged_channels_ping_tally() {
    let store = mem();
    for i in 0..3 {
        store
            .append_turn(
                AppendTurn::new("aria", 10, "user", format!("a{i}"))
                    .consumed(false)
                    .pings_bot(i == 0),
            )
            .unwrap();
    }
    store
        .append_turn(AppendTurn::new("aria", 20, "user", "b").consumed(false))
        .unwrap();
    let got = store.staged_channels("aria").unwrap();
    assert_eq!(got[&10], ChannelUnread(3, 1));
    assert_eq!(got[&20], ChannelUnread(1, 0));
}

#[test]
fn focus_pointers_crud() {
    let store = mem();
    assert!(store.get_focus_pointers("fam").unwrap().is_none());
    store
        .set_focus_pointers("fam", Some(100), Some(200))
        .unwrap();
    let fp = store.get_focus_pointers("fam").unwrap().unwrap();
    assert_eq!(fp.text_channel_id, Some(100));
    assert_eq!(fp.voice_channel_id, Some(200));
    store.set_focus_pointers("fam", Some(3), None).unwrap();
    let fp = store.get_focus_pointers("fam").unwrap().unwrap();
    assert_eq!(fp.text_channel_id, Some(3));
    assert!(fp.voice_channel_id.is_none());
    assert!(store.get_focus_pointers("other").unwrap().is_none());
}

#[test]
fn digest_watermark_crud() {
    let store = mem();
    assert!(store.get_digest_watermark("fam").unwrap().is_none());
    let ts = Utc.with_ymd_and_hms(2025, 3, 15, 9, 0, 0).unwrap();
    store.set_digest_watermark("fam", ts).unwrap();
    assert_eq!(
        iso_utc(store.get_digest_watermark("fam").unwrap().unwrap()),
        iso_utc(ts)
    );
    let ts2 = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
    store.set_digest_watermark("fam", ts2).unwrap();
    assert_eq!(
        iso_utc(store.get_digest_watermark("fam").unwrap().unwrap()),
        iso_utc(ts2)
    );
    assert!(store.get_digest_watermark("other").unwrap().is_none());
}

#[test]
fn recent_cross_channel_oldest_first_and_scoped() {
    let store = mem();
    let base = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
    let t1 = store
        .append_turn(AppendTurn::new("fam", 1, "user", "m0").arrived_at(base))
        .unwrap();
    let t2 = store
        .append_turn(
            AppendTurn::new("fam", 2, "user", "m1").arrived_at(base + chrono::Duration::seconds(1)),
        )
        .unwrap();
    let t3 = store
        .append_turn(
            AppendTurn::new("fam", 1, "user", "m2").arrived_at(base + chrono::Duration::seconds(2)),
        )
        .unwrap();
    let got = store.recent_cross_channel("fam", 10, false).unwrap();
    assert_eq!(
        got.iter().map(|t| t.id).collect::<Vec<_>>(),
        [t1.id, t2.id, t3.id]
    );
    // staged excluded
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "staged").consumed(false))
        .unwrap();
    assert_eq!(
        store.recent_cross_channel("fam", 10, false).unwrap().len(),
        3
    );
}
