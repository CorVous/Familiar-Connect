//! Integration tests for accounts/identity, reply linkage + mentions, alarms,
//! and reactions (including the single-SQL-query performance pin). Ports
//! `test_history_identity.py`, `test_history_replies.py`, `test_history_alarms.py`,
//! and the store parts of `test_message_reactions.py`.

use std::sync::{Arc, Mutex};

use chrono::Utc;
use familiar_connect::history::db::{TraceCallback, Value};
use familiar_connect::history::{AppendTurn, Author, HistoryStore};

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

// --- accounts -------------------------------------------------------------

#[test]
fn upsert_creates_and_updates_row() {
    let store = mem();
    let mut a = author("111", "cass_login", "Cass");
    a.global_name = Some("Cassidy".to_owned());
    store.upsert_account(&a).unwrap();
    let profile = store.get_account_profile("discord:111").unwrap().unwrap();
    assert_eq!(profile.username.as_deref(), Some("cass_login"));
    assert_eq!(profile.global_name.as_deref(), Some("Cassidy"));

    let mut b = author("111", "new_login", "NewName");
    b.global_name = Some("NewGlobal".to_owned());
    store.upsert_account(&b).unwrap();
    let profile = store.get_account_profile("discord:111").unwrap().unwrap();
    assert_eq!(profile.username.as_deref(), Some("new_login"));
    assert_eq!(profile.global_name.as_deref(), Some("NewGlobal"));
}

#[test]
fn upsert_persists_and_coalesces_pronouns_bio() {
    let store = mem();
    let mut a = author("222", "ada_l", "Ada");
    a.pronouns = Some("she/her".to_owned());
    a.bio = Some("Designs analytical engines.".to_owned());
    store.upsert_account(&a).unwrap();
    let profile = store.get_account_profile("discord:222").unwrap().unwrap();
    assert_eq!(profile.pronouns.as_deref(), Some("she/her"));
    assert_eq!(profile.bio.as_deref(), Some("Designs analytical engines."));

    // A later upsert with NULL pronouns/bio must not clobber the richer values.
    let bare = author("222", "ada_l", "Ada");
    store.upsert_account(&bare).unwrap();
    let profile = store.get_account_profile("discord:222").unwrap().unwrap();
    assert_eq!(profile.pronouns.as_deref(), Some("she/her"));
    assert_eq!(profile.bio.as_deref(), Some("Designs analytical engines."));

    assert!(store.get_account_profile("discord:999").unwrap().is_none());
}

// --- guild nicks ----------------------------------------------------------

#[test]
fn guild_nick_upsert_overwrite_and_null() {
    let store = mem();
    store
        .upsert_guild_nick("discord:111", 42, Some("Aria"))
        .unwrap();
    let nick = store
        .conn()
        .query_scalar_string(
            "SELECT nick FROM account_guild_nicks WHERE canonical_key = ? AND guild_id = ?",
            vec![Value::Text("discord:111".to_owned()), Value::Integer(42)],
        )
        .unwrap();
    assert_eq!(nick.as_deref(), Some("Aria"));
    store
        .upsert_guild_nick("discord:111", 42, Some("AriaPrime"))
        .unwrap();
    let nick = store
        .conn()
        .query_scalar_string(
            "SELECT nick FROM account_guild_nicks WHERE canonical_key = ? AND guild_id = ?",
            vec![Value::Text("discord:111".to_owned()), Value::Integer(42)],
        )
        .unwrap();
    assert_eq!(nick.as_deref(), Some("AriaPrime"));
    // NULL nick is a meaningful "no override" (row exists, nick is NULL).
    store.upsert_guild_nick("discord:111", 42, None).unwrap();
    let count = store
        .conn()
        .query_scalar_i64(
            "SELECT COUNT(*) FROM account_guild_nicks WHERE canonical_key = ? AND guild_id = ? AND nick IS NULL",
            vec![Value::Text("discord:111".to_owned()), Value::Integer(42)],
        )
        .unwrap();
    assert_eq!(count, Some(1));
}

// --- resolve_label 5-step chain -------------------------------------------

#[test]
fn resolve_label_chain() {
    let store = mem();
    let mut a = author("111", "cass_login", "Aria");
    a.global_name = Some("Cassidy".to_owned());
    store.upsert_account(&a).unwrap();
    store
        .upsert_guild_nick("discord:111", 42, Some("Aria"))
        .unwrap();
    // 1) guild nick
    assert_eq!(
        store.resolve_label("discord:111", Some(42), None).unwrap(),
        "Aria"
    );
    // 2) global_name in another guild
    assert_eq!(
        store.resolve_label("discord:111", Some(99), None).unwrap(),
        "Cassidy"
    );
    // guild_id None → straight to account global_name
    assert_eq!(
        store.resolve_label("discord:111", None, None).unwrap(),
        "Cassidy"
    );

    // 3) username when no global_name
    let store = mem();
    store
        .upsert_account(&author("111", "cass_login", "cass_login"))
        .unwrap();
    assert_eq!(
        store.resolve_label("discord:111", Some(42), None).unwrap(),
        "cass_login"
    );

    // 5) user_id when no account row
    let store = mem();
    assert_eq!(
        store.resolve_label("discord:999", Some(42), None).unwrap(),
        "999"
    );
}

#[test]
fn resolve_label_snapshot_fallback() {
    let store = mem();
    // No accounts row, but a turn carries the author snapshot.
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "hi").author(author(
            "111",
            "cass_login",
            "SnapName",
        )))
        .unwrap();
    assert_eq!(
        store
            .resolve_label("discord:111", Some(42), Some("fam"))
            .unwrap(),
        "SnapName"
    );
    // latest_author_for returns None on malformed key.
    assert!(
        store
            .latest_author_for("fam", "no-colon")
            .unwrap()
            .is_none()
    );
}

// --- platform message id + reply linkage + mentions -----------------------

#[test]
fn platform_message_id_lookup_and_scope() {
    let store = mem();
    let turn = store
        .append_turn(
            AppendTurn::new("fam", 1, "user", "hi")
                .author(author("111", "cass", "Cass"))
                .platform_message_id("9999000111"),
        )
        .unwrap();
    let looked = store
        .lookup_turn_by_platform_message_id("fam", "9999000111")
        .unwrap()
        .unwrap();
    assert_eq!(looked.id, turn.id);
    assert!(
        store
            .lookup_turn_by_platform_message_id("fam", "ghost")
            .unwrap()
            .is_none()
    );

    // per-familiar scope
    let store = mem();
    store
        .append_turn(
            AppendTurn::new("famA", 1, "user", "A")
                .author(author("111", "c", "C"))
                .platform_message_id("shared"),
        )
        .unwrap();
    store
        .append_turn(
            AppendTurn::new("famB", 1, "user", "B")
                .author(author("111", "c", "C"))
                .platform_message_id("shared"),
        )
        .unwrap();
    let a = store
        .lookup_turn_by_platform_message_id("famA", "shared")
        .unwrap()
        .unwrap();
    let b = store
        .lookup_turn_by_platform_message_id("famB", "shared")
        .unwrap()
        .unwrap();
    assert_ne!(a.id, b.id);
}

#[test]
fn reply_to_message_id_persisted() {
    let store = mem();
    store
        .append_turn(AppendTurn::new("fam", 1, "user", "parent").platform_message_id("aaa"))
        .unwrap();
    store
        .append_turn(
            AppendTurn::new("fam", 1, "user", "child reply")
                .platform_message_id("bbb")
                .reply_to_message_id("aaa"),
        )
        .unwrap();
    let reply_to = store
        .conn()
        .query_scalar_string(
            "SELECT reply_to_message_id FROM turns WHERE platform_message_id = ?",
            vec![Value::Text("bbb".to_owned())],
        )
        .unwrap();
    assert_eq!(reply_to.as_deref(), Some("aaa"));
}

#[test]
fn mentions_record_dedupe_empty() {
    let store = mem();
    let turn = store
        .append_turn(AppendTurn::new("fam", 1, "user", "hey"))
        .unwrap();
    store
        .record_mentions(turn.id, &["discord:222", "discord:333"])
        .unwrap();
    assert_eq!(
        store.mentions_for_turn(turn.id).unwrap(),
        ["discord:222", "discord:333"]
    );

    let store = mem();
    let turn = store
        .append_turn(AppendTurn::new("fam", 1, "user", "hey"))
        .unwrap();
    store
        .record_mentions(turn.id, &["discord:222", "discord:222"])
        .unwrap();
    assert_eq!(store.mentions_for_turn(turn.id).unwrap(), ["discord:222"]);

    let store = mem();
    let turn = store
        .append_turn(AppendTurn::new("fam", 1, "user", "no pings"))
        .unwrap();
    store.record_mentions(turn.id, &[]).unwrap();
    assert!(store.mentions_for_turn(turn.id).unwrap().is_empty());
}

// --- alarms ---------------------------------------------------------------

#[test]
fn alarm_crud_and_guards() {
    let store = mem();
    let id = store
        .insert_alarm(
            "aria",
            555,
            "text",
            "2030-01-01T00:00:00+00:00",
            "wake-test",
            None,
        )
        .unwrap();
    assert!(!id.is_empty());
    let pending = store.list_pending_alarms("aria").unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, id);
    assert_eq!(pending[0].reason, "wake-test");
    assert_eq!(pending[0].channel_kind, "text");
    assert!(pending[0].fired_at.is_none());
    assert!(pending[0].cancelled_at.is_none());

    assert!(
        store
            .mark_alarm_fired(&id, &Utc::now().to_rfc3339())
            .unwrap()
    );
    assert!(store.list_pending_alarms("aria").unwrap().is_empty());

    // cancel path
    let id2 = store
        .insert_alarm("aria", 555, "text", "2030-01-01T00:00:00+00:00", "x", None)
        .unwrap();
    assert!(store.cancel_alarm(&id2, &Utc::now().to_rfc3339()).unwrap());
    assert!(store.list_pending_alarms("aria").unwrap().is_empty());
    // unknown id → false
    assert!(
        !store
            .cancel_alarm("no-such-id", &Utc::now().to_rfc3339())
            .unwrap()
    );
}

#[test]
fn alarm_scoped_and_check_and_turn_id() {
    let store = mem();
    store
        .insert_alarm("fam-a", 1, "text", "2030-01-01T00:00:00+00:00", "a", None)
        .unwrap();
    store
        .insert_alarm("fam-b", 2, "text", "2030-01-01T00:00:00+00:00", "b", None)
        .unwrap();
    assert_eq!(store.list_pending_alarms("fam-a").unwrap().len(), 1);
    assert_eq!(store.list_pending_alarms("fam-b").unwrap().len(), 1);

    // CHECK constraint on channel_kind
    assert!(
        store
            .insert_alarm("aria", 555, "other", "2030-01-01T00:00:00+00:00", "x", None)
            .is_err()
    );

    // originating_turn_id round-trip
    let id = store
        .insert_alarm(
            "aria",
            555,
            "voice",
            "2030-01-01T00:00:00+00:00",
            "x",
            Some("turn-42"),
        )
        .unwrap();
    let pending = store.list_pending_alarms("aria").unwrap();
    let row = pending.iter().find(|r| r.id == id).unwrap();
    assert_eq!(row.originating_turn_id.as_deref(), Some("turn-42"));
}

#[test]
fn turns_tool_columns_round_trip() {
    let store = mem();
    let tool_calls = "[{\"id\":\"c1\",\"type\":\"function\"}]";
    let turn = store
        .append_turn(AppendTurn::new("aria", 555, "assistant", "").tool_calls_json(tool_calls))
        .unwrap();
    let stored = store
        .conn()
        .query_scalar_string(
            "SELECT tool_calls_json FROM turns WHERE id = ?",
            vec![Value::Integer(turn.id)],
        )
        .unwrap();
    assert_eq!(stored.as_deref(), Some(tool_calls));
    let tci = store
        .conn()
        .query_scalar_string(
            "SELECT tool_call_id FROM turns WHERE id = ?",
            vec![Value::Integer(turn.id)],
        )
        .unwrap();
    assert!(tci.is_none());

    let turn2 = store
        .append_turn(AppendTurn::new("aria", 555, "tool", "{\"ok\": true}").tool_call_id("c-abc"))
        .unwrap();
    let tci = store
        .conn()
        .query_scalar_string(
            "SELECT tool_call_id FROM turns WHERE id = ?",
            vec![Value::Integer(turn2.id)],
        )
        .unwrap();
    assert_eq!(tci.as_deref(), Some("c-abc"));
}

// --- reactions ------------------------------------------------------------

#[test]
fn reaction_set_upsert_zero_removes() {
    let store = mem();
    store.set_reaction("fam", "m1", "\u{1f44d}", 3).unwrap();
    assert_eq!(
        store
            .reactions_for_messages("fam", &["m1"])
            .unwrap()
            .get("m1"),
        Some(&vec![("\u{1f44d}".to_owned(), 3)])
    );
    store.set_reaction("fam", "m1", "\u{1f44d}", 5).unwrap();
    assert_eq!(
        store
            .reactions_for_messages("fam", &["m1"])
            .unwrap()
            .get("m1"),
        Some(&vec![("\u{1f44d}".to_owned(), 5)])
    );
    store.set_reaction("fam", "m1", "\u{1f44d}", 0).unwrap();
    assert!(
        store
            .reactions_for_messages("fam", &["m1"])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn reaction_batch_order_scope_unknown() {
    let store = mem();
    store.set_reaction("fam", "m1", "\u{1f389}", 2).unwrap();
    store.set_reaction("fam", "m1", "\u{1f44d}", 5).unwrap();
    store
        .set_reaction("fam", "m1", "\u{2764}\u{fe0f}", 2)
        .unwrap();
    let out = store.reactions_for_messages("fam", &["m1"]).unwrap();
    assert_eq!(out["m1"][0], ("\u{1f44d}".to_owned(), 5));
    let tie: std::collections::HashSet<&str> =
        out["m1"][1..].iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(tie, ["\u{1f389}", "\u{2764}\u{fe0f}"].into_iter().collect());

    // scope + unknown
    let store = mem();
    store.set_reaction("famA", "m1", "\u{1f44d}", 1).unwrap();
    store
        .set_reaction("famB", "m1", "\u{2764}\u{fe0f}", 1)
        .unwrap();
    let out = store
        .reactions_for_messages("famA", &["m1", "nope"])
        .unwrap();
    assert!(!out.contains_key("nope"));
    assert_eq!(out["m1"], vec![("\u{1f44d}".to_owned(), 1)]);
}

#[test]
fn bump_reaction_increment_floor_and_noop() {
    let store = mem();
    store.bump_reaction("fam", "m1", "\u{1f44d}", 1).unwrap();
    store.bump_reaction("fam", "m1", "\u{1f44d}", 1).unwrap();
    assert_eq!(
        store.reactions_for_messages("fam", &["m1"]).unwrap()["m1"],
        vec![("\u{1f44d}".to_owned(), 2)]
    );
    store.bump_reaction("fam", "m1", "\u{1f44d}", -2).unwrap();
    assert!(
        store
            .reactions_for_messages("fam", &["m1"])
            .unwrap()
            .is_empty()
    );

    // negative with no row is a noop
    let store = mem();
    store.bump_reaction("fam", "m1", "\u{1f44d}", -1).unwrap();
    assert!(
        store
            .reactions_for_messages("fam", &["m1"])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn clear_reactions_all_or_one() {
    let store = mem();
    store.set_reaction("fam", "m1", "\u{1f44d}", 2).unwrap();
    store
        .set_reaction("fam", "m1", "\u{2764}\u{fe0f}", 1)
        .unwrap();
    store.clear_reactions("fam", "m1", None).unwrap();
    assert!(
        store
            .reactions_for_messages("fam", &["m1"])
            .unwrap()
            .is_empty()
    );

    let store = mem();
    store.set_reaction("fam", "m1", "\u{1f44d}", 2).unwrap();
    store
        .set_reaction("fam", "m1", "\u{2764}\u{fe0f}", 1)
        .unwrap();
    store
        .clear_reactions("fam", "m1", Some("\u{1f44d}"))
        .unwrap();
    assert_eq!(
        store.reactions_for_messages("fam", &["m1"]).unwrap()["m1"],
        vec![("\u{2764}\u{fe0f}".to_owned(), 1)]
    );
}

#[test]
fn reactions_for_messages_uses_single_query() {
    let store = mem();
    for i in 0..5 {
        store
            .set_reaction("fam", &format!("m{i}"), "\u{1f44d}", 1)
            .unwrap();
    }
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_cb = Arc::clone(&seen);
    let cb: TraceCallback = Arc::new(move |sql: &str| seen_cb.lock().unwrap().push(sql.to_owned()));
    store.conn().set_trace_callback(Some(cb));
    let ids: Vec<&str> = ["m0", "m1", "m2", "m3", "m4"].to_vec();
    store.reactions_for_messages("fam", &ids).unwrap();
    store.conn().set_trace_callback(None);
    let selects: Vec<String> = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|s| s.trim_start().to_uppercase().starts_with("SELECT"))
        .cloned()
        .collect();
    assert_eq!(selects.len(), 1, "{selects:?}");
}
