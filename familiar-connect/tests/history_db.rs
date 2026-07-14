//! Integration tests for the single-owner DB actor (`history::db`).
//!
//! Re-pins the Python `test_turso_compat.py` thread-affinity invariant as "all
//! statements execute on the one owner thread, never the caller's". The
//! `reopen()` / `_conn()` escape-hatch tests are intentionally not ported (the
//! Rust design drops them — see spec 03 "Do not port").

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

use familiar_connect::history::db::{Db, TraceCallback, Value};

#[test]
fn all_statements_execute_on_one_owner_thread() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(&dir.path().join("thread.db")).unwrap();
    let seen: Arc<Mutex<HashSet<ThreadId>>> = Arc::new(Mutex::new(HashSet::new()));
    let seen_cb = Arc::clone(&seen);
    let cb: TraceCallback = Arc::new(move |_sql: &str| {
        seen_cb.lock().unwrap().insert(thread::current().id());
    });
    db.set_trace_callback(Some(cb));

    db.execute("CREATE TABLE t (id INTEGER)", vec![]).unwrap();
    let db = Arc::new(db);
    let worker_db = Arc::clone(&db);
    let worker = thread::spawn(move || {
        worker_db
            .execute("INSERT INTO t VALUES (1)", vec![])
            .unwrap();
        worker_db
            .query_map("SELECT id FROM t", vec![], |r| r.get::<_, i64>(0))
            .unwrap();
    });
    worker.join().unwrap();

    let (count, has_caller) = {
        let guard = seen.lock().unwrap();
        (guard.len(), guard.contains(&thread::current().id()))
    };
    assert_eq!(count, 1, "every statement runs on exactly one thread");
    assert!(!has_caller, "not the caller's thread");
}

#[test]
fn cursor_like_ops_from_worker_thread() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Db::open(&dir.path().join("cur.db")).unwrap());
    db.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
        .unwrap();
    let id = db
        .execute_returning_id("INSERT INTO t (v) VALUES ('a')", vec![])
        .unwrap();
    assert!(id > 0);

    let worker_db = Arc::clone(&db);
    let handle = thread::spawn(move || {
        let one = worker_db
            .query_scalar_string("SELECT v FROM t", vec![])
            .unwrap();
        let all = worker_db
            .query_map("SELECT v FROM t", vec![], |r| r.get::<_, String>(0))
            .unwrap();
        let updated = worker_db
            .execute("UPDATE t SET v='b' WHERE v='a'", vec![])
            .unwrap();
        (one, all, updated)
    });
    let (one, all, updated) = handle.join().unwrap();
    assert_eq!(one.as_deref(), Some("a"));
    assert_eq!(all, vec!["a".to_owned()]);
    assert_eq!(updated, 1);
}

#[test]
fn file_backed_shares_data_across_threads() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Db::open(&dir.path().join("shared.db")).unwrap());
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", vec![])
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", vec![]).unwrap();

    let worker_db = Arc::clone(&db);
    let handle = thread::spawn(move || {
        worker_db
            .query_scalar_i64("SELECT id FROM t", vec![])
            .unwrap()
    });
    assert_eq!(handle.join().unwrap(), Some(1));
}

#[test]
fn calls_after_close_error() {
    let db = Db::open_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", vec![]).unwrap();
    db.close();
    assert!(
        db.execute("INSERT INTO t VALUES (1)", vec![Value::Integer(1)])
            .is_err()
    );
    assert!(db.query_scalar_i64("SELECT 1", vec![]).is_err());
    // close is idempotent
    db.close();
}
