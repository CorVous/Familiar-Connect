//! Ported from `tests/test_bus_router.py` — `TurnRouter` barge-in semantics.

use std::sync::Arc;
use std::time::Duration;

use familiar_connect::bus::TurnRouter;
use tokio::time::Instant;

#[tokio::test]
async fn first_turn_returns_new_scope() {
    let router = TurnRouter::new();
    let scope = router.begin_turn("chan-1", "t-1");
    assert_eq!(scope.turn_id, "t-1");
    assert_eq!(scope.session_id, "chan-1");
    assert!(!scope.is_cancelled());
}

#[tokio::test]
async fn second_turn_cancels_first_same_session() {
    let router = TurnRouter::new();
    let first = router.begin_turn("chan-1", "t-1");
    let second = router.begin_turn("chan-1", "t-2");
    assert!(first.is_cancelled());
    assert!(!second.is_cancelled());
    let active = router.active_scope("chan-1").expect("active scope present");
    assert!(Arc::ptr_eq(&active, &second));
}

#[tokio::test]
async fn turns_in_different_sessions_are_independent() {
    let router = TurnRouter::new();
    let a = router.begin_turn("chan-1", "a");
    let b = router.begin_turn("chan-2", "b");
    assert!(!a.is_cancelled());
    assert!(!b.is_cancelled());
    // Cancelling one does not disturb the other.
    a.cancel();
    assert!(!b.is_cancelled());
}

#[tokio::test]
async fn cancel_propagates_within_50ms() {
    let router = TurnRouter::new();
    let scope = router.begin_turn("s", "t-1");
    let worker = {
        let scope = Arc::clone(&scope);
        tokio::spawn(async move {
            let started = Instant::now();
            scope.wait_cancelled().await;
            started.elapsed()
        })
    };
    tokio::task::yield_now().await;
    router.begin_turn("s", "t-2"); // cancels t-1
    let elapsed = tokio::time::timeout(Duration::from_secs(1), worker)
        .await
        .expect("worker resolves within the timeout")
        .expect("worker task joins cleanly");
    assert!(elapsed < Duration::from_millis(50), "elapsed = {elapsed:?}");
}

#[tokio::test]
async fn end_turn_clears_active() {
    let router = TurnRouter::new();
    let scope = router.begin_turn("s", "t-1");
    router.end_turn(&scope);
    assert!(router.active_scope("s").is_none());
}

#[tokio::test]
async fn end_turn_is_idempotent() {
    let router = TurnRouter::new();
    let scope = router.begin_turn("s", "t-1");
    router.end_turn(&scope);
    router.end_turn(&scope); // must not panic
    assert!(router.active_scope("s").is_none());
}

#[tokio::test]
async fn end_stale_scope_doesnt_clear_newer() {
    let router = TurnRouter::new();
    let old = router.begin_turn("s", "t-1");
    let new = router.begin_turn("s", "t-2");
    router.end_turn(&old);
    let active = router.active_scope("s").expect("active scope present");
    assert!(Arc::ptr_eq(&active, &new));
}

#[tokio::test]
async fn shutdown_cancels_all_active_turns() {
    let router = TurnRouter::new();
    let a = router.begin_turn("s1", "a");
    let b = router.begin_turn("s2", "b");
    router.shutdown();
    assert!(a.is_cancelled());
    assert!(b.is_cancelled());
}
