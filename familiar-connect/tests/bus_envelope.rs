//! `Event` / `TurnScope` envelope.
//!
//! `test_event_is_immutable` is not ported: in Rust immutability is a
//! compile-time property — consumers only ever hold `Arc<Event>` and cannot
//! mutate through a shared reference, so there is no runtime equivalent to
//! exercise. See the port summary's skipped list.

use std::time::Duration;

use familiar_connect::bus::TurnScope;
use tokio::time::Instant;

#[tokio::test]
async fn scope_carries_identity() {
    let scope = TurnScope::new("t-1", "chan-42");
    assert_eq!(scope.turn_id, "t-1");
    assert_eq!(scope.session_id, "chan-42");
    // `started_at` is `Instant::now()`; pin that it is a real, past-or-present
    // stamp (a positive `started_at`).
    assert!(scope.started_at <= Instant::now());
    assert!(!scope.is_cancelled());
}

#[tokio::test]
async fn cancel_sets_flag() {
    let scope = TurnScope::new("t-1", "chan-42");
    assert!(!scope.is_cancelled());
    scope.cancel();
    assert!(scope.is_cancelled());
    // Cancel is idempotent.
    scope.cancel();
    assert!(scope.is_cancelled());
}

#[tokio::test]
async fn wait_cancelled_resolves_after_cancel() {
    let scope = TurnScope::new("t-1", "chan-42");
    let canceller = async {
        tokio::time::sleep(Duration::from_millis(10)).await;
        scope.cancel();
    };
    let waiter = async {
        tokio::time::timeout(Duration::from_millis(500), scope.wait_cancelled())
            .await
            .expect("wait_cancelled should resolve within the timeout");
    };
    tokio::join!(waiter, canceller);
}
