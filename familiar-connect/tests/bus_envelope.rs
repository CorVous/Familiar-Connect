//! Ported from `tests/test_bus_envelope.py` — `Event` / `TurnScope` envelope.
//!
//! `test_event_is_immutable` (Python `setattr` raising on a frozen dataclass) is
//! not ported: in Rust immutability is a compile-time property — consumers only
//! ever hold `Arc<Event>` and cannot mutate through a shared reference, so there
//! is no runtime equivalent to exercise. See the port summary's skipped list.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use chrono::Utc;
use familiar_connect::bus::TurnScope;
use familiar_connect::bus::envelope::{Event, payload};
use tokio::time::Instant;

#[test]
fn event_has_core_fields() {
    let ts = Utc::now();
    let map: HashMap<String, String> = HashMap::from([("text".to_owned(), "hi".to_owned())]);
    let ev = Event {
        event_id: "e-1".to_owned(),
        turn_id: "t-1".to_owned(),
        session_id: "chan-42".to_owned(),
        parent_event_ids: Vec::new(),
        topic: "discord.text".to_owned(),
        timestamp: ts,
        sequence_number: 0,
        payload: payload(map.clone()),
    };
    assert_eq!(ev.event_id, "e-1");
    assert_eq!(ev.turn_id, "t-1");
    assert_eq!(ev.session_id, "chan-42");
    assert!(ev.parent_event_ids.is_empty());
    assert_eq!(ev.topic, "discord.text");
    assert_eq!(ev.timestamp, ts);
    assert_eq!(ev.sequence_number, 0);
    // Payload carried through unchanged (Python asserts `ev.payload == {"text": "hi"}`).
    assert_eq!(
        ev.payload.downcast_ref::<HashMap<String, String>>(),
        Some(&map)
    );
}

#[test]
fn parent_event_ids_carries_lineage() {
    let ev = Event {
        event_id: "e-2".to_owned(),
        turn_id: "t-1".to_owned(),
        session_id: "chan-42".to_owned(),
        parent_event_ids: vec!["e-1".to_owned()],
        topic: "derived".to_owned(),
        timestamp: Utc::now(),
        sequence_number: 1,
        payload: payload(()),
    };
    // Carries the lineage verbatim.
    assert_eq!(ev.parent_event_ids, vec!["e-1".to_owned()]);
    // Python's `test_parent_event_ids_is_tuple_for_hashability` pins
    // `isinstance(ev.parent_event_ids, tuple)` (spec 01 §11): the lineage collection
    // must be a *hashable* type. A frozen dataclass derives `__hash__` from its fields,
    // so a `list` field would break hashing an `Event` at runtime while a `tuple` does
    // not. The Rust analog is that the field's concrete type is `Hash + Eq` — pinned by
    // using the value as a `HashSet<Vec<String>>` key. This fails to compile if the
    // field's type ever stops being hashable (or ceases to be `Vec<String>`).
    let mut lineage_set: HashSet<Vec<String>> = HashSet::new();
    lineage_set.insert(ev.parent_event_ids);
    assert!(lineage_set.contains(&vec!["e-1".to_owned()]));
}

#[tokio::test]
async fn scope_carries_identity() {
    let scope = TurnScope::new("t-1", "chan-42");
    assert_eq!(scope.turn_id, "t-1");
    assert_eq!(scope.session_id, "chan-42");
    // D12: `started_at` is `Instant::now()`; pin that it is a real, past-or-present
    // stamp (replaces Python's `started_at > 0`).
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
