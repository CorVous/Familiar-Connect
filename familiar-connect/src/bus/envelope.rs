//! Event envelope + per-turn cancel scope (subsystem 01).
//!
//! [`Event`] is the immutable, topic-addressed envelope the bus fans out; sharing
//! is safe because consumers only ever observe it behind `Arc<Event>` (read-only).
//! [`TurnScope`] is the per-turn barge-in handle, backed by a
//! `tokio_util::sync::CancellationToken`: `cancel()` is
//! idempotent and `wait_cancelled()` is level-triggered.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Opaque, type-erased event payload.
///
/// A type-erased `Arc<dyn Any + Send + Sync>`, so the bus stays fully
/// payload-agnostic: any producer attaches any concrete payload via
/// [`payload`] / `Arc::new`, and consumers recover it with `downcast_ref`.
pub type Payload = Arc<dyn Any + Send + Sync>;

/// Wrap any value as an opaque [`Payload`].
#[must_use]
pub fn payload<T: Any + Send + Sync>(value: T) -> Payload {
    Arc::new(value)
}

/// Source of unique [`TurnScope`] identities (used by the router's identity guard).
static SCOPE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Immutable, topic-addressed event envelope.
///
/// Deep immutability is a compile-time property in Rust: consumers receive the
/// event as `Arc<Event>` and cannot mutate through a shared reference.
#[derive(Clone)]
pub struct Event {
    /// Unique per publish.
    pub event_id: String,
    /// Scopes derived work; matches the enclosing [`TurnScope`].
    pub turn_id: String,
    /// Usually the channel id (`"discord:{id}"` / `"voice:{id}"`).
    pub session_id: String,
    /// Lineage; empty for source events.
    pub parent_event_ids: Vec<String>,
    /// Routing key (see [`crate::bus::topics`]).
    pub topic: String,
    /// `Utc::now()` at construction.
    pub timestamp: DateTime<Utc>,
    /// Monotonic **per source instance**; not globally monotonic.
    pub sequence_number: u64,
    /// Opaque payload; the bus never inspects it.
    pub payload: Payload,
}

/// Per-turn cancellation handle.
///
/// `cancel()` is idempotent; `wait_cancelled()` is level-triggered — a waiter
/// that starts *after* cancellation returns immediately.
pub struct TurnScope {
    /// The turn this scope guards.
    pub turn_id: String,
    /// The session this scope belongs to.
    pub session_id: String,
    /// Creation time. The this is `Instant::now` unconditionally.
    pub started_at: Instant,
    /// Unique identity; the router's `end_turn` compares this, **not** `turn_id`
    /// (two scopes may legitimately share a `turn_id`).
    id: u64,
    token: CancellationToken,
}

impl TurnScope {
    /// Register a fresh, un-cancelled scope.
    #[must_use]
    pub fn new(turn_id: impl Into<String>, session_id: impl Into<String>) -> Self {
        Self {
            turn_id: turn_id.into(),
            session_id: session_id.into(),
            started_at: Instant::now(),
            id: SCOPE_SEQ.fetch_add(1, Ordering::Relaxed),
            token: CancellationToken::new(),
        }
    }

    /// Signal cancellation. Idempotent.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Whether this scope has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Block until [`cancel`](Self::cancel). Level-triggered.
    pub async fn wait_cancelled(&self) {
        self.token.cancelled().await;
    }

    /// This scope's unique identity (router identity guard).
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// A clone of the underlying cancellation token, for `select!` in pipeline
    /// stages that race work against barge-in.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.token.clone()
    }
}
