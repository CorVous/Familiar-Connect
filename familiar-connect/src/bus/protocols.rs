//! Bus seams: [`BackpressurePolicy`] + the [`EventBus`] / [`StreamSource`] /
//! [`Processor`] traits (subsystem 01).
//!
//! Kept separate from the concrete [`InProcessEventBus`](crate::bus::InProcessEventBus)
//! so a future process-spanning `EventBus` drops in without touching processor
//! code. Capability contracts are Rust traits, so conformance is enforced at
//! compile time.

use std::sync::Arc;

use async_trait::async_trait;

use crate::bus::envelope::Event;
use crate::bus::in_process::Subscription;

/// Per-subscription behaviour when a subscriber's queue is full.
///
/// The wire string values (`"block"`, …) are a stable contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackpressurePolicy {
    /// `publish` awaits until space frees — never loses data. The default.
    #[default]
    Block,
    /// Evict the oldest queued event, enqueue the new one (freshness beats
    /// completeness; e.g. audio).
    DropOldest,
    /// Drop the incoming event when full (a downstream stall must not affect
    /// other subscribers).
    DropNewest,
    /// No bound; the caller accepts the memory risk.
    Unbounded,
}

impl BackpressurePolicy {
    /// Every policy variant, in declaration order. Pins "exactly these four
    /// policies exist".
    pub const ALL: [Self; 4] = [
        Self::Block,
        Self::DropOldest,
        Self::DropNewest,
        Self::Unbounded,
    ];

    /// The wire string value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::DropOldest => "drop_oldest",
            Self::DropNewest => "drop_newest",
            Self::Unbounded => "unbounded",
        }
    }
}

/// Topic-addressed pub/sub surface. **The** swappable seam — a future
/// cross-process bus drops in here.
#[async_trait]
pub trait EventBus: Send + Sync {
    /// Transition `STARTING → RUNNING` (idempotent).
    async fn start(&self);
    /// Drain: close every subscription, then stop.
    async fn shutdown(&self);
    /// Fan an event out to every matching subscription, in registration order.
    async fn publish(&self, event: Event);
    /// Subscribe to `topics`. `maxsize == 0` means "default for policy".
    fn subscribe(
        &self,
        topics: &[&str],
        policy: BackpressurePolicy,
        maxsize: usize,
    ) -> Subscription;
}

/// Produces events onto the bus.
#[async_trait]
pub trait StreamSource: Send + Sync {
    /// Stable identifier for logging/registration.
    fn name(&self) -> &str;
    /// Run until the bus drains; return cleanly on cancel.
    async fn run(&self, bus: Arc<dyn EventBus>);
}

/// Subscribes to one or more topics; optionally re-publishes.
///
/// `handle` returns a `Result`, and the dispatch loop logs-and-continues on
/// `Err` rather than swallowing errors inside the processor.
#[async_trait]
pub trait Processor: Send + Sync {
    /// Stable identifier for logging/registration.
    fn name(&self) -> &str;
    /// Topics consulted once at registration (no dynamic subscription in v1).
    fn topics(&self) -> &[&str];
    /// Handle a single event. An `Err` is logged and swallowed by the dispatcher.
    async fn handle(&self, event: Arc<Event>, bus: &dyn EventBus) -> anyhow::Result<()>;
}
