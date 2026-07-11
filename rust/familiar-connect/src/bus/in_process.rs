//! [`InProcessEventBus`] + [`Lifecycle`] + [`Subscription`] (subsystem 01;
//! Python `bus/bus.py`, renamed to `in_process` to avoid
//! `clippy::module_inception` — DESIGN D-R1/D21).
//!
//! Topic-keyed fan-out: each [`InProcessEventBus::subscribe`] creates an isolated
//! queue whose [`BackpressurePolicy`] governs full-queue behaviour. Fan-out is
//! **sequential in registration order** (`for sub in subs { sub.put(ev).await }`),
//! so a full `BLOCK` subscriber back-pressures the publisher and delays delivery
//! to later-registered subscribers — this head-of-line coupling is test-pinned
//! (spec 01 §6). A `Drop` on the [`Subscription`] handle unsubscribes (D7)
//! without changing observable semantics.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::bus::envelope::Event;
use crate::bus::protocols::{BackpressurePolicy, EventBus};

/// Default per-subscription queue bound for the bounded policies (Python
/// `bus.bus._DEFAULT_MAXSIZE`).
const DEFAULT_MAXSIZE: usize = 64;

/// Bus lifecycle states (Python `Lifecycle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    /// Constructed, not yet started.
    Starting,
    /// Running.
    Running,
    /// Shutting down; subscriptions closed.
    Draining,
    /// Stopped; `publish` is refused.
    Stopped,
}

impl Lifecycle {
    /// The wire string value (matches the Python enum member values).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Draining => "draining",
            Self::Stopped => "stopped",
        }
    }
}

/// A bounded ring for the `DROP_OLDEST` / `DROP_NEWEST` policies.
///
/// The producer must be able to evict, which `mpsc` cannot express from the send
/// side, so these two policies get a hand-rolled `Mutex<VecDeque> + Notify` queue
/// (DESIGN §4.4).
struct DropQueue {
    buf: Mutex<VecDeque<Arc<Event>>>,
    notify: Notify,
    cap: usize,
    drop_oldest: bool,
}

impl DropQueue {
    fn new(cap: usize, drop_oldest: bool) -> Self {
        Self {
            buf: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            cap,
            drop_oldest,
        }
    }

    /// Enqueue per policy. Drops are silent (no log, no counter).
    fn put(&self, event: Arc<Event>) {
        let pushed = {
            let mut buf = self.buf.lock().expect("drop queue mutex poisoned");
            if buf.len() >= self.cap {
                if self.drop_oldest {
                    buf.pop_front();
                    buf.push_back(event);
                    true
                } else {
                    // DROP_NEWEST: drop the incoming event.
                    false
                }
            } else {
                buf.push_back(event);
                true
            }
        };
        if pushed {
            self.notify.notify_one();
        }
    }

    fn try_pop(&self) -> Option<Arc<Event>> {
        self.buf
            .lock()
            .expect("drop queue mutex poisoned")
            .pop_front()
    }
}

/// The send end of a subscription's queue, keyed to its policy.
enum Sink {
    Bounded(mpsc::Sender<Arc<Event>>),
    Unbounded(mpsc::UnboundedSender<Arc<Event>>),
    Drop(Arc<DropQueue>),
}

/// The bus-owned half of a subscription: routing topics, the close signal, and
/// the send end of the queue.
struct SubHandle {
    topics: HashSet<String>,
    closed: CancellationToken,
    sink: Sink,
}

impl SubHandle {
    /// Deliver `event` unless the subscription is closed (closed → silent drop).
    async fn put(&self, event: Arc<Event>) {
        if self.closed.is_cancelled() {
            return;
        }
        match &self.sink {
            // `BLOCK` back-pressures here; a dropped receiver (unsubscribe) makes
            // `send` err, which we treat as a silent drop.
            Sink::Bounded(tx) => {
                let _ = tx.send(event).await;
            }
            Sink::Unbounded(tx) => {
                let _ = tx.send(event);
            }
            Sink::Drop(q) => q.put(event),
        }
    }

    fn close(&self) {
        self.closed.cancel();
    }
}

/// The receive end of a subscription's queue.
enum RxKind {
    Bounded(mpsc::Receiver<Arc<Event>>),
    Unbounded(mpsc::UnboundedReceiver<Arc<Event>>),
    Drop(Arc<DropQueue>),
    /// An always-empty, already-closed subscription (for `EventBus` doubles).
    Closed,
}

/// The consumer handle returned by [`InProcessEventBus::subscribe`].
///
/// [`recv`](Self::recv) yields events until the bus closes the subscription and
/// its queue drains, then returns `None`. Dropping the handle unsubscribes
/// (DESIGN D7): further `publish`es to this subscription are silently discarded,
/// fixing the Python no-unsubscribe leak without changing observable semantics.
pub struct Subscription {
    handle: Arc<SubHandle>,
    rx: RxKind,
}

impl Subscription {
    /// A subscription that is already closed and yields nothing — a convenience
    /// for [`EventBus`] test doubles that never deliver.
    #[must_use]
    pub fn closed() -> Self {
        let token = CancellationToken::new();
        token.cancel();
        Self {
            handle: Arc::new(SubHandle {
                topics: HashSet::new(),
                closed: token,
                sink: Sink::Drop(Arc::new(DropQueue::new(1, false))),
            }),
            rx: RxKind::Closed,
        }
    }

    /// Await the next event, or `None` once the subscription is closed **and** its
    /// queue has drained.
    ///
    /// Semantics (spec 01 §9): while open, wait for the next event or the close
    /// signal, whichever comes first; a racing event wins over close (`biased`
    /// select + a drain-first check). On close, all already-queued events are
    /// drained across successive calls before `None` is returned — so a consumer
    /// mid-`recv` when the bus shuts down still receives every queued event.
    pub async fn recv(&mut self) -> Option<Arc<Event>> {
        let closed = self.handle.closed.clone();
        match &mut self.rx {
            RxKind::Bounded(rx) => loop {
                if closed.is_cancelled() {
                    return rx.try_recv().ok();
                }
                tokio::select! {
                    biased;
                    event = rx.recv() => return event,
                    () = closed.cancelled() => {}
                }
            },
            RxKind::Unbounded(rx) => loop {
                if closed.is_cancelled() {
                    return rx.try_recv().ok();
                }
                tokio::select! {
                    biased;
                    event = rx.recv() => return event,
                    () = closed.cancelled() => {}
                }
            },
            RxKind::Drop(q) => loop {
                if let Some(event) = q.try_pop() {
                    return Some(event);
                }
                if closed.is_cancelled() {
                    return q.try_pop();
                }
                tokio::select! {
                    () = q.notify.notified() => {}
                    () = closed.cancelled() => {}
                }
            },
            RxKind::Closed => None,
        }
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        // Unsubscribe: mark closed so later `publish`es to this subscription are
        // silently dropped and its queue stops accumulating (D7).
        self.handle.closed.cancel();
    }
}

/// Topic-keyed pub/sub, in-process only.
///
/// The only [`EventBus`] impl today; hidden behind the trait so a future
/// process-spanning implementation can drop in without touching processors.
pub struct InProcessEventBus {
    lifecycle: Mutex<Lifecycle>,
    subs: Mutex<Vec<Arc<SubHandle>>>,
}

impl Default for InProcessEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl InProcessEventBus {
    /// A fresh bus in the `STARTING` state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            lifecycle: Mutex::new(Lifecycle::Starting),
            subs: Mutex::new(Vec::new()),
        }
    }

    /// The current lifecycle state (Python's public `lifecycle` attribute).
    #[must_use]
    pub fn lifecycle(&self) -> Lifecycle {
        *self.lifecycle.lock().expect("bus lifecycle mutex poisoned")
    }
}

#[async_trait]
impl EventBus for InProcessEventBus {
    async fn start(&self) {
        let mut lifecycle = self.lifecycle.lock().expect("bus lifecycle mutex poisoned");
        // Transition to RUNNING only from STARTING; any other state is a no-op.
        if *lifecycle == Lifecycle::Starting {
            *lifecycle = Lifecycle::Running;
        }
    }

    async fn shutdown(&self) {
        {
            let mut lifecycle = self.lifecycle.lock().expect("bus lifecycle mutex poisoned");
            if *lifecycle == Lifecycle::Stopped {
                return;
            }
            *lifecycle = Lifecycle::Draining;
        }
        let subs = self.subs.lock().expect("bus subs mutex poisoned").clone();
        for sub in &subs {
            sub.close();
        }
        // One event-loop tick for subscribers to observe close before STOPPED
        // (Python `await asyncio.sleep(0)`); shutdown does not wait for consumers.
        tokio::task::yield_now().await;
        *self.lifecycle.lock().expect("bus lifecycle mutex poisoned") = Lifecycle::Stopped;
    }

    async fn publish(&self, event: Event) {
        // Refused only when STOPPED; publishing while STARTING/DRAINING is allowed
        // (lifecycle is bookkeeping, not a gate — spec 01 §2).
        if self.lifecycle() == Lifecycle::Stopped {
            warn!(target: "familiar_connect.bus.bus", "publish after stop: topic={}", event.topic);
            return;
        }
        let event = Arc::new(event);
        // Snapshot the subscription list, then fan out sequentially without
        // holding the lock across `.await`.
        let subs = self.subs.lock().expect("bus subs mutex poisoned").clone();
        for sub in &subs {
            if sub.topics.contains(event.topic.as_str()) {
                sub.put(Arc::clone(&event)).await;
            }
        }
    }

    fn subscribe(
        &self,
        topics: &[&str],
        policy: BackpressurePolicy,
        maxsize: usize,
    ) -> Subscription {
        let cap = if maxsize > 0 {
            maxsize
        } else {
            DEFAULT_MAXSIZE
        };
        let topics: HashSet<String> = topics.iter().map(|t| (*t).to_owned()).collect();
        let closed = CancellationToken::new();
        let (sink, rx) = match policy {
            BackpressurePolicy::Block => {
                let (tx, rx) = mpsc::channel(cap);
                (Sink::Bounded(tx), RxKind::Bounded(rx))
            }
            BackpressurePolicy::Unbounded => {
                let (tx, rx) = mpsc::unbounded_channel();
                (Sink::Unbounded(tx), RxKind::Unbounded(rx))
            }
            BackpressurePolicy::DropOldest | BackpressurePolicy::DropNewest => {
                let q = Arc::new(DropQueue::new(
                    cap,
                    matches!(policy, BackpressurePolicy::DropOldest),
                ));
                (Sink::Drop(Arc::clone(&q)), RxKind::Drop(q))
            }
        };
        let handle = Arc::new(SubHandle {
            topics,
            closed,
            sink,
        });
        self.subs
            .lock()
            .expect("bus subs mutex poisoned")
            .push(Arc::clone(&handle));
        Subscription { handle, rx }
    }
}
