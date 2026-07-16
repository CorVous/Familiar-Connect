//! `TwitchSource`: queue → bus drain (subsystem 11/10; Python `sources/twitch.py`).
//!
//! Consumes the queue produced by
//! [`TwitchWatcher`](crate::twitch_watcher::TwitchWatcher), wraps each event in
//! an envelope, and publishes on [`TOPIC_TWITCH_EVENT`]. Unbounded topic policy
//! per plan (Twitch volume is low, dropping a cheer is costly). The source is
//! generic over the queued item so it treats the event as opaque, exactly like
//! the Python `asyncio.Queue[object]`; the whole Twitch pipeline is dormant
//! (nothing constructs a watcher/source, nothing consumes `twitch.event`).

use std::sync::{Arc, Mutex};

use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::mpsc::UnboundedReceiver;
use uuid::Uuid;

use crate::bus::envelope::{Event, payload};
use crate::bus::protocols::EventBus;
use crate::bus::topics::TOPIC_TWITCH_EVENT;

/// Payload for a `twitch.event` envelope: `{familiar_id, twitch: <raw event>}`.
///
/// Generic over the queued item — the drain never inspects it (Python passes
/// the raw object straight through).
#[derive(Clone, Debug)]
pub struct TwitchEventPayload<T> {
    /// The owning familiar id.
    pub familiar_id: String,
    /// The raw Twitch event object drained off the queue.
    pub twitch: T,
}

/// Drains a Twitch event queue onto the bus.
pub struct TwitchSource<T> {
    bus: Arc<dyn EventBus>,
    familiar_id: String,
    queue: AsyncMutex<UnboundedReceiver<T>>,
    seq: Mutex<u64>,
}

impl<T: Send + Sync + 'static> TwitchSource<T> {
    /// The stable source name.
    pub const NAME: &'static str = "twitch";

    /// Build a source draining `queue` onto `bus`.
    #[must_use]
    pub fn new(
        bus: Arc<dyn EventBus>,
        familiar_id: impl Into<String>,
        queue: UnboundedReceiver<T>,
    ) -> Self {
        Self {
            bus,
            familiar_id: familiar_id.into(),
            queue: AsyncMutex::new(queue),
            seq: Mutex::new(0),
        }
    }

    /// Forever loop: drain the queue, publish. Task cancellation is the only
    /// clean exit (the loop also ends if every sender is dropped).
    pub async fn run(&self) {
        let mut rx = self.queue.lock().await;
        while let Some(event) = rx.recv().await {
            self.publish(event).await;
        }
    }

    async fn publish(&self, twitch_event: T) {
        let seq = {
            let mut s = self.seq.lock().expect("twitch source seq mutex poisoned");
            *s += 1;
            *s
        };
        let event_id = format!("twitch-{}", &Uuid::new_v4().simple().to_string()[..12]);
        let event = Event {
            event_id: event_id.clone(),
            turn_id: event_id,
            session_id: format!("twitch:{}", self.familiar_id),
            parent_event_ids: Vec::new(),
            topic: TOPIC_TWITCH_EVENT.to_owned(),
            timestamp: chrono::Utc::now(),
            sequence_number: seq,
            payload: payload(TwitchEventPayload {
                familiar_id: self.familiar_id.clone(),
                twitch: twitch_event,
            }),
        };
        self.bus.publish(event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::{TwitchEventPayload, TwitchSource};
    use crate::bus::in_process::InProcessEventBus;
    use crate::bus::protocols::{BackpressurePolicy, EventBus};
    use crate::bus::topics::TOPIC_TWITCH_EVENT;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    #[derive(Clone, Debug)]
    struct FakeTwitchEvent {
        kind: String,
        #[allow(dead_code)]
        detail: String,
    }

    #[tokio::test]
    async fn drains_queue_onto_bus() {
        let bus = Arc::new(InProcessEventBus::new());
        bus.start().await;
        let (tx, rx) = mpsc::unbounded_channel::<FakeTwitchEvent>();
        let source = Arc::new(TwitchSource::new(bus.clone(), "fam", rx));

        let mut sub = bus.subscribe(&[TOPIC_TWITCH_EVENT], BackpressurePolicy::Unbounded, 0);
        let producer = tokio::spawn({
            let s = Arc::clone(&source);
            async move { s.run().await }
        });
        tokio::task::yield_now().await;

        tx.send(FakeTwitchEvent {
            kind: "follow".to_owned(),
            detail: "a".to_owned(),
        })
        .unwrap();
        tx.send(FakeTwitchEvent {
            kind: "cheer".to_owned(),
            detail: "b".to_owned(),
        })
        .unwrap();

        let mut received = Vec::new();
        for _ in 0..2 {
            let ev = timeout(Duration::from_secs(1), sub.recv())
                .await
                .expect("event within timeout")
                .expect("event present");
            received.push(ev);
        }
        producer.abort();
        bus.shutdown().await;

        let kinds: HashSet<String> = received
            .iter()
            .map(|ev| {
                ev.payload
                    .downcast_ref::<TwitchEventPayload<FakeTwitchEvent>>()
                    .expect("twitch payload")
                    .twitch
                    .kind
                    .clone()
            })
            .collect();
        assert_eq!(
            kinds,
            HashSet::from(["follow".to_owned(), "cheer".to_owned()])
        );
        assert!(received.iter().all(|ev| ev.topic == TOPIC_TWITCH_EVENT));
        assert!(received.iter().all(|ev| ev.session_id == "twitch:fam"));
    }

    #[tokio::test]
    async fn run_exits_cleanly_on_cancel() {
        let bus = Arc::new(InProcessEventBus::new());
        bus.start().await;
        let (_tx, rx) = mpsc::unbounded_channel::<FakeTwitchEvent>();
        let source = Arc::new(TwitchSource::new(bus.clone(), "fam", rx));
        let producer = tokio::spawn({
            let s = Arc::clone(&source);
            async move { s.run().await }
        });
        tokio::task::yield_now().await;
        // Simulate shutdown — the run task must be cancellable cleanly.
        producer.abort();
        assert!(producer.await.unwrap_err().is_cancelled());
        bus.shutdown().await;
    }

    #[tokio::test]
    async fn envelope_fields_and_monotonic_seq() {
        let bus = Arc::new(InProcessEventBus::new());
        bus.start().await;
        let (tx, rx) = mpsc::unbounded_channel::<FakeTwitchEvent>();
        let source = Arc::new(TwitchSource::new(bus.clone(), "aria", rx));
        let mut sub = bus.subscribe(&[TOPIC_TWITCH_EVENT], BackpressurePolicy::Unbounded, 0);
        let producer = tokio::spawn({
            let s = Arc::clone(&source);
            async move { s.run().await }
        });
        tokio::task::yield_now().await;
        for i in 0..3 {
            tx.send(FakeTwitchEvent {
                kind: format!("k{i}"),
                detail: String::new(),
            })
            .unwrap();
        }
        let mut seqs = Vec::new();
        for _ in 0..3 {
            let ev = timeout(Duration::from_secs(1), sub.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(ev.turn_id, ev.event_id);
            assert!(ev.event_id.starts_with("twitch-"));
            assert!(ev.parent_event_ids.is_empty());
            seqs.push(ev.sequence_number);
        }
        producer.abort();
        bus.shutdown().await;
        assert_eq!(seqs, vec![1, 2, 3]);
    }
}
