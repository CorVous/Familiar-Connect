//! Ported from `tests/test_bus_protocols.py` — seam conformance + policy set.
//!
//! Python's `isinstance()` structural checks become compile-time trait bounds:
//! `test_missing_method_rejects_structural_check` has no runtime analogue (a type
//! missing `run`/`handle` simply fails to implement the trait and would not
//! compile), so it is not ported — see the port summary's skipped list.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use familiar_connect::bus::envelope::Event;
use familiar_connect::bus::protocols::{BackpressurePolicy, EventBus, Processor, StreamSource};

// The seams store their identity (mirroring the Python `name` / `topics`
// attributes), so `name()` / `topics()` return borrowed fields rather than
// literals — the realistic pattern the subsystem 06/10 impls will follow.
struct DummySource {
    name: String,
}

#[async_trait]
impl StreamSource for DummySource {
    fn name(&self) -> &str {
        &self.name
    }
    async fn run(&self, _bus: Arc<dyn EventBus>) {}
}

struct DummyProcessor {
    name: String,
    topics: Vec<&'static str>,
}

#[async_trait]
impl Processor for DummyProcessor {
    fn name(&self) -> &str {
        &self.name
    }
    fn topics(&self) -> &[&str] {
        &self.topics
    }
    async fn handle(&self, _event: Arc<Event>, _bus: &dyn EventBus) -> anyhow::Result<()> {
        Ok(())
    }
}

#[test]
fn dummy_source_satisfies_trait() {
    // Structural conformance is enforced at compile time; using the value as a
    // trait object proves it satisfies the seam (Python `isinstance(_, StreamSource)`).
    let source = DummySource {
        name: "dummy".to_owned(),
    };
    let source: &dyn StreamSource = &source;
    assert_eq!(source.name(), "dummy");
}

#[test]
fn dummy_processor_satisfies_trait() {
    let processor = DummyProcessor {
        name: "dummy".to_owned(),
        topics: vec!["discord.text"],
    };
    let processor: &dyn Processor = &processor;
    assert_eq!(processor.name(), "dummy");
    assert_eq!(processor.topics(), ["discord.text"].as_slice());
}

#[test]
fn all_policies_listed() {
    // Exactly these four policies exist (Python pins the four member names).
    assert_eq!(BackpressurePolicy::ALL.len(), 4);
    let names: HashSet<&str> = BackpressurePolicy::ALL.iter().map(|p| p.as_str()).collect();
    assert_eq!(
        names,
        HashSet::from(["block", "drop_oldest", "drop_newest", "unbounded"])
    );
}
