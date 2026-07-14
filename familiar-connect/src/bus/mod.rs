//! In-process event bus + turn router (subsystem 01; Python `bus/`).
//!
//! Topic-keyed in-process fan-out: sources publish topic-addressed [`Event`]
//! envelopes; processors consume them via per-subscription queues with
//! per-subscription [`BackpressurePolicy`]. [`TurnRouter`]/[`TurnScope`] express
//! barge-in. This subsystem imports nothing from any other subsystem (layer 0);
//! nearly everything else imports it. See `rust/DESIGN.md` §4.4 and
//! `rust/specs/01-bus-and-diagnostics.md`.

pub mod envelope;
pub mod in_process;
pub mod protocols;
pub mod router;
pub mod topics;

pub use envelope::{Event, TurnScope};
pub use in_process::{InProcessEventBus, Lifecycle, Subscription};
pub use protocols::{BackpressurePolicy, EventBus, Processor, StreamSource};
pub use router::TurnRouter;
