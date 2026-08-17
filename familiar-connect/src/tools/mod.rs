//! Agentic machinery + shipped tools (subsystem 08).
//!
//! registry, loop, builtins, alarms, silent, shift_focus, read_channel,
//! view_image, start_activity. The module is named `agentic` because `loop` is
//! a Rust keyword. The `view_image` trio (`image`, `image_compress`) is gated on
//! the `images` feature; `image_policy` (the fetch-boundary URL gate) is not —
//! `[tools]` config validation reaches it in every build.

pub mod agentic;
pub mod alarm;
pub mod builtins;
pub mod channel_view;
#[cfg(feature = "images")]
pub mod image;
#[cfg(feature = "images")]
pub mod image_compress;
pub mod image_describe;
pub mod image_policy;
pub mod read_channel;
pub mod registry;
pub mod scheduler;
pub mod shift_focus;
pub mod silent;
pub mod start_activity;
pub mod waker;
