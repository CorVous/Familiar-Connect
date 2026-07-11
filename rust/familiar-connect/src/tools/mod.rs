//! Agentic machinery + shipped tools (subsystem 08; Python `tools/`).
//!
//! registry, loop, builtins, alarms, silent, shift_focus, read_channel,
//! view_image, start_activity. `agentic` is Python `tools/loop.py` (`loop` is a
//! Rust keyword).

pub mod agentic;
pub mod alarm;
pub mod builtins;
pub mod channel_view;
pub mod image;
pub mod image_compress;
pub mod image_describe;
pub mod read_channel;
pub mod registry;
pub mod scheduler;
pub mod shift_focus;
pub mod silent;
pub mod start_activity;
pub mod waker;
