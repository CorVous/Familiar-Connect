//! Familiar-Connect — Rust port of the Python Discord voice AI companion bot.
//!
//! Modules mirror the Python `familiar_connect` package layout; every module maps
//! to exactly one porting-spec subsystem (01–11). Stubs only — the porting agents
//! fill them in. See `rust/DESIGN.md` for the module map, port order (topological
//! layers), and the cross-cutting conventions that all porters must follow.

// --- subsystem 01: bus + diagnostics + leaf utilities ---
pub mod bus;
pub mod diagnostics;
pub mod log_style;
pub mod macros;

// --- subsystem 02: config + identity ---
pub mod config;
pub mod familiar;
pub mod identity;
pub mod prompt_fill;
pub mod structured_output;
pub mod subscriptions;

// --- subsystem 03: history store ---
pub mod history;

// --- subsystem 04: embedding + sleep ---
pub mod embedding;
pub mod sleep;

// --- subsystem 05: context assembly ---
pub mod budget;
pub mod context;
pub mod focus;

// --- subsystems 06 + 07: responders + background workers (both under processors/) ---
pub mod processors;
pub mod silence;
pub mod typing_interrupt;

// --- subsystem 08: llm + tools ---
pub mod llm;
pub mod structured_request;
pub mod tools;

// --- subsystem 09: audio path ---
pub mod sentence_streamer;
pub mod stt;
pub mod tts;
pub mod tts_player;
pub mod voice;

// --- subsystem 10: discord shell + cli ---
pub mod bot;
pub mod cli;
pub mod commands;
pub mod sources;

// --- subsystem 11: twitch + activities ---
pub mod activities;
pub mod twitch;
pub mod twitch_watcher;

// --- port-introduced shared helpers (no direct Python source) ---
pub mod support;
