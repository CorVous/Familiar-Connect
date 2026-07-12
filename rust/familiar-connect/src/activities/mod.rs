//! Activities: `activities.toml` loader + the global-absence `ActivityEngine`
//! state machine (subsystem 11; Python `activities/`).
//!
//! Re-exports mirror the Python `activities/__init__.py` public surface so
//! consumers (subsystems 06/07/08/10) import from `crate::activities::` rather
//! than reaching into the submodules. `GateAction` / `GateDecision` are the
//! responder-shared types (`crate::processors`) the engine's gate returns.

pub mod config;
pub mod engine;

pub use config::{ActivitiesConfig, ActivityType, SLEEP_TYPE_ID, load_activities_config};
pub use engine::{
    ACTIVITY_RETURN_MODE, ActivityEngine, RETURN_TURN_MARKER_PREFIX, SLEEP_RETURN_MODE,
};

pub use crate::processors::{GateAction, GateDecision};
