//! Per-session turn routing + cancel-prior-scope barge-in (subsystem 01;
//! Python `bus/router.py`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::bus::envelope::TurnScope;

/// Cancels the prior turn in a session before registering a new one; sessions
/// are fully independent (Python `TurnRouter`).
///
/// Thread-safe: the Python original relied on the GIL + a single event loop, but
/// the Rust runtime is multi-threaded, so the active-scope map is behind a
/// `Mutex` (DESIGN §4.4). Observable semantics are unchanged.
#[derive(Default)]
pub struct TurnRouter {
    active: Mutex<HashMap<String, Arc<TurnScope>>>,
}

impl TurnRouter {
    /// A router with no active turns.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel any active turn in `session_id`, then register and return a fresh
    /// scope. The prior scope is cancelled **before** the new one is registered.
    pub fn begin_turn(&self, session_id: &str, turn_id: &str) -> Arc<TurnScope> {
        let mut active = self.active.lock().expect("turn router mutex poisoned");
        if let Some(prior) = active.get(session_id) {
            prior.cancel();
        }
        let scope = Arc::new(TurnScope::new(turn_id, session_id));
        active.insert(session_id.to_owned(), Arc::clone(&scope));
        scope
    }

    /// Clear `scope` from active — only if the *identical* scope is still active
    /// (identity guard via [`TurnScope::id`], not `turn_id` equality). No-op
    /// otherwise, so ending a stale (superseded) scope never clears the newer one.
    pub fn end_turn(&self, scope: &TurnScope) {
        let mut active = self.active.lock().expect("turn router mutex poisoned");
        if active
            .get(&scope.session_id)
            .is_some_and(|a| a.id() == scope.id())
        {
            active.remove(&scope.session_id);
        }
    }

    /// The active scope for `session_id`, if any.
    #[must_use]
    pub fn active_scope(&self, session_id: &str) -> Option<Arc<TurnScope>> {
        self.active
            .lock()
            .expect("turn router mutex poisoned")
            .get(session_id)
            .cloned()
    }

    /// Cancel every active turn. The map is **not** cleared — post-shutdown
    /// inspection is a documented feature.
    pub fn shutdown(&self) {
        for scope in self
            .active
            .lock()
            .expect("turn router mutex poisoned")
            .values()
        {
            scope.cancel();
        }
    }
}
