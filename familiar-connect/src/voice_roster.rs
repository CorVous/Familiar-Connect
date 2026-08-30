//! Live voice-call roster: who is in the call + a decaying join/leave log
//! (subsystem 05).
//!
//! Standalone and **never feature-gated** so both sides can reach it: the
//! `discord` gateway glue (`bot.rs`) writes membership, the ungated prompt layer
//! (`context::layers::VoiceRosterLayer`) reads it. Same seam shape as
//! [`crate::focus::FocusManager`] — one concrete `Arc`-shared type, no
//! second copy that could drift.
//!
//! Events decay: an entry older than the configured window is pruned on the next
//! read, and the log is hard-capped so a long call with churn cannot grow it
//! without bound. Every state change (including a decay eviction) bumps
//! `revision`, which is the whole invalidation signal the layer's cache key needs
//! — no clock in the key, so a quiet call re-uses the cached render.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::identity::Author;

/// Injectable monotonic clock returning seconds (mirrors [`crate::focus::Clock`]).
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// Default decay window for join/leave events.
pub const DEFAULT_EVENT_WINDOW_SECONDS: f64 = 120.0;

/// Hard cap on retained events — memory bound under join/leave churn.
pub const MAX_EVENTS: usize = 16;

/// What happened to a member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RosterEventKind {
    /// Member entered the call.
    Joined,
    /// Member left the call.
    Left,
}

/// One timestamped roster transition.
#[derive(Clone, Debug, PartialEq)]
pub struct RosterEvent {
    /// Joined / left.
    pub kind: RosterEventKind,
    /// Member label at event time ([`Author::label`]).
    pub label: String,
    /// Clock reading when recorded.
    pub at: f64,
}

/// Pruned read of the roster: current members (join order) + live events.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RosterView {
    /// Member labels, join order.
    pub members: Vec<String>,
    /// Undecayed events, oldest first.
    pub events: Vec<RosterEvent>,
    /// Monotonic state counter — the cache-key signal.
    pub revision: u64,
}

/// Stored member plus its arrival ordinal (render order).
struct Slot {
    author: Author,
    seq: u64,
}

struct State {
    members: BTreeMap<i64, Slot>,
    events: VecDeque<RosterEvent>,
    revision: u64,
    next_seq: u64,
}

/// Shared live-call roster.
pub struct VoiceRoster {
    state: Mutex<State>,
    clock: Clock,
    event_window_seconds: f64,
}

fn monotonic_clock() -> Clock {
    let start = Instant::now();
    Arc::new(move || start.elapsed().as_secs_f64())
}

impl Default for VoiceRoster {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceRoster {
    /// Empty roster, default window, `Instant`-based clock.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                members: BTreeMap::new(),
                events: VecDeque::new(),
                revision: 0,
                next_seq: 0,
            }),
            clock: monotonic_clock(),
            event_window_seconds: DEFAULT_EVENT_WINDOW_SECONDS,
        }
    }

    /// Builder: decay window in seconds.
    #[must_use]
    pub const fn with_event_window_seconds(mut self, seconds: f64) -> Self {
        self.event_window_seconds = seconds;
        self
    }

    /// Builder: inject a clock.
    #[must_use]
    pub fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    /// Record an arrival. A repeat update for a member already in the call is
    /// not re-narrated (voice-state updates also fire on mute / deafen / camera);
    /// a changed [`Author`] still bumps the revision so the line re-renders.
    pub fn member_joined(&self, user_id: i64, author: Author) {
        let now = (self.clock)();
        let mut st = self.lock();
        st.prune(now, self.event_window_seconds);
        if let Some(slot) = st.members.get_mut(&user_id) {
            if slot.author == author {
                return;
            }
            slot.author = author;
        } else {
            let label = author.label();
            let seq = st.next_seq;
            st.next_seq += 1;
            st.members.insert(user_id, Slot { author, seq });
            st.push_event(RosterEvent {
                kind: RosterEventKind::Joined,
                label,
                at: now,
            });
        }
        st.revision += 1;
    }

    /// Record a departure; unknown ids are ignored.
    pub fn member_left(&self, user_id: i64) {
        let now = (self.clock)();
        let mut st = self.lock();
        st.prune(now, self.event_window_seconds);
        let Some(slot) = st.members.remove(&user_id) else {
            return;
        };
        st.push_event(RosterEvent {
            kind: RosterEventKind::Left,
            label: slot.author.label(),
            at: now,
        });
        st.revision += 1;
    }

    /// Replace membership wholesale without narrating (bot joining a call in
    /// progress: nobody "just joined").
    pub fn snapshot(&self, members: impl IntoIterator<Item = (i64, Author)>) {
        let mut st = self.lock();
        st.members.clear();
        st.events.clear();
        st.next_seq = 0;
        for (user_id, author) in members {
            let seq = st.next_seq;
            st.next_seq += 1;
            st.members.insert(user_id, Slot { author, seq });
        }
        st.revision += 1;
    }

    /// Drop every member + event (bot left the call).
    pub fn clear(&self) {
        let mut st = self.lock();
        if st.members.is_empty() && st.events.is_empty() {
            return;
        }
        st.members.clear();
        st.events.clear();
        st.next_seq = 0;
        st.revision += 1;
    }

    /// Cached member by id.
    #[must_use]
    pub fn member(&self, user_id: i64) -> Option<Author> {
        self.lock().members.get(&user_id).map(|s| s.author.clone())
    }

    /// Every member's proper nouns — `all_known_names()` (display / username /
    /// aliases) plus the per-guild nickname — for STT keyterm biasing (#198).
    /// Returned raw (with duplicates); `set_keyterms` normalizes and caps.
    #[must_use]
    pub fn keyterms(&self) -> Vec<String> {
        self.lock()
            .members
            .values()
            .flat_map(|slot| {
                slot.author
                    .all_known_names()
                    .into_iter()
                    .chain(slot.author.guild_nick.clone())
            })
            .collect()
    }

    /// Current member count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().members.len()
    }

    /// Whether the call is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Prune decayed events, then read members (join order) + live events.
    #[must_use]
    pub fn view(&self) -> RosterView {
        let now = (self.clock)();
        let mut st = self.lock();
        st.prune(now, self.event_window_seconds);
        let mut slots: Vec<&Slot> = st.members.values().collect();
        slots.sort_by_key(|s| s.seq);
        RosterView {
            members: slots.iter().map(|s| s.author.label()).collect(),
            events: st.events.iter().cloned().collect(),
            revision: st.revision,
        }
    }

    /// Prune, then the state counter (the layer's whole cache key).
    #[must_use]
    pub fn revision(&self) -> u64 {
        let now = (self.clock)();
        let mut st = self.lock();
        st.prune(now, self.event_window_seconds);
        st.revision
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("voice roster mutex poisoned")
    }
}

impl State {
    /// Drop events older than `window`; an eviction bumps the revision because
    /// the rendered narration just changed with no writer involved.
    fn prune(&mut self, now: f64, window: f64) {
        let mut evicted = false;
        while self.events.front().is_some_and(|e| now - e.at > window) {
            self.events.pop_front();
            evicted = true;
        }
        if evicted {
            self.revision += 1;
        }
    }

    /// Append, evicting the oldest past [`MAX_EVENTS`].
    fn push_event(&mut self, event: RosterEvent) {
        if self.events.len() == MAX_EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }
}
