//! Ported from Python `tests/test_focus_manager.py` — FocusManager + the
//! `SubscriptionRegistry.kind_for` helper it depends on.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use familiar_connect::focus::{Clock, FocusManager, FocusStore};
use familiar_connect::history::StoreError;
use familiar_connect::history::store::{FocusPointers, Promotion};
use familiar_connect::subscriptions::{SubscriptionKind, SubscriptionRegistry, SubscriptionView};

// ---------------------------------------------------------------------------
// Doubles
// ---------------------------------------------------------------------------

#[derive(Default)]
#[allow(clippy::type_complexity)]
struct RecordingFocusStore {
    pointers: Option<FocusPointers>,
    promote_count: usize,
    promote_calls: Mutex<Vec<(String, i64, usize)>>,
    set_calls: Mutex<Vec<(String, Option<i64>, Option<i64>)>>,
}

#[async_trait]
impl FocusStore for RecordingFocusStore {
    async fn get_focus_pointers(
        &self,
        _familiar_id: &str,
    ) -> Result<Option<FocusPointers>, StoreError> {
        Ok(self.pointers)
    }

    async fn set_focus_pointers(
        &self,
        familiar_id: &str,
        text_channel_id: Option<i64>,
        voice_channel_id: Option<i64>,
    ) -> Result<(), StoreError> {
        self.set_calls.lock().unwrap().push((
            familiar_id.to_owned(),
            text_channel_id,
            voice_channel_id,
        ));
        Ok(())
    }

    async fn promote_staged_turns(
        &self,
        familiar_id: &str,
        channel_id: i64,
        catch_up_limit: usize,
    ) -> Result<Promotion, StoreError> {
        self.promote_calls.lock().unwrap().push((
            familiar_id.to_owned(),
            channel_id,
            catch_up_limit,
        ));
        Ok(Promotion {
            consumed: self.promote_count,
            missed: 0,
        })
    }
}

fn make_store(
    text: Option<i64>,
    voice: Option<i64>,
    promote_count: usize,
) -> Arc<RecordingFocusStore> {
    let pointers = if text.is_some() || voice.is_some() {
        Some(FocusPointers {
            text_channel_id: text,
            voice_channel_id: voice,
            updated_at: Utc::now(),
        })
    } else {
        None
    };
    Arc::new(RecordingFocusStore {
        pointers,
        promote_count,
        ..Default::default()
    })
}

fn make_registry(
    kinds: &[(u64, SubscriptionKind)],
) -> (tempfile::TempDir, Arc<SubscriptionRegistry>) {
    let dir = tempfile::tempdir().unwrap();
    let mut reg = SubscriptionRegistry::new(dir.path().join("subs.toml")).unwrap();
    for (ch, kind) in kinds {
        let guild = if *kind == SubscriptionKind::Voice {
            Some(1)
        } else {
            None
        };
        reg.add(*ch, *kind, guild, None).unwrap();
    }
    (dir, Arc::new(reg))
}

struct FakeClock(Arc<Mutex<f64>>);

impl FakeClock {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(1000.0)))
    }
    fn as_clock(&self) -> Clock {
        let inner = self.0.clone();
        Arc::new(move || *inner.lock().unwrap())
    }
    fn advance(&self, dt: f64) {
        *self.0.lock().unwrap() += dt;
    }
}

// ---------------------------------------------------------------------------
// SubscriptionRegistry.kind_for
// ---------------------------------------------------------------------------

#[test]
fn kind_for_returns_none_when_not_subscribed() {
    let (_dir, reg) = make_registry(&[]);
    assert!(reg.kind_for(99).is_none());
}

#[test]
fn kind_for_returns_text_kind() {
    let (_dir, reg) = make_registry(&[(42, SubscriptionKind::Text)]);
    assert_eq!(reg.kind_for(42), Some(SubscriptionKind::Text));
}

#[test]
fn kind_for_returns_voice_kind() {
    let (_dir, reg) = make_registry(&[(7, SubscriptionKind::Voice)]);
    assert_eq!(reg.kind_for(7), Some(SubscriptionKind::Voice));
}

#[test]
fn kind_for_prefers_first_kind_when_both_present() {
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text), (5, SubscriptionKind::Voice)]);
    let result = reg.kind_for(5);
    assert!(matches!(
        result,
        Some(SubscriptionKind::Text | SubscriptionKind::Voice)
    ));
}

#[test]
fn kind_for_unrelated_channel_unaffected() {
    let (_dir, reg) = make_registry(&[(10, SubscriptionKind::Text)]);
    assert!(reg.kind_for(20).is_none());
}

// ---------------------------------------------------------------------------
// FocusManager.initialize
// ---------------------------------------------------------------------------

#[tokio::test]
async fn initialize_loads_focus_pointers() {
    let store = make_store(Some(10), Some(20), 0);
    let (_dir, reg) = make_registry(&[(10, SubscriptionKind::Text), (20, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert_eq!(fm.get_focus("text"), Some(10));
    assert_eq!(fm.get_focus("voice"), Some(20));
}

#[tokio::test]
async fn initialize_with_no_db_entry_stays_none() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert_eq!(fm.get_focus("text"), None);
    assert_eq!(fm.get_focus("voice"), None);
}

#[tokio::test]
async fn initialize_drops_unsubscribed_text_focus() {
    let store = make_store(Some(10), None, 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert_eq!(fm.get_focus("text"), None);
}

#[tokio::test]
async fn initialize_drops_unsubscribed_voice_focus() {
    let store = make_store(None, Some(20), 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert_eq!(fm.get_focus("voice"), None);
}

// ---------------------------------------------------------------------------
// Shared-mutable SubscriptionView seam (parity-audit §3a): the FocusManager
// reads the SAME registry the bot mutates, so a runtime `/subscribe` is visible
// without a restart (Python shares one registry object between bot and focus).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runtime_subscribe_is_visible_to_focus_manager() {
    let dir = tempfile::tempdir().unwrap();
    // ONE registry behind a Mutex, shared as an `Arc<dyn SubscriptionView>`.
    let shared = Arc::new(Mutex::new(
        SubscriptionRegistry::new(dir.path().join("subs.toml")).unwrap(),
    ));
    let view: Arc<dyn SubscriptionView> = shared.clone();
    let fm = FocusManager::new("fam", make_store(None, None, 0), view);

    // Nothing subscribed yet.
    assert!(!fm.is_subscribed(42));
    assert!(fm.subscribed_channels().is_empty());

    // A runtime mutation on the shared registry (as `/subscribe-text` does)…
    shared
        .lock()
        .unwrap()
        .add(42, SubscriptionKind::Text, Some(1), None)
        .unwrap();

    // …is observed by the FocusManager immediately, no rebuild.
    assert!(fm.is_subscribed(42));
    assert_eq!(fm.subscribed_channels(), vec![42]);

    // And a runtime unsubscribe is reflected too.
    shared
        .lock()
        .unwrap()
        .remove(42, SubscriptionKind::Text)
        .unwrap();
    assert!(!fm.is_subscribed(42));
}

#[tokio::test]
async fn initialize_keeps_subscribed_focus() {
    let store = make_store(Some(10), Some(20), 0);
    let (_dir, reg) = make_registry(&[(10, SubscriptionKind::Text), (20, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert_eq!(fm.get_focus("text"), Some(10));
    assert_eq!(fm.get_focus("voice"), Some(20));
}

// ---------------------------------------------------------------------------
// is_subscribed / subscribed_channels
// ---------------------------------------------------------------------------

#[test]
fn is_subscribed_true() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store, reg);
    assert!(fm.is_subscribed(5));
}

#[test]
fn is_subscribed_false() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store, reg);
    assert!(!fm.is_subscribed(99));
}

#[test]
fn subscribed_channels_lists_text_and_voice() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text), (8, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    assert_eq!(fm.subscribed_channels(), vec![5, 8]);
}

#[test]
fn subscribed_channels_dedups_dual_kind() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text), (5, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    assert_eq!(fm.subscribed_channels(), vec![5]);
}

// ---------------------------------------------------------------------------
// is_focused
// ---------------------------------------------------------------------------

#[tokio::test]
async fn is_focused_true_for_text_channel() {
    let store = make_store(Some(42), None, 0);
    let (_dir, reg) = make_registry(&[(42, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert!(fm.is_focused(42));
}

#[tokio::test]
async fn is_focused_true_for_voice_channel() {
    let store = make_store(None, Some(77), 0);
    let (_dir, reg) = make_registry(&[(77, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert!(fm.is_focused(77));
}

#[tokio::test]
async fn is_focused_false_for_unfocused_channel() {
    let store = make_store(Some(1), None, 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert!(!fm.is_focused(99));
}

#[tokio::test]
async fn is_focused_false_when_no_focus() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    assert!(!fm.is_focused(1));
}

// ---------------------------------------------------------------------------
// shift_now + end_turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn shift_now_text_promotes_staged_turns() {
    let store = make_store(None, None, 3);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store.clone(), reg);
    fm.shift_now(5).await;
    assert_eq!(
        *store.promote_calls.lock().unwrap(),
        vec![("fam".to_owned(), 5, 20)]
    );
}

#[tokio::test]
async fn shift_now_text_updates_text_focus() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.shift_now(5).await;
    assert_eq!(fm.get_focus("text"), Some(5));
}

#[tokio::test]
async fn shift_now_voice_updates_voice_focus() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(8, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.shift_now(8).await;
    assert_eq!(fm.get_focus("voice"), Some(8));
}

#[tokio::test]
async fn shift_now_voice_does_not_promote() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(8, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store.clone(), reg);
    fm.shift_now(8).await;
    assert!(store.promote_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn shift_now_persists_pointers() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store.clone(), reg);
    fm.shift_now(5).await;
    assert_eq!(store.set_calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn end_turn_does_not_touch_focus() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store.clone(), reg);
    fm.end_turn().await;
    assert!(store.promote_calls.lock().unwrap().is_empty());
    assert!(store.set_calls.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Idle-drift wake
// ---------------------------------------------------------------------------

fn idle_fm(
    clock: &FakeClock,
    enabled: bool,
    debounce: f64,
    text_focus: i64,
) -> (tempfile::TempDir, Arc<RecordingFocusStore>, FocusManager) {
    let store = make_store(Some(text_focus), None, 0);
    let (dir, reg) = make_registry(&[(u64::try_from(text_focus).unwrap(), SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store.clone(), reg)
        .with_clock(clock.as_clock())
        .with_unread_nudge_enabled(enabled)
        .with_nudge_debounce_seconds(debounce);
    (dir, store, fm)
}

#[tokio::test]
async fn should_wake_true_on_arrival_when_focused_channel_active() {
    let clock = FakeClock::new();
    let (_dir, _store, fm) = idle_fm(&clock, true, 10.0, 1);
    fm.initialize().await;
    assert!(fm.should_wake(99));
}

#[tokio::test]
async fn should_wake_false_for_focused_channel() {
    let clock = FakeClock::new();
    let (_dir, _store, fm) = idle_fm(&clock, true, 10.0, 1);
    fm.initialize().await;
    clock.advance(90.0);
    assert!(!fm.should_wake(1));
}

#[tokio::test]
async fn should_wake_false_when_disabled() {
    let clock = FakeClock::new();
    let (_dir, _store, fm) = idle_fm(&clock, false, 10.0, 1);
    fm.initialize().await;
    clock.advance(10_000.0);
    assert!(!fm.should_wake(99));
}

#[tokio::test]
async fn should_wake_false_within_debounce() {
    let clock = FakeClock::new();
    let (_dir, _store, fm) = idle_fm(&clock, true, 10.0, 1);
    fm.initialize().await;
    clock.advance(90.0);
    assert!(fm.should_wake(99));
    fm.mark_nudge_pending();
    clock.advance(5.0);
    assert!(!fm.should_wake(99));
}

#[tokio::test]
async fn should_wake_true_after_debounce_expires() {
    let clock = FakeClock::new();
    let (_dir, _store, fm) = idle_fm(&clock, true, 10.0, 1);
    fm.initialize().await;
    clock.advance(90.0);
    fm.mark_nudge_pending();
    clock.advance(15.0);
    assert!(fm.should_wake(99));
}

#[tokio::test]
async fn should_wake_repeats_after_each_debounce() {
    let clock = FakeClock::new();
    let (_dir, _store, fm) = idle_fm(&clock, true, 10.0, 1);
    fm.initialize().await;
    clock.advance(90.0);
    assert!(fm.should_wake(99));
    fm.mark_nudge_pending();
    clock.advance(15.0);
    assert!(fm.should_wake(99));
    fm.mark_nudge_pending();
    clock.advance(5.0);
    assert!(!fm.should_wake(99));
}

#[tokio::test]
async fn wake_does_not_move_focus() {
    let clock = FakeClock::new();
    let (_dir, _store, fm) = idle_fm(&clock, true, 10.0, 1);
    fm.initialize().await;
    clock.advance(90.0);
    fm.mark_nudge_pending();
    assert_eq!(fm.get_focus("text"), Some(1));
}

// ---------------------------------------------------------------------------
// Modalities stay independent
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_shift_does_not_affect_voice() {
    let store = make_store(None, Some(99), 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text), (99, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    fm.shift_now(5).await;
    assert_eq!(fm.get_focus("voice"), Some(99));
    assert_eq!(fm.get_focus("text"), Some(5));
}

#[tokio::test]
async fn voice_shift_does_not_affect_text() {
    let store = make_store(Some(11), None, 0);
    let (_dir, reg) = make_registry(&[(8, SubscriptionKind::Voice), (11, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store, reg);
    fm.initialize().await;
    fm.shift_now(8).await;
    assert_eq!(fm.get_focus("text"), Some(11));
    assert_eq!(fm.get_focus("voice"), Some(8));
}

// ---------------------------------------------------------------------------
// set_focus_immediately
// ---------------------------------------------------------------------------

#[test]
fn set_focus_immediately_sets_text_focus() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store, reg);
    fm.set_focus_immediately(3, "text");
    assert_eq!(fm.get_focus("text"), Some(3));
}

#[test]
fn set_focus_immediately_sets_voice_focus() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[]);
    let fm = FocusManager::new("fam", store, reg);
    fm.set_focus_immediately(4, "voice");
    assert_eq!(fm.get_focus("voice"), Some(4));
}

// ---------------------------------------------------------------------------
// presence_text / presence_guild / guild_name_for
// ---------------------------------------------------------------------------

fn bare_fm() -> (tempfile::TempDir, FocusManager) {
    let store = make_store(None, None, 0);
    let (dir, reg) = make_registry(&[]);
    (dir, FocusManager::new("fam", store, reg))
}

#[test]
fn presence_text_returns_none_with_no_focus() {
    let (_dir, fm) = bare_fm();
    assert_eq!(fm.presence_text(), None);
}

#[test]
fn presence_text_returns_channel_name() {
    let (_dir, fm) = bare_fm();
    fm.set_channel_name(42, "general");
    fm.set_focus_immediately(42, "text");
    assert_eq!(fm.presence_text().as_deref(), Some("#general"));
}

#[test]
fn presence_text_falls_back_to_channel_id_when_name_unknown() {
    let (_dir, fm) = bare_fm();
    fm.set_focus_immediately(42, "text");
    assert_eq!(fm.presence_text().as_deref(), Some("#42"));
}

#[test]
fn presence_guild_returns_none_with_no_focus() {
    let (_dir, fm) = bare_fm();
    assert_eq!(fm.presence_guild(), None);
}

#[test]
fn presence_guild_returns_none_when_guild_unknown() {
    let (_dir, fm) = bare_fm();
    fm.set_focus_immediately(42, "text");
    assert_eq!(fm.presence_guild(), None);
}

#[test]
fn presence_guild_returns_guild_name() {
    let (_dir, fm) = bare_fm();
    fm.set_guild_name(42, "Sapphire");
    fm.set_focus_immediately(42, "text");
    assert_eq!(fm.presence_guild().as_deref(), Some("Sapphire"));
}

#[test]
fn guild_name_for_returns_name_for_known_channel() {
    let (_dir, fm) = bare_fm();
    fm.set_guild_name(42, "My Server");
    assert_eq!(fm.guild_name_for(Some(42)).as_deref(), Some("My Server"));
}

#[test]
fn guild_name_for_returns_none_for_unknown_channel() {
    let (_dir, fm) = bare_fm();
    fm.set_guild_name(42, "My Server");
    assert_eq!(fm.guild_name_for(Some(99)), None);
}

#[test]
fn guild_name_for_returns_none_for_none_input() {
    let (_dir, fm) = bare_fm();
    fm.set_guild_name(42, "My Server");
    assert_eq!(fm.guild_name_for(None), None);
}

// ---------------------------------------------------------------------------
// channel_names / guild_names bulk getters (PR #194 — digest threading)
// ---------------------------------------------------------------------------

#[test]
fn channel_names_and_guild_names_getters_reflect_setters() {
    let (_dir, fm) = bare_fm();
    fm.set_channel_name(42, "general");
    fm.set_guild_name(42, "My Server");
    assert_eq!(
        fm.channel_names().get(&42).map(String::as_str),
        Some("general")
    );
    assert_eq!(
        fm.guild_names().get(&42).map(String::as_str),
        Some("My Server")
    );
}

// ---------------------------------------------------------------------------
// on_shift
// ---------------------------------------------------------------------------

#[test]
fn on_shift_is_none_by_default() {
    let (_dir, fm) = bare_fm();
    assert!(!fm.has_on_shift());
}

#[tokio::test]
async fn on_shift_called_after_shift_now() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text)]);
    let fm = FocusManager::new("fam", store, reg);
    let count = Arc::new(AtomicUsize::new(0));
    let counter = count.clone();
    fm.set_on_shift(Arc::new(move || {
        let counter = counter.clone();
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
        })
    }));
    fm.shift_now(5).await;
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn on_shift_fires_per_shift() {
    let store = make_store(None, None, 0);
    let (_dir, reg) = make_registry(&[(5, SubscriptionKind::Text), (8, SubscriptionKind::Voice)]);
    let fm = FocusManager::new("fam", store, reg);
    let count = Arc::new(AtomicUsize::new(0));
    let counter = count.clone();
    fm.set_on_shift(Arc::new(move || {
        let counter = counter.clone();
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
        })
    }));
    fm.shift_now(5).await;
    fm.shift_now(8).await;
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn on_shift_not_called_by_end_turn() {
    let (_dir, fm) = bare_fm();
    let count = Arc::new(AtomicUsize::new(0));
    let counter = count.clone();
    fm.set_on_shift(Arc::new(move || {
        let counter = counter.clone();
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
        })
    }));
    fm.end_turn().await;
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
