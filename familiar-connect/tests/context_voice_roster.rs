//! Live-call roster: the shared `VoiceRoster` state machine and the
//! `VoiceRosterLayer` that puts it in the prompt (subsystem 05, issue: "the bot
//! doesn't know who is in the call").

#[path = "context_helpers/mod.rs"]
mod helpers;

use std::sync::{Arc, Mutex};

use familiar_connect::context::{Layer, VoiceRosterLayer};
use familiar_connect::identity::Author;
use familiar_connect::voice_roster::{Clock, MAX_EVENTS, RosterEventKind, VoiceRoster};

use helpers::{author, tctx, vctx};

/// A settable clock plus the roster reading it.
fn roster_with_clock(window_s: f64) -> (Arc<VoiceRoster>, Arc<Mutex<f64>>) {
    let now = Arc::new(Mutex::new(0.0_f64));
    let handle = now.clone();
    let clock: Clock = Arc::new(move || *handle.lock().expect("clock"));
    let roster = Arc::new(
        VoiceRoster::new()
            .with_clock(clock)
            .with_event_window_seconds(window_s),
    );
    (roster, now)
}

fn advance(now: &Arc<Mutex<f64>>, seconds: f64) {
    *now.lock().expect("clock") += seconds;
}

// ---------------------------------------------------------------------------
// Roster state machine
// ---------------------------------------------------------------------------

#[test]
fn snapshot_seeds_membership_without_narrating() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.snapshot([(1, author("1", "Cor")), (2, author("2", "Cassidy"))]);
    let view = roster.view();
    assert_eq!(view.members, vec!["Cor".to_owned(), "Cassidy".to_owned()]);
    assert!(view.events.is_empty(), "a call in progress has no arrivals");
}

#[test]
fn join_and_leave_are_recorded() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.member_joined(1, author("1", "Cor"));
    roster.member_joined(2, author("2", "Tam"));
    roster.member_left(1);
    let view = roster.view();
    assert_eq!(view.members, vec!["Tam".to_owned()]);
    let kinds: Vec<_> = view.events.iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        vec![
            RosterEventKind::Joined,
            RosterEventKind::Joined,
            RosterEventKind::Left
        ]
    );
    assert_eq!(view.events[2].label, "Cor");
}

#[test]
fn repeat_join_does_not_re_narrate() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.member_joined(1, author("1", "Cor"));
    // Voice-state updates fire on mute/deafen too — same member, same state.
    roster.member_joined(1, author("1", "Cor"));
    assert_eq!(roster.view().events.len(), 1);
}

#[test]
fn leaving_an_absent_member_is_a_no_op() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.member_left(99);
    assert!(roster.view().events.is_empty());
    assert_eq!(roster.revision(), 0);
}

#[test]
fn clear_drops_members_and_events() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.member_joined(1, author("1", "Cor"));
    roster.clear();
    let view = roster.view();
    assert!(view.members.is_empty());
    assert!(view.events.is_empty());
}

#[test]
fn events_decay_after_the_window() {
    let (roster, now) = roster_with_clock(60.0);
    roster.member_joined(1, author("1", "Cor"));
    advance(&now, 59.0);
    assert_eq!(roster.view().events.len(), 1, "still inside the window");
    advance(&now, 2.0);
    assert!(roster.view().events.is_empty(), "decayed past the window");
    // Membership survives decay — only the narration ages out.
    assert_eq!(roster.view().members, vec!["Cor".to_owned()]);
}

#[test]
fn event_log_stays_bounded_under_churn() {
    let (roster, now) = roster_with_clock(1.0e9);
    for i in 0..200_i64 {
        roster.member_joined(i, author(&i.to_string(), &format!("user{i}")));
        roster.member_left(i);
        advance(&now, 0.01);
    }
    assert_eq!(
        roster.view().events.len(),
        MAX_EVENTS,
        "log must be hard-capped, not merely window-trimmed"
    );
}

#[test]
fn member_lookup_and_keyterms_read_the_shared_state() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.member_joined(7, author("7", "Tam"));
    assert_eq!(roster.member(7).map(|a| a.label()), Some("Tam".to_owned()));
    assert!(roster.member(8).is_none());
    assert!(roster.keyterms().contains(&"Tam".to_owned()));
}

// ---------------------------------------------------------------------------
// Invalidation key
// ---------------------------------------------------------------------------

#[tokio::test]
async fn key_is_stable_while_nothing_happens() {
    let (roster, now) = roster_with_clock(60.0);
    roster.member_joined(1, author("1", "Cor"));
    let layer = VoiceRosterLayer::new(roster);
    let first = layer.invalidation_key(&vctx(1)).await;
    advance(&now, 5.0);
    assert_eq!(
        layer.invalidation_key(&vctx(1)).await,
        first,
        "a passing clock alone must not churn the key"
    );
}

#[tokio::test]
async fn key_changes_on_membership_change() {
    let (roster, _now) = roster_with_clock(60.0);
    roster.member_joined(1, author("1", "Cor"));
    let layer = VoiceRosterLayer::new(roster.clone());
    let before = layer.invalidation_key(&vctx(1)).await;
    roster.member_joined(2, author("2", "Tam"));
    let after = layer.invalidation_key(&vctx(1)).await;
    assert_ne!(before, after);
    roster.member_left(2);
    assert_ne!(after, layer.invalidation_key(&vctx(1)).await);
}

#[tokio::test]
async fn key_changes_when_an_event_decays_out() {
    let (roster, now) = roster_with_clock(60.0);
    roster.member_joined(1, author("1", "Cor"));
    let layer = VoiceRosterLayer::new(roster);
    let before = layer.invalidation_key(&vctx(1)).await;
    advance(&now, 61.0);
    assert_ne!(
        before,
        layer.invalidation_key(&vctx(1)).await,
        "the narration disappeared — the cached render is stale"
    );
}

// ---------------------------------------------------------------------------
// Layer rendering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn renders_roster_line_on_a_voice_turn() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.snapshot([
        (1, author("1", "Cor")),
        (2, author("2", "Cassidy")),
        (3, author("3", "Tam")),
    ]);
    let layer = VoiceRosterLayer::new(roster);
    assert_eq!(
        layer.build(&vctx(1)).await,
        "In the call: Cor, Cassidy, Tam."
    );
}

#[tokio::test]
async fn renders_nothing_on_a_text_turn() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.snapshot([(1, author("1", "Cor"))]);
    let layer = VoiceRosterLayer::new(roster);
    assert!(layer.build(&tctx(1)).await.is_empty());
}

#[tokio::test]
async fn empty_roster_renders_nothing() {
    let (roster, _now) = roster_with_clock(120.0);
    let layer = VoiceRosterLayer::new(roster);
    assert!(layer.build(&vctx(1)).await.is_empty());
}

#[tokio::test]
async fn narrates_a_join_and_a_leave() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.snapshot([(1, author("1", "Cor")), (2, author("2", "pixel"))]);
    roster.member_joined(3, author("3", "Tam"));
    roster.member_left(2);
    let layer = VoiceRosterLayer::new(roster);
    assert_eq!(
        layer.build(&vctx(1)).await,
        "In the call: Cor, Tam.\nTam just joined. pixel left."
    );
}

#[tokio::test]
async fn narration_decays_but_the_roster_line_stays() {
    let (roster, now) = roster_with_clock(60.0);
    roster.member_joined(1, author("1", "Cor"));
    let layer = VoiceRosterLayer::new(roster);
    assert!(layer.build(&vctx(1)).await.contains("just joined"));
    advance(&now, 61.0);
    assert_eq!(layer.build(&vctx(1)).await, "In the call: Cor.");
}

#[tokio::test]
async fn roster_line_falls_back_to_the_author_label() {
    let (roster, _now) = roster_with_clock(120.0);
    // No display name → username; no username → user id (Author::label).
    roster.snapshot([
        (
            1,
            Author::new("discord", "1", Some("ada_l".to_owned()), None),
        ),
        (2, Author::new("discord", "2", None, None)),
    ]);
    let layer = VoiceRosterLayer::new(roster);
    assert_eq!(layer.build(&vctx(1)).await, "In the call: ada_l, 2.");
}

#[tokio::test]
async fn roster_line_caps_the_name_list() {
    let (roster, _now) = roster_with_clock(120.0);
    roster.snapshot((0..15_i64).map(|i| (i, author(&i.to_string(), &format!("u{i}")))));
    let rendered = VoiceRosterLayer::new(roster).build(&vctx(1)).await;
    assert!(rendered.ends_with("+3 more."), "got {rendered:?}");
}

#[tokio::test]
async fn narration_renders_at_most_six_events() {
    let (roster, _now) = roster_with_clock(1.0e9);
    for i in 0..10_i64 {
        roster.member_joined(i, author(&i.to_string(), &format!("u{i}")));
    }
    let rendered = VoiceRosterLayer::new(roster).build(&vctx(1)).await;
    let narrated = rendered.matches("just joined").count();
    assert_eq!(narrated, 6, "got {rendered:?}");
    assert!(rendered.contains("u9 just joined."), "newest must survive");
}

// ---------------------------------------------------------------------------
// Keyterm vocabulary: own names + bots + members (#198 follow-up)
//
// Keyterms are a *vocabulary* question ("what proper nouns get spoken"), not a
// roster question ("who is in the call"). A live 263-transcript session missed
// the familiar's own name twice (`Tim` for `Tam`) because the roster filter —
// correct for the prompt line — also starved the keyterm list.
// ---------------------------------------------------------------------------

/// Roster carrying the familiar's own configured names.
fn roster_named(own: &[&str]) -> Arc<VoiceRoster> {
    Arc::new(VoiceRoster::new().with_own_names(own.iter().map(|s| (*s).to_owned())))
}

#[test]
fn own_names_lead_the_keyterms() {
    let roster = roster_named(&["Tam", "Test Familiar"]);
    roster.member_joined(1, author("1", "Kulvar"));
    let terms = roster.keyterms();
    assert_eq!(
        terms.iter().take(2).cloned().collect::<Vec<_>>(),
        vec!["Tam".to_owned(), "Test Familiar".to_owned()],
        "own names must survive the cap ahead of the roster"
    );
    assert!(terms.contains(&"Kulvar".to_owned()));
}

#[test]
fn own_names_are_biased_with_nobody_in_the_call() {
    let roster = roster_named(&["Tam"]);
    assert_eq!(roster.keyterms(), vec!["Tam".to_owned()]);
    // Vocabulary is not membership: an empty call still renders nothing.
    assert!(roster.view().members.is_empty());
}

#[tokio::test]
async fn a_bot_biases_keyterms_but_never_the_roster_line() {
    let roster = roster_named(&["Tam"]);
    roster.member_joined(1, author("1", "Kulvar"));
    // A sibling familiar sharing the guild — said constantly, seated by nobody.
    roster.bot_joined(2, author("2", "Sapphire"));

    let terms = roster.keyterms();
    assert!(terms.contains(&"Sapphire".to_owned()), "got {terms:?}");
    // The human contributes to both halves.
    assert!(terms.contains(&"Kulvar".to_owned()));
    assert_eq!(roster.view().members, vec!["Kulvar".to_owned()]);

    let rendered = VoiceRosterLayer::new(roster).build(&vctx(1)).await;
    assert_eq!(rendered, "In the call: Kulvar.\nKulvar just joined.");
}

#[test]
fn a_bot_narrates_nothing_and_does_not_churn_the_cache_key() {
    let roster = roster_named(&[]);
    let before = roster.view().revision;
    roster.bot_joined(2, author("2", "Sapphire"));
    let after = roster.view();
    assert!(after.events.is_empty(), "a bot arrival is not narration");
    assert_eq!(
        after.revision, before,
        "the rendered line cannot change, so the prompt cache must not drop"
    );
}

#[test]
fn snapshot_bots_replaces_wholesale_and_ignores_call_order() {
    let roster = roster_named(&[]);
    roster.snapshot_bots([(2, author("2", "Sapphire"))]);
    // A re-join must not inherit the last call's bots.
    roster.snapshot_bots([(3, author("3", "Cobalt"))]);
    roster.snapshot([(1, author("1", "Kulvar"))]);
    let terms = roster.keyterms();
    assert!(!terms.contains(&"Sapphire".to_owned()), "got {terms:?}");
    assert!(terms.contains(&"Cobalt".to_owned()));
    assert!(terms.contains(&"Kulvar".to_owned()));
}

#[test]
fn a_departing_bot_leaves_the_keyterms() {
    let roster = roster_named(&[]);
    roster.bot_joined(2, author("2", "Sapphire"));
    // The gateway routes every departure through `member_left`, bot or not.
    roster.member_left(2);
    assert!(roster.keyterms().is_empty());
}

#[test]
fn clear_drops_the_bots_too() {
    let roster = roster_named(&[]);
    roster.bot_joined(2, author("2", "Sapphire"));
    roster.member_joined(1, author("1", "Kulvar"));
    roster.clear();
    assert!(
        roster.keyterms().is_empty(),
        "a stale bot would bias next call"
    );
}

// The whole chain: config jargon still wins the cap, the roster's raw output
// (own names, member names, bot names, duplicates and all) merges behind it,
// and normalization collapses the duplicates.
#[test]
fn roster_keyterms_merge_behind_config_keyterms() {
    use familiar_connect::stt::Transcriber;
    use familiar_connect::stt::deepgram::DeepgramTranscriber;

    let roster = roster_named(&["Tam", "Test Familiar"]);
    roster.member_joined(1, author("1", "Kulvar"));
    roster.bot_joined(2, author("2", "Sapphire"));
    // Overlaps the config list and the member's own username — both must fold.
    roster.member_joined(3, author("3", "Tam"));

    let mut t = DeepgramTranscriber::new("test-key");
    t.keyterms = vec!["lifecycle mesh".to_owned(), "Tam".to_owned()];
    t.set_keyterms(roster.keyterms());

    assert_eq!(
        t.keyterms.iter().take(2).cloned().collect::<Vec<_>>(),
        vec!["lifecycle mesh".to_owned(), "Tam".to_owned()],
        "config keyterms come first so jargon survives the cap"
    );
    // Case-insensitive dedupe: one `Tam`, config spelling wins.
    assert_eq!(
        t.keyterms
            .iter()
            .filter(|k| k.eq_ignore_ascii_case("tam"))
            .count(),
        1
    );
    for expected in ["Test Familiar", "Kulvar", "Sapphire"] {
        assert!(
            t.keyterms.iter().any(|k| k == expected),
            "{expected} missing from {:?}",
            t.keyterms
        );
    }
}
