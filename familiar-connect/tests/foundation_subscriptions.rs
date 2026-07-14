//! Integration tests for `subscriptions::SubscriptionRegistry` (foundation
//! package). Ported from Python `tests/test_subscriptions.py`, plus a byte-exact
//! sidecar-serialization pin for the DESIGN §4 / spec-02 on-disk contract.

use familiar_connect::subscriptions::{Subscription, SubscriptionKind, SubscriptionRegistry};
use std::path::PathBuf;

fn subs_path() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("subs.toml");
    (dir, path)
}

// --- SubscriptionKind -------------------------------------------------------

#[test]
fn kind_has_text_and_voice() {
    assert_eq!(SubscriptionKind::Text.value(), "text");
    assert_eq!(SubscriptionKind::Voice.value(), "voice");
}

// --- Registry core CRUD -----------------------------------------------------

#[test]
fn empty_registry_has_nothing() {
    let (_dir, path) = subs_path();
    let reg = SubscriptionRegistry::new(path).unwrap();
    assert!(reg.all().is_empty());
    assert_eq!(reg.get(1, SubscriptionKind::Text), None);
}

#[test]
fn add_and_get() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();

    let sub = reg.get(42, SubscriptionKind::Text);
    assert_eq!(
        sub,
        Some(Subscription {
            channel_id: 42,
            kind: SubscriptionKind::Text,
            guild_id: Some(999),
        })
    );
}

#[test]
fn add_is_idempotent() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();
    assert_eq!(reg.all().len(), 1);
}

#[test]
fn add_re_add_updates_guild_id() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(42, SubscriptionKind::Text, None, true).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(7), true).unwrap();
    assert_eq!(
        reg.get(42, SubscriptionKind::Text).unwrap().guild_id,
        Some(7)
    );
    assert_eq!(reg.all().len(), 1);
}

#[test]
fn remove_deletes_row() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(42, SubscriptionKind::Text, None, true).unwrap();
    reg.remove(42, SubscriptionKind::Text).unwrap();
    assert_eq!(reg.get(42, SubscriptionKind::Text), None);
}

#[test]
fn remove_of_unknown_is_noop() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    // Must not error.
    reg.remove(42, SubscriptionKind::Text).unwrap();
}

#[test]
fn text_and_voice_in_same_channel_coexist() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(1), true).unwrap();
    reg.add(42, SubscriptionKind::Voice, Some(1), true).unwrap();
    assert_eq!(reg.all().len(), 2);
}

#[test]
fn all_returns_every_subscription() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(1, SubscriptionKind::Text, None, true).unwrap();
    reg.add(2, SubscriptionKind::Text, None, true).unwrap();
    reg.add(3, SubscriptionKind::Voice, None, true).unwrap();
    assert_eq!(reg.all().len(), 3);
}

// --- kind_for ---------------------------------------------------------------

#[test]
fn kind_for_checks_text_before_voice() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(42, SubscriptionKind::Voice, None, true).unwrap();
    assert_eq!(reg.kind_for(42), Some(SubscriptionKind::Voice));
    reg.add(42, SubscriptionKind::Text, None, true).unwrap();
    // Text wins the declaration-order scan when both exist.
    assert_eq!(reg.kind_for(42), Some(SubscriptionKind::Text));
    assert_eq!(reg.kind_for(9999), None);
}

// --- Voice helpers ----------------------------------------------------------

#[test]
fn voice_in_guild_returns_sub() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path).unwrap();
    reg.add(9000, SubscriptionKind::Voice, Some(123), true)
        .unwrap();
    assert!(reg.voice_in_guild(123).is_some());
}

#[test]
fn voice_in_guild_returns_none_when_absent() {
    let (_dir, path) = subs_path();
    let reg = SubscriptionRegistry::new(path).unwrap();
    assert_eq!(reg.voice_in_guild(123), None);
}

// --- Persistence ------------------------------------------------------------

#[test]
fn add_writes_to_disk() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();
    assert!(path.exists());
}

#[test]
fn reload_round_trip() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();
    reg.add(42, SubscriptionKind::Voice, Some(999), true)
        .unwrap();
    reg.add(77, SubscriptionKind::Text, None, true).unwrap();

    let reloaded = SubscriptionRegistry::new(path).unwrap();
    assert!(reloaded.get(42, SubscriptionKind::Text).is_some());
    assert!(reloaded.get(42, SubscriptionKind::Voice).is_some());
    assert!(reloaded.get(77, SubscriptionKind::Text).is_some());
    assert_eq!(reloaded.all().len(), 3);
    // guild_id survived the round trip; the None row reloads as None.
    assert_eq!(
        reloaded.get(42, SubscriptionKind::Text).unwrap().guild_id,
        Some(999)
    );
    assert_eq!(
        reloaded.get(77, SubscriptionKind::Text).unwrap().guild_id,
        None
    );
}

#[test]
fn remove_persists() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();
    reg.remove(42, SubscriptionKind::Text).unwrap();

    let reloaded = SubscriptionRegistry::new(path).unwrap();
    assert_eq!(reloaded.get(42, SubscriptionKind::Text), None);
}

// --- Ephemeral (in-memory-only) subscriptions -------------------------------

#[test]
fn ephemeral_add_is_queryable_but_not_written() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(42, SubscriptionKind::Text, None, false).unwrap();

    assert!(reg.get(42, SubscriptionKind::Text).is_some());
    assert_eq!(reg.kind_for(42), Some(SubscriptionKind::Text));
    assert!(!path.exists());
}

#[test]
fn persisted_mutation_excludes_ephemeral_row() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(42, SubscriptionKind::Text, None, false).unwrap();
    reg.add(77, SubscriptionKind::Text, None, true).unwrap();

    // Live registry still sees both the ephemeral and persisted rows.
    assert_eq!(reg.all().len(), 2);

    let reloaded = SubscriptionRegistry::new(path).unwrap();
    assert!(reloaded.get(77, SubscriptionKind::Text).is_some());
    assert_eq!(reloaded.get(42, SubscriptionKind::Text), None);
    assert_eq!(reloaded.all().len(), 1);
}

#[test]
fn persist_true_promotes_previously_ephemeral_row() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(42, SubscriptionKind::Text, None, false).unwrap();
    reg.add(42, SubscriptionKind::Text, None, true).unwrap();

    let reloaded = SubscriptionRegistry::new(path).unwrap();
    assert!(reloaded.get(42, SubscriptionKind::Text).is_some());
}

#[test]
fn persisted_remove_does_not_resurrect_ephemeral_row() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(77, SubscriptionKind::Text, None, true).unwrap();
    reg.add(42, SubscriptionKind::Text, None, false).unwrap();
    reg.remove(77, SubscriptionKind::Text).unwrap();

    let reloaded = SubscriptionRegistry::new(path).unwrap();
    assert_eq!(reloaded.get(42, SubscriptionKind::Text), None);
}

// --- Byte-exact sidecar format (spec 02 Data formats) -----------------------

#[test]
fn single_row_serializes_byte_exact() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    let expected = "# Persistent subscription registry.\n\
         # Managed by /subscribe-* slash commands; safe to hand-edit while the bot is stopped.\n\
         \n\
         [[subscription]]\n\
         channel_id = 42\n\
         kind = \"text\"\n\
         guild_id = 999\n";
    assert_eq!(written, expected);
}

#[test]
fn multiple_rows_serialize_sorted_with_guild_omitted_when_none() {
    let (_dir, path) = subs_path();
    let mut reg = SubscriptionRegistry::new(path.clone()).unwrap();
    // Insert out of order; the file must come out sorted by (channel_id, kind).
    reg.add(77, SubscriptionKind::Text, None, true).unwrap();
    reg.add(42, SubscriptionKind::Voice, Some(999), true)
        .unwrap();
    reg.add(42, SubscriptionKind::Text, Some(999), true)
        .unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    let expected = "# Persistent subscription registry.\n\
         # Managed by /subscribe-* slash commands; safe to hand-edit while the bot is stopped.\n\
         \n\
         [[subscription]]\n\
         channel_id = 42\n\
         kind = \"text\"\n\
         guild_id = 999\n\
         \n\
         [[subscription]]\n\
         channel_id = 42\n\
         kind = \"voice\"\n\
         guild_id = 999\n\
         \n\
         [[subscription]]\n\
         channel_id = 77\n\
         kind = \"text\"\n";
    assert_eq!(written, expected);
}

// --- Tolerant load / strict syntax ------------------------------------------

#[test]
fn tolerant_load_skips_malformed_rows_but_keeps_valid_ones() {
    let (_dir, path) = subs_path();
    // Hand-written sidecar: some rows are malformed; a TOML-*valid* file with
    // content problems must not brick startup — bad rows are skipped.
    let contents = "\
        [[subscription]]\n\
        channel_id = 10\n\
        kind = \"text\"\n\
        \n\
        [[subscription]]\n\
        channel_id = \"not-an-int\"\n\
        kind = \"text\"\n\
        \n\
        [[subscription]]\n\
        channel_id = 11\n\
        kind = \"bogus\"\n\
        \n\
        [[subscription]]\n\
        channel_id = 12\n\
        kind = \"voice\"\n\
        guild_id = 5\n";
    std::fs::write(&path, contents).unwrap();

    let reg = SubscriptionRegistry::new(path).unwrap();
    // Row 10 (valid text) and row 12 (valid voice) survive; the wrong-typed
    // channel_id and unknown-kind rows are dropped.
    assert_eq!(reg.all().len(), 2);
    assert!(reg.get(10, SubscriptionKind::Text).is_some());
    assert_eq!(
        reg.get(12, SubscriptionKind::Voice).unwrap().guild_id,
        Some(5)
    );
    assert_eq!(reg.get(11, SubscriptionKind::Voice), None);
}

#[test]
fn syntax_error_propagates_as_error() {
    let (_dir, path) = subs_path();
    // Broken TOML syntax (unterminated string) — unlike content problems, this
    // must surface as an error rather than an empty registry.
    std::fs::write(&path, "[[subscription]]\nchannel_id = 1\nkind = \"text").unwrap();
    let result = SubscriptionRegistry::new(path);
    assert!(result.is_err());
}
