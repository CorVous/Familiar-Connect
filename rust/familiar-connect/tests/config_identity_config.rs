//! Integration tests for `config::load_character_config` (subsystem 02;
//! ports `tests/test_config.py`). These exercise the TOML deep-merge over the
//! checked-in `_default/character.toml`, per-section validation, and the
//! byte-stable `ConfigError` message contract.

use familiar_connect::config::{CharacterConfig, ConfigError, load_character_config};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn default_profile() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/familiars/_default/character.toml")
}

/// The registry validator sets, injected per DESIGN D3. These mirror the real
/// `known_projectors()` / `known_embedders()` registries closely enough to load
/// the shipped default profile and exercise the override paths.
fn projectors() -> BTreeSet<String> {
    [
        "rolling_summary",
        "rich_note",
        "people_dossier",
        "reflection",
        "fact_supersede",
        "fact_embedding",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

fn embedders() -> BTreeSet<String> {
    ["off", "hash", "fastembed"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
}

/// Load `target_content` (written to a fresh tmp `character.toml`) merged over
/// the shipped default profile.
fn load(target_content: &str) -> Result<CharacterConfig, ConfigError> {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("character.toml");
    std::fs::write(&path, target_content).unwrap();
    load_character_config(&path, &default_profile(), &projectors(), &embedders())
}

fn load_ok(target_content: &str) -> CharacterConfig {
    load(target_content).expect("config should load")
}

/// Load with the target file absent (defaults-only install) over the shipped
/// default profile.
fn load_missing_target() -> CharacterConfig {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("missing.toml");
    load_character_config(&path, &default_profile(), &projectors(), &embedders())
        .expect("config should load")
}

/// Load a custom target (or none) over a custom defaults file — used to isolate
/// merge behavior from the shipped profile's pins.
fn load_custom(
    target: Option<&str>,
    defaults_content: &str,
) -> Result<CharacterConfig, ConfigError> {
    let dir = TempDir::new().unwrap();
    let defaults = dir.path().join("defaults.toml");
    std::fs::write(&defaults, defaults_content).unwrap();
    let target_path = dir.path().join("character.toml");
    if let Some(c) = target {
        std::fs::write(&target_path, c).unwrap();
    }
    load_character_config(&target_path, &defaults, &projectors(), &embedders())
}

#[track_caller]
fn assert_err(result: Result<CharacterConfig, ConfigError>, needle: &str) {
    let err = result.expect_err("expected ConfigError");
    let msg = err.to_string();
    assert!(
        msg.contains(needle),
        "error {msg:?} did not contain {needle:?}"
    );
}

/// Assert the *entire* `ConfigError` message matches byte-for-byte. Used where
/// the numeric `got {value}` tail must match Python exactly (a negative TOML
/// integer prints as `-1`, not `-1.0`).
#[track_caller]
fn assert_err_eq(result: Result<CharacterConfig, ConfigError>, expected: &str) {
    let err = result.expect_err("expected ConfigError");
    assert_eq!(err.to_string(), expected);
}

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

// ---------------------------------------------------------------------------
// Loading / merge / LLM slots
// ---------------------------------------------------------------------------

#[test]
fn missing_defaults_raises() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("character.toml");
    std::fs::write(&path, "display_tz = 'UTC'\n").unwrap();
    let missing = dir.path().join("missing.toml");
    let result = load_character_config(&path, &missing, &projectors(), &embedders());
    assert_err(result, "default character profile");
}

#[test]
fn defaults_only_roundtrip() {
    let cfg = load_ok("");
    assert_eq!(cfg.display_tz, "UTC");
    for slot in ["fast", "prose", "background"] {
        assert!(cfg.llm.contains_key(slot), "missing slot {slot}");
    }
}

#[test]
fn user_overrides_default() {
    let cfg = load_ok("display_tz = \"America/New_York\"\naliases = [\"m\"]\n");
    assert_eq!(cfg.display_tz, "America/New_York");
    assert_eq!(cfg.aliases, vec!["m".to_owned()]);
}

#[test]
fn unknown_llm_slot_rejected() {
    assert_err(load("[llm.mystery]\nmodel = \"foo\"\n"), "unknown LLM slot");
}

#[test]
fn legacy_main_prose_rejected() {
    assert_err(
        load("[llm.main_prose]\nmodel = \"m\"\n"),
        "unknown LLM slot 'main_prose'",
    );
}

#[test]
fn invalid_display_tz_rejected() {
    assert_err(load("display_tz = \"PST\"\n"), "display_tz");
}

#[test]
fn valid_display_tz_accepted() {
    let cfg = load_ok("display_tz = \"America/Los_Angeles\"\n");
    assert_eq!(cfg.display_tz, "America/Los_Angeles");
}

#[test]
fn temperature_out_of_range() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\ntemperature = 3.0\n"),
        "temperature must be in",
    );
}

#[test]
fn provider_order_parsed() {
    let cfg = load_ok(
        "[llm.prose]\nmodel = \"z-ai/glm-5.1\"\nprovider_order = [\"z-ai\", \"deepinfra\"]\nprovider_allow_fallbacks = false\n",
    );
    let slot = cfg.llm.get("prose").unwrap();
    assert_eq!(
        slot.provider_order,
        Some(vec!["z-ai".to_owned(), "deepinfra".to_owned()])
    );
    assert!(!slot.provider_allow_fallbacks);
}

#[test]
fn sleep_window_and_grace_parse() {
    let cfg = load_ok("[sleep]\nwindow = \"00:00-08:00\"\ngrace_minutes = 45\n");
    assert_eq!(
        cfg.sleep_window,
        Some((
            chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
            chrono::NaiveTime::from_hms_opt(8, 0, 0).unwrap()
        ))
    );
    assert_eq!(cfg.sleep_grace_minutes, 45);
}

#[test]
fn sleep_window_wraps_midnight() {
    let cfg = load_ok("[sleep]\nwindow = \"23:00-07:00\"\n");
    assert_eq!(
        cfg.sleep_window,
        Some((
            chrono::NaiveTime::from_hms_opt(23, 0, 0).unwrap(),
            chrono::NaiveTime::from_hms_opt(7, 0, 0).unwrap()
        ))
    );
}

#[test]
fn sleep_grace_defaults_to_30() {
    let cfg = load_ok("[sleep]\nwindow = \"00:00-08:00\"\n");
    assert_eq!(cfg.sleep_grace_minutes, 30);
}

#[test]
fn sleep_omitted_means_no_window() {
    let cfg = load_ok("");
    assert!(cfg.sleep_window.is_none());
    assert_eq!(cfg.sleep_grace_minutes, 30);
}

#[test]
fn sleep_bad_window_format_rejected() {
    for value in ["00:00", "midnight", "00:00-00:00"] {
        assert_err(load(&format!("[sleep]\nwindow = \"{value}\"\n")), "window");
    }
}

#[test]
fn sleep_bad_grace_rejected() {
    for value in ["0", "-5", "true", "\"30\""] {
        assert_err(
            load(&format!(
                "[sleep]\nwindow = \"00:00-08:00\"\ngrace_minutes = {value}\n"
            )),
            "grace_minutes",
        );
    }
}

#[test]
fn sleep_unknown_key_rejected() {
    assert_err(
        load("[sleep]\nwindow = \"00:00-08:00\"\nbedtime = 5\n"),
        "unknown",
    );
}

#[test]
fn provider_order_omitted_means_none() {
    let defaults =
        "[llm.fast]\nmodel = \"x\"\n[llm.prose]\nmodel = \"x\"\n[llm.background]\nmodel = \"x\"\n";
    let cfg = load_custom(Some("[llm.prose]\nmodel = \"m\"\n"), defaults).unwrap();
    let slot = cfg.llm.get("prose").unwrap();
    assert!(slot.provider_order.is_none());
    assert!(slot.provider_allow_fallbacks);
}

#[test]
fn provider_order_must_be_list_of_strings() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\nprovider_order = [1, 2]\n"),
        "provider_order",
    );
}

#[test]
fn reasoning_levels_parsed() {
    let cfg = load_ok(
        "[llm.fast]\nmodel = \"m\"\nreasoning = \"off\"\n[llm.prose]\nmodel = \"m\"\nreasoning = \"medium\"\n[llm.background]\nmodel = \"m\"\nreasoning = \"high\"\n",
    );
    assert_eq!(
        cfg.llm.get("fast").unwrap().reasoning.as_deref(),
        Some("off")
    );
    assert_eq!(
        cfg.llm.get("prose").unwrap().reasoning.as_deref(),
        Some("medium")
    );
    assert_eq!(
        cfg.llm.get("background").unwrap().reasoning.as_deref(),
        Some("high")
    );
}

#[test]
fn reasoning_none_level_parsed() {
    let cfg = load_ok("[llm.fast]\nmodel = \"m\"\nreasoning = \"none\"\n");
    assert_eq!(
        cfg.llm.get("fast").unwrap().reasoning.as_deref(),
        Some("none")
    );
}

#[test]
fn reasoning_omitted_means_none() {
    let defaults =
        "[llm.fast]\nmodel = \"x\"\n[llm.prose]\nmodel = \"x\"\n[llm.background]\nmodel = \"x\"\n";
    let cfg = load_custom(None, defaults).unwrap();
    assert!(cfg.llm.get("prose").unwrap().reasoning.is_none());
}

#[test]
fn sampling_params_parsed() {
    let cfg = load_ok(
        "[llm.fast]\nmodel = \"m\"\ntop_p = 0.8\ntop_k = 20\npresence_penalty = 1.5\nthink_prepend = true\n",
    );
    let slot = cfg.llm.get("fast").unwrap();
    approx(slot.top_p.unwrap(), 0.8);
    assert_eq!(slot.top_k, Some(20));
    approx(slot.presence_penalty.unwrap(), 1.5);
    assert!(slot.think_prepend);
}

#[test]
fn sampling_params_omitted_mean_provider_default() {
    let cfg = load_missing_target();
    let slot = cfg.llm.get("prose").unwrap();
    assert!(slot.top_p.is_none());
    assert!(slot.top_k.is_none());
    assert!(slot.presence_penalty.is_none());
    assert!(!slot.think_prepend);
}

#[test]
fn top_p_out_of_range() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\ntop_p = 1.5\n"),
        "top_p must be in",
    );
}

#[test]
fn top_k_must_be_positive_int() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\ntop_k = 0\n"),
        "top_k must be",
    );
}

#[test]
fn presence_penalty_out_of_range() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\npresence_penalty = 3.0\n"),
        "presence_penalty must be in",
    );
}

#[test]
fn think_prepend_must_be_bool() {
    assert_err(
        load("[llm.fast]\nmodel = \"m\"\nthink_prepend = \"yes\"\n"),
        "think_prepend must be a bool",
    );
}

#[test]
fn reasoning_default_sentinel_overrides_merged_value() {
    let defaults = "[llm.fast]\nmodel = \"x\"\nreasoning = \"off\"\n[llm.prose]\nmodel = \"x\"\nreasoning = \"medium\"\n[llm.background]\nmodel = \"x\"\n";
    let target = "[llm.fast]\nmodel = \"m\"\nreasoning = \"default\"\n[llm.prose]\nmodel = \"m\"\nreasoning = \"default\"\n";
    let cfg = load_custom(Some(target), defaults).unwrap();
    assert!(cfg.llm.get("fast").unwrap().reasoning.is_none());
    assert!(cfg.llm.get("prose").unwrap().reasoning.is_none());
}

#[test]
fn invalid_reasoning_rejected() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\nreasoning = \"ultra\"\n"),
        "reasoning",
    );
}

#[test]
fn reasoning_must_be_string() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\nreasoning = true\n"),
        "reasoning",
    );
}

#[test]
fn tool_calling_parsed() {
    let cfg = load_ok(
        "[llm.background]\nmodel = \"m\"\ntool_calling = true\n[llm.fast]\nmodel = \"m\"\ntool_calling = false\n",
    );
    assert!(cfg.llm.get("background").unwrap().tool_calling);
    assert!(!cfg.llm.get("fast").unwrap().tool_calling);
}

#[test]
fn tool_calling_omitted_defaults_false() {
    let defaults =
        "[llm.fast]\nmodel = \"x\"\n[llm.prose]\nmodel = \"x\"\n[llm.background]\nmodel = \"x\"\n";
    let cfg = load_custom(None, defaults).unwrap();
    assert!(!cfg.llm.get("prose").unwrap().tool_calling);
}

#[test]
fn tool_calling_must_be_bool() {
    assert_err(
        load("[llm.prose]\nmodel = \"m\"\ntool_calling = \"yes\"\n"),
        "tool_calling",
    );
}

// ---------------------------------------------------------------------------
// Prompt shaping
// ---------------------------------------------------------------------------

#[test]
fn unknown_tts_provider_rejected() {
    assert_err(load("[tts]\nprovider = \"mysterybox\"\n"), "[tts].provider");
}

#[test]
fn post_history_instructions_default_from_profile() {
    let cfg = load_ok("");
    assert!(cfg.post_history_instructions.contains("<silent>"));
    assert!(!cfg.post_history_instructions.trim().is_empty());
}

#[test]
fn post_history_instructions_override() {
    let cfg = load_ok("[prompt]\npost_history_instructions = \"be terse\"\n");
    assert_eq!(cfg.post_history_instructions, "be terse");
}

#[test]
fn post_history_instructions_absent_defaults_empty() {
    let defaults =
        "[llm.fast]\nmodel = \"x\"\n[llm.prose]\nmodel = \"x\"\n[llm.background]\nmodel = \"x\"\n";
    let cfg = load_custom(None, defaults).unwrap();
    assert!(cfg.post_history_instructions.is_empty());
}

#[test]
fn post_history_instructions_must_be_string() {
    assert_err(
        load("[prompt]\npost_history_instructions = 42\n"),
        "post_history_instructions",
    );
}

#[test]
fn prompt_unknown_key_rejected() {
    assert_err(
        load("[prompt]\nmystery = \"x\"\n"),
        "[prompt] has unknown keys",
    );
}

#[test]
fn image_description_constraints_absent_defaults_empty() {
    let cfg = load_ok("");
    assert!(cfg.image_description_constraints.is_empty());
}

#[test]
fn image_description_constraints_override() {
    let cfg = load_ok("[prompt]\nimage_description_constraints = \"no brands\"\n");
    assert_eq!(cfg.image_description_constraints, "no brands");
}

#[test]
fn image_description_constraints_must_be_string() {
    assert_err(
        load("[prompt]\nimage_description_constraints = 42\n"),
        "image_description_constraints",
    );
}

#[test]
fn sleep_prompts_default_from_profile() {
    let cfg = load_ok("");
    assert!(!cfg.sleep_consolidation_system.trim().is_empty());
    assert!(cfg.sleep_stance_system.contains("{self_name}"));
    assert!(cfg.sleep_synthesis_system.contains("{self_name}"));
    assert!(cfg.dream_extraction_clause.contains("{self_name}"));
    assert!(cfg.dream_extraction_clause.contains("{self_key}"));
}

#[test]
fn sleep_consolidation_system_override() {
    let cfg = load_ok("[prompt]\nsleep_consolidation_system = \"custom tidy pass\"\n");
    assert_eq!(cfg.sleep_consolidation_system, "custom tidy pass");
}

#[test]
fn sleep_stance_system_override() {
    let cfg = load_ok("[prompt]\nsleep_stance_system = \"stances for {self_name}\"\n");
    assert_eq!(cfg.sleep_stance_system, "stances for {self_name}");
}

#[test]
fn dream_extraction_clause_override() {
    let cfg =
        load_ok("[prompt]\ndream_extraction_clause = \"dream {self_name} {self_key} {ids}\"\n");
    assert_eq!(
        cfg.dream_extraction_clause,
        "dream {self_name} {self_key} {ids}"
    );
}

#[test]
fn sleep_prompt_must_be_string() {
    assert_err(
        load("[prompt]\nsleep_consolidation_system = 42\n"),
        "sleep_consolidation_system",
    );
}

#[test]
fn default_profile_carries_real_sleep_prompt_prose() {
    let cfg = load_ok("");
    assert!(
        cfg.sleep_consolidation_system
            .contains("memory-consolidation pass")
    );
    assert!(cfg.sleep_stance_system.contains("stance-moment"));
    assert!(cfg.sleep_synthesis_system.contains("settled opinions"));
    assert!(cfg.dream_extraction_clause.contains("dream narration"));
}

// ---------------------------------------------------------------------------
// History windows
// ---------------------------------------------------------------------------

#[test]
fn history_window_split_parsed() {
    let cfg = load_ok("[providers.history]\nvoice_window_size = 25\ntext_window_size = 60\n");
    assert_eq!(cfg.voice_window_size, 25);
    assert_eq!(cfg.text_window_size, 60);
}

#[test]
fn legacy_window_size_rejected() {
    assert_err(
        load("[providers.history]\nwindow_size = 50\n"),
        "window_size",
    );
}

#[test]
fn voice_window_must_be_positive_int() {
    assert_err(
        load("[providers.history]\nvoice_window_size = 0\n"),
        "voice_window_size",
    );
}

#[test]
fn text_window_must_be_positive_int() {
    assert_err(
        load("[providers.history]\ntext_window_size = -1\n"),
        "text_window_size",
    );
}

#[test]
fn text_silence_gap_fold_defaults_to_zero() {
    approx(load_ok("").text_silence_gap_fold_seconds, 0.0);
}

#[test]
fn text_silence_gap_fold_parsed() {
    let cfg = load_ok("[providers.history]\ntext_silence_gap_fold_seconds = 1800\n");
    approx(cfg.text_silence_gap_fold_seconds, 1800.0);
}

#[test]
fn text_silence_gap_fold_rejects_negative() {
    // Ports test_text_silence_gap_fold_rejects_negative. The `-1` is a TOML
    // integer, so the message tail must read `got -1` (int), byte-for-byte with
    // Python's `got {v}` — not the padded `got -1.0`.
    assert_err_eq(
        load("[providers.history]\ntext_silence_gap_fold_seconds = -1\n"),
        "[providers.history].text_silence_gap_fold_seconds must be >= 0, got -1",
    );
}

#[test]
fn text_silence_gap_fold_rejects_negative_float() {
    // A negative TOML float renders through `fmt_num`; `-1.5` is unchanged and
    // matches Python's `got -1.5`.
    assert_err_eq(
        load("[providers.history]\ntext_silence_gap_fold_seconds = -1.5\n"),
        "[providers.history].text_silence_gap_fold_seconds must be >= 0, got -1.5",
    );
}

#[test]
fn coalesce_max_gap_rejects_negative() {
    // No Python ancestor test, but `parse_coalesce_gap` shares the sign-check
    // fix: a negative TOML integer must print `got -1`, not `got -1.0`.
    assert_err_eq(
        load("[providers.history]\ncoalesce_max_gap_seconds = -1\n"),
        "[providers.history].coalesce_max_gap_seconds must be >= 0, got -1",
    );
}

#[test]
fn text_silence_gap_fold_rejects_non_numeric() {
    assert_err(
        load("[providers.history]\ntext_silence_gap_fold_seconds = \"big\"\n"),
        "text_silence_gap_fold_seconds",
    );
}

#[test]
fn text_silence_gap_fold_accepts_zero() {
    approx(
        load_ok("[providers.history]\ntext_silence_gap_fold_seconds = 0\n")
            .text_silence_gap_fold_seconds,
        0.0,
    );
}

// ---------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------

#[test]
fn shipped_default_voice_budget() {
    let cfg = load_ok("");
    let v = cfg.budgets.get("voice").unwrap();
    assert_eq!(v.recent_history_tokens, 3000);
    assert_eq!(v.rag_tokens, 900);
    assert_eq!(v.dossier_tokens, 900);
    assert_eq!(v.summary_tokens, 600);
    assert_eq!(v.max_history_turns, 200);
    assert_eq!(v.max_rag_turns, 10);
    assert_eq!(v.max_rag_facts, 6);
    assert_eq!(v.max_dossier_people, 16);
    assert_eq!(v.total_tokens(), 3000 + 900 + 900 + 600 + 600 + 600);
}

#[test]
fn shipped_default_text_and_background() {
    let cfg = load_ok("");
    let text = cfg.budgets.get("text").unwrap();
    assert_eq!(text.recent_history_tokens, 8000);
    assert_eq!(text.total_tokens(), 8000 + 2400 + 2400 + 1600 + 1600 + 1600);
    let bg = cfg.budgets.get("background").unwrap();
    assert_eq!(bg.recent_history_tokens, 24000);
    assert_eq!(bg.total_tokens(), 24000 + 8000 + 8000 + 4000 + 4000 + 4000);
}

#[test]
fn partial_override_keeps_other_subcaps() {
    let cfg = load_ok("[budget.voice]\nrag_tokens = 5000\n");
    let v = cfg.budgets.get("voice").unwrap();
    assert_eq!(v.rag_tokens, 5000);
    assert_eq!(v.recent_history_tokens, 3000);
    assert_eq!(v.dossier_tokens, 900);
    assert_eq!(v.max_dossier_people, 16);
    assert_eq!(cfg.budgets.get("text").unwrap().recent_history_tokens, 8000);
}

#[test]
fn subcap_overrides_parsed() {
    let cfg = load_ok(
        "[budget.text]\nrecent_history_tokens = 4000\nrag_tokens = 1000\nmax_dossier_people = 12\n",
    );
    let b = cfg.budgets.get("text").unwrap();
    assert_eq!(b.recent_history_tokens, 4000);
    assert_eq!(b.rag_tokens, 1000);
    assert_eq!(b.max_dossier_people, 12);
}

#[test]
fn unknown_tier_rejected() {
    assert_err(
        load("[budget.mystery]\nrecent_history_tokens = 100\n"),
        "unknown budget tier",
    );
}

#[test]
fn budget_unknown_key_rejected() {
    assert_err(load("[budget.voice]\nwobble = 5\n"), "unknown keys");
}

#[test]
fn total_tokens_key_rejected_as_unknown() {
    assert_err(
        load("[budget.voice]\ntotal_tokens = 5000\n"),
        "unknown keys",
    );
}

#[test]
fn negative_cap_rejected() {
    assert_err(
        load("[budget.voice]\nrecent_history_tokens = -1\n"),
        "recent_history_tokens",
    );
}

// ---------------------------------------------------------------------------
// Memory retrieval
// ---------------------------------------------------------------------------

#[test]
fn retrieval_shipped_default_weights() {
    let r = load_ok("").memory_retrieval;
    approx(r.bm25_weight, 1.0);
    approx(r.recency_weight, 0.0);
    approx(r.importance_weight, 0.6);
    approx(r.embedding_weight, 0.0);
}

#[test]
fn retrieval_partial_override() {
    let r = load_ok("[memory.retrieval]\nimportance_weight = 1.5\nrecency_weight = 0.4\n")
        .memory_retrieval;
    approx(r.importance_weight, 1.5);
    approx(r.recency_weight, 0.4);
    approx(r.bm25_weight, 1.0);
}

#[test]
fn retrieval_unknown_key_rejected() {
    assert_err(
        load("[memory.retrieval]\nmagic_weight = 1\n"),
        "unknown keys",
    );
}

#[test]
fn retrieval_negative_weight_rejected() {
    // Ports test_negative_weight_rejected. `-1` is a TOML integer, so the tail
    // must read `got -1` (int), matching Python's `got {v}` before `float(v)`.
    assert_err_eq(
        load("[memory.retrieval]\nimportance_weight = -1\n"),
        "[memory.retrieval].importance_weight must be non-negative, got -1",
    );
}

#[test]
fn retrieval_negative_weight_float_rejected() {
    // Negative float renders through `fmt_num`, matching Python's `got -1.5`.
    assert_err_eq(
        load("[memory.retrieval]\nimportance_weight = -1.5\n"),
        "[memory.retrieval].importance_weight must be non-negative, got -1.5",
    );
}

#[test]
fn retrieval_non_numeric_rejected() {
    assert_err(
        load("[memory.retrieval]\nimportance_weight = \"high\"\n"),
        "non-negative number",
    );
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

#[test]
fn embedding_shipped_default_is_off() {
    assert_eq!(load_ok("").embedding.backend, "off");
}

#[test]
fn embedding_override_to_hash() {
    let cfg = load_ok("[providers.embedding]\nbackend = \"hash\"\ndim = 128\n");
    assert_eq!(cfg.embedding.backend, "hash");
    assert_eq!(cfg.embedding.dim, 128);
}

#[test]
fn embedding_unknown_backend_rejected() {
    assert_err(
        load("[providers.embedding]\nbackend = \"magic\"\n"),
        "is unknown",
    );
}

#[test]
fn embedding_unknown_key_rejected() {
    assert_err(load("[providers.embedding]\nweird = 1\n"), "unknown keys");
}

#[test]
fn embedding_non_positive_dim_rejected() {
    assert_err(load("[providers.embedding]\ndim = 0\n"), "must be > 0");
}

#[test]
fn embedding_non_int_dim_rejected() {
    assert_err(
        load("[providers.embedding]\ndim = \"wide\"\n"),
        "must be a positive integer",
    );
}

#[test]
fn embedding_default_fastembed_model_is_bge_small() {
    let cfg = load_ok("");
    assert_eq!(cfg.embedding.fastembed_model, "BAAI/bge-small-en-v1.5");
    assert!(cfg.embedding.fastembed_cache_dir.is_none());
}

#[test]
fn embedding_fastembed_model_override() {
    let cfg = load_ok(
        "[providers.embedding]\nbackend = \"fastembed\"\nfastembed_model = \"BAAI/bge-base-en-v1.5\"\nfastembed_cache_dir = \"/var/cache/fastembed\"\n",
    );
    assert_eq!(cfg.embedding.backend, "fastembed");
    assert_eq!(cfg.embedding.fastembed_model, "BAAI/bge-base-en-v1.5");
    assert_eq!(
        cfg.embedding.fastembed_cache_dir.as_deref(),
        Some("/var/cache/fastembed")
    );
}

#[test]
fn embedding_empty_fastembed_model_rejected() {
    assert_err(
        load("[providers.embedding]\nfastembed_model = \"\"\n"),
        "non-empty string",
    );
}

#[test]
fn embedding_non_string_cache_dir_rejected() {
    assert_err(
        load("[providers.embedding]\nfastembed_cache_dir = 42\n"),
        "must be a string",
    );
}

// ---------------------------------------------------------------------------
// Channel overrides
// ---------------------------------------------------------------------------

#[test]
fn no_channels_section_is_empty_map() {
    assert!(load_ok("").channels.is_empty());
}

#[test]
fn channel_overrides_parsed() {
    let cfg = load_ok(
        "[channels.12345]\nhistory_window_size = 8\nmessage_rendering = \"name_only\"\nprompt_layers = [\"character_card\", \"operating_mode\"]\n",
    );
    let over = cfg.channels.get(&12345).unwrap();
    assert_eq!(over.history_window_size, Some(8));
    assert_eq!(over.message_rendering.as_deref(), Some("name_only"));
    assert_eq!(
        over.prompt_layers,
        Some(vec![
            "character_card".to_owned(),
            "operating_mode".to_owned()
        ])
    );
}

#[test]
fn voice_window_for_falls_back_to_default() {
    let cfg = load_ok(
        "[providers.history]\nvoice_window_size = 20\n\n[channels.12345]\nhistory_window_size = 8\n",
    );
    assert_eq!(cfg.voice_window_for(Some(12345)), 8);
    assert_eq!(cfg.voice_window_for(Some(99999)), 20);
    assert_eq!(cfg.voice_window_for(None), 20);
}

#[test]
fn text_window_for_falls_back_to_default() {
    let cfg = load_ok(
        "[providers.history]\ntext_window_size = 50\n\n[channels.12345]\nhistory_window_size = 8\n",
    );
    assert_eq!(cfg.text_window_for(Some(12345)), 8);
    assert_eq!(cfg.text_window_for(Some(99999)), 50);
    assert_eq!(cfg.text_window_for(None), 50);
}

#[test]
fn channel_invalid_window_rejected() {
    assert_err(
        load("[channels.12345]\nhistory_window_size = 0\n"),
        "must be positive",
    );
}

#[test]
fn channel_invalid_message_rendering_rejected() {
    assert_err(
        load("[channels.12345]\nmessage_rendering = \"garble\"\n"),
        "message_rendering",
    );
}

#[test]
fn channel_total_tokens_key_ignored() {
    let cfg = load_ok("[channels.12345]\ntotal_tokens = 2000\n");
    assert!(cfg.channels.contains_key(&12345));
}

#[test]
fn budget_for_returns_tier_base() {
    let cfg = load_ok("");
    let base = *cfg.budgets.get("text").unwrap();
    assert_eq!(cfg.budget_for("text"), base);
}

// ---------------------------------------------------------------------------
// Budget curves
// ---------------------------------------------------------------------------

#[test]
fn no_curves_section_empty_map() {
    assert!(load_ok("").budget_curves.is_empty());
}

#[test]
fn curve_parsed() {
    let cfg = load_ok(
        "[budget.model_curves.\"claude-opus-4-7\"]\nrecent_history_tokens = 2.0\nrag_tokens = 1.5\n",
    );
    let curve = cfg.budget_curves.get("claude-opus-4-7").unwrap();
    approx(curve.recent_history_tokens, 2.0);
    approx(curve.rag_tokens, 1.5);
    approx(curve.dossier_tokens, 1.0);
}

#[test]
fn unknown_curve_field_rejected() {
    assert_err(
        load("[budget.model_curves.\"claude-opus-4-7\"]\nno_such_field = 1.5\n"),
        "no_such_field",
    );
}

#[test]
fn total_tokens_curve_field_rejected() {
    assert_err(
        load("[budget.model_curves.\"claude-opus-4-7\"]\ntotal_tokens = 2.0\n"),
        "total_tokens",
    );
}

#[test]
fn non_positive_multiplier_rejected() {
    assert_err(
        load("[budget.model_curves.\"claude-opus-4-7\"]\nrag_tokens = 0.0\n"),
        "rag_tokens",
    );
}

#[test]
fn non_numeric_multiplier_rejected() {
    assert_err(
        load("[budget.model_curves.\"claude-opus-4-7\"]\nrag_tokens = \"big\"\n"),
        "rag_tokens",
    );
}

#[test]
fn budget_for_applies_curve_when_model_matches() {
    let cfg = load_ok(
        "[llm.fast]\nmodel = \"claude-opus-4-7\"\napi_key_env = \"X\"\n\n[budget.model_curves.\"claude-opus-4-7\"]\nrecent_history_tokens = 2.0\n",
    );
    let base = *cfg.budgets.get("voice").unwrap();
    let b = cfg.budget_for("voice");
    assert_eq!(b.recent_history_tokens, base.recent_history_tokens * 2);
    assert_eq!(b.rag_tokens, base.rag_tokens);
    assert_eq!(
        b.total_tokens(),
        base.total_tokens() + base.recent_history_tokens
    );
}

#[test]
fn budget_for_no_curve_returns_base() {
    let cfg = load_ok("");
    let base = *cfg.budgets.get("voice").unwrap();
    assert_eq!(cfg.budget_for("voice"), base);
}

#[test]
fn budget_for_curve_not_applied_for_different_model() {
    let cfg = load_ok(
        "[llm.fast]\nmodel = \"claude-sonnet-4-6\"\napi_key_env = \"X\"\n\n[budget.model_curves.\"claude-opus-4-7\"]\nrecent_history_tokens = 2.0\n",
    );
    let base = *cfg.budgets.get("voice").unwrap();
    assert_eq!(cfg.budget_for("voice"), base);
}

// ---------------------------------------------------------------------------
// Turn detection
// ---------------------------------------------------------------------------

#[test]
fn turn_detection_omitted_section_uses_default() {
    assert_eq!(load_ok("").turn_detection.strategy, "deepgram");
}

#[test]
fn ten_plus_smart_turn_strategy_parsed() {
    let cfg = load_ok("[providers.turn_detection]\nstrategy = \"ten+smart_turn\"\n");
    assert_eq!(cfg.turn_detection.strategy, "ten+smart_turn");
}

#[test]
fn deepgram_strategy_explicit() {
    let cfg = load_ok("[providers.turn_detection]\nstrategy = \"deepgram\"\n");
    assert_eq!(cfg.turn_detection.strategy, "deepgram");
}

#[test]
fn idle_fallback_s_default() {
    approx(load_ok("").turn_detection.local.idle_fallback_s, 1.5);
}

#[test]
fn idle_fallback_s_parsed() {
    let cfg = load_ok("[providers.turn_detection.local]\nidle_fallback_s = 2.5\n");
    approx(cfg.turn_detection.local.idle_fallback_s, 2.5);
}

#[test]
fn idle_fallback_s_must_be_number() {
    assert_err(
        load("[providers.turn_detection.local]\nidle_fallback_s = \"soon\"\n"),
        "[providers.turn_detection.local].idle_fallback_s",
    );
}

#[test]
fn unknown_strategy_rejected() {
    assert_err(
        load("[providers.turn_detection]\nstrategy = \"mystery\"\n"),
        "[providers.turn_detection].strategy",
    );
}

#[test]
fn strategy_must_be_string() {
    assert_err(
        load("[providers.turn_detection]\nstrategy = 42\n"),
        "[providers.turn_detection].strategy",
    );
}

// ---------------------------------------------------------------------------
// STT
// ---------------------------------------------------------------------------

#[test]
fn stt_omitted_section_uses_defaults() {
    let cfg = load_ok("");
    assert_eq!(cfg.stt.backend, "deepgram");
    assert_eq!(cfg.stt.deepgram.endpointing_ms, 500);
    assert_eq!(cfg.stt.deepgram.utterance_end_ms, 1500);
    assert!(cfg.stt.deepgram.keyterms.is_empty());
}

#[test]
fn deepgram_knobs_overridden() {
    let cfg = load_ok(
        "[providers.stt.deepgram]\nmodel = \"nova-2\"\nlanguage = \"es\"\nendpointing_ms = 300\nutterance_end_ms = 1200\nsmart_format = false\npunctuate = false\nkeyterms = [\"lifecycle mesh\", \"Tam\"]\nreplay_buffer_s = 7.5\nkeepalive_interval_s = 2.0\nreconnect_max_attempts = 8\nreconnect_backoff_cap_s = 32.0\nidle_close_s = 45.0\n",
    );
    let dg = &cfg.stt.deepgram;
    assert_eq!(dg.model, "nova-2");
    assert_eq!(dg.language, "es");
    assert_eq!(dg.endpointing_ms, 300);
    assert_eq!(dg.utterance_end_ms, 1200);
    assert!(!dg.smart_format);
    assert!(!dg.punctuate);
    assert_eq!(
        dg.keyterms,
        vec!["lifecycle mesh".to_owned(), "Tam".to_owned()]
    );
    approx(dg.replay_buffer_s, 7.5);
    approx(dg.keepalive_interval_s, 2.0);
    assert_eq!(dg.reconnect_max_attempts, 8);
    approx(dg.reconnect_backoff_cap_s, 32.0);
    approx(dg.idle_close_s, 45.0);
}

#[test]
fn stt_unknown_backend_rejected() {
    assert_err(
        load("[providers.stt]\nbackend = \"mystery\"\n"),
        "[providers.stt].backend",
    );
}

#[test]
fn stt_backend_must_be_string() {
    assert_err(
        load("[providers.stt]\nbackend = 42\n"),
        "[providers.stt].backend",
    );
}

#[test]
fn endpointing_ms_must_be_int() {
    assert_err(
        load("[providers.stt.deepgram]\nendpointing_ms = 1.5\n"),
        "endpointing_ms",
    );
}

#[test]
fn keyterms_must_be_list_of_strings() {
    assert_err(
        load("[providers.stt.deepgram]\nkeyterms = [1, 2]\n"),
        "keyterms",
    );
}

// ---------------------------------------------------------------------------
// Discord text + dm allowlist
// ---------------------------------------------------------------------------

#[test]
fn discord_text_loads_from_toml() {
    let cfg = load_ok(
        "[discord.text]\nrespond_to_typing = false\ntyping_backoff_initial_s = 2.5\ntyping_backoff_max_s = 60.0\n",
    );
    assert!(!cfg.discord_text.respond_to_typing);
    approx(cfg.discord_text.typing_backoff_initial_s, 2.5);
    approx(cfg.discord_text.typing_backoff_max_s, 60.0);
}

#[test]
fn respond_to_typing_must_be_bool() {
    assert_err(
        load("[discord.text]\nrespond_to_typing = \"yes\"\n"),
        "respond_to_typing",
    );
}

#[test]
fn discord_text_unknown_key_rejected() {
    assert_err(load("[discord.text]\nunknown_knob = 1\n"), "unknown");
}

#[test]
fn backoff_max_must_not_be_below_initial() {
    assert_err(
        load("[discord.text]\ntyping_backoff_initial_s = 5.0\ntyping_backoff_max_s = 1.0\n"),
        "typing_backoff_max_s",
    );
}

#[test]
fn dm_allowlist_parsed_into_ints() {
    let cfg = load_ok("[discord]\ndm_allowlist = [111, 222]\n");
    assert_eq!(cfg.dm_allowlist, vec![111, 222]);
}

#[test]
fn dm_allowlist_absent_defaults_empty() {
    assert!(load_ok("").dm_allowlist.is_empty());
}

#[test]
fn dm_allowlist_empty_list_is_valid() {
    assert!(
        load_ok("[discord]\ndm_allowlist = []\n")
            .dm_allowlist
            .is_empty()
    );
}

#[test]
fn dm_allowlist_invalid_rejected() {
    for value in [
        "dm_allowlist = \"x\"",
        "dm_allowlist = [\"a\"]",
        "dm_allowlist = [true]",
        "dm_allowlist = [1.5]",
    ] {
        assert_err(load(&format!("[discord]\n{value}\n")), "dm_allowlist");
    }
}

// ---------------------------------------------------------------------------
// Image tools + shared [llm] scalars
// ---------------------------------------------------------------------------

#[test]
fn llm_slot_parses_image_tools_and_multimodal() {
    let cfg = load_ok("[llm.prose]\nmodel = \"x/y\"\nimage_tools = true\nmultimodal = true\n");
    let slot = cfg.llm.get("prose").unwrap();
    assert!(slot.image_tools);
    assert!(slot.multimodal);
}

#[test]
fn image_tools_defaults_false() {
    let cfg = load_ok("[llm.prose]\nmodel = \"x/y\"\n");
    assert!(!cfg.llm.get("prose").unwrap().image_tools);
    assert!(!cfg.llm.get("prose").unwrap().multimodal);
}

#[test]
fn image_description_model_parsed_at_llm_level() {
    let cfg = load_ok("[llm]\nimage_description_model = \"openai/gpt-4o\"\n");
    assert_eq!(cfg.image_description_model, "openai/gpt-4o");
    // must not be treated as an unknown slot
    assert_eq!(
        cfg.llm.keys().cloned().collect::<BTreeSet<_>>(),
        ["background", "fast", "prose"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    );
}

#[test]
fn image_description_model_defaults_empty() {
    assert!(load_ok("").image_description_model.is_empty());
}

#[test]
fn llm_max_concurrent_loads_from_toml() {
    assert_eq!(
        load_ok("[llm]\nmax_concurrent_requests = 8\n").llm_max_concurrent_requests,
        8
    );
}

#[test]
fn llm_max_concurrent_not_treated_as_slot() {
    let cfg = load_ok("[llm]\nmax_concurrent_requests = 8\n");
    assert_eq!(
        cfg.llm.keys().cloned().collect::<BTreeSet<_>>(),
        ["background", "fast", "prose"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    );
}

#[test]
fn llm_max_concurrent_must_be_positive_int() {
    assert_err(
        load("[llm]\nmax_concurrent_requests = 0\n"),
        "max_concurrent_requests",
    );
}

// ---------------------------------------------------------------------------
// Focus + tools
// ---------------------------------------------------------------------------

#[test]
fn focus_loads_from_toml() {
    let cfg = load_ok(
        "[focus]\nunread_nudge_enabled = false\nnudge_debounce_seconds = 10\ncatch_up_limit = 50\n",
    );
    assert!(!cfg.focus.unread_nudge_enabled);
    approx(cfg.focus.nudge_debounce_seconds, 10.0);
    assert_eq!(cfg.focus.catch_up_limit, 50);
}

#[test]
fn focus_catch_up_limit_must_be_positive_int() {
    assert_err(load("[focus]\ncatch_up_limit = 0\n"), "catch_up_limit");
}

#[test]
fn focus_catch_up_limit_rejects_non_int() {
    assert_err(load("[focus]\ncatch_up_limit = 2.5\n"), "catch_up_limit");
}

#[test]
fn focus_must_be_bool() {
    assert_err(
        load("[focus]\nunread_nudge_enabled = 1\n"),
        "unread_nudge_enabled",
    );
}

#[test]
fn focus_unknown_key_rejected() {
    assert_err(load("[focus]\nbogus = 1\n"), "unknown");
}

#[test]
fn tools_loads_from_toml() {
    assert_eq!(
        load_ok("[tools]\nloop_max_iterations = 9\n")
            .tools
            .loop_max_iterations,
        9
    );
}

#[test]
fn tools_must_be_positive_int() {
    assert_err(
        load("[tools]\nloop_max_iterations = 0\n"),
        "loop_max_iterations",
    );
}

#[test]
fn tools_unknown_key_rejected() {
    assert_err(load("[tools]\nbogus = 1\n"), "unknown");
}

// ---------------------------------------------------------------------------
// Memory worker configs
// ---------------------------------------------------------------------------

#[test]
fn memory_workers_load_from_toml() {
    let cfg = load_ok(
        "[providers.memory.rolling_summary]\nturns_threshold = 4\ntick_interval_s = 1.5\n[providers.memory.rich_note]\nbatch_size = 3\ntick_interval_s = 7\nparticipants_max = 12\n[providers.memory.people_dossier]\ntick_interval_s = 11.0\n[providers.memory.reflection]\nturns_threshold = 8\nmax_reflections_per_tick = 1\nmax_turns_per_tick = 25\nrecent_facts_limit = 5\ntick_interval_s = 90.0\n[providers.memory.fact_supersede]\nbatch_size = 2\ntick_interval_s = 120.0\npriors_max = 6\n",
    );
    let mem = &cfg.memory_providers;
    assert_eq!(mem.rolling_summary.turns_threshold, 4);
    approx(mem.rolling_summary.tick_interval_s, 1.5);
    assert_eq!(mem.rich_note.batch_size, 3);
    approx(mem.rich_note.tick_interval_s, 7.0);
    assert_eq!(mem.rich_note.participants_max, 12);
    approx(mem.people_dossier.tick_interval_s, 11.0);
    assert_eq!(mem.reflection.turns_threshold, 8);
    assert_eq!(mem.reflection.max_reflections_per_tick, 1);
    assert_eq!(mem.reflection.max_turns_per_tick, 25);
    assert_eq!(mem.reflection.recent_facts_limit, 5);
    approx(mem.reflection.tick_interval_s, 90.0);
    assert_eq!(mem.fact_supersede.batch_size, 2);
    approx(mem.fact_supersede.tick_interval_s, 120.0);
    assert_eq!(mem.fact_supersede.priors_max, 6);
}

#[test]
fn memory_worker_partial_override_keeps_other_knobs() {
    let cfg = load_ok("[providers.memory.rich_note]\nbatch_size = 3\n");
    assert_eq!(cfg.memory_providers.rich_note.batch_size, 3);
    approx(cfg.memory_providers.rich_note.tick_interval_s, 15.0);
}

#[test]
fn memory_worker_must_be_positive() {
    assert_err(
        load("[providers.memory.rich_note]\nbatch_size = 0\n"),
        "batch_size",
    );
}

#[test]
fn memory_worker_unknown_knob_rejected() {
    assert_err(
        load("[providers.memory.rolling_summary]\nbogus = 1\n"),
        "unknown",
    );
}

#[test]
fn memory_worker_unknown_subtable_rejected() {
    assert_err(
        load("[providers.memory.bogus_worker]\ntick_interval_s = 1\n"),
        "bogus_worker",
    );
}

// ---------------------------------------------------------------------------
// Default-profile drift
// ---------------------------------------------------------------------------

#[test]
fn default_profile_enables_every_default_projector() {
    let cfg = load_ok("");
    assert_eq!(
        cfg.memory_providers.projectors,
        vec![
            "rolling_summary".to_owned(),
            "rich_note".to_owned(),
            "people_dossier".to_owned(),
            "reflection".to_owned(),
            "fact_supersede".to_owned(),
        ]
    );
}

#[test]
fn history_window_fallbacks_match_dataclass_defaults() {
    let cfg = load_custom(Some(""), "").unwrap();
    assert_eq!(
        cfg.voice_window_size,
        CharacterConfig::default().voice_window_size
    );
    assert_eq!(
        cfg.text_window_size,
        CharacterConfig::default().text_window_size
    );
}
