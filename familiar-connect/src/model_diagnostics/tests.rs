//! Unit tests for the startup model-configuration diagnostics (#218).
//!
//! Pure: no test here performs a network call. The catalog fixtures mirror the
//! shape verified against the live `GET https://openrouter.ai/api/v1/models`.

use std::collections::BTreeMap;

use super::{
    AuditReport, CapabilityMismatch, MismatchKind, ModelCapabilities, audit_slots, compare_slot,
    focus_unreachable_message, log_audit, mismatch_line, parse_model_catalog,
};
use crate::config::LLMSlotConfig;
use crate::diagnostics::testutil::{Capture, install_capture, singleton_guard, strip_ansi};
use tracing::Level;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn slot(model: &str) -> LLMSlotConfig {
    LLMSlotConfig {
        model: model.to_owned(),
        ..LLMSlotConfig::default()
    }
}

fn caps(id: &str, modalities: &[&str], params: &[&str]) -> ModelCapabilities {
    ModelCapabilities {
        id: id.to_owned(),
        input_modalities: modalities.iter().map(|s| (*s).to_owned()).collect(),
        supported_parameters: params.iter().map(|s| (*s).to_owned()).collect(),
    }
}

/// Text-only, tool-capable.
fn text_tools() -> ModelCapabilities {
    caps("vendor/text", &["text"], &["temperature", "tools"])
}

/// Image-capable, no tools.
fn vision_no_tools() -> ModelCapabilities {
    caps("vendor/vision", &["text", "image"], &["temperature"])
}

// ---------------------------------------------------------------------------
// parse_model_catalog
// ---------------------------------------------------------------------------

#[test]
fn parses_the_verified_openrouter_shape() {
    // Trimmed verbatim from a live GET /models response.
    let body = r#"{
      "data": [
        {
          "id": "qwen/qwen3-vl",
          "canonical_slug": "qwen/qwen3-vl-20260812",
          "name": "Qwen: Qwen3 VL",
          "context_length": 1048576,
          "architecture": {
            "modality": "text+image->text",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "tokenizer": "Qwen",
            "instruct_type": null
          },
          "pricing": {"prompt": "0.0000001"},
          "supported_parameters": ["temperature", "tool_choice", "tools", "top_p"]
        }
      ],
      "total_count": 1
    }"#;
    let catalog = parse_model_catalog(body).expect("data array parses");
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].id, "qwen/qwen3-vl");
    assert_eq!(catalog[0].input_modalities, vec!["text", "image"]);
    assert!(
        catalog[0]
            .supported_parameters
            .contains(&"tools".to_owned())
    );
}

#[test]
fn unparseable_response_yields_none() {
    assert_eq!(parse_model_catalog("<html>502 Bad Gateway</html>"), None);
    assert_eq!(parse_model_catalog(""), None);
    assert_eq!(parse_model_catalog(r#"{"error": "unauthorized"}"#), None);
}

#[test]
fn malformed_entries_are_skipped_not_fatal() {
    // Missing architecture, wrong-typed supported_parameters, no id, non-object.
    let body = r#"{"data": [
      {"id": "a/b"},
      {"id": "c/d", "architecture": {"input_modalities": "image"},
       "supported_parameters": "tools"},
      {"architecture": {"input_modalities": ["image"]}},
      "nonsense",
      {"id": "e/f", "architecture": {"input_modalities": ["text", 7, "image"]},
       "supported_parameters": ["tools", null]}
    ]}"#;
    let catalog = parse_model_catalog(body).expect("data array parses");
    assert_eq!(
        catalog,
        vec![
            caps("a/b", &[], &[]),
            caps("c/d", &[], &[]),
            caps("e/f", &["text", "image"], &["tools"]),
        ]
    );
}

// ---------------------------------------------------------------------------
// compare_slot — declared but unsupported (ERROR direction)
// ---------------------------------------------------------------------------

#[test]
fn tool_calling_declared_on_a_model_without_tools_is_a_mismatch() {
    let mut cfg = slot("vendor/vision");
    cfg.tool_calling = true;
    cfg.multimodal = true; // matches the model; isolates the tools mismatch
    let found = compare_slot("prose", &cfg, &vision_no_tools());
    assert_eq!(found.len(), 1, "one mismatch: {found:?}");
    assert_eq!(found[0].flag, "tool_calling");
    assert_eq!(found[0].kind, MismatchKind::DeclaredUnsupported);
    assert_eq!(found[0].slot, "prose");
    assert_eq!(found[0].model, "vendor/vision");
    assert_eq!(
        found[0].remediation,
        "set [llm.prose].tool_calling = false, or pick a model whose \
         supported_parameters include 'tools'"
    );
}

#[test]
fn multimodal_declared_on_a_text_only_model_is_a_mismatch() {
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = true;
    cfg.multimodal = true;
    let found = compare_slot("prose", &cfg, &text_tools());
    assert_eq!(found.len(), 1, "one mismatch: {found:?}");
    assert_eq!(found[0].flag, "multimodal");
    assert_eq!(found[0].kind, MismatchKind::DeclaredUnsupported);
    assert_eq!(
        found[0].remediation,
        "set [llm.prose].multimodal = false, or pick a model whose \
         input_modalities include 'image'"
    );
}

#[test]
fn image_tools_declared_on_a_text_only_model_is_a_mismatch() {
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = true;
    cfg.image_tools = true;
    let found = compare_slot("prose", &cfg, &text_tools());
    assert_eq!(found.len(), 1, "one mismatch: {found:?}");
    assert_eq!(found[0].flag, "image_tools");
    assert_eq!(found[0].kind, MismatchKind::DeclaredUnsupported);
    assert_eq!(
        found[0].remediation,
        "set [llm.prose].image_tools = false, or pick a model whose \
         input_modalities include 'image'"
    );
}

// ---------------------------------------------------------------------------
// compare_slot — the inverse (advisory / INFO direction)
// ---------------------------------------------------------------------------

#[test]
fn tools_supported_but_not_declared_is_advisory() {
    let cfg = slot("vendor/text"); // tool_calling defaults false
    let found = compare_slot("fast", &cfg, &text_tools());
    assert_eq!(found.len(), 1, "one advisory: {found:?}");
    assert_eq!(found[0].flag, "tool_calling");
    assert_eq!(found[0].kind, MismatchKind::SupportedNotDeclared);
    assert_eq!(
        found[0].remediation,
        "model supports tools; set [llm.fast].tool_calling = true to use them"
    );
}

#[test]
fn image_input_supported_but_not_declared_is_one_advisory() {
    let mut cfg = slot("vendor/vision");
    cfg.tool_calling = false;
    let found = compare_slot("prose", &cfg, &vision_no_tools());
    // Model has no tools and tool_calling is off — nothing to say there. One
    // image advisory, not two (multimodal + image_tools are the same fact).
    assert_eq!(found.len(), 1, "one advisory: {found:?}");
    assert_eq!(found[0].flag, "multimodal");
    assert_eq!(found[0].kind, MismatchKind::SupportedNotDeclared);
    assert_eq!(
        found[0].remediation,
        "model accepts image input; set [llm.prose].multimodal = true \
         (and image_tools = true for view_image) to use it"
    );
}

#[test]
fn a_fully_matched_slot_reports_nothing() {
    let mut cfg = slot("vendor/vl");
    cfg.tool_calling = true;
    cfg.multimodal = true;
    cfg.image_tools = true;
    let model = caps("vendor/vl", &["text", "image"], &["temperature", "tools"]);
    assert_eq!(compare_slot("prose", &cfg, &model), Vec::new());
}

#[test]
fn declaring_only_image_tools_silences_the_image_advisory() {
    let mut cfg = slot("vendor/vision");
    cfg.image_tools = true;
    let found = compare_slot("prose", &cfg, &vision_no_tools());
    assert_eq!(found, Vec::new(), "no advisory when image input is used");
}

// ---------------------------------------------------------------------------
// audit_slots
// ---------------------------------------------------------------------------

#[test]
fn unknown_model_is_recorded_not_flagged() {
    let mut slots = BTreeMap::new();
    let mut cfg = slot("vendor/never-heard-of-it");
    cfg.tool_calling = true;
    cfg.multimodal = true;
    slots.insert("prose".to_owned(), cfg);
    let report = audit_slots(&slots, &[text_tools()]);
    assert_eq!(report.mismatches, Vec::new(), "silence on unknown ids");
    assert_eq!(
        report.unknown_models,
        vec![("prose".to_owned(), "vendor/never-heard-of-it".to_owned())]
    );
}

#[test]
fn an_empty_catalog_flags_nothing() {
    let mut slots = BTreeMap::new();
    let mut cfg = slot("vendor/text");
    cfg.multimodal = true;
    slots.insert("prose".to_owned(), cfg);
    assert_eq!(audit_slots(&slots, &[]).mismatches, Vec::new());
}

#[test]
fn a_variant_suffixed_model_matches_its_base_id() {
    let mut slots = BTreeMap::new();
    let mut cfg = slot("vendor/text:free");
    cfg.multimodal = true;
    cfg.tool_calling = true;
    slots.insert("fast".to_owned(), cfg);
    let report = audit_slots(&slots, &[text_tools()]);
    assert_eq!(report.unknown_models, Vec::new());
    assert_eq!(report.mismatches.len(), 1);
    assert_eq!(report.mismatches[0].flag, "multimodal");
    assert_eq!(
        report.mismatches[0].model, "vendor/text:free",
        "the message names the configured string, not the base id"
    );
}

#[test]
fn slots_with_an_empty_model_are_ignored() {
    let mut slots = BTreeMap::new();
    slots.insert("prose".to_owned(), slot(""));
    let report = audit_slots(&slots, &[text_tools()]);
    assert_eq!(report, AuditReport::default());
}

// ---------------------------------------------------------------------------
// Log levels
// ---------------------------------------------------------------------------

fn mismatch(kind: MismatchKind) -> CapabilityMismatch {
    CapabilityMismatch {
        slot: "prose".to_owned(),
        model: "vendor/text".to_owned(),
        flag: "tool_calling",
        kind,
        remediation: "do the thing".to_owned(),
    }
}

#[test]
fn mismatch_line_names_slot_model_and_remediation() {
    let line = strip_ansi(&mismatch_line(&mismatch(MismatchKind::DeclaredUnsupported)));
    assert!(line.starts_with("[Config]"), "{line}");
    assert!(line.contains("slot=prose"), "{line}");
    assert!(line.contains("model=vendor/text"), "{line}");
    assert!(line.contains("capability=tool_calling"), "{line}");
    assert!(line.contains("fix=do the thing"), "{line}");
}

#[test]
fn declared_unsupported_logs_error_and_the_inverse_logs_info() {
    let _guard = singleton_guard();
    let cap = Capture::default();
    let _sub = install_capture(&cap);
    log_audit(&AuditReport {
        mismatches: vec![
            mismatch(MismatchKind::DeclaredUnsupported),
            mismatch(MismatchKind::SupportedNotDeclared),
        ],
        unknown_models: vec![("fast".to_owned(), "vendor/mystery".to_owned())],
    });
    let recs = cap.records();
    let levels: Vec<Level> = recs.iter().map(|r| r.level).collect();
    assert_eq!(
        levels,
        vec![Level::ERROR, Level::INFO, Level::DEBUG],
        "{recs:?}"
    );
    assert!(
        strip_ansi(&recs[2].message).contains("vendor/mystery"),
        "{:?}",
        recs[2]
    );
}

// ---------------------------------------------------------------------------
// Focus reachability (#221)
// ---------------------------------------------------------------------------

#[test]
fn tool_calling_off_on_text_names_the_ping_fallback() {
    let cfg = slot("vendor/text"); // tool_calling defaults false
    let msg = focus_unreachable_message("text", "prose", &cfg, true).expect("flagged");
    assert_eq!(
        msg,
        "[llm.prose].tool_calling = false disables every tool call on the text \
         surface, so shift_focus is unreachable — the familiar can only follow a \
         direct ping to another text channel, never move on its own judgement — \
         set [llm.prose].tool_calling = true to let the familiar change channels"
    );
}

#[test]
fn tool_calling_off_on_voice_pins_focus_for_the_session() {
    // Voice has no direct-ping fallback, so the stronger claim holds there.
    let cfg = slot("vendor/text");
    let msg = focus_unreachable_message("voice", "fast", &cfg, false).expect("flagged");
    assert_eq!(
        msg,
        "[llm.fast].tool_calling = false disables every tool call on the voice \
         surface, so shift_focus is unreachable and the voice channel focus is \
         fixed for the session — set [llm.fast].tool_calling = true to let the \
         familiar change channels"
    );
}

#[test]
fn tool_calling_on_is_silent() {
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = true;
    assert_eq!(
        focus_unreachable_message("voice", "fast", &cfg, false),
        None
    );
}
