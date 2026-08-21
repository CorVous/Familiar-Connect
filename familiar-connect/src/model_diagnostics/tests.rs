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
    // Matches the model and silences the image advisory; isolates the tools
    // mismatch.
    cfg.multimodal = Some(true);
    cfg.image_tools = true;
    let found = compare_slot("prose", &cfg, &vision_no_tools());
    assert_eq!(found.len(), 1, "one mismatch: {found:?}");
    assert_eq!(found[0].flag, "tool_calling");
    assert_eq!(found[0].kind, MismatchKind::DeclaredUnsupported);
    assert_eq!(found[0].slot, "prose");
    assert_eq!(found[0].model, "vendor/vision");
    assert_eq!(
        found[0].remediation,
        "pick a model whose supported_parameters include 'tools' — \
         [llm.prose].tool_calling cannot be turned off"
    );
}

#[test]
fn multimodal_declared_on_a_text_only_model_is_a_mismatch() {
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = true;
    cfg.multimodal = Some(true);
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

/// Tool calling is mandatory, so a model that supports it earns no advisory.
#[test]
fn tools_supported_and_declared_says_nothing() {
    let cfg = slot("vendor/text"); // tool_calling defaults true
    assert_eq!(compare_slot("fast", &cfg, &text_tools()), Vec::new());
}

#[test]
fn image_input_supported_but_not_declared_is_one_advisory() {
    let mut cfg = slot("vendor/vision");
    cfg.tool_calling = false;
    let found = compare_slot("prose", &cfg, &vision_no_tools());
    // Model has no tools and tool_calling is off — nothing to say there. The
    // advisory is about `image_tools` alone: `multimodal` is unset, and unset
    // now auto-detects, so there is nothing to advise.
    assert_eq!(found.len(), 1, "one advisory: {found:?}");
    assert_eq!(found[0].flag, "image_tools");
    assert_eq!(found[0].kind, MismatchKind::SupportedNotDeclared);
    assert_eq!(
        found[0].remediation,
        "model accepts image input; set [llm.prose].image_tools = true \
         to register view_image (multimodal auto-detects)"
    );
}

#[test]
fn an_unset_multimodal_is_never_flagged_on_a_text_only_model() {
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = true;
    // multimodal unset: auto-detection will answer `false` from this catalog,
    // so there is no operator claim to contradict.
    let found = compare_slot("prose", &cfg, &text_tools());
    assert_eq!(found, Vec::new(), "unset is not a declaration: {found:?}");
}

#[test]
fn an_explicit_multimodal_false_is_never_flagged() {
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = true;
    cfg.multimodal = Some(false);
    assert_eq!(compare_slot("prose", &cfg, &text_tools()), Vec::new());
}

#[test]
fn a_fully_matched_slot_reports_nothing() {
    let mut cfg = slot("vendor/vl");
    cfg.tool_calling = true;
    cfg.multimodal = Some(true);
    cfg.image_tools = true;
    let model = caps("vendor/vl", &["text", "image"], &["temperature", "tools"]);
    assert_eq!(compare_slot("prose", &cfg, &model), Vec::new());
}

#[test]
fn declaring_only_image_tools_silences_the_image_advisory() {
    let mut cfg = slot("vendor/vision");
    // Isolate the image side: this model lists no `tools` parameter.
    cfg.tool_calling = false;
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
    cfg.multimodal = Some(true);
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
    cfg.multimodal = Some(true);
    slots.insert("prose".to_owned(), cfg);
    assert_eq!(audit_slots(&slots, &[]).mismatches, Vec::new());
}

#[test]
fn a_variant_suffixed_model_matches_its_base_id() {
    let mut slots = BTreeMap::new();
    let mut cfg = slot("vendor/text:free");
    cfg.multimodal = Some(true);
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
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = false;
    let msg = focus_unreachable_message("text", "prose", &cfg, true).expect("flagged");
    assert_eq!(
        msg,
        "[llm.prose].tool_calling = false is unsupported: silence and shift_focus \
         are both tool calls, so the text surface can never decline to reply and \
         can only follow a direct ping to another text channel, never move on its \
         own judgement — set [llm.prose].tool_calling = true"
    );
}

#[test]
fn tool_calling_off_on_voice_pins_focus_for_the_session() {
    // Voice has no direct-ping fallback, so the stronger claim holds there.
    let mut cfg = slot("vendor/text");
    cfg.tool_calling = false;
    let msg = focus_unreachable_message("voice", "fast", &cfg, false).expect("flagged");
    assert_eq!(
        msg,
        "[llm.fast].tool_calling = false is unsupported: silence and shift_focus \
         are both tool calls, so the voice surface can never decline to reply and \
         the voice channel focus is fixed for the session — set \
         [llm.fast].tool_calling = true"
    );
}

#[test]
fn tool_calling_on_is_silent() {
    let cfg = slot("vendor/text"); // tool_calling defaults true
    assert_eq!(
        focus_unreachable_message("voice", "fast", &cfg, false),
        None
    );
}

// ---------------------------------------------------------------------------
// Catalog cache (#204) — every case is local file I/O; nothing here dials out.
// ---------------------------------------------------------------------------

mod cache {
    use super::{caps, slot, text_tools, vision_no_tools};
    use crate::model_diagnostics::cache::{
        CACHE_TTL_HOURS, CachedCatalog, apply_detected_multimodal, default_cache_path,
        detect_multimodal, is_stale, read_cache, write_cache,
    };
    use chrono::{Duration, TimeZone as _, Utc};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn now() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 8, 16, 12, 0, 0).unwrap()
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("models.json");
        let models = vec![text_tools(), vision_no_tools()];
        write_cache(&path, &models, now()).expect("write");
        let got = read_cache(&path).expect("read back");
        assert_eq!(got.models, models);
        assert_eq!(got.fetched_at, crate::support::time::iso_utc(now()));
    }

    #[test]
    fn a_missing_cache_reads_as_absent() {
        let dir = TempDir::new().unwrap();
        assert_eq!(read_cache(&dir.path().join("nope.json")), None);
    }

    #[test]
    fn a_corrupt_cache_reads_as_absent_not_fatal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models.json");
        for junk in ["", "{", "not json at all", r#"{"models": 3}"#] {
            std::fs::write(&path, junk).unwrap();
            assert_eq!(read_cache(&path), None, "junk {junk:?} must read as absent");
        }
    }

    #[test]
    fn a_corrupt_cache_is_overwritten_by_the_next_write() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(&path, "{{{").unwrap();
        write_cache(&path, &[text_tools()], now()).expect("write over junk");
        assert_eq!(read_cache(&path).expect("read back").models, [text_tools()]);
    }

    #[test]
    fn freshness_is_measured_against_the_ttl() {
        let stamp = crate::support::time::iso_utc(now());
        assert!(!is_stale(&stamp, now(), CACHE_TTL_HOURS));
        assert!(!is_stale(
            &stamp,
            now() + Duration::hours(CACHE_TTL_HOURS - 1),
            CACHE_TTL_HOURS
        ));
        assert!(is_stale(
            &stamp,
            now() + Duration::hours(CACHE_TTL_HOURS),
            CACHE_TTL_HOURS
        ));
    }

    #[test]
    fn an_unparseable_stamp_counts_as_stale() {
        assert!(is_stale("", now(), CACHE_TTL_HOURS));
        assert!(is_stale("yesterday", now(), CACHE_TTL_HOURS));
    }

    #[test]
    fn a_future_stamp_counts_as_stale() {
        // Clock skew or a mangled file must not pin the cache forever.
        let ahead = crate::support::time::iso_utc(now() + Duration::hours(1));
        assert!(is_stale(&ahead, now(), CACHE_TTL_HOURS));
    }

    #[test]
    fn detect_reads_input_modalities_and_falls_back_to_the_base_id() {
        let catalog = [text_tools(), vision_no_tools()];
        assert_eq!(detect_multimodal(&catalog, "vendor/vision"), Some(true));
        assert_eq!(detect_multimodal(&catalog, "vendor/text"), Some(false));
        assert_eq!(
            detect_multimodal(&catalog, "vendor/vision:free"),
            Some(true)
        );
        assert_eq!(detect_multimodal(&catalog, "vendor/unknown"), None);
    }

    fn slots(
        pairs: &[(&str, &str, Option<bool>)],
    ) -> BTreeMap<String, crate::config::LLMSlotConfig> {
        pairs
            .iter()
            .map(|(name, model, mm)| {
                let mut cfg = slot(model);
                cfg.multimodal = *mm;
                ((*name).to_owned(), cfg)
            })
            .collect()
    }

    #[test]
    fn unset_slots_follow_the_catalog() {
        let mut s = slots(&[
            ("prose", "vendor/vision", None),
            ("fast", "vendor/text", None),
        ]);
        let applied = apply_detected_multimodal(&mut s, &[text_tools(), vision_no_tools()]);
        assert_eq!(s["prose"].multimodal, Some(true));
        assert_eq!(s["fast"].multimodal, Some(false));
        assert_eq!(
            applied,
            vec![("fast".to_owned(), false), ("prose".to_owned(), true)]
        );
    }

    #[test]
    fn an_explicit_false_beats_a_catalog_that_says_vision() {
        let mut s = slots(&[("prose", "vendor/vision", Some(false))]);
        let applied = apply_detected_multimodal(&mut s, &[vision_no_tools()]);
        assert_eq!(s["prose"].multimodal, Some(false), "override untouched");
        assert!(applied.is_empty(), "no decision to report");
        assert!(!s["prose"].resolve_multimodal(Some(true)));
    }

    #[test]
    fn an_explicit_true_survives_a_catalog_that_says_text_only() {
        let mut s = slots(&[("prose", "vendor/text", Some(true))]);
        apply_detected_multimodal(&mut s, &[text_tools()]);
        assert_eq!(s["prose"].multimodal, Some(true));
    }

    #[test]
    fn a_model_the_catalog_does_not_list_keeps_the_configured_value() {
        let mut s = slots(&[("prose", "vendor/unheard", None)]);
        assert!(apply_detected_multimodal(&mut s, &[text_tools()]).is_empty());
        assert_eq!(s["prose"].multimodal, None);
        assert!(!s["prose"].resolve_multimodal(None));
    }

    #[test]
    fn no_cache_and_no_network_falls_back_to_the_configured_values() {
        // First-ever run: nothing on disk, nothing fetched.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models.json");
        assert_eq!(read_cache(&path), None);
        let mut s = slots(&[
            ("prose", "vendor/vision", None),
            ("fast", "vendor/text", Some(true)),
        ]);
        crate::model_diagnostics::cache::resolve_capabilities_from_cache(&mut s, &path);
        assert_eq!(s["prose"].multimodal, None, "unset stays unset");
        assert!(!s["prose"].resolve_multimodal(None), "and resolves false");
        assert_eq!(s["fast"].multimodal, Some(true), "explicit survives");
    }

    #[test]
    fn a_written_cache_drives_the_next_boot_resolution() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models.json");
        write_cache(&path, &[caps("vendor/vl", &["text", "image"], &[])], now()).unwrap();
        let mut s = slots(&[("prose", "vendor/vl", None)]);
        crate::model_diagnostics::cache::resolve_capabilities_from_cache(&mut s, &path);
        assert_eq!(s["prose"].multimodal, Some(true));
    }

    #[test]
    fn the_cache_lives_under_the_platform_cache_dir_not_the_familiars_root() {
        let path = default_cache_path();
        assert_eq!(path.file_name().unwrap(), "openrouter-models.json");
        assert!(path.is_absolute() || path.starts_with("data"));
        assert!(
            !path.to_string_lossy().contains("familiars"),
            "regenerable catalog must not sit in the state tree: {}",
            path.display()
        );
    }

    #[test]
    fn the_cached_shape_is_stable_json() {
        // Pinned so an older binary can still read a newer cache's fields.
        let cached = CachedCatalog {
            fetched_at: "2026-08-16T12:00:00.000000+00:00".to_owned(),
            models: vec![caps("a/b", &["text"], &["tools"])],
        };
        let body = serde_json::to_string(&cached).unwrap();
        assert_eq!(
            body,
            r#"{"fetched_at":"2026-08-16T12:00:00.000000+00:00","models":[{"id":"a/b","input_modalities":["text"],"supported_parameters":["tools"]}]}"#
        );
        assert_eq!(
            serde_json::from_str::<CachedCatalog>(&body).unwrap(),
            cached
        );
    }
}

// ---------------------------------------------------------------------------
// Refresh driver (#204) — wiremock only; no test reaches a real host.
// ---------------------------------------------------------------------------

#[cfg(feature = "net")]
mod refresh {
    use super::{caps, slot};
    use crate::model_diagnostics::cache::{CACHE_TTL_HOURS, read_cache, write_cache};
    use crate::model_diagnostics::run_capability_audit;
    use chrono::{Duration as ChronoDuration, Utc};
    use std::collections::BTreeMap;
    use std::time::Duration;
    use tempfile::TempDir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn one_slot(model: &str) -> BTreeMap<String, crate::config::LLMSlotConfig> {
        std::iter::once(("prose".to_owned(), slot(model))).collect()
    }

    #[tokio::test]
    async fn a_refresh_writes_the_catalog_back_to_disk() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"data":[{"id":"vendor/vl","architecture":{"input_modalities":["text","image"]},
                   "supported_parameters":["tools"]}]}"#,
            ))
            .mount(&server)
            .await;
        let dir = TempDir::new().unwrap();
        let cache = dir.path().join("models.json");
        run_capability_audit(
            "sk".to_owned(),
            server.uri(),
            one_slot("vendor/vl"),
            cache.clone(),
        )
        .await;
        let written = read_cache(&cache).expect("refresh wrote the cache");
        assert_eq!(
            written.models,
            [caps("vendor/vl", &["text", "image"], &["tools"])]
        );
        assert!(!written.fetched_at.is_empty());
    }

    #[tokio::test]
    async fn a_stale_cache_triggers_a_refresh() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"data":[{"id":"vendor/new","architecture":{"input_modalities":["text"]}}]}"#,
            ))
            .mount(&server)
            .await;
        let dir = TempDir::new().unwrap();
        let cache = dir.path().join("models.json");
        let long_ago = Utc::now() - ChronoDuration::hours(CACHE_TTL_HOURS + 1);
        write_cache(&cache, &[caps("vendor/old", &["text"], &[])], long_ago).unwrap();
        run_capability_audit(
            "sk".to_owned(),
            server.uri(),
            one_slot("vendor/new"),
            cache.clone(),
        )
        .await;
        assert_eq!(
            read_cache(&cache).expect("cache").models,
            [caps("vendor/new", &["text"], &[])],
            "the stale entry was replaced"
        );
    }

    #[tokio::test]
    async fn an_unreachable_catalog_leaves_the_last_known_good_intact() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let dir = TempDir::new().unwrap();
        let cache = dir.path().join("models.json");
        let long_ago = Utc::now() - ChronoDuration::hours(CACHE_TTL_HOURS + 1);
        write_cache(&cache, &[caps("vendor/old", &["text"], &[])], long_ago).unwrap();
        run_capability_audit(
            "sk".to_owned(),
            server.uri(),
            one_slot("vendor/old"),
            cache.clone(),
        )
        .await;
        assert_eq!(
            read_cache(&cache).expect("cache").models,
            [caps("vendor/old", &["text"], &[])],
            "a failed refresh must not clobber the cache"
        );
    }

    #[tokio::test]
    async fn a_fresh_cache_short_circuits_the_network_entirely() {
        // TEST-NET-1: nothing listens and nothing routes. Reaching it at all
        // would blow the timeout — a fresh cache must not even try.
        let dir = TempDir::new().unwrap();
        let cache = dir.path().join("models.json");
        write_cache(
            &cache,
            &[caps("vendor/vl", &["text", "image"], &[])],
            Utc::now(),
        )
        .unwrap();
        // Generous: this guards against a HANG (a real dial-out to TEST-NET-1
        // blocks until connect timeout), not against slowness. A tight bound
        // here just flakes under load.
        tokio::time::timeout(
            Duration::from_secs(30),
            run_capability_audit(
                "sk".to_owned(),
                "http://192.0.2.1:9".to_owned(),
                one_slot("vendor/vl"),
                cache,
            ),
        )
        .await
        .expect("audited from cache without dialing out");
    }
}

// ---------------------------------------------------------------------------
// Boot path never blocks (#204)
// ---------------------------------------------------------------------------

#[test]
fn capability_resolution_on_the_boot_path_is_local_only() {
    use crate::model_diagnostics::cache::{resolve_capabilities_from_cache, write_cache};
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let cache = dir.path().join("models.json");
    write_cache(
        &cache,
        &[caps("vendor/vl", &["text", "image"], &[])],
        chrono::Utc::now(),
    )
    .unwrap();
    let mut slots: BTreeMap<String, LLMSlotConfig> =
        std::iter::once(("prose".to_owned(), slot("vendor/vl"))).collect();

    // Synchronous, no runtime, no client, no URL in the signature — the type
    // system already forbids a fetch here, so that is the real guarantee. A
    // wall-clock bound here only added load-sensitive flake (it tripped once
    // under concurrent builds) and pinned nothing the signature does not.
    resolve_capabilities_from_cache(&mut slots, &cache);
    assert_eq!(slots["prose"].multimodal, Some(true));
}
