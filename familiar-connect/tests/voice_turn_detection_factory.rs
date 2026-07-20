//! `create_local_turn_detector` tests (ported from Python
//! `tests/test_turn_detection_factory.py`).
//!
//! Smart Turn weights are resolved through the [`ModelDownloader`] seam — the
//! Rust analogue of Python's mocked `hf_hub_download`. The scripted downloader
//! records its `(repo_id, filename)` args and returns a canned path/error so CI
//! never touches the network.

// The scripted downloader hands back a `(handle, recorded-calls)` tuple.
#![allow(clippy::type_complexity)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use familiar_connect::config::LocalTurnConfig;
use familiar_connect::voice::turn_detection::{ModelDownloader, create_local_turn_detector_with};

/// Downloader that returns a canned result and records its call args.
struct ScriptedDownloader {
    result: Result<PathBuf, String>,
    calls: Arc<Mutex<Vec<(String, String)>>>,
}

impl ScriptedDownloader {
    fn ok(path: PathBuf) -> (Self, Arc<Mutex<Vec<(String, String)>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                result: Ok(path),
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }

    fn err(reason: &str) -> Self {
        Self {
            result: Err(reason.to_owned()),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ModelDownloader for ScriptedDownloader {
    fn download(&self, repo_id: &str, filename: &str) -> Result<PathBuf, String> {
        self.calls
            .lock()
            .unwrap()
            .push((repo_id.to_owned(), filename.to_owned()));
        self.result.clone()
    }
}

#[test]
fn returns_detector_with_downloaded_path() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("smart-turn-v3.2-cpu.onnx");
    std::fs::write(&model, b"fake").unwrap();

    let (downloader, calls) = ScriptedDownloader::ok(model.clone());
    let result = create_local_turn_detector_with(&LocalTurnConfig::default(), &downloader);

    let detector = result.expect("detector built");
    assert_eq!(detector.smart_turn_path, model);
    // Default repo + filename flow through verbatim.
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0],
        (
            "pipecat-ai/smart-turn-v3".to_owned(),
            "smart-turn-v3.2-cpu.onnx".to_owned()
        )
    );
}

#[test]
fn returns_none_when_download_fails() {
    let downloader = ScriptedDownloader::err("offline");
    let result = create_local_turn_detector_with(&LocalTurnConfig::default(), &downloader);
    assert!(result.is_none());
}

#[test]
fn returns_none_when_downloaded_path_missing() {
    // The downloader returns a path that no longer exists (cache rot).
    let (downloader, _) = ScriptedDownloader::ok(PathBuf::from("/nonexistent/x.onnx"));
    let result = create_local_turn_detector_with(&LocalTurnConfig::default(), &downloader);
    assert!(result.is_none());
}

#[test]
fn default_knobs_applied() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("smart-turn-v3.2-cpu.onnx");
    std::fs::write(&model, b"fake").unwrap();

    let (downloader, _) = ScriptedDownloader::ok(model);
    let detector =
        create_local_turn_detector_with(&LocalTurnConfig::default(), &downloader).unwrap();

    assert_eq!(detector.silence_ms, 200);
    assert_eq!(detector.speech_start_ms, 100);
    assert!((detector.vad_threshold - 0.5).abs() < 1e-9);
    assert!((detector.smart_turn_threshold - 0.5).abs() < 1e-9);
    assert_eq!(detector.vad_hop_size, 256);
    assert!((detector.idle_fallback_s - 1.5).abs() < 1e-9);
}

#[test]
fn passes_config_through() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("smart-turn-v3.2-gpu.onnx");
    std::fs::write(&model, b"fake").unwrap();

    let cfg = LocalTurnConfig {
        smart_turn_repo_id: "pipecat-ai/smart-turn-v3".to_owned(),
        smart_turn_filename: "smart-turn-v3.2-gpu.onnx".to_owned(),
        silence_ms: 300,
        speech_start_ms: 150,
        vad_threshold: 0.7,
        smart_turn_threshold: 0.6,
        vad_hop_size: 160,
        idle_fallback_s: 2.5,
    };
    let (downloader, calls) = ScriptedDownloader::ok(model);
    let detector = create_local_turn_detector_with(&cfg, &downloader).unwrap();

    assert_eq!(detector.silence_ms, 300);
    assert_eq!(detector.speech_start_ms, 150);
    assert!((detector.vad_threshold - 0.7).abs() < 1e-9);
    assert!((detector.smart_turn_threshold - 0.6).abs() < 1e-9);
    assert_eq!(detector.vad_hop_size, 160);
    assert!((detector.idle_fallback_s - 2.5).abs() < 1e-9);
    assert_eq!(
        calls.lock().unwrap()[0],
        (
            "pipecat-ai/smart-turn-v3".to_owned(),
            "smart-turn-v3.2-gpu.onnx".to_owned()
        )
    );
}

#[test]
fn custom_repo_id_flows_through() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("custom.onnx");
    std::fs::write(&model, b"fake").unwrap();

    let cfg = LocalTurnConfig {
        smart_turn_repo_id: "acme/custom-turn".to_owned(),
        smart_turn_filename: "custom.onnx".to_owned(),
        ..LocalTurnConfig::default()
    };
    let (downloader, calls) = ScriptedDownloader::ok(model);
    create_local_turn_detector_with(&cfg, &downloader).unwrap();

    assert_eq!(
        calls.lock().unwrap()[0],
        ("acme/custom-turn".to_owned(), "custom.onnx".to_owned())
    );
}
