//! Factory for the V1 phase-2 local turn-detection chain (subsystem 09; Python
//! `voice/turn_detection/factory.py`).
//!
//! Bundles the Smart Turn ONNX path + thresholds.
//! [`LocalTurnDetector::make_endpointer`] is called once per Discord user —
//! [`TenVad`](super::ten_vad::TenVad) is per-user (its native handle accumulates
//! state across frames) while the [`SmartTurnDetector`] classifier is stateless
//! and shared, lazily loaded on first use.
//!
//! Smart Turn weights are fetched from HuggingFace via a [`ModelDownloader`]
//! seam (the `local-turn`-gated production impl calls `hf_hub_download`; tests
//! inject a scripted double, DESIGN §4.8). [`create_local_turn_detector`]
//! returns [`None`] — never an error — when weights can't be resolved, so the
//! bot degrades to Deepgram-only endpointing (spec 09 §29).

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::LocalTurnConfig;
use crate::log_style as ls;
use crate::voice::turn_detection::endpointer::{OnTurnComplete, UtteranceEndpointer};
use crate::voice::turn_detection::smart_turn::{SmartTurnDetector, SmartTurnError};
use crate::voice::turn_detection::ten_vad::{TenVad, VadError};

/// Log target mirroring the Python logger name (grep-stable for ops).
const LOG_TARGET: &str = "familiar_connect.voice.turn_detection.factory";

/// Error building a live endpointer (native VAD or ONNX Smart Turn missing).
#[derive(Debug, thiserror::Error)]
pub enum TurnDetectionError {
    /// The TEN-VAD backend could not be constructed.
    #[error(transparent)]
    Vad(#[from] VadError),
    /// The Smart Turn ONNX model could not be loaded.
    #[error(transparent)]
    SmartTurn(#[from] SmartTurnError),
}

/// Resolves a model file from a HuggingFace repo to a local (cached) path.
///
/// The one seam `create_local_turn_detector` depends on — mirrors Python's
/// `hf_hub_download(repo_id, filename) -> path`. `Err(reason)` covers any
/// download / network / FS failure; the reason string is logged.
pub trait ModelDownloader {
    /// Resolve `filename` in `repo_id`, returning the on-disk path.
    ///
    /// # Errors
    /// Any download, network, or filesystem failure — as a human-readable reason.
    fn download(&self, repo_id: &str, filename: &str) -> Result<PathBuf, String>;
}

/// Per-process bundle: a shared Smart Turn classifier + a per-user TenVAD
/// factory.
///
/// The Smart Turn ONNX session loads once (lazily, on first
/// [`make_endpointer`](Self::make_endpointer)) and is shared; each call builds a
/// fresh [`TenVad`] (stateful native handle).
pub struct LocalTurnDetector {
    /// Resolved Smart Turn ONNX path.
    pub smart_turn_path: PathBuf,
    /// Silence ms after speech before Smart Turn fires.
    pub silence_ms: i64,
    /// Consecutive speech ms before "speaking" latches.
    pub speech_start_ms: i64,
    /// Smart Turn completion-probability cut.
    pub smart_turn_threshold: f64,
    /// TEN-VAD activation threshold.
    pub vad_threshold: f64,
    /// TEN-VAD samples per frame (160 or 256).
    pub vad_hop_size: i64,
    /// Idle gap before the audio pump force-completes a stranded turn. Read by
    /// the wiring layer; the endpointer is frame-driven and can't time out
    /// during silence (Discord halts RTP), so the pump owns this fallback.
    pub idle_fallback_s: f64,
    /// Lazily-loaded shared Smart Turn classifier.
    smart_turn: Mutex<Option<Arc<SmartTurnDetector>>>,
}

impl fmt::Debug for LocalTurnDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalTurnDetector")
            .field("smart_turn_path", &self.smart_turn_path)
            .field("silence_ms", &self.silence_ms)
            .field("speech_start_ms", &self.speech_start_ms)
            .field("smart_turn_threshold", &self.smart_turn_threshold)
            .field("vad_threshold", &self.vad_threshold)
            .field("vad_hop_size", &self.vad_hop_size)
            .field("idle_fallback_s", &self.idle_fallback_s)
            .finish_non_exhaustive()
    }
}

impl LocalTurnDetector {
    /// Build a detector from a resolved model path + typed config knobs.
    #[must_use]
    pub const fn from_config(smart_turn_path: PathBuf, config: &LocalTurnConfig) -> Self {
        Self {
            smart_turn_path,
            silence_ms: config.silence_ms,
            speech_start_ms: config.speech_start_ms,
            smart_turn_threshold: config.smart_turn_threshold,
            vad_threshold: config.vad_threshold,
            vad_hop_size: config.vad_hop_size,
            idle_fallback_s: config.idle_fallback_s,
            smart_turn: Mutex::new(None),
        }
    }

    /// Build a fresh per-user endpointer.
    ///
    /// A fresh [`TenVad`] is constructed per call (stateful native handle); the
    /// Smart Turn classifier is loaded once and shared.
    ///
    /// # Errors
    /// [`TurnDetectionError`] when the native VAD or the ONNX Smart Turn backend
    /// is unavailable (e.g. `local-turn` off, or `ten-vad-sys` not yet linked).
    #[allow(clippy::cast_possible_truncation)]
    pub fn make_endpointer(
        &self,
        on_turn_complete: OnTurnComplete,
    ) -> Result<UtteranceEndpointer, TurnDetectionError> {
        let smart_turn = self.load_smart_turn()?;
        let vad = TenVad::new(
            16000,
            usize::try_from(self.vad_hop_size).unwrap_or(TenVad::DEFAULT_HOP_SIZE),
            self.vad_threshold as f32,
        )?;
        Ok(UtteranceEndpointer::new(
            Box::new(vad),
            smart_turn,
            on_turn_complete,
            self.silence_ms,
            self.speech_start_ms,
        ))
    }

    /// Lazily load + share the Smart Turn classifier.
    ///
    /// The lock is deliberately held across the model load: this is a
    /// double-checked lazy init, so a concurrent first caller must wait rather
    /// than load a second copy of the ONNX session.
    #[allow(clippy::cast_possible_truncation, clippy::significant_drop_tightening)]
    fn load_smart_turn(&self) -> Result<Arc<SmartTurnDetector>, SmartTurnError> {
        let mut guard = self.smart_turn.lock().expect("smart_turn lock poisoned");
        if let Some(existing) = guard.as_ref() {
            return Ok(Arc::clone(existing));
        }
        let detector = Arc::new(SmartTurnDetector::from_path(
            &self.smart_turn_path,
            self.smart_turn_threshold as f32,
        )?);
        *guard = Some(Arc::clone(&detector));
        Ok(detector)
    }
}

/// Build a [`LocalTurnDetector`] from typed *config*, resolving weights through
/// *downloader*.
///
/// Returns [`None`] (with a warning) on any download / FS error, or when the
/// resolved path is missing on disk (cache rot) — the bot then falls back to
/// Deepgram-only endpointing rather than crashing.
#[must_use]
pub fn create_local_turn_detector_with(
    config: &LocalTurnConfig,
    downloader: &dyn ModelDownloader,
) -> Option<LocalTurnDetector> {
    let resolved =
        match downloader.download(&config.smart_turn_repo_id, &config.smart_turn_filename) {
            Ok(path) => path,
            Err(reason) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "{} {} {} {} {} {}",
                    ls::tag("🎙️  Voice", ls::Y),
                    ls::kv_styled("local_turn_detection", "disabled", ls::W, ls::LY),
                    ls::kv_styled("reason", "smart_turn_download_failed", ls::W, ls::LW),
                    ls::kv_styled("repo", &config.smart_turn_repo_id, ls::W, ls::LW),
                    ls::kv_styled("file", &config.smart_turn_filename, ls::W, ls::LW),
                    ls::kv_styled("exc", &reason, ls::W, ls::LW),
                );
                return None;
            }
        };
    if !resolved.exists() {
        // Cache rot — the downloader returned a path that's gone.
        tracing::warn!(
            target: LOG_TARGET,
            "{} {} {} {}",
            ls::tag("🎙️  Voice", ls::Y),
            ls::kv_styled("local_turn_detection", "disabled", ls::W, ls::LY),
            ls::kv_styled("reason", "smart_turn_cache_missing", ls::W, ls::LW),
            ls::kv_styled("path", &resolved.display().to_string(), ls::W, ls::LW),
        );
        return None;
    }
    let detector = LocalTurnDetector::from_config(resolved, config);
    tracing::info!(
        target: LOG_TARGET,
        "{} {} {} {} {} {} {}",
        ls::tag("🎙️  Voice", ls::G),
        ls::kv_styled("local_turn_detection", "enabled", ls::W, ls::LG),
        ls::kv_styled("vad", "ten-vad", ls::W, ls::LG),
        ls::kv_styled("smart_turn", &config.smart_turn_filename, ls::W, ls::LW),
        ls::kv_styled("hop_size", &detector.vad_hop_size.to_string(), ls::W, ls::LW),
        ls::kv_styled("silence_ms", &detector.silence_ms.to_string(), ls::W, ls::LW),
        ls::kv_styled(
            "speech_start_ms",
            &detector.speech_start_ms.to_string(),
            ls::W,
            ls::LW
        ),
    );
    Some(detector)
}

/// Build a [`LocalTurnDetector`] from typed *config*, fetching Smart Turn
/// weights from HuggingFace.
///
/// Returns [`None`] (never an error) on any failure — missing `local-turn`
/// backend, download failure, or cache rot — each with a distinct warning.
#[must_use]
pub fn create_local_turn_detector(config: &LocalTurnConfig) -> Option<LocalTurnDetector> {
    #[cfg(feature = "local-turn")]
    {
        create_local_turn_detector_with(config, &hf::HfHubDownloader)
    }
    #[cfg(not(feature = "local-turn"))]
    {
        let _ = config;
        tracing::warn!(
            target: LOG_TARGET,
            "{} {} {} {}",
            ls::tag("🎙️  Voice", ls::Y),
            ls::kv_styled("local_turn_detection", "disabled", ls::W, ls::LY),
            ls::kv_styled("reason", "huggingface_hub_missing", ls::W, ls::LW),
            ls::kv_styled("hint", "enable the local-turn feature", ls::W, ls::LW),
        );
        None
    }
}

#[cfg(feature = "local-turn")]
mod hf {
    //! Production [`ModelDownloader`] over `hf-hub` (feature `local-turn`).

    use super::ModelDownloader;
    use std::path::PathBuf;

    /// Downloads Smart Turn weights via the HuggingFace Hub cache.
    pub struct HfHubDownloader;

    impl ModelDownloader for HfHubDownloader {
        fn download(&self, repo_id: &str, filename: &str) -> Result<PathBuf, String> {
            let api = hf_hub::api::sync::Api::new().map_err(|e| e.to_string())?;
            api.model(repo_id.to_owned())
                .get(filename)
                .map_err(|e| e.to_string())
        }
    }
}
