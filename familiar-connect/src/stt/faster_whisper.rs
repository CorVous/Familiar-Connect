//! faster-whisper buffer-and-finalize backend (subsystem 09; Python
//! `stt/faster_whisper.py`; feature `local-stt`).
//!
//! STUB — intentionally unimplemented. The Python backend runs a CTranslate2
//! `WhisperModel` locally with the same buffer-and-finalize shape as
//! [`crate::stt::parakeet`]: `send_audio` resamples 48 kHz Discord PCM to 16 kHz
//! mono and appends to a per-instance buffer; `finalize` runs inference on the
//! buffered audio (consuming the lazy segment generator, joining `seg.text`) and
//! emits a single final [`crate::stt::TranscriptionResult`]. Like Parakeet it
//! exposes **no** `endpointing_ms` seam and shares its loaded model handle across
//! `clone()`.
//!
//! No Rust CTranslate2/faster-whisper runtime is wired; the replacement engine
//! (whisper-rs / sherpa-onnx behind the same contract, DESIGN §6 / spec 09
//! F.43–46) has not been chosen, so the `local-stt` feature carries no engine
//! yet. Until one lands, [`crate::stt::factory::create_transcriber`] degrades a
//! `faster_whisper` selection with
//! [`crate::stt::SttError::LocalSttUnavailable`], mirroring the Python
//! lazy-import failure path. When an engine is picked, the contract + its tests
//! port here.
