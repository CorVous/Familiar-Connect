//! Parakeet-TDT buffer-and-finalize backend (subsystem 09; Python
//! `stt/parakeet.py`; feature `local-stt`).
//!
//! STUB — intentionally unimplemented. The Python backend runs NeMo's
//! `EncDecRNNTBPEModel` locally with buffer-and-finalize semantics: `send_audio`
//! resamples 48 kHz Discord PCM to 16 kHz mono (via [`crate::voice::audio`]'s
//! `Resampler48to16`) and appends to a per-instance buffer; `finalize` runs
//! inference on the buffered audio and emits a single final
//! [`crate::stt::TranscriptionResult`]. It deliberately exposes **no**
//! `endpointing_ms` seam (so the wiring's support check routes it to the local
//! turn detector), and its `clone()` shares the loaded model handle.
//!
//! No Rust NeMo runtime exists; the intended replacement (whisper-rs /
//! sherpa-onnx behind the same buffer-and-finalize contract, DESIGN §6 /
//! spec 09 F.43–46) has not been chosen, so the `local-stt` feature carries no
//! engine yet. Until one lands, [`crate::stt::factory::create_transcriber`]
//! degrades a `parakeet` selection with [`crate::stt::SttError::LocalSttUnavailable`]
//! — the parity of the Python lazy-import failure path. When an engine is
//! picked, the buffer-and-finalize contract + its tests port here.
