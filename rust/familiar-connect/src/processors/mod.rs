//! Bus processors: reply loops (subsystem 06) + watermark-driven memory
//! projectors / background workers (subsystem 07). Python `processors/`.

// subsystem 06 — reply loops + projector registry + debug logger
pub mod debug_logger;
pub mod history_writer;
pub mod projectors;
pub mod text_responder;
pub mod voice_responder;

// subsystem 07 — the six memory-projector workers
pub mod fact_embedding_worker;
pub mod fact_extractor;
pub mod fact_supersede_worker;
pub mod people_dossier_worker;
pub mod reflection_worker;
pub mod summary_worker;
