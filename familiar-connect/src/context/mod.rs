//! Prompt assembly: layered system prompt, recent history, final reminder
//! (subsystem 05).
//!
//! Re-exports the assembler, the nine concrete layers, and the final-reminder
//! builder at the module root.

pub mod assembler;
pub mod final_reminder;
pub mod layers;

pub use assembler::{AssembledPrompt, Assembler, AssemblerBuilder, AssemblyContext};
pub use final_reminder::FinalReminder;
pub use layers::{
    ChannelResolver, CharacterCardLayer, ConversationSummaryLayer, Layer, LorebookEntry,
    LorebookLayer, OperatingModeLayer, PeopleDossierLayer, RagContextLayer, RecentHistoryLayer,
    ReflectionLayer, VoiceRosterLayer,
};
