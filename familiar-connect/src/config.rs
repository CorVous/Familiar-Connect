//! `CharacterConfig` loader: TOML deep-merge + validation + `ConfigError`
//! (subsystem 02).
//!
//! The process-wide "load once, immutable thereafter" layer: parses
//! `character.toml` deep-merged over the checked-in default profile into a
//! fully validated, frozen [`CharacterConfig`]. Validation is hand-rolled over
//! `toml::Value` — the per-section unknown-key policy is deliberately
//! inconsistent (some sections reject unknown keys, `[tts]` /
//! `[providers.stt.*]` / `[channels.<id>]` ignore them), so serde
//! `deny_unknown_fields` cannot reproduce the contract. Error messages are
//! byte-stable test contracts; reproduce the phrases exactly.
//!
//! The two registry lookups (`known_projectors`,
//! `known_embedders`) are injected as `&BTreeSet<String>` parameters, keeping
//! this a near-leaf module. The fastembed model → native-dim table
//! ([`FastembedNativeDim`]) is injected the same way through the parse layer;
//! only [`load_character_config`] names the concrete table.

use crate::budget::{ModelBudgetCurve, TierBudget};
use chrono::NaiveTime;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::LazyLock;
use toml::{Table, Value};

// ---------------------------------------------------------------------------
// Error type + constants
// ---------------------------------------------------------------------------

/// Injected lookup: fastembed model name → native output dim, `None` when the
/// name is unmapped (dim knowable only after the runtime probe). Keeps the
/// model table out of this module.
pub type FastembedNativeDim = fn(&str) -> Option<usize>;

/// Malformed config file or unknown value reference. The single error type for
/// every config problem; callers match on message substrings (byte-stable).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(pub String);

/// Canonical LLM call-site slot names.
pub const LLM_SLOT_NAMES: [&str; 3] = ["fast", "prose", "background"];
/// `[llm]` keys that are shared settings, not slot tables.
const LLM_SHARED_KEYS: [&str; 3] = [
    "image_description_model",
    "image_caption_model",
    "max_concurrent_requests",
];
/// Canonical assembly-tier names.
pub const BUDGET_TIER_NAMES: [&str; 3] = ["voice", "text", "background"];
/// Allowed values for `[llm.<slot>].reasoning`.
pub const REASONING_LEVELS: [&str; 6] = ["off", "none", "low", "medium", "high", "default"];

const TTS_PROVIDERS: [&str; 1] = ["cartesia"];
/// Providers that existed as unwired stubs and were removed; named so an
/// upgraded profile gets a migration hint instead of a bare "unknown".
const REMOVED_TTS_PROVIDERS: [&str; 2] = ["azure", "gemini"];
const TURN_STRATEGIES: [&str; 2] = ["deepgram", "ten+smart_turn"];
const STT_BACKENDS: [&str; 3] = ["deepgram", "parakeet", "faster_whisper"];
const MESSAGE_RENDERINGS: [&str; 2] = ["prefixed", "name_only"];
const DEFAULT_PROJECTORS: [&str; 5] = [
    "rolling_summary",
    "rich_note",
    "people_dossier",
    "reflection",
    "fact_supersede",
];
const PROMPT_FIELDS: [&str; 15] = [
    "post_history_instructions",
    "image_description_constraints",
    "operating_mode_voice",
    "operating_mode_text",
    "voice_tool_ack",
    "shift_focus_coaching",
    "start_activity_description",
    "rolling_summary_system",
    "reflection_system",
    "dossier_self_system",
    "dossier_other_system",
    "sleep_consolidation_system",
    "sleep_stance_system",
    "sleep_synthesis_system",
    "dream_extraction_clause",
];
const RETRIEVAL_FIELDS: [&str; 4] = [
    "bm25_weight",
    "recency_weight",
    "importance_weight",
    "embedding_weight",
];
const BUDGET_FIELDS: [&str; 12] = [
    "recent_history_tokens",
    "rag_tokens",
    "dossier_tokens",
    "summary_tokens",
    "reflection_tokens",
    "lorebook_tokens",
    "max_history_turns",
    "max_rag_turns",
    "max_rag_facts",
    "max_dossier_people",
    "max_reflections",
    "max_lorebook_entries",
];

// ---------------------------------------------------------------------------
// Sub-config types
// ---------------------------------------------------------------------------

/// Per-call-site LLM config from `[llm.<slot>]`.
#[derive(Clone, Debug, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "these are independent boolean knobs, not a state enum"
)]
pub struct LLMSlotConfig {
    /// Model string (required, non-empty).
    pub model: String,
    /// Sampling temperature in `[0, 2]`; `None` = provider default.
    pub temperature: Option<f64>,
    /// Nucleus sampling `top_p` in `[0, 1]`.
    pub top_p: Option<f64>,
    /// Top-k sampling (>= 1).
    pub top_k: Option<i64>,
    /// Presence penalty in `[-2, 2]`.
    pub presence_penalty: Option<f64>,
    /// Qwen3 no-think stabiliser.
    pub think_prepend: bool,
    /// OpenRouter provider routing override; `None` = default.
    pub provider_order: Option<Vec<String>>,
    /// Allow OpenRouter provider fallbacks (default true).
    pub provider_allow_fallbacks: bool,
    /// OpenRouter reasoning effort; `None` = model default (`"default"` → None).
    pub reasoning: Option<String>,
    /// Surface-only tool-calling flag. Defaults `true` and `false` is refused
    /// at load: silence is a tool call, so a tool-less slot cannot decline to
    /// reply.
    pub tool_calling: bool,
    /// Gate for `view_image` registration.
    ///
    /// Deliberately NOT tri-state (unlike [`multimodal`](Self::multimodal)):
    /// it is an intent knob, not a capability fact. `view_image` works on a
    /// text-only model too — the substitution description stands in for the
    /// image — and an operator may not want image fetches at all on a
    /// vision-capable one. Neither direction is inferable from the catalog, so
    /// it stays an explicit opt-in.
    pub image_tools: bool,
    /// Send image content blocks in tool-result messages. Tri-state: `None` =
    /// auto-detect from the OpenRouter catalog, `Some(_)` = operator override
    /// detection must never contradict.
    pub multimodal: Option<bool>,
}

impl LLMSlotConfig {
    /// Resolve [`multimodal`](Self::multimodal) against catalog metadata.
    ///
    /// Explicit config always wins. Unset follows `detected`; with no catalog
    /// (`None`) the answer is `false` — the pre-detection default.
    #[must_use]
    pub fn resolve_multimodal(&self, detected: Option<bool>) -> bool {
        self.multimodal.or(detected).unwrap_or(false)
    }
}

impl Default for LLMSlotConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            temperature: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            think_prepend: false,
            provider_order: None,
            provider_allow_fallbacks: true,
            reasoning: None,
            tool_calling: true,
            image_tools: false,
            multimodal: None,
        }
    }
}

/// V1 local-turn-detection knobs from `[providers.turn_detection.local]`.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalTurnConfig {
    /// Smart Turn HuggingFace repo id.
    pub smart_turn_repo_id: String,
    /// Smart Turn ONNX filename.
    pub smart_turn_filename: String,
    /// Silence ms after speech before Smart Turn fires.
    pub silence_ms: i64,
    /// Consecutive speech ms before "speaking" latches.
    pub speech_start_ms: i64,
    /// TEN-VAD activation threshold.
    pub vad_threshold: f64,
    /// Smart Turn completion-probability cut.
    pub smart_turn_threshold: f64,
    /// Samples per VAD frame.
    pub vad_hop_size: i64,
    /// Idle gap before the pump force-completes a stranded turn.
    pub idle_fallback_s: f64,
}

impl Default for LocalTurnConfig {
    fn default() -> Self {
        Self {
            smart_turn_repo_id: "pipecat-ai/smart-turn-v3".to_owned(),
            smart_turn_filename: "smart-turn-v3.2-cpu.onnx".to_owned(),
            silence_ms: 200,
            speech_start_ms: 100,
            vad_threshold: 0.5,
            smart_turn_threshold: 0.5,
            vad_hop_size: 256,
            idle_fallback_s: 1.5,
        }
    }
}

/// Turn-detection strategy from `[providers.turn_detection]`.
#[derive(Clone, Debug, PartialEq)]
pub struct TurnDetectionConfig {
    /// `"deepgram"` | `"ten+smart_turn"`.
    pub strategy: String,
    /// Local endpointer knobs (always parsed).
    pub local: LocalTurnConfig,
}

impl Default for TurnDetectionConfig {
    fn default() -> Self {
        Self {
            strategy: "deepgram".to_owned(),
            local: LocalTurnConfig::default(),
        }
    }
}

/// Deepgram knobs from `[providers.stt.deepgram]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepgramSTTConfig {
    /// Deepgram model.
    pub model: String,
    /// Language code.
    pub language: String,
    /// Silence ms before a segment finalizes.
    pub endpointing_ms: i64,
    /// Speech-end grace window ms.
    pub utterance_end_ms: i64,
    /// Enable smart formatting.
    pub smart_format: bool,
    /// Enable punctuation.
    pub punctuate: bool,
    /// Bias keyterms.
    pub keyterms: Vec<String>,
    /// Replay buffer seconds.
    pub replay_buffer_s: f64,
    /// Keepalive interval seconds.
    pub keepalive_interval_s: f64,
    /// Reconnect attempt cap.
    pub reconnect_max_attempts: i64,
    /// Reconnect backoff cap seconds.
    pub reconnect_backoff_cap_s: f64,
    /// Idle-close seconds (0 disables).
    pub idle_close_s: f64,
}

impl Default for DeepgramSTTConfig {
    fn default() -> Self {
        Self {
            model: "nova-3".to_owned(),
            language: "en".to_owned(),
            endpointing_ms: 500,
            utterance_end_ms: 1500,
            smart_format: true,
            punctuate: true,
            keyterms: Vec::new(),
            replay_buffer_s: 5.0,
            keepalive_interval_s: 3.0,
            reconnect_max_attempts: 5,
            reconnect_backoff_cap_s: 16.0,
            idle_close_s: 30.0,
        }
    }
}

/// Parakeet knobs from `[providers.stt.parakeet]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ParakeetSTTConfig {
    /// NeMo model name.
    pub model_name: String,
    /// Device selector.
    pub device: String,
    /// Idle-close seconds.
    pub idle_close_s: f64,
}

impl Default for ParakeetSTTConfig {
    fn default() -> Self {
        Self {
            model_name: "nvidia/parakeet-tdt-0.6b-v3".to_owned(),
            device: "auto".to_owned(),
            idle_close_s: 30.0,
        }
    }
}

/// FasterWhisper knobs from `[providers.stt.faster_whisper]`.
#[derive(Clone, Debug, PartialEq)]
pub struct FasterWhisperSTTConfig {
    /// Model size.
    pub model_size: String,
    /// Device selector.
    pub device: String,
    /// Compute type.
    pub compute_type: String,
    /// Language code.
    pub language: String,
    /// Idle-close seconds.
    pub idle_close_s: f64,
}

impl Default for FasterWhisperSTTConfig {
    fn default() -> Self {
        Self {
            model_size: "small".to_owned(),
            device: "auto".to_owned(),
            compute_type: "auto".to_owned(),
            language: "en".to_owned(),
            idle_close_s: 30.0,
        }
    }
}

/// STT backend selector + per-backend knobs from `[providers.stt]`.
#[derive(Clone, Debug, PartialEq)]
pub struct STTConfig {
    /// `"deepgram"` | `"parakeet"` | `"faster_whisper"`.
    pub backend: String,
    /// Deepgram knobs.
    pub deepgram: DeepgramSTTConfig,
    /// Parakeet knobs.
    pub parakeet: ParakeetSTTConfig,
    /// FasterWhisper knobs.
    pub faster_whisper: FasterWhisperSTTConfig,
}

impl Default for STTConfig {
    fn default() -> Self {
        Self {
            backend: "deepgram".to_owned(),
            deepgram: DeepgramSTTConfig::default(),
            parakeet: ParakeetSTTConfig::default(),
            faster_whisper: FasterWhisperSTTConfig::default(),
        }
    }
}

/// Text-to-speech config from `[tts]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TTSConfig {
    /// `"cartesia"` — the only implemented backend, and the default.
    pub provider: String,
    /// Cartesia voice id.
    pub cartesia_voice_id: Option<String>,
    /// Cartesia model.
    pub cartesia_model: Option<String>,
    /// Greeting lines (stringified).
    pub greetings: Vec<String>,
}

impl Default for TTSConfig {
    fn default() -> Self {
        Self {
            provider: "cartesia".to_owned(),
            cartesia_voice_id: None,
            cartesia_model: None,
            greetings: Vec::new(),
        }
    }
}

/// `[discord.text]` knobs — typing-indicator + interruption behavior.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscordTextConfig {
    /// Treat other users' typing as interruptions (default true).
    pub respond_to_typing: bool,
    /// Initial backoff seconds when another bot is typing.
    pub typing_backoff_initial_s: f64,
    /// Max backoff seconds (>= initial).
    pub typing_backoff_max_s: f64,
}

impl Default for DiscordTextConfig {
    fn default() -> Self {
        Self {
            respond_to_typing: true,
            typing_backoff_initial_s: 1.0,
            typing_backoff_max_s: 30.0,
        }
    }
}

/// Ranking weights from `[memory.retrieval]` (M2).
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryRetrievalConfig {
    /// BM25 quality weight.
    pub bm25_weight: f64,
    /// Recency weight.
    pub recency_weight: f64,
    /// Importance weight.
    pub importance_weight: f64,
    /// Embedding cosine-similarity weight.
    pub embedding_weight: f64,
}

impl Default for MemoryRetrievalConfig {
    fn default() -> Self {
        Self {
            bm25_weight: 1.0,
            recency_weight: 0.0,
            importance_weight: 0.0,
            embedding_weight: 0.0,
        }
    }
}

/// `[providers.memory.rolling_summary]` worker knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct RollingSummaryConfig {
    /// New turns per channel before the summary refreshes.
    pub turns_threshold: i64,
    /// Idle interval between worker ticks.
    pub tick_interval_s: f64,
}

impl Default for RollingSummaryConfig {
    fn default() -> Self {
        Self {
            turns_threshold: 10,
            tick_interval_s: 5.0,
        }
    }
}

/// `[providers.memory.rich_note]` (fact extractor) worker knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct RichNoteConfig {
    /// Unprocessed turns per extraction tick.
    pub batch_size: i64,
    /// Idle interval between worker ticks.
    pub tick_interval_s: f64,
    /// Cap on participant manifest rows.
    pub participants_max: i64,
    /// Self-capability post-filter on/off. `false` keeps every extracted fact.
    pub self_capability_filter: bool,
    /// Regex replacing the built-in self-capability matcher; empty keeps the
    /// built-in. Validated at load — an uncompilable pattern fails startup.
    pub self_capability_pattern: String,
}

impl Default for RichNoteConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            tick_interval_s: 15.0,
            participants_max: 30,
            self_capability_filter: true,
            self_capability_pattern: String::new(),
        }
    }
}

/// `[providers.memory.people_dossier]` worker knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct PeopleDossierConfig {
    /// Idle interval between worker ticks.
    pub tick_interval_s: f64,
}

impl Default for PeopleDossierConfig {
    fn default() -> Self {
        Self {
            tick_interval_s: 20.0,
        }
    }
}

/// `[providers.memory.reflection]` worker knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflectionConfig {
    /// New turns before a reflection tick fires.
    pub turns_threshold: i64,
    /// Cap on reflections written per tick.
    pub max_reflections_per_tick: i64,
    /// Window cap on turns fed to the prompt.
    pub max_turns_per_tick: i64,
    /// Recent facts included in the prompt.
    pub recent_facts_limit: i64,
    /// Idle interval between worker ticks.
    pub tick_interval_s: f64,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            turns_threshold: 20,
            max_reflections_per_tick: 3,
            max_turns_per_tick: 50,
            recent_facts_limit: 20,
            tick_interval_s: 60.0,
        }
    }
}

/// `[providers.memory.fact_supersede]` worker knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct FactSupersedeConfig {
    /// New facts evaluated per tick.
    pub batch_size: i64,
    /// Idle interval between worker ticks.
    pub tick_interval_s: f64,
    /// Cap on prior facts shown to the LLM per subject.
    pub priors_max: i64,
}

impl Default for FactSupersedeConfig {
    fn default() -> Self {
        Self {
            batch_size: 5,
            tick_interval_s: 60.0,
            priors_max: 20,
        }
    }
}

/// Memory-projector selector + per-projector knobs from `[providers.memory]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryProvidersConfig {
    /// Ordered projector names. Empty disables all projection.
    pub projectors: Vec<String>,
    /// Rolling-summary knobs.
    pub rolling_summary: RollingSummaryConfig,
    /// Rich-note (fact extractor) knobs.
    pub rich_note: RichNoteConfig,
    /// People-dossier knobs.
    pub people_dossier: PeopleDossierConfig,
    /// Reflection knobs.
    pub reflection: ReflectionConfig,
    /// Fact-supersede knobs.
    pub fact_supersede: FactSupersedeConfig,
}

impl Default for MemoryProvidersConfig {
    fn default() -> Self {
        Self {
            projectors: default_projectors(),
            rolling_summary: RollingSummaryConfig::default(),
            rich_note: RichNoteConfig::default(),
            people_dossier: PeopleDossierConfig::default(),
            reflection: ReflectionConfig::default(),
            fact_supersede: FactSupersedeConfig::default(),
        }
    }
}

/// `[focus]` knobs — attentional unread-nudge controls.
#[derive(Clone, Debug, PartialEq)]
pub struct FocusConfig {
    /// Enable the unread nudge (default true).
    pub unread_nudge_enabled: bool,
    /// Rapid arrivals within this window share one nudge.
    pub nudge_debounce_seconds: f64,
    /// Staged turns caught up on when attention lands.
    pub catch_up_limit: i64,
}

impl Default for FocusConfig {
    fn default() -> Self {
        Self {
            unread_nudge_enabled: true,
            nudge_debounce_seconds: 30.0,
            catch_up_limit: 20,
        }
    }
}

/// `[voice]` knobs — live-call roster rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceConfig {
    /// Join/leave events older than this stop being narrated.
    pub roster_event_window_seconds: f64,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            roster_event_window_seconds: crate::voice_roster::DEFAULT_EVENT_WINDOW_SECONDS,
        }
    }
}

/// `[tools]` knobs — agentic tool-loop behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolsConfig {
    /// Hard cap on agentic-loop iterations per turn.
    pub loop_max_iterations: i64,
    /// Let `view_image` fetch hosts outside `trusted_image_hosts`. Private and
    /// reserved addresses stay refused either way.
    pub allow_untrusted_image_urls: bool,
    /// Hosts `view_image` may fetch: exact names, or `*.suffix` patterns.
    pub trusted_image_hosts: Vec<String>,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            loop_max_iterations: 5,
            allow_untrusted_image_urls: false,
            trusted_image_hosts: crate::tools::image_policy::DEFAULT_TRUSTED_IMAGE_HOSTS
                .iter()
                .map(|h| (*h).to_owned())
                .collect(),
        }
    }
}

/// Embedder backend selector from `[providers.embedding]` (M6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingConfig {
    /// Backend name (registered); `"off"` disables the seam.
    pub backend: String,
    /// Dimensionality for backends that accept one (`hash`). Under
    /// `fastembed` the model owns it: parsing resolves an unset `dim` to the
    /// model's native dim and rejects one that contradicts it.
    pub dim: i64,
    /// FastEmbed model name.
    pub fastembed_model: String,
    /// FastEmbed cache dir override (`""` → None).
    pub fastembed_cache_dir: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: "off".to_owned(),
            dim: 256,
            fastembed_model: "BAAI/bge-small-en-v1.5".to_owned(),
            fastembed_cache_dir: None,
        }
    }
}

/// Per-channel overrides for latency-sensitive knobs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChannelOverrides {
    /// Overrides the tier default history window.
    pub history_window_size: Option<i64>,
    /// Ordered layer names; `None` inherits default order.
    pub prompt_layers: Option<Vec<String>>,
    /// `"prefixed"` | `"name_only"`; `None` inherits default.
    pub message_rendering: Option<String>,
}

fn default_projectors() -> Vec<String> {
    DEFAULT_PROJECTORS.iter().map(|s| (*s).to_owned()).collect()
}

fn default_budgets() -> BTreeMap<String, TierBudget> {
    BUDGET_TIER_NAMES
        .iter()
        .map(|t| ((*t).to_owned(), TierBudget::default()))
        .collect()
}

// ---------------------------------------------------------------------------
// CharacterConfig
// ---------------------------------------------------------------------------

/// Config loaded once per install from `character.toml`.
#[derive(Clone, Debug, PartialEq)]
pub struct CharacterConfig {
    /// Hard cap on voice-tier history turns.
    pub voice_window_size: i64,
    /// Hard cap on text-tier history turns.
    pub text_window_size: i64,
    /// Same-speaker voice-fragment coalesce gap (seconds); 0 disables.
    pub recent_history_coalesce_max_gap_seconds: f64,
    /// Fold text turns before a silence gap this large into summary; 0 disables.
    pub text_silence_gap_fold_seconds: f64,
    /// Rows the `llm_calls` mirror keeps per familiar; 0 disables mirroring.
    pub llm_mirror_calls: i64,
    /// IANA timezone name (validated at load, stored as the string).
    pub display_tz: String,
    /// Sleep window `(start, end)` (may wrap midnight); `None` disarmed.
    pub sleep_window: Option<(NaiveTime, NaiveTime)>,
    /// Force-sleep grace minutes after window start.
    pub sleep_grace_minutes: i64,
    /// Configured aliases.
    pub aliases: Vec<String>,
    /// Per-slot LLM configs (present only if the merged TOML defines them).
    pub llm: BTreeMap<String, LLMSlotConfig>,
    /// Text-to-speech config.
    pub tts: TTSConfig,
    /// Per-channel overrides keyed by channel id.
    pub channels: BTreeMap<i64, ChannelOverrides>,
    /// Turn-detection config.
    pub turn_detection: TurnDetectionConfig,
    /// STT config.
    pub stt: STTConfig,
    /// Per-tier token envelopes.
    pub budgets: BTreeMap<String, TierBudget>,
    /// Per-model multipliers keyed by model name.
    pub budget_curves: BTreeMap<String, ModelBudgetCurve>,
    /// Discord text-channel behavior.
    pub discord_text: DiscordTextConfig,
    /// Discord user ids whose DMs are engaged.
    pub dm_allowlist: Vec<i64>,
    /// Free-text block appended to the trailing reminder.
    pub post_history_instructions: String,
    /// Appended to the neutral image-description base prompt.
    pub image_description_constraints: String,
    /// Per-mode operating directive, voice tier. One source for the
    /// `operating_mode` layer and the final reminder.
    pub operating_mode_voice: String,
    /// Per-mode operating directive, text tier.
    pub operating_mode_text: String,
    /// Voice-tier nudge to speak before calling a tool.
    pub voice_tool_ack: String,
    /// `shift_focus` clause spliced onto the unread digest.
    pub shift_focus_coaching: String,
    /// Roleplay guidance carried by `start_activity`'s tool description.
    pub start_activity_description: String,
    /// Static rolling-summary instruction text.
    pub rolling_summary_system: String,
    /// Static reflection persona (the reply contract stays in code).
    pub reflection_system: String,
    /// Static self-dossier instruction text (`{self_name}`).
    pub dossier_self_system: String,
    /// Static other-person dossier instruction text (`{display_name}`).
    pub dossier_other_system: String,
    /// Static sleep-consolidation instruction text.
    pub sleep_consolidation_system: String,
    /// Static sleep-stance instruction text.
    pub sleep_stance_system: String,
    /// Static sleep-synthesis instruction text.
    pub sleep_synthesis_system: String,
    /// Static fact-extractor dream-framing clause.
    pub dream_extraction_clause: String,
    /// Retrieval ranking weights.
    pub memory_retrieval: MemoryRetrievalConfig,
    /// Memory-projector selection + knobs.
    pub memory_providers: MemoryProvidersConfig,
    /// Embedder backend selection.
    pub embedding: EmbeddingConfig,
    /// Substitution model: describes the image for a slot that cannot see it.
    /// `""` disables.
    pub image_description_model: String,
    /// Persistence model: the durable caption written to history (and thus to
    /// facts / summaries / dossiers). `""` falls back to
    /// [`image_description_model`](Self::image_description_model).
    pub image_caption_model: String,
    /// Process-wide cap on concurrent LLM requests.
    pub llm_max_concurrent_requests: i64,
    /// Attentional unread-nudge controls.
    pub focus: FocusConfig,
    /// Agentic tool-loop knobs.
    pub tools: ToolsConfig,
    /// Live-call roster knobs.
    pub voice: VoiceConfig,
}

impl Default for CharacterConfig {
    fn default() -> Self {
        Self {
            voice_window_size: 100,
            text_window_size: 200,
            recent_history_coalesce_max_gap_seconds: 45.0,
            text_silence_gap_fold_seconds: 0.0,
            llm_mirror_calls: DEFAULT_LLM_MIRROR_CALLS,
            display_tz: "UTC".to_owned(),
            sleep_window: None,
            sleep_grace_minutes: 30,
            aliases: Vec::new(),
            llm: BTreeMap::new(),
            tts: TTSConfig::default(),
            channels: BTreeMap::new(),
            turn_detection: TurnDetectionConfig::default(),
            stt: STTConfig::default(),
            budgets: default_budgets(),
            budget_curves: BTreeMap::new(),
            discord_text: DiscordTextConfig::default(),
            dm_allowlist: Vec::new(),
            post_history_instructions: String::new(),
            image_description_constraints: String::new(),
            operating_mode_voice: String::new(),
            operating_mode_text: String::new(),
            voice_tool_ack: String::new(),
            shift_focus_coaching: String::new(),
            start_activity_description: String::new(),
            rolling_summary_system: String::new(),
            reflection_system: String::new(),
            dossier_self_system: String::new(),
            dossier_other_system: String::new(),
            sleep_consolidation_system: String::new(),
            sleep_stance_system: String::new(),
            sleep_synthesis_system: String::new(),
            dream_extraction_clause: String::new(),
            memory_retrieval: MemoryRetrievalConfig::default(),
            memory_providers: MemoryProvidersConfig::default(),
            embedding: EmbeddingConfig::default(),
            image_description_model: String::new(),
            image_caption_model: String::new(),
            llm_max_concurrent_requests: 4,
            focus: FocusConfig::default(),
            tools: ToolsConfig::default(),
            voice: VoiceConfig::default(),
        }
    }
}

/// Default `llm_calls` retention: rows kept per familiar.
///
/// An assembled prompt runs ~15 KB and the mirror stores it twice (the
/// `system_prompt` projection plus the verbatim message array), so a row costs
/// ~30 KB — roughly 30 MB at this cap, or ~20 sessions of 50 calls. Enough to
/// answer "what did the prompt contain on turn X" days later without the table
/// outgrowing the history it annotates.
pub const DEFAULT_LLM_MIRROR_CALLS: i64 = 1000;

/// Tier → LLM slot mapping (matches the responder wiring).
fn tier_to_slot(tier: &str) -> Option<&'static str> {
    match tier {
        "voice" => Some("fast"),
        "text" => Some("prose"),
        "background" => Some("background"),
        _ => None,
    }
}

impl CharacterConfig {
    /// Overrides for `channel_id`; empty (all-`None`) if `None` or unknown.
    #[must_use]
    pub fn for_channel(&self, channel_id: Option<i64>) -> ChannelOverrides {
        channel_id.map_or_else(ChannelOverrides::default, |id| {
            self.channels.get(&id).cloned().unwrap_or_default()
        })
    }

    /// Voice-tier window for `channel_id`; the channel override wins.
    #[must_use]
    pub fn voice_window_for(&self, channel_id: Option<i64>) -> i64 {
        self.for_channel(channel_id)
            .history_window_size
            .unwrap_or(self.voice_window_size)
    }

    /// Text-tier window for `channel_id`; the channel override wins.
    #[must_use]
    pub fn text_window_for(&self, channel_id: Option<i64>) -> i64 {
        self.for_channel(channel_id)
            .history_window_size
            .unwrap_or(self.text_window_size)
    }

    /// Effective budget for `tier`, with the model curve applied when the tier's
    /// active model has one registered.
    ///
    /// # Panics
    /// Panics if `tier` is not one of the canonical tier names (callers only
    /// pass canonical names).
    #[must_use]
    pub fn budget_for(&self, tier: &str) -> TierBudget {
        let base = *self
            .budgets
            .get(tier)
            .expect("budget tier must be one of voice/text/background");
        if let Some(slot) = tier_to_slot(tier) {
            if let Some(slot_cfg) = self.llm.get(slot) {
                if let Some(curve) = self.budget_curves.get(&slot_cfg.model) {
                    return base.apply_curve(curve);
                }
            }
        }
        base
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Load a [`CharacterConfig`] from `path`, merged over `defaults_path`.
///
/// The default profile must exist (else a `ConfigError` containing "default
/// character profile"); a missing target file is treated as `{}`. The registry
/// validator sets (`known_projectors`, `known_embedders`) are injected rather
/// than looked up via a module cycle.
pub fn load_character_config(
    path: &Path,
    defaults_path: &Path,
    known_projectors: &BTreeSet<String>,
    known_embedders: &BTreeSet<String>,
) -> Result<CharacterConfig, ConfigError> {
    // The one lookup this entry point resolves itself: the fastembed
    // model → native-dim table is static metadata, and adding a fifth
    // parameter would churn every caller. Parsing below stays injection-only.
    let fastembed_native_dim: FastembedNativeDim = crate::embedding::fastembed_native_dim;
    let Some(defaults_data) = read_toml(defaults_path)? else {
        return Err(ConfigError(format!(
            "default character profile not found at {}. This file is a required repo asset — check your install.",
            defaults_path.display()
        )));
    };
    let target_data = read_toml(path)?.unwrap_or_default();
    let merged = deep_merge(&defaults_data, &target_data);
    parse_character_config(
        &merged,
        known_projectors,
        known_embedders,
        fastembed_native_dim,
    )
}

fn read_toml(path: &Path) -> Result<Option<Table>, ConfigError> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| ConfigError(format!("failed to read config at {}: {e}", path.display())))?;
    match toml::from_str::<Table>(&text) {
        Ok(t) => Ok(Some(t)),
        Err(e) => Err(ConfigError(format!(
            "failed to parse TOML config at {}: {e}",
            path.display()
        ))),
    }
}

/// Deep-merge `override_` over `base`; recurse only when both values are tables.
/// Any non-table override (including a list) replaces the base value wholesale.
fn deep_merge(base: &Table, override_: &Table) -> Table {
    let mut result = Table::new();
    for (key, base_value) in base {
        if let Some(override_value) = override_.get(key) {
            if let (Value::Table(b), Value::Table(o)) = (base_value, override_value) {
                result.insert(key.clone(), Value::Table(deep_merge(b, o)));
            } else {
                result.insert(key.clone(), override_value.clone());
            }
        } else {
            result.insert(key.clone(), base_value.clone());
        }
    }
    for (key, value) in override_ {
        if !base.contains_key(key) {
            result.insert(key.clone(), value.clone());
        }
    }
    result
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat field-by-field parse sequence; splitting it would scatter the error-message contract"
)]
fn parse_character_config(
    data: &Table,
    known_projectors: &BTreeSet<String>,
    known_embedders: &BTreeSet<String>,
    fastembed_native_dim: FastembedNativeDim,
) -> Result<CharacterConfig, ConfigError> {
    let providers = expect_table(data.get("providers"), "[providers]")?;
    let history_section = expect_table(providers.get("history"), "[providers.history]")?;
    let (voice_window_size, text_window_size) = parse_history_windows(history_section)?;
    let recent_history_coalesce_max_gap_seconds = parse_coalesce_gap(history_section)?;
    let text_silence_gap_fold_seconds = parse_text_silence(history_section)?;
    let llm_mirror_calls = parse_llm_mirror_calls(history_section)?;

    let display_tz = data
        .get("display_tz")
        .map_or_else(|| "UTC".to_owned(), value_str);
    if display_tz.parse::<chrono_tz::Tz>().is_err() {
        return Err(ConfigError(format!(
            "invalid display_tz '{display_tz}' — use an IANA name like 'America/Los_Angeles'"
        )));
    }

    let sleep_raw = expect_table(data.get("sleep"), "[sleep]")?;
    let (sleep_window, sleep_grace_minutes) = parse_sleep_config(sleep_raw)?;

    let aliases = match data.get("aliases") {
        None => Vec::new(),
        Some(Value::Array(arr)) => arr.iter().map(value_str).collect(),
        Some(other) => {
            return Err(ConfigError(format!(
                "aliases must be a list of strings, got {}",
                value_type_name(other)
            )));
        }
    };

    let llm_raw = expect_table(data.get("llm"), "[llm]")?;
    let image_description_model = match llm_raw.get("image_description_model") {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            return Err(ConfigError(
                "[llm].image_description_model must be a string".into(),
            ));
        }
    };
    let image_caption_model = match llm_raw.get("image_caption_model") {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            return Err(ConfigError(
                "[llm].image_caption_model must be a string".into(),
            ));
        }
    };
    let llm_max_concurrent_requests = match llm_raw.get("max_concurrent_requests") {
        None => 4,
        Some(Value::Integer(n)) if *n > 0 => *n,
        Some(Value::Integer(n)) => {
            return Err(ConfigError(format!(
                "[llm].max_concurrent_requests must be positive, got {n}"
            )));
        }
        Some(other) => {
            return Err(ConfigError(format!(
                "[llm].max_concurrent_requests must be a positive integer, got {}",
                value_type_name(other)
            )));
        }
    };
    let mut llm_slots_raw = Table::new();
    for (k, v) in llm_raw {
        if !LLM_SHARED_KEYS.contains(&k.as_str()) {
            llm_slots_raw.insert(k.clone(), v.clone());
        }
    }
    let llm = parse_llm_slots(&llm_slots_raw)?;

    let tts = parse_tts_config(expect_table(data.get("tts"), "[tts]")?)?;
    let channels = parse_channel_overrides(expect_table(data.get("channels"), "[channels]")?)?;
    let turn_detection = parse_turn_detection_config(expect_table(
        providers.get("turn_detection"),
        "[providers.turn_detection]",
    )?)?;
    let stt = parse_stt_config(expect_table(providers.get("stt"), "[providers.stt]")?)?;
    let memory_providers = parse_memory_providers(
        expect_table(providers.get("memory"), "[providers.memory]")?,
        known_projectors,
    )?;
    let embedding = parse_embedding_config(
        expect_table(providers.get("embedding"), "[providers.embedding]")?,
        known_embedders,
        fastembed_native_dim,
    )?;

    let budget_raw = expect_table(data.get("budget"), "[budget]")?;
    let model_curves_raw = match budget_raw.get("model_curves") {
        None => empty_table(),
        Some(Value::Table(t)) => t,
        Some(_) => return Err(ConfigError("[budget.model_curves] must be a table".into())),
    };
    let mut budget_tiers_raw = budget_raw.clone();
    budget_tiers_raw.remove("model_curves");
    let budgets = parse_budgets(&budget_tiers_raw)?;
    let budget_curves = parse_budget_curves(model_curves_raw)?;

    let discord_raw = expect_table(data.get("discord"), "[discord]")?;
    let discord_text =
        parse_discord_text_config(expect_table(discord_raw.get("text"), "[discord.text]")?)?;
    let dm_allowlist = match discord_raw.get("dm_allowlist") {
        None => Vec::new(),
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for uid in arr {
                match uid {
                    Value::Integer(n) => out.push(*n),
                    other => {
                        return Err(ConfigError(format!(
                            "[discord].dm_allowlist entries must be integer user IDs, got {}",
                            value_type_name(other)
                        )));
                    }
                }
            }
            out
        }
        Some(other) => {
            return Err(ConfigError(format!(
                "[discord].dm_allowlist must be a list of integer user IDs, got {}",
                value_type_name(other)
            )));
        }
    };

    let prompt = parse_prompt_config(expect_table(data.get("prompt"), "[prompt]")?)?;

    let memory_raw = expect_table(data.get("memory"), "[memory]")?;
    let memory_retrieval = parse_memory_retrieval(expect_table(
        memory_raw.get("retrieval"),
        "[memory.retrieval]",
    )?)?;

    let focus = parse_focus_config(expect_table(data.get("focus"), "[focus]")?)?;
    let tools = parse_tools_config(expect_table(data.get("tools"), "[tools]")?)?;
    let voice = parse_voice_config(expect_table(data.get("voice"), "[voice]")?)?;

    Ok(CharacterConfig {
        voice_window_size,
        text_window_size,
        recent_history_coalesce_max_gap_seconds,
        text_silence_gap_fold_seconds,
        llm_mirror_calls,
        display_tz,
        sleep_window,
        sleep_grace_minutes,
        aliases,
        llm,
        tts,
        channels,
        turn_detection,
        stt,
        budgets,
        budget_curves,
        discord_text,
        dm_allowlist,
        post_history_instructions: prompt.post_history_instructions,
        image_description_constraints: prompt.image_description_constraints,
        operating_mode_voice: prompt.operating_mode_voice,
        operating_mode_text: prompt.operating_mode_text,
        voice_tool_ack: prompt.voice_tool_ack,
        shift_focus_coaching: prompt.shift_focus_coaching,
        start_activity_description: prompt.start_activity_description,
        rolling_summary_system: prompt.rolling_summary_system,
        reflection_system: prompt.reflection_system,
        dossier_self_system: prompt.dossier_self_system,
        dossier_other_system: prompt.dossier_other_system,
        sleep_consolidation_system: prompt.sleep_consolidation_system,
        sleep_stance_system: prompt.sleep_stance_system,
        sleep_synthesis_system: prompt.sleep_synthesis_system,
        dream_extraction_clause: prompt.dream_extraction_clause,
        memory_retrieval,
        memory_providers,
        embedding,
        image_description_model,
        image_caption_model,
        llm_max_concurrent_requests,
        focus,
        tools,
        voice,
    })
}

/// Parse `"HH:MM-HH:MM"` → `(start, end)`; may wrap midnight, `start == end`
/// rejected. Shared by `[sleep].window` and the activities catalog's
/// `active_hours`.
pub fn parse_hhmm_range(value: &Value, key: &str) -> Result<(NaiveTime, NaiveTime), ConfigError> {
    let err = || {
        ConfigError(format!(
            "{key} must be 'HH:MM-HH:MM' (may wrap midnight, start != end), got {}",
            value_repr(value)
        ))
    };
    let Some(s) = value.as_str() else {
        return Err(err());
    };
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        return Err(err());
    }
    let mut parsed: Vec<NaiveTime> = Vec::with_capacity(2);
    for part in &parts {
        let pieces: Vec<&str> = part.split(':').collect();
        // Count Unicode scalars, not bytes. The ASCII-digit gate still
        // dominates: any two-scalar piece that passes `is_ascii_digit` has byte
        // length 2 as well, so this is behaviour-neutral for ASCII and only
        // aligns with the convention. Non-ASCII Unicode digits (which a more
        // lenient parser would accept) remain rejected — the blessed
        // ASCII-digit deviation, consistent with the structured_output gate.
        if pieces.len() != 2
            || !pieces
                .iter()
                .all(|p| p.chars().count() == 2 && p.bytes().all(|b| b.is_ascii_digit()))
        {
            return Err(err());
        }
        let hour: u32 = pieces[0].parse().map_err(|_| err())?;
        let minute: u32 = pieces[1].parse().map_err(|_| err())?;
        if hour > 23 || minute > 59 {
            return Err(err());
        }
        parsed.push(NaiveTime::from_hms_opt(hour, minute, 0).ok_or_else(err)?);
    }
    let (start, end) = (parsed[0], parsed[1]);
    if start == end {
        return Err(err());
    }
    Ok((start, end))
}

// ---------------------------------------------------------------------------
// Shared value helpers
// ---------------------------------------------------------------------------

static EMPTY_TABLE: LazyLock<Table> = LazyLock::new(Table::new);

fn empty_table() -> &'static Table {
    LazyLock::force(&EMPTY_TABLE)
}

const fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "str",
        Value::Integer(_) => "int",
        Value::Float(_) => "float",
        Value::Boolean(_) => "bool",
        Value::Datetime(_) => "datetime",
        Value::Array(_) => "list",
        Value::Table(_) => "dict",
    }
}

#[allow(
    clippy::float_cmp,
    reason = "fract() == 0.0 exactly detects integer-valued floats for %g-style formatting"
)]
fn fmt_num(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

#[allow(
    clippy::float_cmp,
    reason = "fract() == 0.0 exactly detects integer-valued bounds for %g formatting"
)]
fn fmt_g(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{x:.0}")
    } else {
        format!("{x}")
    }
}

fn value_repr(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{s}'"),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => fmt_num(*f),
        Value::Boolean(b) => (if *b { "True" } else { "False" }).to_owned(),
        Value::Datetime(d) => format!("'{d}'"),
        Value::Array(_) => "[...]".to_owned(),
        Value::Table(_) => "{...}".to_owned(),
    }
}

fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => fmt_num(*f),
        Value::Boolean(b) => (if *b { "True" } else { "False" }).to_owned(),
        Value::Datetime(d) => d.to_string(),
        other => value_repr(other),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "config numeric values are small; i64→f64 is exact in range"
)]
const fn int_as_f64(n: i64) -> f64 {
    n as f64
}

/// Return the table at `value`, an empty table when absent, or a
/// `"{label} must be a table, got {type}"` error.
fn expect_table<'a>(value: Option<&'a Value>, label: &str) -> Result<&'a Table, ConfigError> {
    match value {
        None => Ok(empty_table()),
        Some(Value::Table(t)) => Ok(t),
        Some(other) => Err(ConfigError(format!(
            "{label} must be a table, got {}",
            value_type_name(other)
        ))),
    }
}

fn check_unknown_keys(section: &Table, allowed: &[&str], prefix: &str) -> Result<(), ConfigError> {
    let mut unknown: Vec<&str> = section
        .keys()
        .map(String::as_str)
        .filter(|k| !allowed.contains(k))
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return Err(ConfigError(format!(
            "{prefix} has unknown keys: {}",
            unknown.join(", ")
        )));
    }
    Ok(())
}

fn sorted_join(items: &[&str]) -> String {
    let mut v: Vec<&str> = items.to_vec();
    v.sort_unstable();
    v.join(", ")
}

fn valid_or_none(set: &BTreeSet<String>) -> String {
    if set.is_empty() {
        "(none)".to_owned()
    } else {
        set.iter().cloned().collect::<Vec<_>>().join(", ")
    }
}

// -- generic field readers (no positivity / range check) --------------------

fn field_string_nonempty(
    raw: &Table,
    prefix: &str,
    key: &str,
    default: &str,
) -> Result<String, ConfigError> {
    match raw.get(key) {
        None => Ok(default.to_owned()),
        Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        Some(_) => Err(ConfigError(format!(
            "{prefix}.{key} must be a non-empty string"
        ))),
    }
}

fn field_int(raw: &Table, prefix: &str, key: &str, default: i64) -> Result<i64, ConfigError> {
    match raw.get(key) {
        None => Ok(default),
        Some(Value::Integer(n)) => Ok(*n),
        Some(other) => Err(ConfigError(format!(
            "{prefix}.{key} must be an integer, got {}",
            value_type_name(other)
        ))),
    }
}

fn field_number(raw: &Table, prefix: &str, key: &str, default: f64) -> Result<f64, ConfigError> {
    match raw.get(key) {
        None => Ok(default),
        Some(Value::Integer(n)) => Ok(int_as_f64(*n)),
        Some(Value::Float(f)) => Ok(*f),
        Some(other) => Err(ConfigError(format!(
            "{prefix}.{key} must be a number, got {}",
            value_type_name(other)
        ))),
    }
}

fn field_bool(raw: &Table, prefix: &str, key: &str, default: bool) -> Result<bool, ConfigError> {
    Ok(field_bool_opt(raw, prefix, key)?.unwrap_or(default))
}

/// Tri-state bool: absent stays `None` so a caller can tell "unset" from an
/// explicit `false`. Same error string as [`field_bool`].
fn field_bool_opt(raw: &Table, prefix: &str, key: &str) -> Result<Option<bool>, ConfigError> {
    match raw.get(key) {
        None => Ok(None),
        Some(Value::Boolean(b)) => Ok(Some(*b)),
        Some(other) => Err(ConfigError(format!(
            "{prefix}.{key} must be a bool, got {}",
            value_type_name(other)
        ))),
    }
}

// -- positivity / range readers ---------------------------------------------

fn positive_int(
    prefix: &str,
    raw: &Table,
    key: &str,
    default: i64,
    positive_kind: bool,
) -> Result<i64, ConfigError> {
    match raw.get(key) {
        None => Ok(default),
        Some(Value::Integer(n)) if *n > 0 => Ok(*n),
        Some(Value::Integer(n)) => Err(ConfigError(format!(
            "{prefix}.{key} must be positive, got {n}"
        ))),
        Some(other) => {
            let kind = if positive_kind {
                "a positive integer"
            } else {
                "an integer"
            };
            Err(ConfigError(format!(
                "{prefix}.{key} must be {kind}, got {}",
                value_type_name(other)
            )))
        }
    }
}

fn positive_float(prefix: &str, raw: &Table, key: &str, default: f64) -> Result<f64, ConfigError> {
    match raw.get(key) {
        None => Ok(default),
        Some(Value::Integer(n)) if *n > 0 => Ok(int_as_f64(*n)),
        Some(Value::Integer(n)) => Err(ConfigError(format!(
            "{prefix}.{key} must be positive, got {n}"
        ))),
        Some(Value::Float(f)) if *f > 0.0 => Ok(*f),
        Some(Value::Float(f)) => Err(ConfigError(format!(
            "{prefix}.{key} must be positive, got {}",
            fmt_num(*f)
        ))),
        Some(other) => Err(ConfigError(format!(
            "{prefix}.{key} must be a number, got {}",
            value_type_name(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Section parsers
// ---------------------------------------------------------------------------

fn parse_sleep_config(raw: &Table) -> Result<(Option<(NaiveTime, NaiveTime)>, i64), ConfigError> {
    check_unknown_keys(raw, &["window", "grace_minutes"], "[sleep]")?;
    let window = match raw.get("window") {
        Some(v) => Some(parse_hhmm_range(v, "window")?),
        None => None,
    };
    let grace = match raw.get("grace_minutes") {
        None => 30,
        Some(Value::Integer(n)) if *n > 0 => *n,
        Some(other) => {
            return Err(ConfigError(format!(
                "[sleep].grace_minutes must be a positive integer, got {}",
                value_repr(other)
            )));
        }
    };
    Ok((window, grace))
}

fn parse_history_windows(raw: &Table) -> Result<(i64, i64), ConfigError> {
    if raw.contains_key("window_size") {
        return Err(ConfigError(
            "[providers.history].window_size has been split into voice_window_size and \
             text_window_size. Replace the legacy key with both. See \
             docs/architecture/tuning.md § History windows."
                .to_owned(),
        ));
    }
    let voice = positive_int("[providers.history]", raw, "voice_window_size", 100, true)?;
    let text = positive_int("[providers.history]", raw, "text_window_size", 200, true)?;
    Ok((voice, text))
}

/// `[providers.history].llm_mirror_calls` — rows the `llm_calls` mirror keeps.
///
/// `0` switches mirroring off; anything negative is a mistake, not a synonym for
/// unbounded (an unbounded mirror is not offered).
fn parse_llm_mirror_calls(raw: &Table) -> Result<i64, ConfigError> {
    match raw.get("llm_mirror_calls") {
        None => Ok(DEFAULT_LLM_MIRROR_CALLS),
        Some(Value::Integer(n)) if *n >= 0 => Ok(*n),
        Some(Value::Integer(n)) => Err(ConfigError(format!(
            "[providers.history].llm_mirror_calls must be >= 0, got {n}"
        ))),
        Some(other) => Err(ConfigError(format!(
            "[providers.history].llm_mirror_calls must be a non-negative integer, got {}",
            value_type_name(other)
        ))),
    }
}

fn parse_coalesce_gap(raw: &Table) -> Result<f64, ConfigError> {
    match raw.get("coalesce_max_gap_seconds") {
        None => Ok(45.0),
        // The `v < 0` sign check runs on the raw TOML value before the
        // float conversion, so a negative integer prints as an int (`got -1`), not a
        // padded float (`got -1.0`). Mirror that per-arm.
        Some(Value::Integer(n)) if *n < 0 => Err(ConfigError(format!(
            "[providers.history].coalesce_max_gap_seconds must be >= 0, got {n}"
        ))),
        Some(Value::Integer(n)) => Ok(int_as_f64(*n)),
        Some(Value::Float(f)) if *f < 0.0 => Err(ConfigError(format!(
            "[providers.history].coalesce_max_gap_seconds must be >= 0, got {}",
            fmt_num(*f)
        ))),
        Some(Value::Float(f)) => Ok(*f),
        Some(other) => Err(ConfigError(format!(
            "[providers.history].coalesce_max_gap_seconds must be a number, got {}",
            value_type_name(other)
        ))),
    }
}

fn parse_text_silence(raw: &Table) -> Result<f64, ConfigError> {
    match raw.get("text_silence_gap_fold_seconds") {
        None => Ok(0.0),
        // Sign check on the raw value (see `parse_coalesce_gap`): integers
        // print as ints, floats through `fmt_num`.
        Some(Value::Integer(n)) if *n < 0 => Err(ConfigError(format!(
            "[providers.history].text_silence_gap_fold_seconds must be >= 0, got {n}"
        ))),
        Some(Value::Integer(n)) => Ok(int_as_f64(*n)),
        Some(Value::Float(f)) if *f < 0.0 => Err(ConfigError(format!(
            "[providers.history].text_silence_gap_fold_seconds must be >= 0, got {}",
            fmt_num(*f)
        ))),
        Some(Value::Float(f)) => Ok(*f),
        Some(other) => Err(ConfigError(format!(
            "[providers.history].text_silence_gap_fold_seconds must be a number, got {}",
            value_type_name(other)
        ))),
    }
}

fn check_temperature(name: &str, value: f64) -> Result<f64, ConfigError> {
    if (0.0..=2.0).contains(&value) {
        Ok(value)
    } else {
        Err(ConfigError(format!(
            "[llm.{name}].temperature must be in [0, 2], got {}",
            fmt_num(value)
        )))
    }
}

fn parse_ranged_float(
    slot: &str,
    section: &Table,
    key: &str,
    lo: f64,
    hi: f64,
) -> Result<Option<f64>, ConfigError> {
    let value = match section.get(key) {
        None => return Ok(None),
        Some(Value::Integer(n)) => int_as_f64(*n),
        Some(Value::Float(f)) => *f,
        Some(other) => {
            return Err(ConfigError(format!(
                "[llm.{slot}].{key} must be a number, got {}",
                value_type_name(other)
            )));
        }
    };
    if (lo..=hi).contains(&value) {
        Ok(Some(value))
    } else {
        Err(ConfigError(format!(
            "[llm.{slot}].{key} must be in [{}, {}], got {}",
            fmt_g(lo),
            fmt_g(hi),
            fmt_num(value)
        )))
    }
}

fn parse_provider_order(
    slot: &str,
    raw: Option<&Value>,
) -> Result<Option<Vec<String>>, ConfigError> {
    match raw {
        None => Ok(None),
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::String(s) if !s.is_empty() => out.push(s.clone()),
                    other => {
                        return Err(ConfigError(format!(
                            "[llm.{slot}].provider_order entries must be non-empty strings, got {}",
                            value_repr(other)
                        )));
                    }
                }
            }
            Ok(Some(out))
        }
        Some(other) => Err(ConfigError(format!(
            "[llm.{slot}].provider_order must be a list of strings, got {}",
            value_type_name(other)
        ))),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat field-by-field parse sequence; splitting it would scatter the error-message contract"
)]
fn parse_llm_slots(raw: &Table) -> Result<BTreeMap<String, LLMSlotConfig>, ConfigError> {
    let mut slots = BTreeMap::new();
    for (name, section) in raw {
        if !LLM_SLOT_NAMES.contains(&name.as_str()) {
            return Err(ConfigError(format!(
                "unknown LLM slot '{name}'; valid slots: {}",
                sorted_join(&LLM_SLOT_NAMES)
            )));
        }
        let Value::Table(section) = section else {
            return Err(ConfigError(format!(
                "[llm.{name}] must be a table, got {}",
                value_type_name(section)
            )));
        };
        let model = match section.get("model") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => {
                return Err(ConfigError(format!(
                    "[llm.{name}].model must be a non-empty string"
                )));
            }
        };
        let temperature = match section.get("temperature") {
            None => None,
            Some(Value::Integer(n)) => Some(check_temperature(name, int_as_f64(*n))?),
            Some(Value::Float(f)) => Some(check_temperature(name, *f)?),
            Some(other) => {
                return Err(ConfigError(format!(
                    "[llm.{name}].temperature must be a number, got {}",
                    value_type_name(other)
                )));
            }
        };
        let top_p = parse_ranged_float(name, section, "top_p", 0.0, 1.0)?;
        let presence_penalty = parse_ranged_float(name, section, "presence_penalty", -2.0, 2.0)?;
        let top_k = match section.get("top_k") {
            None => None,
            Some(Value::Integer(n)) if *n >= 1 => Some(*n),
            Some(other) => {
                return Err(ConfigError(format!(
                    "[llm.{name}].top_k must be a positive integer, got {}",
                    value_repr(other)
                )));
            }
        };
        let prefix = format!("[llm.{name}]");
        let think_prepend = field_bool(section, &prefix, "think_prepend", false)?;
        let provider_order = parse_provider_order(name, section.get("provider_order"))?;
        let provider_allow_fallbacks =
            field_bool(section, &prefix, "provider_allow_fallbacks", true)?;
        let reasoning = match section.get("reasoning") {
            None => None,
            Some(Value::String(s)) => {
                if !REASONING_LEVELS.contains(&s.as_str()) {
                    return Err(ConfigError(format!(
                        "[llm.{name}].reasoning '{s}' unknown; valid options: {}",
                        sorted_join(&REASONING_LEVELS)
                    )));
                }
                if s == "default" {
                    None
                } else {
                    Some(s.clone())
                }
            }
            Some(other) => {
                return Err(ConfigError(format!(
                    "[llm.{name}].reasoning must be a string, got {}",
                    value_type_name(other)
                )));
            }
        };
        let tool_calling = field_bool(section, &prefix, "tool_calling", true)?;
        if !tool_calling {
            return Err(ConfigError(format!(
                "[llm.{name}].tool_calling = false is unsupported: silence and \
                 every other decision the familiar declines to speak for are \
                 tool calls, so a slot without tool calling can never stay \
                 quiet — remove the key (it defaults to true) or set \
                 [llm.{name}].tool_calling = true"
            )));
        }
        let image_tools = field_bool(section, &prefix, "image_tools", false)?;
        let multimodal = field_bool_opt(section, &prefix, "multimodal")?;
        slots.insert(
            name.clone(),
            LLMSlotConfig {
                model,
                temperature,
                top_p,
                top_k,
                presence_penalty,
                think_prepend,
                provider_order,
                provider_allow_fallbacks,
                reasoning,
                tool_calling,
                image_tools,
                multimodal,
            },
        );
    }
    Ok(slots)
}

fn parse_tts_config(raw: &Table) -> Result<TTSConfig, ConfigError> {
    let provider = match raw.get("provider") {
        None => "cartesia".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(ConfigError(format!(
                "[tts].provider must be a string, got {}",
                value_type_name(other)
            )));
        }
    };
    if REMOVED_TTS_PROVIDERS.contains(&provider.as_str()) {
        return Err(ConfigError(format!(
            "[tts].provider '{provider}' is no longer supported — it was an \
             unwired stub and has been removed; use \"cartesia\""
        )));
    }
    if !TTS_PROVIDERS.contains(&provider.as_str()) {
        return Err(ConfigError(format!(
            "[tts].provider '{provider}' unknown; valid options: {}",
            sorted_join(&TTS_PROVIDERS)
        )));
    }
    let greetings = match raw.get("greetings") {
        None => Vec::new(),
        Some(Value::Array(arr)) => arr.iter().map(value_str).collect(),
        Some(other) => {
            return Err(ConfigError(format!(
                "[tts].greetings must be a list of strings, got {}",
                value_type_name(other)
            )));
        }
    };
    Ok(TTSConfig {
        provider,
        cartesia_voice_id: tts_opt_string(raw, "cartesia_voice_id")?,
        cartesia_model: tts_opt_string(raw, "cartesia_model")?,
        greetings,
    })
}

fn tts_opt_string(raw: &Table, key: &str) -> Result<Option<String>, ConfigError> {
    match raw.get(key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(ConfigError(format!("[tts].{key} must be a string"))),
    }
}

fn parse_channel_overrides(raw: &Table) -> Result<BTreeMap<i64, ChannelOverrides>, ConfigError> {
    let mut out = BTreeMap::new();
    for (key, section) in raw {
        let channel_id: i64 = key.parse().map_err(|_| {
            ConfigError(format!(
                "[channels.{key}] key must be an integer channel id"
            ))
        })?;
        let Value::Table(section) = section else {
            return Err(ConfigError(format!(
                "[channels.{key}] must be a table, got {}",
                value_type_name(section)
            )));
        };
        let history_window_size = match section.get("history_window_size") {
            None => None,
            Some(Value::Integer(n)) if *n > 0 => Some(*n),
            Some(Value::Integer(n)) => {
                return Err(ConfigError(format!(
                    "[channels.{key}].history_window_size must be positive, got {n}"
                )));
            }
            Some(other) => {
                return Err(ConfigError(format!(
                    "[channels.{key}].history_window_size must be an integer, got {}",
                    value_type_name(other)
                )));
            }
        };
        let prompt_layers = match section.get("prompt_layers") {
            None => None,
            Some(Value::Array(arr)) if arr.iter().all(|x| x.as_str().is_some()) => Some(
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect(),
            ),
            Some(_) => {
                return Err(ConfigError(format!(
                    "[channels.{key}].prompt_layers must be a list of strings"
                )));
            }
        };
        let message_rendering = match section.get("message_rendering") {
            None => None,
            Some(Value::String(s)) => {
                if !MESSAGE_RENDERINGS.contains(&s.as_str()) {
                    return Err(ConfigError(format!(
                        "[channels.{key}].message_rendering='{s}' unknown; valid options: {}",
                        sorted_join(&MESSAGE_RENDERINGS)
                    )));
                }
                Some(s.clone())
            }
            Some(_) => {
                return Err(ConfigError(format!(
                    "[channels.{key}].message_rendering must be a string"
                )));
            }
        };
        out.insert(
            channel_id,
            ChannelOverrides {
                history_window_size,
                prompt_layers,
                message_rendering,
            },
        );
    }
    Ok(out)
}

fn parse_turn_detection_config(raw: &Table) -> Result<TurnDetectionConfig, ConfigError> {
    let strategy = match raw.get("strategy") {
        None => "deepgram".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(ConfigError(format!(
                "[providers.turn_detection].strategy must be a string, got {}",
                value_type_name(other)
            )));
        }
    };
    if !TURN_STRATEGIES.contains(&strategy.as_str()) {
        return Err(ConfigError(format!(
            "[providers.turn_detection].strategy '{strategy}' unknown; valid options: {}",
            sorted_join(&TURN_STRATEGIES)
        )));
    }
    let local = parse_local_turn_config(expect_table(
        raw.get("local"),
        "[providers.turn_detection.local]",
    )?)?;
    Ok(TurnDetectionConfig { strategy, local })
}

fn parse_local_turn_config(raw: &Table) -> Result<LocalTurnConfig, ConfigError> {
    let d = LocalTurnConfig::default();
    let p = "[providers.turn_detection.local]";
    Ok(LocalTurnConfig {
        smart_turn_repo_id: field_string_nonempty(
            raw,
            p,
            "smart_turn_repo_id",
            &d.smart_turn_repo_id,
        )?,
        smart_turn_filename: field_string_nonempty(
            raw,
            p,
            "smart_turn_filename",
            &d.smart_turn_filename,
        )?,
        silence_ms: field_int(raw, p, "silence_ms", d.silence_ms)?,
        speech_start_ms: field_int(raw, p, "speech_start_ms", d.speech_start_ms)?,
        vad_threshold: field_number(raw, p, "vad_threshold", d.vad_threshold)?,
        smart_turn_threshold: field_number(raw, p, "smart_turn_threshold", d.smart_turn_threshold)?,
        vad_hop_size: field_int(raw, p, "vad_hop_size", d.vad_hop_size)?,
        idle_fallback_s: field_number(raw, p, "idle_fallback_s", d.idle_fallback_s)?,
    })
}

fn parse_stt_config(raw: &Table) -> Result<STTConfig, ConfigError> {
    let backend = match raw.get("backend") {
        None => "deepgram".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(ConfigError(format!(
                "[providers.stt].backend must be a string, got {}",
                value_type_name(other)
            )));
        }
    };
    if !STT_BACKENDS.contains(&backend.as_str()) {
        return Err(ConfigError(format!(
            "[providers.stt].backend '{backend}' unknown; valid options: {}",
            sorted_join(&STT_BACKENDS)
        )));
    }
    Ok(STTConfig {
        backend,
        deepgram: parse_deepgram(expect_table(
            raw.get("deepgram"),
            "[providers.stt.deepgram]",
        )?)?,
        parakeet: parse_parakeet(expect_table(
            raw.get("parakeet"),
            "[providers.stt.parakeet]",
        )?)?,
        faster_whisper: parse_faster_whisper(expect_table(
            raw.get("faster_whisper"),
            "[providers.stt.faster_whisper]",
        )?)?,
    })
}

fn parse_deepgram(raw: &Table) -> Result<DeepgramSTTConfig, ConfigError> {
    let d = DeepgramSTTConfig::default();
    let p = "[providers.stt.deepgram]";
    let keyterms = match raw.get("keyterms") {
        None => d.keyterms.clone(),
        Some(Value::Array(arr)) if arr.iter().all(|x| x.as_str().is_some()) => arr
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        Some(_) => {
            return Err(ConfigError(format!(
                "{p}.keyterms must be a list of strings"
            )));
        }
    };
    Ok(DeepgramSTTConfig {
        model: field_string_nonempty(raw, p, "model", &d.model)?,
        language: field_string_nonempty(raw, p, "language", &d.language)?,
        endpointing_ms: field_int(raw, p, "endpointing_ms", d.endpointing_ms)?,
        utterance_end_ms: field_int(raw, p, "utterance_end_ms", d.utterance_end_ms)?,
        smart_format: field_bool(raw, p, "smart_format", d.smart_format)?,
        punctuate: field_bool(raw, p, "punctuate", d.punctuate)?,
        keyterms,
        replay_buffer_s: field_number(raw, p, "replay_buffer_s", d.replay_buffer_s)?,
        keepalive_interval_s: field_number(raw, p, "keepalive_interval_s", d.keepalive_interval_s)?,
        reconnect_max_attempts: field_int(
            raw,
            p,
            "reconnect_max_attempts",
            d.reconnect_max_attempts,
        )?,
        reconnect_backoff_cap_s: field_number(
            raw,
            p,
            "reconnect_backoff_cap_s",
            d.reconnect_backoff_cap_s,
        )?,
        idle_close_s: field_number(raw, p, "idle_close_s", d.idle_close_s)?,
    })
}

fn parse_parakeet(raw: &Table) -> Result<ParakeetSTTConfig, ConfigError> {
    let d = ParakeetSTTConfig::default();
    let p = "[providers.stt.parakeet]";
    Ok(ParakeetSTTConfig {
        model_name: field_string_nonempty(raw, p, "model_name", &d.model_name)?,
        device: field_string_nonempty(raw, p, "device", &d.device)?,
        idle_close_s: field_number(raw, p, "idle_close_s", d.idle_close_s)?,
    })
}

fn parse_faster_whisper(raw: &Table) -> Result<FasterWhisperSTTConfig, ConfigError> {
    let d = FasterWhisperSTTConfig::default();
    let p = "[providers.stt.faster_whisper]";
    Ok(FasterWhisperSTTConfig {
        model_size: field_string_nonempty(raw, p, "model_size", &d.model_size)?,
        device: field_string_nonempty(raw, p, "device", &d.device)?,
        compute_type: field_string_nonempty(raw, p, "compute_type", &d.compute_type)?,
        language: field_string_nonempty(raw, p, "language", &d.language)?,
        idle_close_s: field_number(raw, p, "idle_close_s", d.idle_close_s)?,
    })
}

// -- memory providers -------------------------------------------------------

fn mem_pos_int(section: &Table, name: &str, key: &str, default: i64) -> Result<i64, ConfigError> {
    match section.get(key) {
        None => Ok(default),
        Some(Value::Integer(n)) if *n > 0 => Ok(*n),
        Some(Value::Integer(n)) => Err(ConfigError(format!(
            "[providers.memory.{name}].{key} must be positive, got {n}"
        ))),
        Some(other) => Err(ConfigError(format!(
            "[providers.memory.{name}].{key} must be a positive integer, got {}",
            value_type_name(other)
        ))),
    }
}

fn mem_pos_float(section: &Table, name: &str, key: &str, default: f64) -> Result<f64, ConfigError> {
    match section.get(key) {
        None => Ok(default),
        Some(Value::Integer(n)) if *n > 0 => Ok(int_as_f64(*n)),
        Some(Value::Integer(n)) => Err(ConfigError(format!(
            "[providers.memory.{name}].{key} must be positive, got {n}"
        ))),
        Some(Value::Float(f)) if *f > 0.0 => Ok(*f),
        Some(Value::Float(f)) => Err(ConfigError(format!(
            "[providers.memory.{name}].{key} must be positive, got {}",
            fmt_num(*f)
        ))),
        Some(other) => Err(ConfigError(format!(
            "[providers.memory.{name}].{key} must be a positive number, got {}",
            value_type_name(other)
        ))),
    }
}

/// Read + compile-check `self_capability_pattern`. Empty means "keep the
/// built-in matcher"; anything else must compile here so a typo fails startup
/// rather than the extractor's first tick.
fn self_capability_pattern(section: &Table, prefix: &str) -> Result<String, ConfigError> {
    let key = "self_capability_pattern";
    let pattern = match section.get(key) {
        None => return Ok(String::new()),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(ConfigError(format!(
                "{prefix}.{key} must be a string, got {}",
                value_type_name(other)
            )));
        }
    };
    if !pattern.is_empty() && regex::Regex::new(&pattern).is_err() {
        return Err(ConfigError(format!(
            "{prefix}.{key} must be a valid regex, got '{pattern}'"
        )));
    }
    Ok(pattern)
}

fn worker_section<'a>(raw: &'a Table, name: &str) -> Result<&'a Table, ConfigError> {
    match raw.get(name) {
        None => Ok(empty_table()),
        Some(Value::Table(t)) => Ok(t),
        Some(other) => Err(ConfigError(format!(
            "[providers.memory.{name}] must be a table, got {}",
            value_type_name(other)
        ))),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "per-worker knobs are enumerated explicitly rather than derived by reflection"
)]
fn parse_memory_providers(
    raw: &Table,
    known_projectors: &BTreeSet<String>,
) -> Result<MemoryProvidersConfig, ConfigError> {
    check_unknown_keys(
        raw,
        &[
            "projectors",
            "rolling_summary",
            "rich_note",
            "people_dossier",
            "reflection",
            "fact_supersede",
        ],
        "[providers.memory]",
    )?;

    let rs = worker_section(raw, "rolling_summary")?;
    check_unknown_keys(
        rs,
        &["turns_threshold", "tick_interval_s"],
        "[providers.memory.rolling_summary]",
    )?;
    let rolling_defaults = RollingSummaryConfig::default();
    let rolling_summary = RollingSummaryConfig {
        turns_threshold: mem_pos_int(
            rs,
            "rolling_summary",
            "turns_threshold",
            rolling_defaults.turns_threshold,
        )?,
        tick_interval_s: mem_pos_float(
            rs,
            "rolling_summary",
            "tick_interval_s",
            rolling_defaults.tick_interval_s,
        )?,
    };

    let rn = worker_section(raw, "rich_note")?;
    check_unknown_keys(
        rn,
        &[
            "batch_size",
            "tick_interval_s",
            "participants_max",
            "self_capability_filter",
            "self_capability_pattern",
        ],
        "[providers.memory.rich_note]",
    )?;
    let rich_defaults = RichNoteConfig::default();
    let rich_prefix = "[providers.memory.rich_note]";
    let rich_note = RichNoteConfig {
        batch_size: mem_pos_int(rn, "rich_note", "batch_size", rich_defaults.batch_size)?,
        tick_interval_s: mem_pos_float(
            rn,
            "rich_note",
            "tick_interval_s",
            rich_defaults.tick_interval_s,
        )?,
        participants_max: mem_pos_int(
            rn,
            "rich_note",
            "participants_max",
            rich_defaults.participants_max,
        )?,
        self_capability_filter: field_bool(
            rn,
            rich_prefix,
            "self_capability_filter",
            rich_defaults.self_capability_filter,
        )?,
        self_capability_pattern: self_capability_pattern(rn, rich_prefix)?,
    };

    let pd = worker_section(raw, "people_dossier")?;
    check_unknown_keys(
        pd,
        &["tick_interval_s"],
        "[providers.memory.people_dossier]",
    )?;
    let dossier_defaults = PeopleDossierConfig::default();
    let people_dossier = PeopleDossierConfig {
        tick_interval_s: mem_pos_float(
            pd,
            "people_dossier",
            "tick_interval_s",
            dossier_defaults.tick_interval_s,
        )?,
    };

    let rf = worker_section(raw, "reflection")?;
    check_unknown_keys(
        rf,
        &[
            "turns_threshold",
            "max_reflections_per_tick",
            "max_turns_per_tick",
            "recent_facts_limit",
            "tick_interval_s",
        ],
        "[providers.memory.reflection]",
    )?;
    let reflection_defaults = ReflectionConfig::default();
    let reflection = ReflectionConfig {
        turns_threshold: mem_pos_int(
            rf,
            "reflection",
            "turns_threshold",
            reflection_defaults.turns_threshold,
        )?,
        max_reflections_per_tick: mem_pos_int(
            rf,
            "reflection",
            "max_reflections_per_tick",
            reflection_defaults.max_reflections_per_tick,
        )?,
        max_turns_per_tick: mem_pos_int(
            rf,
            "reflection",
            "max_turns_per_tick",
            reflection_defaults.max_turns_per_tick,
        )?,
        recent_facts_limit: mem_pos_int(
            rf,
            "reflection",
            "recent_facts_limit",
            reflection_defaults.recent_facts_limit,
        )?,
        tick_interval_s: mem_pos_float(
            rf,
            "reflection",
            "tick_interval_s",
            reflection_defaults.tick_interval_s,
        )?,
    };

    let fs = worker_section(raw, "fact_supersede")?;
    check_unknown_keys(
        fs,
        &["batch_size", "tick_interval_s", "priors_max"],
        "[providers.memory.fact_supersede]",
    )?;
    let supersede_defaults = FactSupersedeConfig::default();
    let fact_supersede = FactSupersedeConfig {
        batch_size: mem_pos_int(
            fs,
            "fact_supersede",
            "batch_size",
            supersede_defaults.batch_size,
        )?,
        tick_interval_s: mem_pos_float(
            fs,
            "fact_supersede",
            "tick_interval_s",
            supersede_defaults.tick_interval_s,
        )?,
        priors_max: mem_pos_int(
            fs,
            "fact_supersede",
            "priors_max",
            supersede_defaults.priors_max,
        )?,
    };

    let projectors = match raw.get("projectors") {
        None => default_projectors(),
        Some(Value::Array(arr)) if arr.iter().all(|x| x.as_str().is_some()) => {
            let names: Vec<String> = arr
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            for name in &names {
                if !known_projectors.contains(name) {
                    return Err(ConfigError(format!(
                        "[providers.memory].projectors lists unknown memory projector '{name}'; valid: {}",
                        valid_or_none(known_projectors)
                    )));
                }
            }
            names
        }
        Some(_) => {
            return Err(ConfigError(
                "[providers.memory].projectors must be a list of strings".to_owned(),
            ));
        }
    };

    Ok(MemoryProvidersConfig {
        projectors,
        rolling_summary,
        rich_note,
        people_dossier,
        reflection,
        fact_supersede,
    })
}

fn parse_embedding_config(
    raw: &Table,
    known_embedders: &BTreeSet<String>,
    fastembed_native_dim: FastembedNativeDim,
) -> Result<EmbeddingConfig, ConfigError> {
    check_unknown_keys(
        raw,
        &["backend", "dim", "fastembed_model", "fastembed_cache_dir"],
        "[providers.embedding]",
    )?;
    let d = EmbeddingConfig::default();
    let backend = match raw.get("backend") {
        None => d.backend.clone(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(ConfigError(format!(
                "[providers.embedding].backend must be a string, got {}",
                value_type_name(other)
            )));
        }
    };
    if !known_embedders.contains(&backend) {
        return Err(ConfigError(format!(
            "[providers.embedding].backend = '{backend}' is unknown; valid: {}",
            valid_or_none(known_embedders)
        )));
    }
    let fastembed_model = match raw.get("fastembed_model") {
        None => d.fastembed_model.clone(),
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(other) => {
            return Err(ConfigError(format!(
                "[providers.embedding].fastembed_model must be a non-empty string, got {}",
                value_type_name(other)
            )));
        }
    };
    // The selected model's native dim, when knowable without a runtime probe.
    // Only `fastembed` is model-bound; `hash` owns its `dim` outright, and an
    // unmapped model name stays unchecked (dim known only after the probe).
    let native_dim: Option<i64> = if backend == "fastembed" {
        fastembed_native_dim(&fastembed_model).and_then(|n| i64::try_from(n).ok())
    } else {
        None
    };
    let dim = match raw.get("dim") {
        // Unset: adopt the model's native dim so the struct never advertises a
        // width that is wrong for the selected model.
        None => native_dim.unwrap_or(d.dim),
        Some(Value::Integer(n)) if *n > 0 => {
            if let Some(native) = native_dim
                && *n != native
            {
                return Err(ConfigError(format!(
                    "[providers.embedding].dim = {n} contradicts fastembed_model \
                     '{fastembed_model}' (native dim {native}); remove `dim` or set it to {native}."
                )));
            }
            *n
        }
        Some(Value::Integer(n)) => {
            return Err(ConfigError(format!(
                "[providers.embedding].dim must be > 0, got {n}"
            )));
        }
        Some(other) => {
            return Err(ConfigError(format!(
                "[providers.embedding].dim must be a positive integer, got {}",
                value_type_name(other)
            )));
        }
    };
    let fastembed_cache_dir = match raw.get("fastembed_cache_dir") {
        None => d.fastembed_cache_dir,
        Some(Value::String(s)) => {
            if s.is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        Some(other) => {
            return Err(ConfigError(format!(
                "[providers.embedding].fastembed_cache_dir must be a string, got {}",
                value_type_name(other)
            )));
        }
    };
    Ok(EmbeddingConfig {
        backend,
        dim,
        fastembed_model,
        fastembed_cache_dir,
    })
}

// -- budgets ----------------------------------------------------------------

fn parse_budgets(raw: &Table) -> Result<BTreeMap<String, TierBudget>, ConfigError> {
    let mut out = default_budgets();
    for (tier, section) in raw {
        if !BUDGET_TIER_NAMES.contains(&tier.as_str()) {
            return Err(ConfigError(format!(
                "unknown budget tier '{tier}'; valid tiers: {}",
                sorted_join(&BUDGET_TIER_NAMES)
            )));
        }
        let Value::Table(section) = section else {
            return Err(ConfigError(format!(
                "[budget.{tier}] must be a table, got {}",
                value_type_name(section)
            )));
        };
        let default = out.get(tier).copied().unwrap_or_default();
        out.insert(tier.clone(), parse_tier_budget(tier, section, &default)?);
    }
    Ok(out)
}

fn parse_tier_budget(
    tier: &str,
    raw: &Table,
    default: &TierBudget,
) -> Result<TierBudget, ConfigError> {
    check_unknown_keys(raw, &BUDGET_FIELDS, &format!("[budget.{tier}]"))?;
    let get = |key: &str, fallback: i64| -> Result<i64, ConfigError> {
        match raw.get(key) {
            None => Ok(fallback),
            Some(Value::Integer(n)) if *n > 0 => Ok(*n),
            Some(Value::Integer(n)) => Err(ConfigError(format!(
                "[budget.{tier}].{key} must be positive, got {n}"
            ))),
            Some(other) => Err(ConfigError(format!(
                "[budget.{tier}].{key} must be a positive integer, got {}",
                value_type_name(other)
            ))),
        }
    };
    Ok(TierBudget {
        recent_history_tokens: get("recent_history_tokens", default.recent_history_tokens)?,
        rag_tokens: get("rag_tokens", default.rag_tokens)?,
        dossier_tokens: get("dossier_tokens", default.dossier_tokens)?,
        summary_tokens: get("summary_tokens", default.summary_tokens)?,
        reflection_tokens: get("reflection_tokens", default.reflection_tokens)?,
        lorebook_tokens: get("lorebook_tokens", default.lorebook_tokens)?,
        max_history_turns: get("max_history_turns", default.max_history_turns)?,
        max_rag_turns: get("max_rag_turns", default.max_rag_turns)?,
        max_rag_facts: get("max_rag_facts", default.max_rag_facts)?,
        max_dossier_people: get("max_dossier_people", default.max_dossier_people)?,
        max_reflections: get("max_reflections", default.max_reflections)?,
        max_lorebook_entries: get("max_lorebook_entries", default.max_lorebook_entries)?,
    })
}

fn set_curve_field(curve: &mut ModelBudgetCurve, key: &str, value: f64) {
    match key {
        "recent_history_tokens" => curve.recent_history_tokens = value,
        "rag_tokens" => curve.rag_tokens = value,
        "dossier_tokens" => curve.dossier_tokens = value,
        "summary_tokens" => curve.summary_tokens = value,
        "reflection_tokens" => curve.reflection_tokens = value,
        "lorebook_tokens" => curve.lorebook_tokens = value,
        "max_history_turns" => curve.max_history_turns = value,
        "max_rag_turns" => curve.max_rag_turns = value,
        "max_rag_facts" => curve.max_rag_facts = value,
        "max_dossier_people" => curve.max_dossier_people = value,
        "max_reflections" => curve.max_reflections = value,
        "max_lorebook_entries" => curve.max_lorebook_entries = value,
        _ => {}
    }
}

fn parse_budget_curves(raw: &Table) -> Result<BTreeMap<String, ModelBudgetCurve>, ConfigError> {
    let mut out = BTreeMap::new();
    for (model_name, section) in raw {
        let Value::Table(section) = section else {
            return Err(ConfigError(format!(
                "[budget.model_curves.'{model_name}'] must be a table, got {}",
                value_type_name(section)
            )));
        };
        check_unknown_keys(
            section,
            &BUDGET_FIELDS,
            &format!("[budget.model_curves.'{model_name}']"),
        )?;
        let mut curve = ModelBudgetCurve::default();
        for (key, value) in section {
            let fv = match value {
                Value::Integer(n) => int_as_f64(*n),
                Value::Float(f) => *f,
                other => {
                    return Err(ConfigError(format!(
                        "[budget.model_curves.'{model_name}'].{key} must be a positive number, got {}",
                        value_type_name(other)
                    )));
                }
            };
            if fv <= 0.0 {
                return Err(ConfigError(format!(
                    "[budget.model_curves.'{model_name}'].{key} must be positive, got {}",
                    fmt_num(fv)
                )));
            }
            set_curve_field(&mut curve, key, fv);
        }
        out.insert(model_name.clone(), curve);
    }
    Ok(out)
}

// -- discord.text / prompt / memory.retrieval / focus / tools ---------------

fn parse_discord_text_config(raw: &Table) -> Result<DiscordTextConfig, ConfigError> {
    check_unknown_keys(
        raw,
        &[
            "respond_to_typing",
            "typing_backoff_initial_s",
            "typing_backoff_max_s",
        ],
        "[discord.text]",
    )?;
    let d = DiscordTextConfig::default();
    let respond_to_typing = field_bool(
        raw,
        "[discord.text]",
        "respond_to_typing",
        d.respond_to_typing,
    )?;
    let typing_backoff_initial_s = positive_float(
        "[discord.text]",
        raw,
        "typing_backoff_initial_s",
        d.typing_backoff_initial_s,
    )?;
    let typing_backoff_max_s = positive_float(
        "[discord.text]",
        raw,
        "typing_backoff_max_s",
        d.typing_backoff_max_s,
    )?;
    if typing_backoff_max_s < typing_backoff_initial_s {
        return Err(ConfigError(format!(
            "[discord.text].typing_backoff_max_s must be >= typing_backoff_initial_s, got {} < {}",
            fmt_num(typing_backoff_max_s),
            fmt_num(typing_backoff_initial_s)
        )));
    }
    Ok(DiscordTextConfig {
        respond_to_typing,
        typing_backoff_initial_s,
        typing_backoff_max_s,
    })
}

struct PromptFields {
    post_history_instructions: String,
    image_description_constraints: String,
    operating_mode_voice: String,
    operating_mode_text: String,
    voice_tool_ack: String,
    shift_focus_coaching: String,
    start_activity_description: String,
    rolling_summary_system: String,
    reflection_system: String,
    dossier_self_system: String,
    dossier_other_system: String,
    sleep_consolidation_system: String,
    sleep_stance_system: String,
    sleep_synthesis_system: String,
    dream_extraction_clause: String,
}

fn parse_prompt_config(raw: &Table) -> Result<PromptFields, ConfigError> {
    check_unknown_keys(raw, &PROMPT_FIELDS, "[prompt]")?;
    let get = |key: &str| -> Result<String, ConfigError> {
        match raw.get(key) {
            None => Ok(String::new()),
            Some(Value::String(s)) => Ok(s.trim().to_owned()),
            Some(other) => Err(ConfigError(format!(
                "[prompt].{key} must be a string, got {}",
                value_type_name(other)
            ))),
        }
    };
    Ok(PromptFields {
        post_history_instructions: get("post_history_instructions")?,
        image_description_constraints: get("image_description_constraints")?,
        operating_mode_voice: get("operating_mode_voice")?,
        operating_mode_text: get("operating_mode_text")?,
        voice_tool_ack: get("voice_tool_ack")?,
        shift_focus_coaching: get("shift_focus_coaching")?,
        start_activity_description: get("start_activity_description")?,
        rolling_summary_system: get("rolling_summary_system")?,
        reflection_system: get("reflection_system")?,
        dossier_self_system: get("dossier_self_system")?,
        dossier_other_system: get("dossier_other_system")?,
        sleep_consolidation_system: get("sleep_consolidation_system")?,
        sleep_stance_system: get("sleep_stance_system")?,
        sleep_synthesis_system: get("sleep_synthesis_system")?,
        dream_extraction_clause: get("dream_extraction_clause")?,
    })
}

fn parse_memory_retrieval(raw: &Table) -> Result<MemoryRetrievalConfig, ConfigError> {
    check_unknown_keys(raw, &RETRIEVAL_FIELDS, "[memory.retrieval]")?;
    let d = MemoryRetrievalConfig::default();
    let get = |key: &str, fallback: f64| -> Result<f64, ConfigError> {
        match raw.get(key) {
            None => Ok(fallback),
            // Sign check on the raw value before `float(v)` (see
            // `parse_coalesce_gap`): a negative integer prints as `got -1`, not
            // `got -1.0`.
            Some(Value::Integer(n)) if *n < 0 => Err(ConfigError(format!(
                "[memory.retrieval].{key} must be non-negative, got {n}"
            ))),
            Some(Value::Integer(n)) => Ok(int_as_f64(*n)),
            Some(Value::Float(f)) if *f < 0.0 => Err(ConfigError(format!(
                "[memory.retrieval].{key} must be non-negative, got {}",
                fmt_num(*f)
            ))),
            Some(Value::Float(f)) => Ok(*f),
            Some(other) => Err(ConfigError(format!(
                "[memory.retrieval].{key} must be a non-negative number, got {}",
                value_type_name(other)
            ))),
        }
    };
    Ok(MemoryRetrievalConfig {
        bm25_weight: get("bm25_weight", d.bm25_weight)?,
        recency_weight: get("recency_weight", d.recency_weight)?,
        importance_weight: get("importance_weight", d.importance_weight)?,
        embedding_weight: get("embedding_weight", d.embedding_weight)?,
    })
}

fn parse_focus_config(raw: &Table) -> Result<FocusConfig, ConfigError> {
    check_unknown_keys(
        raw,
        &[
            "unread_nudge_enabled",
            "nudge_debounce_seconds",
            "catch_up_limit",
        ],
        "[focus]",
    )?;
    let d = FocusConfig::default();
    Ok(FocusConfig {
        unread_nudge_enabled: field_bool(
            raw,
            "[focus]",
            "unread_nudge_enabled",
            d.unread_nudge_enabled,
        )?,
        nudge_debounce_seconds: positive_float(
            "[focus]",
            raw,
            "nudge_debounce_seconds",
            d.nudge_debounce_seconds,
        )?,
        catch_up_limit: positive_int("[focus]", raw, "catch_up_limit", d.catch_up_limit, false)?,
    })
}

fn parse_voice_config(raw: &Table) -> Result<VoiceConfig, ConfigError> {
    check_unknown_keys(raw, &["roster_event_window_seconds"], "[voice]")?;
    let d = VoiceConfig::default();
    Ok(VoiceConfig {
        roster_event_window_seconds: positive_float(
            "[voice]",
            raw,
            "roster_event_window_seconds",
            d.roster_event_window_seconds,
        )?,
    })
}

fn parse_tools_config(raw: &Table) -> Result<ToolsConfig, ConfigError> {
    check_unknown_keys(
        raw,
        &[
            "loop_max_iterations",
            "allow_untrusted_image_urls",
            "trusted_image_hosts",
        ],
        "[tools]",
    )?;
    let d = ToolsConfig::default();
    Ok(ToolsConfig {
        loop_max_iterations: positive_int(
            "[tools]",
            raw,
            "loop_max_iterations",
            d.loop_max_iterations,
            true,
        )?,
        allow_untrusted_image_urls: field_bool(
            raw,
            "[tools]",
            "allow_untrusted_image_urls",
            d.allow_untrusted_image_urls,
        )?,
        trusted_image_hosts: parse_trusted_image_hosts(raw, d.trusted_image_hosts)?,
    })
}

/// Bare hostnames (or `*.suffix` patterns), lowercased. Rejects anything
/// carrying a scheme, path, or port at load rather than silently never
/// matching.
fn parse_trusted_image_hosts(
    raw: &Table,
    default: Vec<String>,
) -> Result<Vec<String>, ConfigError> {
    let Some(value) = raw.get("trusted_image_hosts") else {
        return Ok(default);
    };
    let Value::Array(entries) = value else {
        return Err(ConfigError(format!(
            "[tools].trusted_image_hosts must be a list of strings, got {}",
            value_type_name(value)
        )));
    };
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        match entry {
            Value::String(s) if crate::tools::image_policy::is_host_pattern(s) => {
                out.push(s.to_ascii_lowercase());
            }
            other => {
                return Err(ConfigError(format!(
                    "[tools].trusted_image_hosts entries must be bare hostnames, optionally '*.'-prefixed, got {}",
                    value_repr(other)
                )));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        BUDGET_TIER_NAMES, ChannelOverrides, CharacterConfig, DeepgramSTTConfig, DiscordTextConfig,
        EmbeddingConfig, FactSupersedeConfig, FocusConfig, LLM_SLOT_NAMES, MemoryProvidersConfig,
        MemoryRetrievalConfig, PeopleDossierConfig, ReflectionConfig, RichNoteConfig,
        RollingSummaryConfig, STTConfig, TTSConfig, ToolsConfig, TurnDetectionConfig,
        default_projectors, parse_tts_config,
    };
    use crate::budget::TierBudget;
    use std::collections::BTreeSet;
    use toml::Table;

    #[test]
    fn tiered_slots() {
        let got: BTreeSet<&str> = LLM_SLOT_NAMES.into_iter().collect();
        let want: BTreeSet<&str> = ["fast", "prose", "background"].into_iter().collect();
        assert_eq!(got, want);
    }

    #[test]
    fn character_config_defaults() {
        let cfg = CharacterConfig::default();
        assert_eq!(cfg.display_tz, "UTC");
        assert!(cfg.aliases.is_empty());
        assert_eq!(cfg.voice_window_size, 100);
        assert_eq!(cfg.text_window_size, 200);
        assert!(cfg.llm.is_empty());
        assert!(cfg.sleep_window.is_none());
        assert_eq!(cfg.sleep_grace_minutes, 30);
    }

    #[test]
    fn tts_defaults_to_the_only_wired_provider() {
        // #N1: the default must be synthesizable, not a stub backend.
        assert_eq!(TTSConfig::default().provider, "cartesia");
        assert_eq!(
            parse_tts_config(&Table::new())
                .expect("empty [tts] parses")
                .provider,
            "cartesia"
        );
    }

    #[test]
    fn removed_tts_providers_name_the_removal() {
        // #N1: azure/gemini stubs deleted — a stale profile gets a migration hint.
        for provider in ["azure", "gemini"] {
            let mut raw = Table::new();
            raw.insert("provider".to_owned(), provider.into());
            let err = parse_tts_config(&raw).expect_err("removed provider rejected");
            assert_eq!(
                err.0,
                format!(
                    "[tts].provider '{provider}' is no longer supported — it was an \
                     unwired stub and has been removed; use \"cartesia\""
                )
            );
        }
    }

    #[test]
    fn unknown_tts_provider_lists_the_only_option() {
        let mut raw = Table::new();
        raw.insert("provider".to_owned(), "foo".into());
        let err = parse_tts_config(&raw).expect_err("unknown provider rejected");
        assert_eq!(
            err.0,
            "[tts].provider 'foo' unknown; valid options: cartesia"
        );
    }

    #[test]
    fn discord_text_defaults_present() {
        let d = DiscordTextConfig::default();
        assert!(d.respond_to_typing);
        assert!(d.typing_backoff_initial_s > 0.0);
        assert!(d.typing_backoff_max_s >= d.typing_backoff_initial_s);
        assert_eq!(CharacterConfig::default().discord_text, d);
    }

    #[test]
    fn focus_defaults_present() {
        let f = FocusConfig::default();
        assert!(f.unread_nudge_enabled);
        assert!((f.nudge_debounce_seconds - 30.0).abs() < f64::EPSILON);
        assert_eq!(f.catch_up_limit, 20);
        assert_eq!(CharacterConfig::default().focus, f);
    }

    #[test]
    fn tools_defaults_present() {
        assert_eq!(ToolsConfig::default().loop_max_iterations, 5);
        assert_eq!(CharacterConfig::default().tools, ToolsConfig::default());
    }

    #[test]
    fn llm_max_concurrent_default_is_four() {
        assert_eq!(CharacterConfig::default().llm_max_concurrent_requests, 4);
    }

    #[test]
    fn memory_worker_dataclass_defaults_match_legacy_hardcodes() {
        assert_eq!(
            RollingSummaryConfig::default(),
            RollingSummaryConfig {
                turns_threshold: 10,
                tick_interval_s: 5.0
            }
        );
        assert_eq!(
            RichNoteConfig::default(),
            RichNoteConfig {
                batch_size: 10,
                tick_interval_s: 15.0,
                participants_max: 30,
                self_capability_filter: true,
                self_capability_pattern: String::new(),
            }
        );
        assert_eq!(
            PeopleDossierConfig::default(),
            PeopleDossierConfig {
                tick_interval_s: 20.0
            }
        );
        assert_eq!(
            ReflectionConfig::default(),
            ReflectionConfig {
                turns_threshold: 20,
                max_reflections_per_tick: 3,
                max_turns_per_tick: 50,
                recent_facts_limit: 20,
                tick_interval_s: 60.0
            }
        );
        assert_eq!(
            FactSupersedeConfig::default(),
            FactSupersedeConfig {
                batch_size: 5,
                tick_interval_s: 60.0,
                priors_max: 20
            }
        );
    }

    #[test]
    fn memory_retrieval_dataclass_default_is_bm25_only() {
        let r = MemoryRetrievalConfig::default();
        assert!((r.bm25_weight - 1.0).abs() < f64::EPSILON);
        assert!(r.recency_weight.abs() < f64::EPSILON);
        assert!(r.importance_weight.abs() < f64::EPSILON);
        assert!(r.embedding_weight.abs() < f64::EPSILON);
    }

    #[test]
    fn embedding_dataclass_default_is_off() {
        let e = EmbeddingConfig::default();
        assert_eq!(e.backend, "off");
        assert_eq!(e.dim, 256);
        assert_eq!(e.fastembed_model, "BAAI/bge-small-en-v1.5");
        assert!(e.fastembed_cache_dir.is_none());
    }

    #[test]
    fn stt_and_turn_detection_defaults() {
        let s = STTConfig::default();
        assert_eq!(s.backend, "deepgram");
        assert_eq!(DeepgramSTTConfig::default(), s.deepgram);
        let dg = DeepgramSTTConfig::default();
        assert_eq!(dg.model, "nova-3");
        assert_eq!(dg.language, "en");
        assert_eq!(dg.endpointing_ms, 500);
        assert_eq!(dg.utterance_end_ms, 1500);
        assert!(dg.smart_format);
        assert!(dg.punctuate);
        assert!(dg.keyterms.is_empty());
        assert_eq!(dg.reconnect_max_attempts, 5);
        assert_eq!(TurnDetectionConfig::default().strategy, "deepgram");
    }

    #[test]
    fn channel_overrides_default_is_all_none() {
        let over = ChannelOverrides::default();
        assert!(over.history_window_size.is_none());
        assert!(over.prompt_layers.is_none());
        assert!(over.message_rendering.is_none());
    }

    #[test]
    fn memory_providers_default_projectors_are_the_five() {
        assert_eq!(
            MemoryProvidersConfig::default().projectors,
            default_projectors()
        );
        assert_eq!(
            default_projectors(),
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
    fn budget_tier_names_are_canonical() {
        let got: BTreeSet<&str> = BUDGET_TIER_NAMES.into_iter().collect();
        let want: BTreeSet<&str> = ["voice", "text", "background"].into_iter().collect();
        assert_eq!(got, want);
    }

    #[test]
    fn budget_for_returns_tier_base_with_no_slot() {
        let cfg = CharacterConfig::default();
        assert_eq!(cfg.budget_for("voice"), TierBudget::default());
        assert_eq!(cfg.budget_for("text"), TierBudget::default());
    }

    #[test]
    fn window_for_falls_back_to_tier_default() {
        let cfg = CharacterConfig::default();
        assert_eq!(cfg.voice_window_for(None), 100);
        assert_eq!(cfg.text_window_for(None), 200);
        assert_eq!(cfg.voice_window_for(Some(999)), 100);
    }
}
