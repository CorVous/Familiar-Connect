//! Topic string constants (subsystem 01; Python `bus/topics.py`).
//!
//! One file, grep-friendly. Every topic the system routes on lives here. Dotted
//! namespaces group related topics; the prefix names the subsystem. The
//! declared-but-unused topics are the reserved namespace for planned streaming —
//! keep them (spec 01 Data formats).

// --- Inputs ---
/// Discord text messages (and synthetic alarm wakes).
pub const TOPIC_DISCORD_TEXT: &str = "discord.text";
/// Discord voice-state changes (declared, unused).
pub const TOPIC_DISCORD_VOICE_STATE: &str = "discord.voice.state";

/// Raw voice audio frames (declared, unused — audio bypasses the bus today).
pub const TOPIC_VOICE_AUDIO_RAW: &str = "voice.audio.raw";
/// Interim voice transcripts.
pub const TOPIC_VOICE_TRANSCRIPT_PARTIAL: &str = "voice.transcript.partial";
/// Finalized voice transcripts.
pub const TOPIC_VOICE_TRANSCRIPT_FINAL: &str = "voice.transcript.final";
/// A user started speaking (barge-in signal).
pub const TOPIC_VOICE_ACTIVITY_START: &str = "voice.activity.start";
/// A user stopped speaking.
pub const TOPIC_VOICE_ACTIVITY_END: &str = "voice.activity.end";

/// Twitch EventSub notifications.
pub const TOPIC_TWITCH_EVENT: &str = "twitch.event";

// --- LLM / TTS outputs (declared, unused) ---
/// Streaming LLM response chunk.
pub const TOPIC_LLM_RESPONSE_CHUNK: &str = "llm.response.chunk";
/// Final LLM response.
pub const TOPIC_LLM_RESPONSE_FINAL: &str = "llm.response.final";

/// Streaming TTS audio chunk.
pub const TOPIC_TTS_AUDIO_CHUNK: &str = "tts.audio.chunk";
/// Final TTS audio.
pub const TOPIC_TTS_AUDIO_FINAL: &str = "tts.audio.final";

// --- Tool-driven wakes ---
/// A scheduled alarm fired.
pub const TOPIC_ALARM_FIRED: &str = "alarm.fired";
