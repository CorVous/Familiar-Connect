//! `TtsPlayer` seam + Discord/logging/mock playback impls
//! (subsystem 09; Python `tts_player/`).

pub mod discord_player;
pub mod logging_player;
pub mod mock;
pub mod protocol;

pub use discord_player::{AudioSource, DiscordVoicePlayer, PlayError, VoiceClientLike};
pub use logging_player::LoggingTTSPlayer;
pub use mock::MockTTSPlayer;
pub use protocol::TtsPlayer;
