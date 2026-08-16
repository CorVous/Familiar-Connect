//! Topic constant sanity.

use std::collections::HashSet;

use familiar_connect::bus::topics;

#[test]
// The `!values.is_empty()` check pins that the topic list is non-empty
// defined"` guard against an empty `dir(topics)`. In Rust the topic list is a
// fixed array, so non-emptiness is a compile-time constant; keep the assertion
// for parity and silence the resulting always-true lint.
#[allow(clippy::const_is_empty)]
fn topics_are_strings_and_unique() {
    let values: [&str; 13] = [
        topics::TOPIC_DISCORD_TEXT,
        topics::TOPIC_DISCORD_VOICE_STATE,
        topics::TOPIC_VOICE_AUDIO_RAW,
        topics::TOPIC_VOICE_TRANSCRIPT_PARTIAL,
        topics::TOPIC_VOICE_TRANSCRIPT_FINAL,
        topics::TOPIC_VOICE_ACTIVITY_START,
        topics::TOPIC_VOICE_ACTIVITY_END,
        topics::TOPIC_TWITCH_EVENT,
        topics::TOPIC_LLM_RESPONSE_CHUNK,
        topics::TOPIC_LLM_RESPONSE_FINAL,
        topics::TOPIC_TTS_AUDIO_CHUNK,
        topics::TOPIC_TTS_AUDIO_FINAL,
        topics::TOPIC_ALARM_FIRED,
    ];
    assert!(!values.is_empty(), "no topics defined");
    let unique: HashSet<&str> = values.iter().copied().collect();
    assert_eq!(unique.len(), values.len(), "duplicate topic string");
}

#[test]
fn core_topics_present() {
    assert_eq!(topics::TOPIC_DISCORD_TEXT, "discord.text");
    assert_eq!(
        topics::TOPIC_VOICE_TRANSCRIPT_FINAL,
        "voice.transcript.final"
    );
    assert_eq!(topics::TOPIC_TWITCH_EVENT, "twitch.event");
}
