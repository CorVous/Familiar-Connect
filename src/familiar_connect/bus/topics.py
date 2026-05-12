"""Topic string constants for the event bus.

One file, grep-friendly. Every topic the system routes on lives here.
Dotted namespaces group related topics; prefix is the subsystem.
"""

from __future__ import annotations

# Inputs --------------------------------------------------------------------
TOPIC_DISCORD_TEXT = "discord.text"
TOPIC_DISCORD_VOICE_STATE = "discord.voice.state"

TOPIC_VOICE_AUDIO_RAW = "voice.audio.raw"
TOPIC_VOICE_TRANSCRIPT_PARTIAL = "voice.transcript.partial"
TOPIC_VOICE_TRANSCRIPT_FINAL = "voice.transcript.final"
TOPIC_VOICE_ACTIVITY_START = "voice.activity.start"
TOPIC_VOICE_ACTIVITY_END = "voice.activity.end"

TOPIC_TWITCH_EVENT = "twitch.event"

# LLM / TTS outputs ---------------------------------------------------------
TOPIC_LLM_RESPONSE_CHUNK = "llm.response.chunk"
TOPIC_LLM_RESPONSE_FINAL = "llm.response.final"

TOPIC_TTS_AUDIO_CHUNK = "tts.audio.chunk"
TOPIC_TTS_AUDIO_FINAL = "tts.audio.final"
