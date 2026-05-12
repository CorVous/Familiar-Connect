"""TTS playback abstraction.

:class:`TTSPlayer` is the speak surface used by
:class:`familiar_connect.processors.voice_responder.VoiceResponder`.
Production wraps Cartesia/Azure/Gemini + Discord voice playback via
:class:`DiscordVoicePlayer`. Tests use :class:`MockTTSPlayer`, which
records audio duration played before cancellation.
"""

from __future__ import annotations

from familiar_connect.tts_player.discord_player import DiscordVoicePlayer
from familiar_connect.tts_player.logging_player import LoggingTTSPlayer
from familiar_connect.tts_player.mock import MockTTSPlayer
from familiar_connect.tts_player.protocol import TTSPlayer

__all__ = [
    "DiscordVoicePlayer",
    "LoggingTTSPlayer",
    "MockTTSPlayer",
    "TTSPlayer",
]
