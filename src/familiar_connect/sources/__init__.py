"""Event sources — Discord, Twitch, voice, etc.

See plan § Design.2 *StreamSource and Processor protocols*.
"""

from __future__ import annotations

from familiar_connect.sources.discord_text import DiscordTextSource
from familiar_connect.sources.twitch import TwitchSource
from familiar_connect.sources.voice import VoiceSource

__all__ = ["DiscordTextSource", "TwitchSource", "VoiceSource"]
