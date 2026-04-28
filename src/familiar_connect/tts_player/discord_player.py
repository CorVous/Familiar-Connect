"""Discord-voice :class:`TTSPlayer` — synthesize and push PCM through pycord.

Wraps a TTS client (Cartesia / Azure / Gemini), converts the returned
mono 48 kHz PCM to stereo (Discord's required format), and feeds it
through ``voice_client.play(...)`` via :class:`discord.PCMAudio`.

Polls ``vc.is_playing()`` so :meth:`TurnScope.is_cancelled` cuts
playback within ~20 ms when a new turn arrives.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING, Protocol

import discord

from familiar_connect import log_style as ls
from familiar_connect.voice.audio import mono_to_stereo

if TYPE_CHECKING:
    from collections.abc import Callable

    from familiar_connect.bus.envelope import TurnScope
    from familiar_connect.tts import TTSResult

_logger = logging.getLogger("familiar_connect.tts_player.discord")

_POLL_S = 0.02


class _TTSClient(Protocol):
    """Minimal surface ``DiscordVoicePlayer`` needs from a TTS client."""

    async def synthesize(self, text: str) -> TTSResult: ...


class _VoiceClientLike(Protocol):
    """Minimal surface used from pycord's ``VoiceClient``.

    Kept narrow so tests can pass a :class:`unittest.mock.MagicMock`
    without inheriting the full ``discord.VoiceClient`` API.
    """

    def is_connected(self) -> bool: ...
    def is_playing(self) -> bool: ...
    def play(self, source: discord.AudioSource) -> None: ...
    def stop(self) -> None: ...


class DiscordVoicePlayer:
    """Synthesize and play through a live Discord voice client."""

    def __init__(
        self,
        *,
        tts_client: _TTSClient,
        get_voice_client: Callable[[], _VoiceClientLike | None],
    ) -> None:
        self._tts = tts_client
        self._get_voice_client = get_voice_client
        # serialize playback. per-user scopes (voice_responder) let two
        # speakers' replies coexist, but the ``VoiceClient`` is single-
        # track — concurrent ``vc.play`` raises ``ClientException(
        # 'Already playing audio.')``. this lock makes the second
        # ``speak`` await the first's playback completion.
        self._play_lock = asyncio.Lock()

    async def speak(self, text: str, *, scope: TurnScope) -> None:
        if scope.is_cancelled():
            return
        # defense-in-depth: Cartesia 400s on empty/whitespace transcript.
        if not text.strip():
            _logger.warning(
                f"{ls.tag('Player', ls.Y)} "
                f"{ls.kv('skip', 'empty_text', vc=ls.LY)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
            )
            return
        try:
            result = await self._tts.synthesize(text)
        except Exception as exc:  # noqa: BLE001 — TTS errors must not crash loop
            _logger.warning(
                f"{ls.tag('Player', ls.R)} "
                f"{ls.kv('synthesize_error', repr(exc), vc=ls.R)}"
            )
            return
        if scope.is_cancelled():
            return

        vc = self._get_voice_client()
        if vc is None or not vc.is_connected():
            _logger.warning(
                f"{ls.tag('Player', ls.Y)} {ls.kv('skip', 'no_voice_client', vc=ls.LY)}"
            )
            return

        stereo = mono_to_stereo(result.audio)
        source = discord.PCMAudio(io.BytesIO(stereo))

        async with self._play_lock:
            # scope may have been cancelled while we waited for the
            # voice client to free up.
            if scope.is_cancelled():
                return
            _logger.info(
                f"{ls.tag('🔊 Say', ls.G)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
                f"{ls.kv('bytes', str(len(stereo)), vc=ls.LW)}"
            )
            vc.play(source)

            # Poll for completion or cancellation. ``vc.is_playing`` flips
            # to False once py-cord has drained the source.
            while vc.is_playing():
                if scope.is_cancelled():
                    _logger.info(
                        f"{ls.tag('🔊 Cut', ls.Y)} "
                        f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                    )
                    vc.stop()
                    return
                await asyncio.sleep(_POLL_S)

    async def stop(self) -> None:
        vc = self._get_voice_client()
        if vc is None:
            return
        try:
            playing = vc.is_playing()
        except Exception:  # noqa: BLE001
            return
        if playing:
            vc.stop()
