"""Discord-voice :class:`TTSPlayer` - synthesize and push PCM through pycord.

Wraps a TTS client (Cartesia / Azure / Gemini) and feeds Discord-format
stereo s16le @ 48 kHz PCM through ``voice_client.play(...)``.

Two synthesis paths:

* **Streaming** - when the client exposes ``synthesize_stream(text)``,
  bytes go into a :class:`StreamingPCMSource` as they arrive so
  ``vc.play`` starts within ~one Cartesia TTFB instead of waiting for
  the full utterance. Cuts ``voice.tts_to_playback`` from full-sentence
  synthesis time (1.5-3 s for a long sentence) to ~150 ms.
* **Buffered** - fallback for clients without a streaming method
  (Azure, Gemini). Identical to the prior behaviour.

Polls ``vc.is_playing()`` so :meth:`TurnScope.is_cancelled` cuts
playback within ~20 ms when a new turn arrives.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from typing import TYPE_CHECKING, Protocol

import discord

from familiar_connect import log_style as ls
from familiar_connect.diagnostics.voice_budget import (
    PHASE_PLAYBACK_START,
    get_voice_budget_recorder,
)
from familiar_connect.voice.audio import StreamingPCMSource, mono_to_stereo

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

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
        if hasattr(self._tts, "synthesize_stream"):
            await self._speak_streaming(text, scope=scope)
        else:
            await self._speak_buffered(text, scope=scope)

    # ------------------------------------------------------------------
    # streaming path — feed bytes into ``StreamingPCMSource`` as they arrive
    # ------------------------------------------------------------------

    async def _speak_streaming(self, text: str, *, scope: TurnScope) -> None:
        source = StreamingPCMSource()
        feed_task: asyncio.Task[int] | None = None
        async with self._play_lock:
            if scope.is_cancelled():
                return
            vc = self._get_voice_client()
            if vc is None or not vc.is_connected():
                _logger.warning(
                    f"{ls.tag('Player', ls.Y)} "
                    f"{ls.kv('skip', 'no_voice_client', vc=ls.LY)}"
                )
                return

            stream = self._tts.synthesize_stream(text)  # ty: ignore[unresolved-attribute]
            stream_iter = aiter(stream)
            try:
                first_chunk = await anext(stream_iter)
            except StopAsyncIteration:
                _logger.warning(
                    f"{ls.tag('Player', ls.Y)} "
                    f"{ls.kv('skip', 'empty_stream', vc=ls.LY)} "
                    f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                )
                return
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    f"{ls.tag('Player', ls.R)} "
                    f"{ls.kv('synthesize_error', repr(exc), vc=ls.R)}"
                )
                return
            if scope.is_cancelled():
                aclose = getattr(stream_iter, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()
                return

            source.feed(mono_to_stereo(first_chunk))
            feed_task = asyncio.create_task(
                self._drain_stream(stream_iter, source, scope),
                name=f"tts-stream-{scope.turn_id}",
            )

            _logger.info(
                f"{ls.tag('🔊 Say', ls.G)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
                f"{ls.kv('mode', 'stream', vc=ls.LM)}"
            )
            try:
                vc.play(source)
            except discord.ClientException as exc:
                # belt-and-braces — should not happen with the lock held
                _logger.warning(
                    f"{ls.tag('Player', ls.R)} "
                    f"{ls.kv('play_error', repr(exc), vc=ls.R)}"
                )
                source.close_input()
                if not feed_task.done():
                    feed_task.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await feed_task
                return
            get_voice_budget_recorder().record(
                turn_id=scope.turn_id, phase=PHASE_PLAYBACK_START
            )

            try:
                while vc.is_playing():
                    if scope.is_cancelled():
                        _logger.info(
                            f"{ls.tag('🔊 Cut', ls.Y)} "
                            f"{ls.kv('turn', scope.turn_id, vc=ls.LC)}"
                        )
                        vc.stop()
                        return
                    await asyncio.sleep(_POLL_S)
            finally:
                # ensure the feeder unwinds — under cancellation pycord's
                # cleanup already closed the source, but on natural drain
                # the producer may still be appending tail bytes.
                source.close_input()
                if feed_task is not None and not feed_task.done():
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await feed_task

    @staticmethod
    async def _drain_stream(
        stream_iter: AsyncIterator[bytes],
        source: StreamingPCMSource,
        scope: TurnScope,
    ) -> int:
        """Feed remaining stream chunks into ``source`` until exhausted."""
        total = 0
        try:
            async for chunk in stream_iter:
                if scope.is_cancelled():
                    break
                source.feed(mono_to_stereo(chunk))
                total += len(chunk)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                f"{ls.tag('Player', ls.R)} {ls.kv('stream_error', repr(exc), vc=ls.R)}"
            )
        finally:
            source.close_input()
        return total

    # ------------------------------------------------------------------
    # buffered path — Azure / Gemini today (no streaming surface)
    # ------------------------------------------------------------------

    async def _speak_buffered(self, text: str, *, scope: TurnScope) -> None:
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
            if scope.is_cancelled():
                return
            _logger.info(
                f"{ls.tag('🔊 Say', ls.G)} "
                f"{ls.kv('turn', scope.turn_id, vc=ls.LC)} "
                f"{ls.kv('bytes', str(len(stereo)), vc=ls.LW)}"
            )
            vc.play(source)
            get_voice_budget_recorder().record(
                turn_id=scope.turn_id, phase=PHASE_PLAYBACK_START
            )
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
