"""Tests for :class:`DiscordVoicePlayer`.

Verifies:

* synthesize-then-play flow,
* mono PCM is converted to stereo before pycord sees it,
* scope cancellation mid-playback calls ``vc.stop()`` quickly,
* missing / disconnected voice client is logged and skipped (no crash),
* :meth:`stop` calls ``vc.stop`` only when playing.
"""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import MagicMock

import pytest

from familiar_connect.bus.envelope import TurnScope
from familiar_connect.tts import TTSResult
from familiar_connect.tts_player.discord_player import DiscordVoicePlayer


class _StubTTS:
    """Minimal TTS client returning fixed PCM."""

    def __init__(self, audio: bytes) -> None:
        self._audio = audio
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> TTSResult:
        self.calls.append(text)
        return TTSResult(audio=self._audio, timestamps=[])


def _mono_pcm(samples: int = 4) -> bytes:
    """Generate ``samples`` int16 samples of test PCM."""
    return struct.pack(f"<{samples}h", *range(samples))


def _voice_client(
    *,
    connected: bool = True,
    play_durations: int = 1,
) -> MagicMock:
    """Build a MagicMock voice client.

    ``play_durations`` controls how many ``is_playing()`` calls return
    True before flipping False — simulates audio playing for that many
    polls before completing naturally.
    """
    vc = MagicMock(name="voice_client")
    vc.is_connected.return_value = connected
    counter = {"n": 0}

    def _is_playing() -> bool:
        counter["n"] += 1
        return counter["n"] <= play_durations

    vc.is_playing.side_effect = _is_playing
    return vc


class TestSpeakHappyPath:
    @pytest.mark.asyncio
    async def test_synthesizes_and_plays(self) -> None:
        tts = _StubTTS(audio=_mono_pcm(8))
        vc = _voice_client(play_durations=2)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("hello world", scope=scope)

        assert tts.calls == ["hello world"]
        vc.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_pcm_converted_to_stereo(self) -> None:
        """Mono PCM gets duplicated L+R before reaching ``vc.play()``."""
        mono = _mono_pcm(samples=4)  # 8 bytes
        tts = _StubTTS(audio=mono)
        vc = _voice_client(play_durations=1)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("x", scope=scope)

        source = vc.play.call_args[0][0]
        # Read whatever pycord would read from the source's underlying buffer.
        # The DiscordVoicePlayer wraps a BytesIO of stereo PCM.
        stereo_bytes = source.stream.getvalue()
        assert len(stereo_bytes) == len(mono) * 2


class TestSpeakSkipPaths:
    @pytest.mark.asyncio
    async def test_no_voice_client_skips(self) -> None:
        tts = _StubTTS(audio=_mono_pcm())
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: None)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("hello", scope=scope)

        # synthesized but had nowhere to play
        assert tts.calls == ["hello"]

    @pytest.mark.asyncio
    async def test_disconnected_voice_client_skips(self) -> None:
        tts = _StubTTS(audio=_mono_pcm())
        vc = _voice_client(connected=False)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("hello", scope=scope)

        vc.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_cancelled_scope_short_circuits(self) -> None:
        tts = _StubTTS(audio=_mono_pcm())
        vc = _voice_client()
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)
        scope.cancel()

        await player.speak("hi", scope=scope)
        vc.play.assert_not_called()


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_cancel_during_playback_calls_vc_stop(self) -> None:
        tts = _StubTTS(audio=_mono_pcm(8))
        # is_playing always True until vc.stop() is called explicitly
        vc = MagicMock(name="voice_client")
        vc.is_connected.return_value = True
        playing = {"v": True}
        vc.is_playing.side_effect = lambda: playing["v"]

        def _stop() -> None:
            playing["v"] = False

        vc.stop.side_effect = _stop

        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        async def cancel_soon() -> None:
            await asyncio.sleep(0.03)
            scope.cancel()

        cancel_task = asyncio.create_task(cancel_soon())
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await player.speak("a long utterance", scope=scope)
        elapsed_ms = int((loop.time() - t0) * 1000)
        await cancel_task

        vc.stop.assert_called()
        assert elapsed_ms < 200, f"barge-in took {elapsed_ms}ms"


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_calls_vc_stop_when_playing(self) -> None:
        tts = _StubTTS(audio=_mono_pcm())
        vc = MagicMock()
        vc.is_playing.return_value = True
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        await player.stop()
        vc.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_playing(self) -> None:
        tts = _StubTTS(audio=_mono_pcm())
        vc = MagicMock()
        vc.is_playing.return_value = False
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        await player.stop()
        vc.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_no_voice_client(self) -> None:
        tts = _StubTTS(audio=_mono_pcm())
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: None)
        # Should not raise.
        await player.stop()
