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
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import discord
import pytest

from familiar_connect.bus.envelope import TurnScope
from familiar_connect.diagnostics.voice_budget import (
    PHASE_PLAYBACK_START,
    get_voice_budget_recorder,
    reset_voice_budget_recorder,
)
from familiar_connect.tts import TTSResult
from familiar_connect.tts_player.discord_player import DiscordVoicePlayer
from familiar_connect.voice.audio import (
    DISCORD_FRAME_SIZE,
    StreamingPCMSource,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
    async def test_empty_text_skips_synthesize(self) -> None:
        """Whitespace-only text never hits the TTS provider.

        Cartesia returns HTTP 400 for empty ``transcript``; defend in
        depth so any caller bug doesn't spam upstream.
        """
        tts = _StubTTS(audio=_mono_pcm())
        vc = _voice_client()
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("   \n\t", scope=scope)

        assert tts.calls == []
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


class TestConcurrentSpeak:
    """Cross-user replies share one voice client.

    Per-user scope (responder) lets two finals from different speakers
    each spawn their own ``speak``. The Discord ``VoiceClient`` is
    single-track; ``vc.play()`` while audio is already playing raises
    ``ClientException('Already playing audio.')``. The player must
    serialize.
    """

    @pytest.mark.asyncio
    async def test_concurrent_speaks_do_not_collide(self) -> None:
        tts = _StubTTS(audio=_mono_pcm(8))

        # mimic real pycord: play() raises while audio is already
        # playing; is_playing returns True for several polls so each
        # playback occupies the voice client across event-loop ticks
        # — that's the race window where the second speak() arrives.
        state = {"playing": False, "polls_remaining": 0}
        plays: list[object] = []

        def _play(source: object) -> None:
            if state["playing"]:
                msg = "Already playing audio."
                raise discord.ClientException(msg)
            state["playing"] = True
            state["polls_remaining"] = 3
            plays.append(source)

        def _is_playing() -> bool:
            if state["polls_remaining"] > 0:
                state["polls_remaining"] -= 1
                return True
            state["playing"] = False
            return False

        vc = MagicMock(name="voice_client")
        vc.is_connected.return_value = True
        vc.play.side_effect = _play
        vc.is_playing.side_effect = _is_playing

        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        s1 = TurnScope(turn_id="t1", session_id="voice:1:user:101", started_at=0.0)
        s2 = TurnScope(turn_id="t2", session_id="voice:1:user:202", started_at=0.0)

        await asyncio.gather(
            player.speak("alice reply", scope=s1),
            player.speak("bob reply", scope=s2),
        )

        assert len(plays) == 2
        assert tts.calls == ["alice reply", "bob reply"]

    @pytest.mark.asyncio
    async def test_cancel_then_immediate_speak_does_not_collide(self) -> None:
        """Barge-in should not race the next speaker's ``vc.play()``.

        Reproduces the prod ``ClientException('Already playing audio.')``
        from the trace: speaker A is mid-playback when scope_A is
        cancelled; ``vc.stop()`` flips the stop flag but pycord's audio
        thread takes a tick or two to actually release the player. If
        ``speak()`` returns immediately on stop and B's ``speak()``
        acquires the lock at that moment, B's ``vc.play()`` raises.
        The fix drains ``is_playing()`` after ``vc.stop()`` before
        releasing the lock.
        """
        tts = _StubTTS(audio=_mono_pcm(8))

        # mimic real pycord: stop() flips a sticky flag, is_playing
        # only returns False after several polls (audio-thread tick lag);
        # play() raises if is_playing is still True; absent any stop,
        # is_playing naturally drains after some polls so the test
        # eventually terminates.
        state = {
            "playing": False,
            "stop_lag_polls": 0,
            "stopping": False,
            "natural_polls": 0,
        }
        plays: list[object] = []

        def _play(source: object) -> None:
            if state["playing"]:
                msg = "Already playing audio."
                raise discord.ClientException(msg)
            state["playing"] = True
            state["stopping"] = False
            state["stop_lag_polls"] = 0
            state["natural_polls"] = 8
            plays.append(source)

        def _is_playing() -> bool:
            if state["stopping"]:
                state["stop_lag_polls"] -= 1
                if state["stop_lag_polls"] <= 0:
                    state["playing"] = False
                    state["stopping"] = False
            elif state["playing"]:
                state["natural_polls"] -= 1
                if state["natural_polls"] <= 0:
                    state["playing"] = False
            return bool(state["playing"])

        def _stop() -> None:
            if state["playing"] and not state["stopping"]:
                # 4 polls of post-stop lag — within the bounded
                # ``_await_stop_drain`` wait but enough that releasing
                # the lock immediately would race B's ``vc.play``.
                state["stopping"] = True
                state["stop_lag_polls"] = 4

        vc = MagicMock(name="voice_client")
        vc.is_connected.return_value = True
        vc.play.side_effect = _play
        vc.is_playing.side_effect = _is_playing
        vc.stop.side_effect = _stop

        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope_a = TurnScope(turn_id="ta", session_id="voice:1:user:101", started_at=0.0)
        scope_b = TurnScope(turn_id="tb", session_id="voice:1:user:202", started_at=0.0)

        async def cancel_a() -> None:
            await asyncio.sleep(0.03)
            scope_a.cancel()

        cancel_task = asyncio.create_task(cancel_a())
        # B's speak is launched concurrently; the lock serializes, but
        # the post-stop drain is what prevents the race.
        await asyncio.gather(
            player.speak("alice", scope=scope_a),
            player.speak("bob", scope=scope_b),
        )
        await cancel_task

        # Both plays must succeed — no ClientException raised.
        assert len(plays) == 2


class TestVoiceBudget:
    """``DiscordVoicePlayer`` stamps ``playback_start`` once per turn.

    See :mod:`familiar_connect.diagnostics.voice_budget`. Closes the
    funnel: enables ``voice.tts_to_playback`` + ``voice.total`` spans.
    """

    @pytest.mark.asyncio
    async def test_play_records_playback_start(self) -> None:
        reset_voice_budget_recorder()
        tts = _StubTTS(audio=_mono_pcm(8))
        vc = _voice_client(play_durations=2)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t-budget", session_id="voice:1", started_at=0.0)

        await player.speak("hello", scope=scope)

        rec = get_voice_budget_recorder()
        assert "t-budget" in rec._turns
        assert PHASE_PLAYBACK_START in rec._turns["t-budget"]


class _StreamingStubTTS:
    """TTS stub exposing both ``synthesize`` and ``synthesize_stream``.

    The streaming variant yields ``chunks`` one at a time, awaiting
    ``per_chunk_delay`` between each. ``record_stream_consumed`` flips
    once the iterator has fully drained — tests assert producer-side
    progress.
    """

    def __init__(
        self,
        chunks: list[bytes],
        *,
        per_chunk_delay: float = 0.0,
    ) -> None:
        self._chunks = chunks
        self._delay = per_chunk_delay
        self.calls: list[str] = []
        self.stream_consumed = False
        self.cancelled_after: int | None = None

    async def synthesize(self, text: str) -> TTSResult:
        self.calls.append(text)
        return TTSResult(audio=b"".join(self._chunks), timestamps=[])

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        yielded = 0
        try:
            for chunk in self._chunks:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield chunk
                yielded += 1
            self.stream_consumed = True
        except GeneratorExit:
            self.cancelled_after = yielded
            raise


def _streaming_voice_client(*, play_durations: int = 4) -> MagicMock:
    """MagicMock vc that simulates ``play(source)`` + finite ``is_playing``.

    ``play_durations`` controls how many ``is_playing()`` calls return
    True before flipping False (covers the ``vc.play`` → poll-loop window
    in :meth:`DiscordVoicePlayer._speak_streaming`).
    """
    vc = MagicMock(name="streaming_voice_client")
    vc.is_connected.return_value = True
    counter = {"n": 0}

    def _is_playing() -> bool:
        counter["n"] += 1
        return counter["n"] <= play_durations

    vc.is_playing.side_effect = _is_playing
    return vc


class TestSpeakStreaming:
    """Streaming path: bytes feed into ``StreamingPCMSource`` as they arrive."""

    @pytest.mark.asyncio
    async def test_uses_streaming_when_client_supports_it(self) -> None:
        chunks = [b"\x10" * DISCORD_FRAME_SIZE, b"\x20" * DISCORD_FRAME_SIZE]
        tts = _StreamingStubTTS(chunks)
        vc = _streaming_voice_client(play_durations=2)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("hello world", scope=scope)

        assert tts.calls == ["hello world"]
        vc.play.assert_called_once()
        played = vc.play.call_args[0][0]
        assert isinstance(played, StreamingPCMSource)

    @pytest.mark.asyncio
    async def test_chunks_fed_to_source_as_stereo(self) -> None:
        """Each mono s16le sample is duplicated L+R into the source."""
        # one s16le sample = 2 bytes, duplicated as L+R into 4 bytes stereo
        chunks = [b"\x01\x02", b"\x03\x04"]
        tts = _StreamingStubTTS(chunks)
        vc = _streaming_voice_client(play_durations=4)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("x", scope=scope)

        source = vc.play.call_args[0][0]
        assert tts.stream_consumed is True
        all_bytes = b""
        while True:
            frame = source.read()
            if not frame:
                break
            all_bytes += frame
        # sample 1 (b"\x01\x02") → b"\x01\x02\x01\x02" (L+R)
        # sample 2 (b"\x03\x04") → b"\x03\x04\x03\x04"
        assert all_bytes[:8] == b"\x01\x02\x01\x02\x03\x04\x03\x04"

    @pytest.mark.asyncio
    async def test_play_called_before_stream_drained(self) -> None:
        """``vc.play`` fires once the first chunk is buffered, not after EOS."""
        chunks = [b"\xaa" * 8, b"\xbb" * 8, b"\xcc" * 8]
        tts = _StreamingStubTTS(chunks, per_chunk_delay=0.01)
        vc = _streaming_voice_client(play_durations=10)

        # Capture the producer-side state at the moment ``vc.play`` is
        # called. Streaming-correctness requires play to fire while the
        # producer is still in flight (chunks 2+ haven't been yielded).
        play_seen_consumed: list[bool] = []

        def _capture(_source: object) -> None:
            play_seen_consumed.append(tts.stream_consumed)

        vc.play.side_effect = _capture

        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("x", scope=scope)
        assert play_seen_consumed == [False]

    @pytest.mark.asyncio
    async def test_records_playback_start_in_budget(self) -> None:
        reset_voice_budget_recorder()
        chunks = [b"\x10" * 4]
        tts = _StreamingStubTTS(chunks)
        vc = _streaming_voice_client(play_durations=1)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t-stream", session_id="voice:1", started_at=0.0)

        await player.speak("hello", scope=scope)

        rec = get_voice_budget_recorder()
        assert "t-stream" in rec._turns
        assert PHASE_PLAYBACK_START in rec._turns["t-stream"]

    @pytest.mark.asyncio
    async def test_cancellation_during_stream_calls_vc_stop(self) -> None:
        # many chunks with delay → drain task stays busy past cancel
        chunks = [b"\x33" * 4 for _ in range(20)]
        tts = _StreamingStubTTS(chunks, per_chunk_delay=0.01)

        vc = MagicMock(name="vc")
        vc.is_connected.return_value = True
        playing = {"v": True}
        vc.is_playing.side_effect = lambda: playing["v"]
        vc.stop.side_effect = lambda: playing.update(v=False)

        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        async def cancel_soon() -> None:
            await asyncio.sleep(0.03)
            scope.cancel()

        cancel_task = asyncio.create_task(cancel_soon())
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await player.speak("long", scope=scope)
        elapsed_ms = int((loop.time() - t0) * 1000)
        await cancel_task

        vc.stop.assert_called()
        assert elapsed_ms < 200, f"barge-in took {elapsed_ms}ms"

    @pytest.mark.asyncio
    async def test_empty_stream_logs_skip_no_play(self) -> None:
        tts = _StreamingStubTTS(chunks=[])
        vc = _streaming_voice_client(play_durations=1)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("nothing", scope=scope)
        vc.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_error_on_first_chunk_logs_no_play(self) -> None:
        class _BoomTTS:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
                self.calls.append(text)
                msg = "ws auth failed"
                raise RuntimeError(msg)
                yield b""  # unreachable; needed so this is an async generator

            async def synthesize(self, _text: str) -> TTSResult:
                return TTSResult(audio=b"", timestamps=[])

        tts = _BoomTTS()
        vc = _streaming_voice_client(play_durations=1)
        player = DiscordVoicePlayer(tts_client=tts, get_voice_client=lambda: vc)  # ty: ignore[invalid-argument-type]
        scope = TurnScope(turn_id="t", session_id="voice:1", started_at=0.0)

        await player.speak("hi", scope=scope)
        vc.play.assert_not_called()
        assert tts.calls == ["hi"]


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
