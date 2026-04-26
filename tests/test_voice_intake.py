"""Tests for voice intake lifecycle: ``_start_voice_intake`` / ``_stop``.

Covers the per-channel wiring that ``/subscribe-voice`` performs:
recording sink attached, transcriber started, audio pump + voice
source running. Mocks the voice client and transcriber surfaces.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_connect.bot import (
    BotHandle,
    VoiceRuntime,
    _start_voice_intake,
    _stop_voice_intake,
)


def _make_handle() -> BotHandle:
    bot = MagicMock()
    return BotHandle(bot=bot, send_text=AsyncMock())


def _make_familiar(*, transcriber: object | None) -> MagicMock:
    fam = MagicMock()
    fam.id = "fam"
    fam.transcriber = transcriber
    fam.bus = MagicMock()
    return fam


class TestStartVoiceIntake:
    @pytest.mark.asyncio
    async def test_returns_none_when_transcriber_unavailable(self) -> None:
        """No transcriber → bot still joined for playback only; no intake."""
        handle = _make_handle()
        familiar = _make_familiar(transcriber=None)
        vc = MagicMock()

        rt = await _start_voice_intake(
            handle=handle,
            familiar=familiar,
            voice_client=vc,
            channel_id=10,
        )
        assert rt is None
        assert handle.voice_runtime == {}
        vc.start_recording.assert_not_called()

    @pytest.mark.asyncio
    async def test_attaches_sink_and_starts_transcriber(self) -> None:
        handle = _make_handle()
        transcriber = MagicMock()
        transcriber.start = AsyncMock()
        transcriber.send_audio = AsyncMock()
        familiar = _make_familiar(transcriber=transcriber)
        vc = MagicMock()

        rt = await _start_voice_intake(
            handle=handle,
            familiar=familiar,
            voice_client=vc,
            channel_id=10,
        )
        try:
            assert isinstance(rt, VoiceRuntime)
            assert handle.voice_runtime[10] is rt
            vc.start_recording.assert_called_once()
            transcriber.start.assert_awaited_once()
            assert not rt.pump_task.done()
            assert not rt.source_task.done()
        finally:
            await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

    @pytest.mark.asyncio
    async def test_idempotent_for_same_channel(self) -> None:
        handle = _make_handle()
        transcriber = MagicMock()
        transcriber.start = AsyncMock()
        transcriber.send_audio = AsyncMock()
        familiar = _make_familiar(transcriber=transcriber)
        vc = MagicMock()

        rt1 = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        rt2 = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        try:
            assert rt1 is rt2
            assert vc.start_recording.call_count == 1
            transcriber.start.assert_awaited_once()
        finally:
            await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)


class TestStopVoiceIntake:
    @pytest.mark.asyncio
    async def test_cancels_tasks_and_stops_transcriber(self) -> None:
        handle = _make_handle()
        transcriber = MagicMock()
        transcriber.start = AsyncMock()
        transcriber.send_audio = AsyncMock()
        transcriber.stop = AsyncMock()
        familiar = _make_familiar(transcriber=transcriber)
        vc = MagicMock()

        await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        rt = handle.voice_runtime[10]
        assert isinstance(rt, VoiceRuntime)

        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        assert 10 not in handle.voice_runtime
        vc.stop_recording.assert_called_once()
        transcriber.stop.assert_awaited_once()
        assert rt.pump_task.cancelled() or rt.pump_task.done()
        assert rt.source_task.cancelled() or rt.source_task.done()

    @pytest.mark.asyncio
    async def test_noop_when_channel_not_active(self) -> None:
        handle = _make_handle()
        familiar = _make_familiar(transcriber=MagicMock())
        # Should not raise; nothing registered for channel 99.
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=99)


class TestPumpAudio:
    @pytest.mark.asyncio
    async def test_audio_queue_drains_into_transcriber(self) -> None:
        """Bytes pushed into the sink's queue reach transcriber.send_audio."""
        handle = _make_handle()
        transcriber = MagicMock()
        transcriber.start = AsyncMock()
        transcriber.send_audio = AsyncMock()
        familiar = _make_familiar(transcriber=transcriber)
        vc = MagicMock()

        rt = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        assert isinstance(rt, VoiceRuntime)

        # Push two chunks onto the audio queue the sink shares with the pump.
        await rt.audio_queue.put((1, b"\x00\x01\x02\x03"))
        await rt.audio_queue.put((1, b"\x04\x05"))
        # Yield enough loop ticks for the pump to consume.
        for _ in range(5):
            await asyncio.sleep(0)
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        sent = [c.args[0] for c in transcriber.send_audio.await_args_list]
        assert b"\x00\x01\x02\x03" in sent
        assert b"\x04\x05" in sent
