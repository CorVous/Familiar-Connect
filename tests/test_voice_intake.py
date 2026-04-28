"""Tests for voice intake lifecycle: ``_start_voice_intake`` / ``_stop``.

Covers the per-channel wiring that ``/subscribe-voice`` performs:
recording sink attached, transcriber started, audio pump + voice
source running. Mocks the voice client and transcriber surfaces.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_connect import bot as bot_module
from familiar_connect.bot import (
    BotHandle,
    VoiceRuntime,
    _on_recording_done,
    _prefetch_voice_member,
    _start_voice_intake,
    _stop_voice_intake,
    create_bot,
)
from familiar_connect.identity import Author
from familiar_connect.transcription import TranscriptionResult


def _make_handle() -> BotHandle:
    bot = MagicMock()
    return BotHandle(bot=bot, send_text=AsyncMock())


def _make_familiar(*, transcriber: object | None) -> MagicMock:
    fam = MagicMock()
    fam.id = "fam"
    fam.transcriber = transcriber
    fam.bus = MagicMock()
    return fam


def _make_template_transcriber() -> tuple[MagicMock, list[MagicMock]]:
    """Template transcriber whose ``clone()`` returns a fresh per-user mock.

    Returned list captures every clone produced — tests inspect it to
    verify per-user dispatch.
    """
    clones: list[MagicMock] = []

    def _make_clone() -> MagicMock:
        c = MagicMock()
        c.start = AsyncMock()
        c.send_audio = AsyncMock()
        c.stop = AsyncMock()
        clones.append(c)
        return c

    template = MagicMock()
    template.clone = MagicMock(side_effect=_make_clone)
    # template is never started directly — clones are. but keep these
    # as AsyncMocks so accidental calls don't blow up the tests.
    template.start = AsyncMock()
    template.send_audio = AsyncMock()
    template.stop = AsyncMock()
    return template, clones


async def _drain_loop(ticks: int = 10) -> None:
    """Yield enough loop turns for the pump to consume queued chunks."""
    for _ in range(ticks):
        await asyncio.sleep(0)


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
    async def test_attaches_sink_and_arms_pump(self) -> None:
        """Sink attached, pump + source live; clones lazy until first audio."""
        handle = _make_handle()
        template, clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)
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
            # template never started directly — clones own the WS
            template.start.assert_not_awaited()
            assert clones == []
            assert not rt.pump_task.done()
            assert not rt.source_task.done()
        finally:
            await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

    @pytest.mark.asyncio
    async def test_idempotent_for_same_channel(self) -> None:
        handle = _make_handle()
        template, _clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)
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
        finally:
            await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)


class TestStopVoiceIntake:
    @pytest.mark.asyncio
    async def test_cancels_tasks_and_stops_every_clone(self) -> None:
        handle = _make_handle()
        template, clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)
        vc = MagicMock()

        await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        rt = handle.voice_runtime[10]
        assert isinstance(rt, VoiceRuntime)
        # spawn one clone via audio so stop has something to tear down
        await rt.audio_queue.put((1, b"\x00"))
        await _drain_loop()

        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        assert 10 not in handle.voice_runtime
        vc.stop_recording.assert_called_once()
        assert len(clones) == 1
        clones[0].stop.assert_awaited_once()
        assert rt.pump_task.cancelled() or rt.pump_task.done()
        assert rt.source_task.cancelled() or rt.source_task.done()

    @pytest.mark.asyncio
    async def test_noop_when_channel_not_active(self) -> None:
        handle = _make_handle()
        familiar = _make_familiar(transcriber=MagicMock())
        # Should not raise; nothing registered for channel 99.
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=99)


class TestRecordingDoneCallback:
    """pycord's ``start_recording`` callback contract.

    Voice client calls ``asyncio.run_coroutine_threadsafe(callback(sink,
    *args), self.loop)`` (``discord/voice_client.py:915``) when
    recording ends. A plain ``def`` returns ``None`` and triggers
    ``TypeError: A coroutine object is required``.
    """

    def test_on_recording_done_is_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(_on_recording_done)


class TestPumpAudio:
    @pytest.mark.asyncio
    async def test_audio_queue_drains_into_transcriber(self) -> None:
        """Bytes pushed into the sink's queue reach a per-user transcriber clone."""
        handle = _make_handle()
        template, clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)
        vc = MagicMock()

        rt = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        assert isinstance(rt, VoiceRuntime)

        # Push two chunks for the same user onto the audio queue.
        await rt.audio_queue.put((1, b"\x00\x01\x02\x03"))
        await rt.audio_queue.put((1, b"\x04\x05"))
        await _drain_loop()
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        # Exactly one clone should have been produced for user 1.
        assert len(clones) == 1
        sent = [c.args[0] for c in clones[0].send_audio.await_args_list]
        assert b"\x00\x01\x02\x03" in sent
        assert b"\x04\x05" in sent
        # The template itself never accepts audio — only the clone does.
        template.send_audio.assert_not_awaited()


class TestPerUserDispatch:
    """Discord delivers per-SSRC audio; each user gets their own Deepgram WS.

    Avoids mixed-stream endpointing where one speaker's pause finalizes
    another's mid-sentence, and gives each transcript inherent
    attribution to the Discord user_id.
    """

    @pytest.mark.asyncio
    async def test_distinct_users_get_distinct_transcribers(self) -> None:
        """Two user_ids → two clones; each receives only their own audio."""
        handle = _make_handle()
        template, clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)
        vc = MagicMock()

        rt = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        assert isinstance(rt, VoiceRuntime)

        await rt.audio_queue.put((101, b"alice-1"))
        await rt.audio_queue.put((202, b"bob-1"))
        await rt.audio_queue.put((101, b"alice-2"))
        await rt.audio_queue.put((202, b"bob-2"))
        await _drain_loop()
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        assert len(clones) == 2
        all_sent = [
            [c.args[0] for c in clone.send_audio.await_args_list] for clone in clones
        ]
        # find which clone got alice's audio vs bob's
        alice_chunks = next(s for s in all_sent if b"alice-1" in s)
        bob_chunks = next(s for s in all_sent if b"bob-1" in s)
        assert alice_chunks == [b"alice-1", b"alice-2"]
        assert bob_chunks == [b"bob-1", b"bob-2"]

    @pytest.mark.asyncio
    async def test_same_user_reuses_transcriber(self) -> None:
        """Repeated chunks from one user_id stay on one clone."""
        handle = _make_handle()
        template, clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)
        vc = MagicMock()

        rt = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        assert isinstance(rt, VoiceRuntime)

        for i in range(5):
            await rt.audio_queue.put((42, bytes([i])))
        await _drain_loop()
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        assert len(clones) == 1
        assert clones[0].send_audio.await_count == 5

    @pytest.mark.asyncio
    async def test_stop_stops_every_clone(self) -> None:
        """Tearing down the runtime stops every per-user transcriber."""
        handle = _make_handle()
        template, clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)
        vc = MagicMock()

        await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        await rt_audio_put(handle, 10, [(1, b"a"), (2, b"b"), (3, b"c")])
        await _drain_loop()
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        assert len(clones) == 3
        for clone in clones:
            clone.stop.assert_awaited()

    @pytest.mark.asyncio
    async def test_results_carry_user_id_through_to_shared_queue(self) -> None:
        """Fan-in tags every result with the originating Discord user_id."""
        handle = _make_handle()

        # Custom template whose clone captures the queue handed to start()
        # so the test can push synthetic results into it.
        per_clone_queues: list[asyncio.Queue[TranscriptionResult]] = []
        clones: list[MagicMock] = []

        def _make_clone() -> MagicMock:
            c = MagicMock()

            async def _start(q: asyncio.Queue[TranscriptionResult]) -> None:  # noqa: RUF029
                per_clone_queues.append(q)

            c.start = AsyncMock(side_effect=_start)
            c.send_audio = AsyncMock()
            c.stop = AsyncMock()
            clones.append(c)
            return c

        template = MagicMock()
        template.clone = MagicMock(side_effect=_make_clone)
        template.start = AsyncMock()
        template.stop = AsyncMock()

        familiar = _make_familiar(transcriber=template)
        vc = MagicMock()

        rt = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        assert isinstance(rt, VoiceRuntime)

        # Cancel the VoiceSource so it doesn't drain result_queue out
        # from under the test — we're inspecting the fan-in's output
        # directly here.
        rt.source_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await rt.source_task

        # Trigger lazy clone creation for two users
        await rt.audio_queue.put((101, b"a"))
        await rt.audio_queue.put((202, b"b"))
        await _drain_loop()
        assert len(per_clone_queues) == 2

        # Push a fake transcription result onto each per-user queue;
        # the fan-in should tag with user_id and forward to result_queue.
        await per_clone_queues[0].put(
            TranscriptionResult(
                text="alice talking",
                is_final=True,
                start=0.0,
                end=1.0,
                confidence=0.9,
            )
        )
        await per_clone_queues[1].put(
            TranscriptionResult(
                text="bob talking",
                is_final=True,
                start=0.0,
                end=1.0,
                confidence=0.9,
            )
        )
        await _drain_loop()

        forwarded: list[TranscriptionResult] = []
        while not rt.result_queue.empty():
            forwarded.append(rt.result_queue.get_nowait())

        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        by_text = {r.text: r for r in forwarded}
        assert by_text["alice talking"].user_id == 101
        assert by_text["bob talking"].user_id == 202


async def rt_audio_put(
    handle: BotHandle, channel_id: int, items: list[tuple[int, bytes]]
) -> None:
    rt: Any = handle.voice_runtime[channel_id]
    for item in items:
        await rt.audio_queue.put(item)


class TestVoiceMemberCache:
    """Voice-only members aren't in the guild member cache.

    Without the privileged ``members`` intent, ``guild.get_member()``
    only knows users it has seen via other events (text messages,
    voice state changes). A voice-only side cache populated from
    voice state events plus a background ``guild.fetch_member()`` on
    first audio from a new user_id keeps voice turn attribution
    working without requiring privileged intents.
    """

    @pytest.mark.asyncio
    async def test_resolve_member_consults_voice_cache_first(self) -> None:
        """Cache hit short-circuits ``guild.get_member`` lookup."""
        familiar = MagicMock()
        familiar.id = "fam"
        familiar.bus = MagicMock()
        familiar.subscriptions = MagicMock()
        handle = create_bot(familiar)

        cached = Author(
            platform="discord",
            user_id="42",
            username="vox",
            display_name="VoxOnly",
        )
        handle.voice_members[42] = cached

        assert handle.resolve_member is not None
        resolved = handle.resolve_member(10, 42)
        assert resolved is cached

    @pytest.mark.asyncio
    async def test_pump_schedules_member_prefetch_on_new_user(self) -> None:
        """First audio from a user_id should fire a background member fetch."""
        handle = _make_handle()
        member = MagicMock()
        member.id = 999
        member.name = "voxer"
        member.display_name = "VoxOnly"

        guild = MagicMock()
        guild.get_member = MagicMock(return_value=None)
        guild.fetch_member = AsyncMock(return_value=member)

        channel = MagicMock()
        channel.guild = guild
        handle.bot.get_channel = MagicMock(return_value=channel)  # ty: ignore[invalid-assignment]

        await _prefetch_voice_member(handle=handle, channel_id=10, user_id=999)
        assert 999 in handle.voice_members
        assert handle.voice_members[999].display_name == "VoxOnly"
        guild.fetch_member.assert_awaited_once_with(999)

    @pytest.mark.asyncio
    async def test_prefetch_skips_when_already_cached(self) -> None:
        """Repeated prefetch for a known user_id must not re-hit Discord."""
        handle = _make_handle()
        handle.voice_members[7] = Author(
            platform="discord",
            user_id="7",
            username="known",
            display_name="Known",
        )

        guild = MagicMock()
        guild.get_member = MagicMock(return_value=None)
        guild.fetch_member = AsyncMock()
        channel = MagicMock()
        channel.guild = guild
        handle.bot.get_channel = MagicMock(return_value=channel)  # ty: ignore[invalid-assignment]

        await _prefetch_voice_member(handle=handle, channel_id=10, user_id=7)
        guild.fetch_member.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prefetch_uses_get_member_when_cached_in_guild(self) -> None:
        """If guild already has the member, no fetch is needed."""
        handle = _make_handle()
        member = MagicMock()
        member.id = 13
        member.name = "g"
        member.display_name = "GuildKnown"

        guild = MagicMock()
        guild.get_member = MagicMock(return_value=member)
        guild.fetch_member = AsyncMock()
        channel = MagicMock()
        channel.guild = guild
        handle.bot.get_channel = MagicMock(return_value=channel)  # ty: ignore[invalid-assignment]

        await _prefetch_voice_member(handle=handle, channel_id=10, user_id=13)
        assert handle.voice_members[13].display_name == "GuildKnown"
        guild.fetch_member.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_intake_pump_triggers_prefetch_per_new_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the pump sees a new user_id, prefetch is scheduled."""
        handle = _make_handle()
        template, _clones = _make_template_transcriber()
        familiar = _make_familiar(transcriber=template)

        prefetched: list[int] = []

        async def fake_prefetch(  # noqa: RUF029 — matches real signature
            *, handle: BotHandle, channel_id: int, user_id: int
        ) -> None:
            del handle, channel_id
            prefetched.append(user_id)

        monkeypatch.setattr(bot_module, "_prefetch_voice_member", fake_prefetch)

        vc = MagicMock()
        rt = await _start_voice_intake(
            handle=handle, familiar=familiar, voice_client=vc, channel_id=10
        )
        assert isinstance(rt, VoiceRuntime)
        for _ in range(3):
            await rt.audio_queue.put((101, b"x"))
        await rt.audio_queue.put((202, b"y"))
        await _drain_loop(20)
        await _stop_voice_intake(handle=handle, familiar=familiar, channel_id=10)

        # exactly one prefetch per distinct user_id
        assert sorted(prefetched) == [101, 202]
