"""Tests for the RecordingSink that bridges threaded audio capture to asyncio."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import MagicMock

from discord.sinks import Sink

from familiar_connect.voice.recording_sink import RecordingSink


class TestRecordingSink:
    def test_is_subclass_of_sink(self) -> None:
        """RecordingSink is a proper subclass of discord.sinks.Sink."""
        assert issubclass(RecordingSink, Sink)

    def test_write_puts_user_id_and_mono_tuple_on_queue(self) -> None:
        """write() puts a (user_id, mono_bytes) tuple on the queue."""
        loop = MagicMock()
        queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        sink = RecordingSink(loop=loop, audio_queue=queue)

        # Stereo frame: L=100, R=100 -> mono=100
        stereo = struct.pack("<hh", 100, 100)
        sink.write(stereo, user=12345)

        # Verify call_soon_threadsafe was used for thread-safe bridging.
        loop.call_soon_threadsafe.assert_called_once()
        args = loop.call_soon_threadsafe.call_args
        assert args[0][0] == queue.put_nowait
        expected_mono = struct.pack("<h", 100)
        assert args[0][1] == (12345, expected_mono)

    def test_write_uses_call_soon_threadsafe(self) -> None:
        """write() bridges to asyncio via call_soon_threadsafe."""
        loop = MagicMock()
        queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        sink = RecordingSink(loop=loop, audio_queue=queue)

        stereo = struct.pack("<hh", 0, 0)
        sink.write(stereo, user=99)

        loop.call_soon_threadsafe.assert_called_once()

    def test_write_preserves_different_user_ids(self) -> None:
        """Different users get different user_id tags in the tuple."""
        loop = MagicMock()
        queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        sink = RecordingSink(loop=loop, audio_queue=queue)

        stereo = struct.pack("<hh", 50, 50)
        sink.write(stereo, user=111)
        sink.write(stereo, user=222)

        assert loop.call_soon_threadsafe.call_count == 2
        first_call = loop.call_soon_threadsafe.call_args_list[0]
        second_call = loop.call_soon_threadsafe.call_args_list[1]
        assert first_call[0][1][0] == 111
        assert second_call[0][1][0] == 222

    def test_cleanup_sets_finished(self) -> None:
        """cleanup() sets the finished flag."""
        loop = MagicMock()
        queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        sink = RecordingSink(loop=loop, audio_queue=queue)
        sink.cleanup()
        assert sink.finished is True
