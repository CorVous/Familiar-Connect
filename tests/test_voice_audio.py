"""Tests for voice audio conversion utilities."""

from __future__ import annotations

import struct
import threading
import time

import pytest

from familiar_connect.voice.audio import (
    DISCORD_FRAME_SIZE,
    StreamingPCMSource,
    mono_to_stereo,
    stereo_to_mono,
)


class TestMonoToStereo:
    def test_doubles_length(self) -> None:
        """Output is exactly twice the length of mono input."""
        mono = bytes(8)  # 4 samples of 16-bit audio
        stereo = mono_to_stereo(mono)
        assert len(stereo) == 16

    def test_each_stereo_pair_matches_source_sample(self) -> None:
        """Each stereo frame duplicates the mono sample into left and right."""
        # Two 16-bit samples: 0x0102 and 0x0304
        mono = struct.pack("<hh", 0x0102, 0x0304)
        stereo = mono_to_stereo(mono)
        # Unpack as four 16-bit samples: L0, R0, L1, R1
        samples = struct.unpack("<hhhh", stereo)
        assert samples[0] == samples[1] == 0x0102  # first sample duplicated
        assert samples[2] == samples[3] == 0x0304  # second sample duplicated

    def test_empty_input_returns_empty(self) -> None:
        """Empty mono input yields empty stereo output."""
        assert mono_to_stereo(b"") == b""

    def test_raises_on_odd_length(self) -> None:
        """Odd-length input is not valid 16-bit PCM — raises ValueError."""
        with pytest.raises(ValueError, match=r"even"):
            mono_to_stereo(b"\x00\x01\x02")

    def test_single_sample(self) -> None:
        """A single mono sample becomes one stereo frame (4 bytes)."""
        mono = struct.pack("<h", -1000)
        stereo = mono_to_stereo(mono)
        assert len(stereo) == 4
        left, right = struct.unpack("<hh", stereo)
        assert left == right == -1000

    def test_preserves_sample_values(self) -> None:
        """All sample values are preserved correctly after conversion."""
        samples = [100, -200, 32767, -32768, 0]
        mono = struct.pack(f"<{len(samples)}h", *samples)
        stereo = mono_to_stereo(mono)
        stereo_samples = struct.unpack(f"<{len(samples) * 2}h", stereo)
        for i, original in enumerate(samples):
            assert stereo_samples[i * 2] == original  # left
            assert stereo_samples[i * 2 + 1] == original  # right


class TestStereoToMono:
    def test_halves_length(self) -> None:
        """Stereo input of 16 bytes produces mono output of 8 bytes."""
        stereo = bytes(16)  # 4 stereo frames (L/R pairs)
        mono = stereo_to_mono(stereo)
        assert len(mono) == 8

    def test_averages_left_and_right(self) -> None:
        """Left=100, Right=200 averages to mono=150."""
        stereo = struct.pack("<hh", 100, 200)
        mono = stereo_to_mono(stereo)
        (sample,) = struct.unpack("<h", mono)
        assert sample == 150

    def test_empty_returns_empty(self) -> None:
        """Empty stereo input yields empty mono output."""
        assert stereo_to_mono(b"") == b""

    def test_raises_on_invalid_length(self) -> None:
        """Length not divisible by 4 is not valid stereo — raises ValueError."""
        with pytest.raises(ValueError, match=r"divisible by 4"):
            stereo_to_mono(b"\x00\x01\x02")

    def test_single_stereo_frame(self) -> None:
        """One L/R pair produces one mono sample."""
        stereo = struct.pack("<hh", -1000, -1000)
        mono = stereo_to_mono(stereo)
        assert len(mono) == 2
        (sample,) = struct.unpack("<h", mono)
        assert sample == -1000

    def test_roundtrip(self) -> None:
        """stereo_to_mono(mono_to_stereo(data)) recovers the original data."""
        original = struct.pack("<5h", 100, -200, 32767, -32768, 0)
        roundtripped = stereo_to_mono(mono_to_stereo(original))
        assert roundtripped == original


class TestDiscordFrameSize:
    def test_frame_size_is_3840(self) -> None:
        """Discord expects 3840-byte frames (48kHz, 16-bit, stereo, 20ms)."""
        assert DISCORD_FRAME_SIZE == 3840


class TestStreamingPCMSource:
    """Thread-safe buffer that feeds 20 ms PCM frames as they arrive."""

    def test_read_returns_full_frame_when_buffered(self) -> None:
        src = StreamingPCMSource()
        src.feed(b"\xab" * (DISCORD_FRAME_SIZE * 2))
        first = src.read()
        second = src.read()
        assert len(first) == DISCORD_FRAME_SIZE
        assert len(second) == DISCORD_FRAME_SIZE
        assert first == b"\xab" * DISCORD_FRAME_SIZE

    def test_read_returns_empty_when_closed_and_drained(self) -> None:
        src = StreamingPCMSource()
        src.close_input()
        assert src.read() == b""

    def test_partial_frame_zero_padded_on_close(self) -> None:
        src = StreamingPCMSource()
        partial = b"\x01\x02\x03\x04"
        src.feed(partial)
        src.close_input()
        out = src.read()
        assert len(out) == DISCORD_FRAME_SIZE
        assert out[: len(partial)] == partial
        assert out[len(partial) :] == b"\x00" * (DISCORD_FRAME_SIZE - len(partial))
        # next read drains
        assert src.read() == b""

    def test_read_blocks_until_data_arrives(self) -> None:
        """Reader thread waits on the condition until the producer feeds."""
        src = StreamingPCMSource()
        results: list[bytes] = []

        def reader() -> None:
            results.append(src.read())

        t = threading.Thread(target=reader)
        t.start()
        # give the reader time to enter cond.wait
        time.sleep(0.05)
        assert t.is_alive(), "reader should be blocked"
        src.feed(b"\x42" * DISCORD_FRAME_SIZE)
        t.join(timeout=1.0)
        assert results == [b"\x42" * DISCORD_FRAME_SIZE]

    def test_close_unblocks_reader(self) -> None:
        """Close while reader blocked → reader returns empty bytes."""
        src = StreamingPCMSource()
        results: list[bytes] = []

        def reader() -> None:
            results.append(src.read())

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.05)
        src.close_input()
        t.join(timeout=1.0)
        assert results == [b""]

    def test_is_opus_false(self) -> None:
        assert StreamingPCMSource().is_opus() is False

    def test_cleanup_closes_input(self) -> None:
        """Pycord calls cleanup on stop; should release any blocked reader."""
        src = StreamingPCMSource()
        src.cleanup()
        assert src.read() == b""

    def test_feed_empty_is_noop(self) -> None:
        """Empty feed shouldn't notify or pollute the buffer."""
        src = StreamingPCMSource()
        src.feed(b"")
        src.close_input()
        assert src.read() == b""
