"""Tests for voice audio conversion utilities."""

from __future__ import annotations

import struct

import pytest

from familiar_connect.voice.audio import (
    DISCORD_FRAME_SIZE,
    mono_to_stereo,
    stereo_to_mono,
    upsample_2x,
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


class TestUpsample2x:
    def test_doubles_length(self) -> None:
        """Output is exactly twice the length of input."""
        data = bytes(4)  # 2 samples of 16-bit audio
        assert len(upsample_2x(data)) == 8

    def test_each_sample_duplicated(self) -> None:
        """Each 16-bit sample appears twice consecutively in output."""
        data = struct.pack("<hh", 0x0102, 0x0304)
        result = upsample_2x(data)
        samples = struct.unpack("<hhhh", result)
        assert samples[0] == samples[1] == 0x0102
        assert samples[2] == samples[3] == 0x0304

    def test_empty_input(self) -> None:
        """Empty input returns empty bytes."""
        assert upsample_2x(b"") == b""

    def test_odd_length_raises(self) -> None:
        """Odd-length input is not valid 16-bit PCM — raises ValueError."""
        with pytest.raises(ValueError, match=r"even"):
            upsample_2x(b"\x00\x01\x02")

    def test_single_sample(self) -> None:
        """Single 16-bit sample produces two identical copies."""
        data = struct.pack("<h", -32000)
        result = upsample_2x(data)
        assert len(result) == 4
        s0, s1 = struct.unpack("<hh", result)
        assert s0 == s1 == -32000


class TestDiscordFrameSize:
    def test_frame_size_is_3840(self) -> None:
        """Discord expects 3840-byte frames (48kHz, 16-bit, stereo, 20ms)."""
        assert DISCORD_FRAME_SIZE == 3840
