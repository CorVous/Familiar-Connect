"""Tests for voice audio conversion utilities."""

from __future__ import annotations

import struct

import pytest

from familiar_connect.voice.audio import DISCORD_FRAME_SIZE, mono_to_stereo


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


class TestDiscordFrameSize:
    def test_frame_size_is_3840(self) -> None:
        """Discord expects 3840-byte frames (48kHz, 16-bit, stereo, 20ms)."""
        assert DISCORD_FRAME_SIZE == 3840
