"""Tests for 48 kHz → 16 kHz resampler used by the local turn-detection chain.

Discord delivers 48 kHz mono int16 PCM after the recording sink's
stereo→mono conversion. TEN-VAD wants 16 kHz mono int16. 3:1
decimation with a boxcar pre-filter (average each 3 samples) is
cheap and good enough — TEN-VAD is forgiving of the residual aliasing
above 8 kHz.

The resampler is stream-stateful: callers feed arbitrary chunk
lengths, and any sample remainder (1 or 2 samples mod 3) carries
over to the next ``feed`` call.
"""

from __future__ import annotations

import struct

import pytest

from familiar_connect.voice.audio import Resampler48to16


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class TestResampler48to16Basic:
    def test_empty_input_returns_empty(self) -> None:
        r = Resampler48to16()
        assert r.feed(b"") == b""

    def test_three_input_samples_yield_one_output_sample(self) -> None:
        """3:1 decimation — three int16 in (6 bytes), one int16 out (2 bytes)."""
        r = Resampler48to16()
        out = r.feed(_pcm([100, 200, 300]))
        # Boxcar avg → int(round((100+200+300)/3)) == 200
        (sample,) = struct.unpack("<h", out)
        assert sample == 200

    def test_output_length_is_one_third_input(self) -> None:
        """Six aligned input samples → two output samples."""
        r = Resampler48to16()
        out = r.feed(_pcm([0, 0, 0, 100, 100, 100]))
        assert len(out) == 4  # 2 int16 samples


class TestResampler48to16Streaming:
    def test_carries_remainder_across_feed_calls(self) -> None:
        """Two samples then one sample → the boundary sample is averaged correctly."""
        r = Resampler48to16()
        # 4 samples in two chunks (2 + 2). Aligned groups:
        #   group0 = (a0, a1, b0)  group1 = leftover (b1) — held for next feed
        out1 = r.feed(_pcm([100, 200]))
        out2 = r.feed(_pcm([300, 400]))
        # First feed has only 2 samples → no full triplet → no output yet.
        assert out1 == b""
        # After second feed: triplet (100,200,300) closes; (400,) carries.
        (sample,) = struct.unpack("<h", out2)
        assert sample == 200  # avg(100,200,300)

    def test_close_drains_carry_with_zero_padding(self) -> None:
        """``close()`` pads any held remainder with zeros and emits the last sample."""
        r = Resampler48to16()
        r.feed(_pcm([300, 300]))  # 2 samples held
        out = r.close()
        # avg(300, 300, 0) → 200
        (sample,) = struct.unpack("<h", out)
        assert sample == 200

    def test_close_when_aligned_emits_nothing(self) -> None:
        r = Resampler48to16()
        r.feed(_pcm([0, 0, 0]))
        assert r.close() == b""

    def test_reset_drops_held_remainder(self) -> None:
        r = Resampler48to16()
        r.feed(_pcm([100, 200]))  # held
        r.reset()
        # Fresh feed of one full triplet
        out = r.feed(_pcm([90, 90, 90]))
        (sample,) = struct.unpack("<h", out)
        assert sample == 90


class TestResampler48to16Validation:
    def test_rejects_odd_byte_length(self) -> None:
        r = Resampler48to16()
        with pytest.raises(ValueError, match="even"):
            r.feed(b"\x00")


class TestResampler48to16Numerics:
    def test_clipped_to_int16_range(self) -> None:
        """Avg of three identical int16 max values is still int16 max."""
        r = Resampler48to16()
        out = r.feed(_pcm([32767, 32767, 32767]))
        (sample,) = struct.unpack("<h", out)
        assert sample == 32767

    def test_negative_values_average_correctly(self) -> None:
        r = Resampler48to16()
        out = r.feed(_pcm([-100, -200, -300]))
        (sample,) = struct.unpack("<h", out)
        assert sample == -200
