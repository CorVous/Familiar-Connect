"""Tests for voice audio conversion utilities."""

from __future__ import annotations

import struct
import threading
import time

import pytest

from familiar_connect.voice import audio as audio_mod
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

    @pytest.mark.parametrize(
        "samples",
        [
            [],
            [0],
            [-32768],
            [32767],
            [0x0102, 0x0304],
            [100, -200, 32767, -32768, 0],
            [-1, 1, -32768, 32767, 12345, -12345, 0],
        ],
    )
    def test_byte_identical_to_reference_duplication(self, samples: list[int]) -> None:
        """Output is byte-for-byte the per-int16-sample duplication.

        Independent reference: each little-endian 2-byte sample appears
        twice in order ([s0,s0,s1,s1,...]), built here without touching
        the implementation under test, across empty/single/edge int16s.
        """
        mono = struct.pack(f"<{len(samples)}h", *samples)
        expected = b"".join(mono[i : i + 2] * 2 for i in range(0, len(mono), 2))
        assert mono_to_stereo(mono) == expected

    def test_fallback_matches_numpy_when_numpy_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pure-Python fallback is byte-identical to the numpy path.

        ``voice.audio`` is base-eager (imported via ``voice/__init__``), but
        numpy is an optional extra — so the module must import and convert
        without it (docs build, base-only install). Force the numpy-absent
        branch and confirm it matches the numpy result and still validates.
        """
        samples = [-1, 1, -32768, 32767, 12345, -12345, 0, 0x0102]
        mono = struct.pack(f"<{len(samples)}h", *samples)
        numpy_result = mono_to_stereo(mono)

        monkeypatch.setattr(audio_mod, "np", None)
        assert audio_mod.mono_to_stereo(mono) == numpy_result
        with pytest.raises(ValueError, match="even"):
            audio_mod.mono_to_stereo(b"\x01")


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


class TestStreamingPCMSourceJitterBuffer:
    """Opt-in pre-roll + underrun padding for bursty providers (Azure).

    Cartesia's steady cadence keeps the defaults (no pre-roll, block on
    underrun). Azure delivers in synthesis-paced bursts that starve the
    buffer; these knobs smooth playback so pycord's 20 ms clock stays
    monotonic instead of rushing to catch up after a stall.
    """

    def test_preroll_blocks_until_threshold_met(self) -> None:
        """First read waits for ``prebuffer_bytes`` before any frame."""
        src = StreamingPCMSource(prebuffer_bytes=DISCORD_FRAME_SIZE * 2)
        results: list[bytes] = []

        def reader() -> None:
            results.append(src.read())

        t = threading.Thread(target=reader)
        t.start()
        # Sub-threshold feed: one frame's worth, below the 2-frame gate.
        src.feed(b"\x11" * DISCORD_FRAME_SIZE)
        time.sleep(0.05)
        assert t.is_alive(), "reader should still be pre-roll-blocked"
        # Cross the threshold; reader unblocks and returns the first frame.
        src.feed(b"\x11" * DISCORD_FRAME_SIZE)
        t.join(timeout=1.0)
        assert results == [b"\x11" * DISCORD_FRAME_SIZE]

    def test_preroll_only_gates_first_read(self) -> None:
        """Once primed, later reads do not re-apply the pre-roll gate."""
        src = StreamingPCMSource(prebuffer_bytes=DISCORD_FRAME_SIZE)
        src.feed(b"\x22" * (DISCORD_FRAME_SIZE * 2))
        first = src.read()  # primes
        second = src.read()  # must not block on the gate again
        assert first == b"\x22" * DISCORD_FRAME_SIZE
        assert second == b"\x22" * DISCORD_FRAME_SIZE

    def test_eos_overrides_preroll(self) -> None:
        """A short reply closing below threshold still plays (EOS wins)."""
        src = StreamingPCMSource(prebuffer_bytes=DISCORD_FRAME_SIZE * 4)
        partial = b"\x07\x08\x09\x0a"
        src.feed(partial)
        src.close_input()
        out = src.read()
        assert len(out) == DISCORD_FRAME_SIZE
        assert out[: len(partial)] == partial
        assert src.read() == b""

    def test_underrun_pads_silence_without_blocking(self) -> None:
        """Primed, empty, open + pad_underrun → a silent frame, no block."""
        src = StreamingPCMSource(pad_underrun=True)
        src.feed(b"\x33" * DISCORD_FRAME_SIZE)
        primed = src.read()
        assert primed == b"\x33" * DISCORD_FRAME_SIZE
        # Buffer now empty and NOT closed: must return silence, not block.
        results: list[bytes] = []

        def reader() -> None:
            results.append(src.read())

        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=1.0)
        assert not t.is_alive(), "pad_underrun read must not block"
        assert results == [b"\x00" * DISCORD_FRAME_SIZE]

    def test_eos_overrides_underrun_padding(self) -> None:
        """Closed + empty returns empty bytes — no infinite silence."""
        src = StreamingPCMSource(pad_underrun=True)
        src.feed(b"\x44" * DISCORD_FRAME_SIZE)
        assert src.read() == b"\x44" * DISCORD_FRAME_SIZE
        src.close_input()
        assert src.read() == b""

    def test_default_underrun_still_blocks(self) -> None:
        """Defaults (pad_underrun=False) keep the block-on-underrun path."""
        src = StreamingPCMSource()
        src.feed(b"\x55" * DISCORD_FRAME_SIZE)
        assert src.read() == b"\x55" * DISCORD_FRAME_SIZE
        results: list[bytes] = []

        def reader() -> None:
            results.append(src.read())

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.05)
        assert t.is_alive(), "default underrun must block, not pad"
        src.close_input()  # release the blocked reader for cleanup
        t.join(timeout=1.0)
        assert results == [b""]
