"""Audio conversion utilities for Discord voice playback."""

from __future__ import annotations

import struct

# Discord requires 48kHz, 16-bit signed, stereo PCM in 20ms frames.
# 48000 samples/s x 2 channels x 2 bytes/sample x 0.020 s = 3840 bytes/frame.
DISCORD_FRAME_SIZE = 3840


# 3:1 decimation — 48000 / 16000.
_RESAMPLE_RATIO: int = 3


class Resampler48to16:
    """Streaming 48 kHz → 16 kHz int16 PCM resampler.

    Stateful: holds up to two int16 samples between calls so callers
    can feed arbitrary chunk lengths. Each output sample is the
    integer mean of three consecutive input samples (boxcar pre-filter
    + 3:1 decimation). Silero VAD is forgiving of the residual high-
    frequency aliasing above the 8 kHz Nyquist.

    Use :meth:`close` to flush a partial triplet (zero-padded) at
    end-of-stream; :meth:`reset` drops held state without emitting.
    """

    def __init__(self) -> None:
        # held int samples carried into the next feed call (0, 1, or 2)
        self._carry: list[int] = []

    def reset(self) -> None:
        """Drop held remainder; next ``feed`` starts a fresh triplet."""
        self._carry = []

    def feed(self, pcm_48k: bytes) -> bytes:
        """Resample arbitrary-length 48 kHz int16 PCM; return 16 kHz int16 PCM.

        :raises ValueError: If *pcm_48k* has odd length (not int16-aligned).
        """
        if len(pcm_48k) % 2 != 0:
            msg = f"PCM data length must be even, got {len(pcm_48k)}"
            raise ValueError(msg)
        n_in = len(pcm_48k) // 2
        if n_in == 0 and not self._carry:
            return b""
        samples = list(self._carry)
        if n_in:
            samples.extend(struct.unpack(f"<{n_in}h", pcm_48k))
        n_full = len(samples) // _RESAMPLE_RATIO
        out = bytearray(n_full * 2)
        for i in range(n_full):
            base = i * _RESAMPLE_RATIO
            avg = (
                samples[base] + samples[base + 1] + samples[base + 2]
            ) // _RESAMPLE_RATIO
            struct.pack_into("<h", out, i * 2, avg)
        self._carry = samples[n_full * _RESAMPLE_RATIO :]
        return bytes(out)

    def close(self) -> bytes:
        """Flush any held remainder, zero-padding to a full triplet."""
        if not self._carry:
            return b""
        samples = self._carry + [0] * (_RESAMPLE_RATIO - len(self._carry))
        avg = sum(samples) // _RESAMPLE_RATIO
        self._carry = []
        return struct.pack("<h", avg)


def mono_to_stereo(data: bytes) -> bytes:
    """Duplicate each 16-bit sample into L+R, producing 2x output.

    :raises ValueError: If *data* has odd length.
    """
    if len(data) % 2 != 0:
        msg = f"PCM data length must be even, got {len(data)}"
        raise ValueError(msg)

    result = bytearray(len(data) * 2)
    for i in range(0, len(data), 2):
        sample = data[i : i + 2]
        out_off = i * 2
        result[out_off : out_off + 2] = sample  # left
        result[out_off + 2 : out_off + 4] = sample  # right
    return bytes(result)


def stereo_to_mono(data: bytes) -> bytes:
    """Average L+R int16 samples into mono, producing 0.5x output.

    :raises ValueError: If *data* length not divisible by 4.
    """
    if len(data) % 4 != 0:
        msg = f"Stereo PCM data length must be divisible by 4, got {len(data)}"
        raise ValueError(msg)

    n_frames = len(data) // 4
    result = bytearray(n_frames * 2)
    for i in range(n_frames):
        offset = i * 4
        (left, right) = struct.unpack_from("<hh", data, offset)
        avg = (left + right) // 2
        struct.pack_into("<h", result, i * 2, avg)
    return bytes(result)
