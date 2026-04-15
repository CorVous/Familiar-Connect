"""Audio conversion utilities for Discord voice playback."""

from __future__ import annotations

import struct

# Discord requires 48kHz, 16-bit signed, stereo PCM in 20ms frames.
# 48000 samples/s x 2 channels x 2 bytes/sample x 0.020 s = 3840 bytes/frame.
DISCORD_FRAME_SIZE = 3840


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


def upsample_2x(data: bytes) -> bytes:
    """Duplicate each 16-bit sample, doubling the effective sample rate.

    :raises ValueError: If *data* has odd length.
    """
    if len(data) % 2 != 0:
        msg = f"PCM data length must be even, got {len(data)}"
        raise ValueError(msg)
    result = bytearray(len(data) * 2)
    for i in range(0, len(data), 2):
        sample = data[i : i + 2]
        out_off = i * 2
        result[out_off : out_off + 2] = sample
        result[out_off + 2 : out_off + 4] = sample
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
