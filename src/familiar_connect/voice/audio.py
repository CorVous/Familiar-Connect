"""Audio conversion utilities for Discord voice playback."""

from __future__ import annotations

import struct

# Discord requires 48kHz, 16-bit signed, stereo PCM in 20ms frames.
# 48000 samples/s x 2 channels x 2 bytes/sample x 0.020 s = 3840 bytes/frame.
DISCORD_FRAME_SIZE = 3840


def mono_to_stereo(data: bytes) -> bytes:
    """Convert mono 16-bit PCM to stereo by duplicating each sample.

    Each 2-byte (16-bit little-endian) sample from *data* is written twice —
    once for the left channel and once for the right — producing an output
    buffer twice the length of the input.

    :param data: Mono 16-bit signed little-endian PCM bytes.
    :return: Stereo 16-bit PCM bytes (interleaved L/R pairs).
    :raises ValueError: If *data* has an odd number of bytes (not valid 16-bit PCM).
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
    """Convert stereo 16-bit PCM to mono by averaging left and right channels.

    Each 4-byte stereo frame (2 bytes left + 2 bytes right) is reduced to a
    single 2-byte mono sample by averaging the two int16 values.

    :param data: Stereo 16-bit signed little-endian PCM bytes.
    :return: Mono 16-bit PCM bytes (half the length of input).
    :raises ValueError: If *data* length is not divisible by 4 (not valid stereo).
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
