"""Audio conversion utilities for Discord voice playback."""

from __future__ import annotations

import struct
import threading

import discord

try:  # numpy is an optional extra (voice backends); base/docs installs lack it
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in numpy-less envs
    np = None  # type: ignore[assignment]

# Discord requires 48kHz s16le stereo PCM in 20ms frames.
# 48000 * 2ch * 2B * 0.020s = 3840 bytes/frame.
DISCORD_FRAME_SIZE = 3840


# 3:1 decimation — 48000 / 16000.
_RESAMPLE_RATIO: int = 3


class Resampler48to16:
    """Streaming 48 kHz → 16 kHz int16 PCM resampler.

    Stateful: holds up to two int16 samples between calls so callers
    can feed arbitrary chunk lengths. Each output sample = integer
    mean of three consecutive inputs (boxcar pre-filter + 3:1
    decimation). TEN-VAD tolerates residual aliasing above 8 kHz Nyquist.

    :meth:`close` flushes a partial triplet (zero-padded) at EOS;
    :meth:`reset` drops held state without emitting.
    """

    def __init__(self) -> None:
        # Held int samples carried into next feed (0, 1, or 2)
        self._carry: list[int] = []

    def reset(self) -> None:
        """Drop held remainder; next ``feed`` starts a fresh triplet."""
        self._carry = []

    def feed(self, pcm_48k: bytes) -> bytes:
        """Resample arbitrary-length 48 kHz int16 PCM → 16 kHz int16 PCM.

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
        """Flush held remainder, zero-padding to a full triplet."""
        if not self._carry:
            return b""
        samples = self._carry + [0] * (_RESAMPLE_RATIO - len(self._carry))
        avg = sum(samples) // _RESAMPLE_RATIO
        self._carry = []
        return struct.pack("<h", avg)


def mono_to_stereo(data: bytes) -> bytes:
    """Duplicate each int16 sample into L+R; produces 2x output.

    Uses numpy when available (the voice playback path always installs it);
    falls back to a byte-identical pure-Python loop so this base-eager module
    still imports in numpy-less environments (docs build, base-only install).

    :raises ValueError: If *data* has odd length.
    """
    if len(data) % 2 != 0:
        msg = f"PCM data length must be even, got {len(data)}"
        raise ValueError(msg)

    if np is not None:
        # Vectorized: int16 view → duplicate each sample into L+R. Little-endian
        # pinned via "<i2" so output bytes are host-independent. Releases the GIL
        # over the buffer instead of looping per-sample in Python.
        return np.frombuffer(data, dtype="<i2").repeat(2).tobytes()

    # Pure-Python fallback (numpy-less envs): duplicate each 2-byte sample.
    result = bytearray(len(data) * 2)
    for i in range(0, len(data), 2):
        sample = data[i : i + 2]
        out_off = i * 2
        result[out_off : out_off + 2] = sample  # Left
        result[out_off + 2 : out_off + 4] = sample  # Right
    return bytes(result)


class StreamingPCMSource(discord.AudioSource):
    """Thread-safe streaming PCM source for ``vc.play``.

    Producer (asyncio task) calls :meth:`feed` with stereo s16le bytes
    from the TTS stream; pycord's ``AudioPlayer`` thread drains 20 ms
    frames via :meth:`read`. :meth:`close_input` flips an EOS flag —
    once buffer drains after that, ``read`` returns ``b""`` and pycord
    stops the player.

    Without the close flag, ``read`` blocks on the condvar until the
    producer feeds more — pycord's player thread blocks, pausing
    playback (no underrun silence) until next chunk. With Cartesia's
    ~140 ms TTFB, the first ``read`` returns within one or two of
    pycord's 20 ms ticks.

    Two opt-in jitter-buffer knobs smooth bursty producers (Azure, which
    delivers in synthesis-paced bursts) without touching the steady-cadence
    default (Cartesia):

    * ``prebuffer_bytes`` — the first ``read`` blocks until at least this
      many bytes are buffered (or EOS), building a cushion before
      playback starts. ``0`` (default) starts immediately.
    * ``pad_underrun`` — in steady state, an empty-but-open buffer returns
      one frame of silence instead of blocking, so pycord's 20 ms clock
      never overshoots and rushes to catch up. ``False`` (default) keeps
      the block-on-underrun behavior. EOS always overrides padding so
      playback ends instead of emitting silence forever.
    """

    def __init__(
        self,
        *,
        prebuffer_bytes: int = 0,
        pad_underrun: bool = False,
    ) -> None:
        self._buf = bytearray()
        # Condition guards ``_buf``, ``_closed``, and ``_primed``
        self._cond = threading.Condition()
        self._closed = False
        self._prebuffer_bytes = prebuffer_bytes
        self._pad_underrun = pad_underrun
        self._primed = prebuffer_bytes <= 0

    def feed(self, data: bytes) -> None:
        """Append ``data`` (stereo s16le); notify reader."""
        if not data:
            return
        with self._cond:
            self._buf.extend(data)
            self._cond.notify()

    def close_input(self) -> None:
        """Signal EOS; reader drains then returns empty bytes."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def read(self) -> bytes:
        with self._cond:
            if not self._primed:
                # Pre-roll: build a cushion before the first frame plays.
                # EOS overrides so a short reply still plays out.
                while len(self._buf) < self._prebuffer_bytes and not self._closed:
                    self._cond.wait()
                self._primed = True

            if len(self._buf) < DISCORD_FRAME_SIZE and not self._closed:
                # Underrun while still producing. Pad with silence (opt-in)
                # to keep pycord's clock monotonic; otherwise block until fed.
                if self._pad_underrun:
                    return b"\x00" * DISCORD_FRAME_SIZE
                while len(self._buf) < DISCORD_FRAME_SIZE and not self._closed:
                    self._cond.wait()

            if len(self._buf) >= DISCORD_FRAME_SIZE:
                out = bytes(self._buf[:DISCORD_FRAME_SIZE])
                del self._buf[:DISCORD_FRAME_SIZE]
                return out
            if self._buf:
                # Zero-pad final partial frame so pycord plays it before stop
                out = bytes(self._buf) + b"\x00" * (DISCORD_FRAME_SIZE - len(self._buf))
                self._buf.clear()
                return out
            return b""

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        # Called by pycord on player stop; release any blocked reader
        self.close_input()


def stereo_to_mono(data: bytes) -> bytes:
    """Average L+R int16 samples into mono; produces 0.5x output.

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
