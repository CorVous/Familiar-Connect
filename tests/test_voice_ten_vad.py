"""Tests for the TEN-VAD-based speech activity detector.

Exercises the pure detector logic with a mocked ``TenVad`` backend so
tests are deterministic and don't depend on the native library.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pytest

from familiar_connect.voice.ten_vad import (
    DEFAULT_HANGOVER_MS,
    DEFAULT_HOP_SAMPLES,
    TenVadDetector,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


# 48 kHz int16 mono — one chunk of N samples
def _pcm48(n_samples: int, value: int = 0) -> bytes:
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


class _FakeTenVad:
    """Deterministic stand-in for ``ten_vad.TenVad``.

    ``process(frame)`` returns a scripted ``(prob, flag)`` per call,
    cycling the last entry forever.
    """

    def __init__(self, script: list[tuple[float, int]]) -> None:
        self._script = script
        self._i = 0

    def process(self, frame: np.ndarray) -> tuple[float, int]:
        _ = frame  # unused in fake
        idx = min(self._i, len(self._script) - 1)
        self._i += 1
        return self._script[idx]


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[int], _FakeTenVad],
) -> dict[int, _FakeTenVad]:
    """Patch ``TenVad`` so each new instance is produced by *factory*.

    Returns a dict mapping instance-index -> fake so tests can assert
    on per-user state.
    """
    created: dict[int, _FakeTenVad] = {}
    counter = {"n": 0}

    def _make(*_args: object, **_kwargs: object) -> _FakeTenVad:
        fake = factory(counter["n"])
        created[counter["n"]] = fake
        counter["n"] += 1
        return fake

    monkeypatch.setattr("familiar_connect.voice.ten_vad.TenVad", _make)
    return created


class TestTenVadDetector:
    def test_feed_emits_speech_start_on_first_speech_frame(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """First positive VAD flag fires ``on_speech_start`` once."""
        _install_fake(monkeypatch, lambda _i: _FakeTenVad([(0.9, 1)]))

        starts: list[int] = []
        ends: list[int] = []
        det = TenVadDetector(
            on_speech_start=starts.append,
            on_speech_end=ends.append,
            hop_samples=DEFAULT_HOP_SAMPLES,
            hangover_ms=DEFAULT_HANGOVER_MS,
        )

        # feed enough 48 kHz audio to produce >= 1 hop of 16 kHz output
        # 960 samples @ 48 kHz == 20 ms; 10 chunks == 200 ms  → ample
        for _ in range(10):
            det.feed(42, _pcm48(960, 5000))

        assert starts == [42]
        assert ends == []

    def test_feed_emits_speech_end_after_hangover(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After ``hangover_ms`` of silence frames, ``on_speech_end`` fires."""
        # speech for the first 5 hops, silence forever after
        script = [(0.9, 1)] * 5 + [(0.05, 0)]
        _install_fake(monkeypatch, lambda _i: _FakeTenVad(script))

        starts: list[int] = []
        ends: list[int] = []
        det = TenVadDetector(
            on_speech_start=starts.append,
            on_speech_end=ends.append,
            hop_samples=DEFAULT_HOP_SAMPLES,
            hangover_ms=96,  # 6 hops @ 16 ms
        )

        # feed plenty of audio: 2 s
        for _ in range(100):
            det.feed(7, _pcm48(960, 1000))

        assert starts == [7]
        assert ends == [7]

    def test_per_user_state_is_independent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two users drive independent VAD instances and edge events."""
        # user A always speaking, user B always silent
        scripts = {
            0: [(0.9, 1)],
            1: [(0.05, 0)],
        }
        _install_fake(monkeypatch, lambda i: _FakeTenVad(scripts[i]))

        starts: list[int] = []
        ends: list[int] = []
        det = TenVadDetector(
            on_speech_start=starts.append,
            on_speech_end=ends.append,
        )

        for _ in range(10):
            det.feed(111, _pcm48(960, 5000))  # first-seen → TenVad #0
            det.feed(222, _pcm48(960, 0))  # second-seen → TenVad #1

        assert starts == [111]
        assert ends == []

    def test_no_redundant_start_events(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """While already speaking, further speech hops do not re-emit start."""
        _install_fake(monkeypatch, lambda _i: _FakeTenVad([(0.9, 1)]))

        starts: list[int] = []
        det = TenVadDetector(
            on_speech_start=starts.append,
            on_speech_end=lambda _u: None,
        )

        for _ in range(50):
            det.feed(9, _pcm48(960, 5000))

        assert starts == [9]

    def test_speech_end_restarts_on_next_speech(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After end fires, a fresh speech burst emits a new start."""
        # speech → silence (long) → speech again
        script = (
            [(0.9, 1)] * 3
            + [(0.05, 0)] * 20  # > hangover
            + [(0.9, 1)] * 5
        )
        _install_fake(monkeypatch, lambda _i: _FakeTenVad(script))

        starts: list[int] = []
        ends: list[int] = []
        det = TenVadDetector(
            on_speech_start=starts.append,
            on_speech_end=ends.append,
            hangover_ms=96,
        )

        for _ in range(200):
            det.feed(1, _pcm48(960, 1000))

        assert starts == [1, 1]
        assert ends == [1]

    def test_odd_chunk_lengths_rejected(self) -> None:
        """Feeding a chunk with odd byte count raises (invalid int16)."""
        det = TenVadDetector(
            on_speech_start=lambda _u: None,
            on_speech_end=lambda _u: None,
        )
        with pytest.raises(ValueError, match="even"):
            det.feed(1, b"\x00\x00\x00")
