"""Tests for image_compress module."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from familiar_connect.tools.image_compress import (
    _MAX_GIF_FRAMES,
    ImageTooLargeError,
    _open_as_rgb,
    compress_for_description,
    compress_to_jpeg,
    encode_base64_jpeg,
)


def _make_png(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    """Return PNG bytes for a solid-color image."""
    img = Image.new(mode, (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gif(n_frames: int = 6, width: int = 80, height: int = 60) -> bytes:
    """Return animated GIF bytes with ``n_frames`` distinct frames."""
    # convert from RGB so each frame has a distinct colour Pillow can distinguish
    frames = [
        Image.new("RGB", (width, height), color=(i * 40, 0, 0)).convert("P")
        for i in range(n_frames)
    ]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=100,
    )
    return buf.getvalue()


def test_compress_shrinks_oversized() -> None:
    raw = _make_png(4000, 3000)
    out = compress_to_jpeg(raw)

    img = Image.open(io.BytesIO(out))
    assert max(img.size) == 1024
    assert len(out) <= 1_000_000


def test_compress_handles_rgba() -> None:
    raw = _make_png(200, 200, mode="RGBA")
    # JPEG has no alpha channel; compression must convert without error
    out = compress_to_jpeg(raw)
    assert len(out) > 0


def test_compress_small_image_unchanged_size() -> None:
    """Small images that already fit should pass through without upscaling."""
    raw = _make_png(64, 64)
    out = compress_to_jpeg(raw)

    img = Image.open(io.BytesIO(out))
    # thumbnail only downscales; 64x64 stays 64x64
    assert max(img.size) <= 1024


def test_compress_raises_when_unreachable() -> None:
    raw = _make_png(100, 100)
    with pytest.raises(ImageTooLargeError):
        compress_to_jpeg(raw, ceiling=1)


def test_gif_single_frame_no_tiling() -> None:
    raw = _make_gif(n_frames=1)
    img = _open_as_rgb(raw)
    # single frame — no tiling, width unchanged
    assert img.size == (80, 60)


def test_gif_animated_tiles_frames_horizontally() -> None:
    raw = _make_gif(n_frames=6, width=80, height=60)
    img = _open_as_rgb(raw)
    # capped at _MAX_GIF_FRAMES frames tiled side-by-side
    assert img.height == 60
    assert img.width == 80 * _MAX_GIF_FRAMES


def test_gif_fewer_frames_than_max() -> None:
    raw = _make_gif(n_frames=2, width=80, height=60)
    img = _open_as_rgb(raw)
    assert img.width == 80 * 2
    assert img.height == 60


def test_gif_compress_produces_valid_jpeg() -> None:
    raw = _make_gif(n_frames=6)
    out = compress_to_jpeg(raw)
    assert out[:3] == b"\xff\xd8\xff"
    assert len(out) <= 1_000_000


def test_gif_compress_for_description_produces_valid_jpeg() -> None:
    raw = _make_gif(n_frames=6)
    out = compress_for_description(raw)
    assert out[:3] == b"\xff\xd8\xff"
    assert len(out) <= 4_000_000


def test_encode_base64_jpeg_returns_ascii_string() -> None:
    raw = _make_png(50, 50)
    encoded = encode_base64_jpeg(raw)
    assert isinstance(encoded, str)
    # must be valid base64
    decoded = base64.b64decode(encoded)
    assert decoded[:3] == b"\xff\xd8\xff"  # JPEG magic bytes
