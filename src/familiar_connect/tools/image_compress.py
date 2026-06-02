"""JPEG compression for images fetched by the view_image tool."""

from __future__ import annotations

import base64
import io

from PIL import Image

_MAX_EDGE = 1024
_START_QUALITY = 85
_START_QUALITY_DESCRIBE = 95
_MIN_QUALITY = 20
_QUALITY_STEP = 5
_SIZE_CEILING = 1_000_000
_SIZE_CEILING_DESCRIBE = 4_000_000


class ImageTooLargeError(Exception):
    """Image cannot be compressed under the size ceiling."""


def compress_to_jpeg(raw: bytes, *, ceiling: int = _SIZE_CEILING) -> bytes:
    """Open ``raw`` bytes, resize to max 1024px edge, encode as JPEG.

    Iterates quality from :data:`_START_QUALITY` down to
    :data:`_MIN_QUALITY` in steps of :data:`_QUALITY_STEP` until the
    output is under ``ceiling`` bytes.

    :raises ImageTooLargeError: no quality setting achieves target size.
    """
    img = Image.open(io.BytesIO(raw))
    # convert before thumbnail to avoid palette-mode resize artifacts;
    # RGB also strips alpha (JPEG has no alpha channel)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # thumbnail only downscales — preserves images already within limits
    img.thumbnail((_MAX_EDGE, _MAX_EDGE))

    for quality in range(_START_QUALITY, _MIN_QUALITY - 1, -_QUALITY_STEP):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= ceiling:
            return data

    msg = f"image cannot be compressed under {ceiling} bytes"
    raise ImageTooLargeError(msg)


def compress_for_description(raw: bytes) -> bytes:
    """High-quality JPEG for vision model description; 4MB ceiling.

    Same 1024px resize, but starts at quality 95 to preserve detail that
    matters for accurate descriptions. Description is stored permanently;
    prose payload uses the tighter :func:`compress_to_jpeg` instead.
    """
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((_MAX_EDGE, _MAX_EDGE))

    for quality in range(_START_QUALITY_DESCRIBE, _MIN_QUALITY - 1, -_QUALITY_STEP):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= _SIZE_CEILING_DESCRIBE:
            return data

    msg = f"image cannot be compressed under {_SIZE_CEILING_DESCRIBE} bytes"
    raise ImageTooLargeError(msg)


def encode_base64_jpeg(raw: bytes) -> str:
    """Compress ``raw`` to JPEG and base64-encode."""
    return base64.b64encode(compress_to_jpeg(raw)).decode("ascii")
