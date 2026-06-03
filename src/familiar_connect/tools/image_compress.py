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
_MAX_GIF_FRAMES = 4


class ImageTooLargeError(Exception):
    """Image cannot be compressed under the size ceiling."""


def _open_as_rgb(raw: bytes) -> Image.Image:
    """Open image bytes as RGB; tile evenly-spaced frames for animated GIFs.

    Animated GIFs produce a horizontal strip of up to :data:`_MAX_GIF_FRAMES`
    evenly-spaced frames so the vision model sees the arc of the animation
    rather than only frame 0. Static images (including single-frame GIFs)
    are converted to RGB directly.
    """
    img = Image.open(io.BytesIO(raw))
    n_frames = getattr(img, "n_frames", 1)

    if n_frames <= 1:
        return img.convert("RGB")

    count = min(_MAX_GIF_FRAMES, n_frames)
    indices = (
        [0]
        if count == 1
        else [round(i * (n_frames - 1) / (count - 1)) for i in range(count)]
    )

    frames: list[Image.Image] = []
    for idx in indices:
        img.seek(idx)
        frames.append(img.convert("RGB"))

    # scale all frames to the tallest frame's height, then stitch horizontally
    h = max(f.height for f in frames)
    parts: list[Image.Image] = []
    for f in frames:
        if f.height != h:
            scale = h / f.height
            parts.append(f.resize((int(f.width * scale), h), Image.Resampling.LANCZOS))
        else:
            parts.append(f)

    composite = Image.new("RGB", (sum(p.width for p in parts), h))
    x = 0
    for p in parts:
        composite.paste(p, (x, 0))
        x += p.width
    return composite


def compress_to_jpeg(raw: bytes, *, ceiling: int = _SIZE_CEILING) -> bytes:
    """Open ``raw`` bytes, resize to max 1024px edge, encode as JPEG.

    Animated GIFs are tiled as a frame strip via :func:`_open_as_rgb`.
    Iterates quality from :data:`_START_QUALITY` down to
    :data:`_MIN_QUALITY` in steps of :data:`_QUALITY_STEP` until the
    output is under ``ceiling`` bytes.

    :raises ImageTooLargeError: no quality setting achieves target size.
    """
    img = _open_as_rgb(raw)
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

    Same 1024px resize and GIF frame tiling as :func:`compress_to_jpeg`,
    but starts at quality 95. Description is stored permanently; prose
    payload uses the tighter :func:`compress_to_jpeg` instead.
    """
    img = _open_as_rgb(raw)
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
