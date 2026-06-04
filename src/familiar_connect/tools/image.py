"""view_image tool — fetch and inspect Discord images."""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

import httpx

from familiar_connect import log_style as ls
from familiar_connect.tools.image_compress import (
    compress_for_description,
    compress_to_jpeg,
)
from familiar_connect.tools.image_describe import describe_image
from familiar_connect.tools.registry import ImageResult, Tool

if TYPE_CHECKING:
    from familiar_connect.tools.registry import ToolContext

_logger = logging.getLogger(__name__)

_FETCH_TIMEOUT_S = 15.0
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20 MB cap before compression
_TOOL_TIMEOUT_S = 30.0

_ALLOWED_CONTENT_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
})


async def _fetch_image_bytes(url: str) -> bytes:
    """Fetch raw bytes from ``url``; validates content-type is image.

    :raises ValueError: non-image content-type.
    :raises ValueError: download exceeds size cap.
    """
    client = httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S, follow_redirects=True)
    async with client, client.stream("GET", url) as response:
        response.raise_for_status()
        ct = response.headers.get("content-type", "")
        # strip params like "image/jpeg; charset=..."
        ct_base = ct.split(";")[0].strip().lower()
        if ct_base not in _ALLOWED_CONTENT_TYPES:
            msg = f"non-image content-type: {ct!r}"
            raise ValueError(msg)
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes(chunk_size=65536):
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                msg = f"download exceeds {_MAX_DOWNLOAD_BYTES} byte cap"
                raise ValueError(msg)
            chunks.append(chunk)
    return b"".join(chunks)


async def _view_image_handler(
    args: dict,
    ctx: ToolContext,
    describe_constraints: str = "",
) -> str | ImageResult:
    """Fetch and optionally describe an image referenced by placeholder."""
    img_id = args.get("image_id")
    if not isinstance(img_id, str):
        return json.dumps({"error": "image_id must be a string"})

    url = ctx.images.get(img_id)
    if url is None:
        return json.dumps({"error": f"unknown image id {img_id!r}"})

    try:
        raw = await _fetch_image_bytes(url)
    except Exception as exc:  # noqa: BLE001 — surface as tool error
        _logger.warning(
            f"{ls.tag('Image', ls.Y)} "
            f"{ls.kv('fetch_error', repr(exc), vc=ls.LY)} "
            f"{ls.kv('img_id', img_id, vc=ls.LC)}"
        )
        return json.dumps({"error": f"fetch failed: {exc}"})

    # describe at high quality (JPEG 95, 4MB) — result persists
    # compress separately at lower ceiling for prose model payload
    if ctx.description_llm is None:
        desc = "(no description model configured)"
    else:
        try:
            desc_jpeg = compress_for_description(raw)
            desc_b64 = base64.b64encode(desc_jpeg).decode("ascii")
            desc = await describe_image(
                llm=ctx.description_llm,
                jpeg_base64=desc_b64,
                constraints=describe_constraints,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                f"{ls.tag('Image', ls.Y)} "
                f"{ls.kv('describe_error', repr(exc), vc=ls.LY)} "
                f"{ls.kv('img_id', img_id, vc=ls.LC)}"
            )
            desc = "(description unavailable)"

    try:
        jpeg = compress_to_jpeg(raw)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"compression failed: {exc}"})

    b64 = base64.b64encode(jpeg).decode("ascii")
    return ImageResult(description=desc, jpeg_base64=b64)


def build_view_image_tool(describe_constraints: str = "") -> Tool:
    """Return the view_image :class:`Tool` descriptor.

    ``describe_constraints`` (per-familiar, from
    ``[prompt].image_description_constraints``) bind into the handler at
    construction — static for the familiar's lifetime, so closed over
    here rather than carried on per-turn ``ToolContext``.
    """

    async def _handler(args: dict, ctx: ToolContext) -> str | ImageResult:
        return await _view_image_handler(
            args, ctx, describe_constraints=describe_constraints
        )

    return Tool(
        name="view_image",
        description=(
            "Fetch and look at an image referenced by its [image: img_N (filename)] "
            "placeholder. Pass the `image_id` exactly as shown (e.g. `img_0`). "
            "Returns a description and the image itself when the model supports vision."
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "Image id from the [image: img_N] placeholder.",
                },
            },
            "required": ["image_id"],
        },
        handler=_handler,
        timeout_s=_TOOL_TIMEOUT_S,
    )
