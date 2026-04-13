"""Character Card V3 loader.

Parses TavernAI V3 card from PNG tEXt chunk (keyword ``ccv3``,
base64-encoded JSON). V2-only cards raise :class:`CharacterCardError`.
Stdlib only (struct, base64, json).
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass
from pathlib import Path

PathLike = str | Path

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CharacterCardError(Exception):
    """Raised when a character card cannot be loaded or parsed."""


@dataclass
class CharacterCard:
    """Fields extracted from a TavernAI Character Card V3."""

    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    creator_notes: str = ""


def load_card(source: bytes | PathLike) -> CharacterCard:
    """Load a Character Card V3 from a PNG file or raw bytes.

    :param source: Path to a ``.png`` file (str or Path) or raw PNG bytes.
    :raises CharacterCardError: If the source is not a valid V3 card PNG.
    :return: Parsed CharacterCard.
    """
    if isinstance(source, (str, Path)):
        try:
            data = Path(source).read_bytes()
        except OSError as exc:
            msg = f"Cannot read file: {source}"
            raise CharacterCardError(msg) from exc
    else:
        data = source

    chunks = _parse_png_chunks(data)
    return _extract_card(chunks)


# ---------------------------------------------------------------------------
# PNG parsing (stdlib only)
# ---------------------------------------------------------------------------


def _parse_png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Extract (chunk_type, chunk_data) pairs from PNG bytes.

    :raises CharacterCardError: If data is not valid PNG.
    """
    if not data.startswith(_PNG_SIGNATURE):
        msg = "Not a PNG file (invalid signature)"
        raise CharacterCardError(msg)

    offset = 8  # skip 8-byte PNG signature
    chunks: list[tuple[bytes, bytes]] = []

    while offset < len(data):
        if offset + 8 > len(data):
            msg = "Truncated PNG: cannot read chunk header"
            raise CharacterCardError(msg)

        (length,) = struct.unpack_from(">I", data, offset)
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]

        if len(chunk_data) < length:
            msg = f"Truncated PNG: chunk {chunk_type!r} data too short"
            raise CharacterCardError(msg)

        chunks.append((chunk_type, chunk_data))
        offset += 8 + length + 4  # header + data + CRC

    return chunks


def _extract_card(chunks: list[tuple[bytes, bytes]]) -> CharacterCard:
    """Find and parse V3 card from PNG chunks.

    Accepts ``ccv3`` (pure V3) and ``chara`` with
    ``spec: chara_card_v3`` (SillyTavern default).
    """
    chara_text: str | None = None  # raw text from 'chara' chunk, if any

    for chunk_type, chunk_data in chunks:
        if chunk_type != b"tEXt":
            continue

        sep = chunk_data.find(b"\x00")
        if sep == -1:
            continue

        keyword = chunk_data[:sep].decode("latin-1", errors="replace")
        text = chunk_data[sep + 1 :].decode("latin-1", errors="replace")

        if keyword.lower() == "ccv3":
            # pure V3 chunk — use immediately
            return _parse_v3_json(text)

        if keyword.lower() == "chara":
            chara_text = text

    # fall back to 'chara' chunk if its JSON declares V3
    if chara_text is not None:
        return _parse_v3_json(chara_text)

    msg = (
        "No character card found in this PNG. "
        "Expected a tEXt chunk with keyword 'ccv3' or 'chara' (V3 JSON)."
    )
    raise CharacterCardError(msg)


def _parse_v3_json(encoded_text: str) -> CharacterCard:
    """Decode base64 and parse V3 card JSON.

    :raises CharacterCardError: If decode, parse, or spec validation fails.
    """
    try:
        raw_json = base64.b64decode(encoded_text.encode("latin-1")).decode("utf-8")
    except Exception as exc:
        msg = f"Failed to base64-decode character card: {exc}"
        raise CharacterCardError(msg) from exc

    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        msg = f"Character card JSON is invalid: {exc}"
        raise CharacterCardError(msg) from exc

    spec = obj.get("spec", "")
    if spec != "chara_card_v3":
        msg = f"Expected spec='chara_card_v3', got {spec!r}. Only V3 supported."
        raise CharacterCardError(msg)

    data = obj.get("data", {})

    return CharacterCard(
        name=data.get("name", ""),
        description=data.get("description", ""),
        personality=data.get("personality", ""),
        scenario=data.get("scenario", ""),
        first_mes=data.get("first_mes", ""),
        mes_example=data.get("mes_example", ""),
        system_prompt=data.get("system_prompt", ""),
        post_history_instructions=data.get("post_history_instructions", ""),
        creator_notes=data.get("creator_notes", ""),
    )
