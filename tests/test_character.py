"""Tests for the Character Card V3 PNG loader."""

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest

from familiar_connect.character import (
    CharacterCard,
    CharacterCardError,
    load_card,
)

# ---------------------------------------------------------------------------
# Helpers to build minimal valid PNGs in memory
# ---------------------------------------------------------------------------

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Return a single PNG chunk (length + type + data + CRC)."""
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def _minimal_ihdr() -> bytes:
    """Return a 1x1 greyscale IHDR chunk."""
    # width=1, height=1, bit_depth=8, colour_type=0 (greyscale),
    # compression=0, filter=0, interlace=0
    data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return _png_chunk(b"IHDR", data)


def _text_chunk(keyword: str, text: str) -> bytes:
    """Return a tEXt chunk with the given keyword and text."""
    data = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    return _png_chunk(b"tEXt", data)


def _idat_chunk() -> bytes:
    """Return a minimal IDAT chunk (1x1 greyscale pixel)."""
    raw = b"\x00\x00"  # filter byte 0 + one pixel value 0
    compressed = zlib.compress(raw)
    return _png_chunk(b"IDAT", compressed)


def _iend_chunk() -> bytes:
    return _png_chunk(b"IEND", b"")


def _make_card_json(
    name: str = "Aria",
    description: str = "A helpful spirit.",
    personality: str = "Kind and wise.",
    scenario: str = "A cozy library.",
    first_mes: str = "Hello there!",
    mes_example: str = "<START>\n{{user}}: Hi\n{{char}}: Hello!",
    system_prompt: str = "",
    post_history_instructions: str = "",
    creator_notes: str = "",
) -> str:
    return json.dumps({
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": name,
            "description": description,
            "personality": personality,
            "scenario": scenario,
            "first_mes": first_mes,
            "mes_example": mes_example,
            "system_prompt": system_prompt,
            "post_history_instructions": post_history_instructions,
            "creator_notes": creator_notes,
            "character_book": None,
            "extensions": {},
        },
    })


def _make_v3_png(card_json: str) -> bytes:
    """Build a minimal PNG with a ccv3 tEXt chunk."""
    encoded = base64.b64encode(card_json.encode("utf-8")).decode("latin-1")
    return (
        _PNG_SIG
        + _minimal_ihdr()
        + _text_chunk("ccv3", encoded)
        + _idat_chunk()
        + _iend_chunk()
    )


def _make_v2_only_png() -> bytes:
    """Build a PNG with a 'chara' tEXt chunk containing non-V3 JSON."""
    # V2 cards have no 'spec' field (or a different one)
    card = {"spec": "chara_card_v2", "name": "Old", "description": "V2 only"}
    encoded = base64.b64encode(json.dumps(card).encode()).decode("latin-1")
    return (
        _PNG_SIG
        + _minimal_ihdr()
        + _text_chunk("chara", encoded)
        + _idat_chunk()
        + _iend_chunk()
    )


def _make_no_card_png() -> bytes:
    """Build a PNG with no character card chunk at all."""
    return _PNG_SIG + _minimal_ihdr() + _idat_chunk() + _iend_chunk()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def v3_png_bytes() -> bytes:
    return _make_v3_png(_make_card_json())


@pytest.fixture
def v3_png_file(tmp_path: Path, v3_png_bytes: bytes) -> Path:
    p = tmp_path / "test_card.png"
    p.write_bytes(v3_png_bytes)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadCardFromBytes:
    def test_returns_character_card(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert isinstance(card, CharacterCard)

    def test_name_parsed(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert card.name == "Aria"

    def test_description_parsed(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert card.description == "A helpful spirit."

    def test_personality_parsed(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert card.personality == "Kind and wise."

    def test_scenario_parsed(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert card.scenario == "A cozy library."

    def test_first_mes_parsed(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert card.first_mes == "Hello there!"

    def test_mes_example_parsed(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert "{{user}}" in card.mes_example

    def test_system_prompt_defaults_empty(self, v3_png_bytes: bytes) -> None:
        card = load_card(v3_png_bytes)
        assert not card.system_prompt

    def test_post_history_instructions_defaults_empty(
        self, v3_png_bytes: bytes
    ) -> None:
        card = load_card(v3_png_bytes)
        assert not card.post_history_instructions


class TestLoadCardFromPath:
    def test_load_from_path_object(self, v3_png_file: Path) -> None:
        card = load_card(v3_png_file)
        assert card.name == "Aria"

    def test_load_from_string_path(self, v3_png_file: Path) -> None:
        card = load_card(str(v3_png_file))
        assert card.name == "Aria"


class TestLoadCardErrors:
    def test_v2_only_raises(self) -> None:
        png = _make_v2_only_png()
        with pytest.raises(CharacterCardError, match=r"[Vv]3"):
            load_card(png)

    def test_no_card_chunk_raises(self) -> None:
        png = _make_no_card_png()
        with pytest.raises(CharacterCardError):
            load_card(png)

    def test_not_a_png_raises(self) -> None:
        with pytest.raises(CharacterCardError):
            load_card(b"this is not a png")

    def test_truncated_png_raises(self) -> None:
        with pytest.raises(CharacterCardError):
            load_card(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)


class TestCaseInsensitiveKeyword:
    def test_uppercase_ccv3_keyword_accepted(self) -> None:
        """The chunk keyword search is case-insensitive."""
        card_json = _make_card_json(name="CasedCard")
        encoded = base64.b64encode(card_json.encode("utf-8")).decode("latin-1")
        png = (
            _PNG_SIG
            + _minimal_ihdr()
            + _text_chunk("CCV3", encoded)
            + _idat_chunk()
            + _iend_chunk()
        )
        card = load_card(png)
        assert card.name == "CasedCard"


# ---------------------------------------------------------------------------
# Integration test against real Sapphire card (skipped if file absent)
# ---------------------------------------------------------------------------

_SAPPHIRE_PATH = (
    r"C:/Users/User/OneDrive/Documents/Writing/SillyTavern"
    r"/characters/sapphire/Sapphire_0.1.0.card.v3.png"
)


@pytest.mark.skipif(
    not Path(_SAPPHIRE_PATH).exists(),
    reason="Sapphire character card not available on this machine",
)
def test_load_sapphire_card() -> None:
    card = load_card(_SAPPHIRE_PATH)
    assert card.name  # non-empty
    assert isinstance(card, CharacterCard)
