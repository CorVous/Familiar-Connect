"""Tests for the SillyTavern preset loader and prompt assembler."""

import json
from pathlib import Path

import pytest

from familiar_connect.character import CharacterCard, load_card
from familiar_connect.preset import PresetError, assemble_prompt, load_preset

# ---------------------------------------------------------------------------
# Minimal preset structures for unit tests
# ---------------------------------------------------------------------------


def _make_minimal_preset(
    prompts: list[dict],
    order: list[dict],
    character_id: int = 100001,
) -> dict:
    """Build a minimal preset dict matching SillyTavern's JSON format."""
    return {
        "prompts": prompts,
        "prompt_order": [{"character_id": character_id, "order": order}],
    }


_CARD = CharacterCard(
    name="Aria",
    description="A celestial being.",
    personality="Warm and thoughtful.",
    scenario="A moonlit rooftop.",
    first_mes="Hello, wanderer.",
    mes_example="<START>\n{{user}}: Hi\n{{char}}: Greetings!",
    system_prompt="",
    post_history_instructions="",
    creator_notes="",
)


# ---------------------------------------------------------------------------
# load_preset
# ---------------------------------------------------------------------------


class TestLoadPreset:
    def test_load_from_dict_returns_same(self) -> None:
        preset = _make_minimal_preset([], [])
        result = load_preset(preset)
        assert result is preset

    def test_load_from_path(self, tmp_path: Path) -> None:
        preset = _make_minimal_preset([], [])
        p = tmp_path / "preset.json"
        p.write_text(json.dumps(preset), encoding="utf-8")
        result = load_preset(str(p))
        assert result["prompts"] == []

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(PresetError):
            load_preset("/nonexistent/path/preset.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(PresetError):
            load_preset(str(p))


# ---------------------------------------------------------------------------
# assemble_prompt — basic
# ---------------------------------------------------------------------------


class TestAssemblePromptBasic:
    def test_empty_order_returns_empty(self) -> None:
        preset = _make_minimal_preset([], [])
        result = assemble_prompt(preset, _CARD)
        assert not result

    def test_disabled_entry_skipped(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "abc",
                    "system_prompt": False,
                    "marker": False,
                    "content": "Should not appear",
                    "role": "system",
                }
            ],
            order=[{"identifier": "abc", "enabled": False}],
        )
        result = assemble_prompt(preset, _CARD)
        assert not result

    def test_custom_prompt_content_included(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "abc",
                    "system_prompt": False,
                    "marker": False,
                    "content": "You are a helpful assistant.",
                    "role": "system",
                }
            ],
            order=[{"identifier": "abc", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert result == "You are a helpful assistant."

    def test_multiple_sections_joined(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "a",
                    "system_prompt": False,
                    "marker": False,
                    "content": "First.",
                    "role": "system",
                },
                {
                    "identifier": "b",
                    "system_prompt": False,
                    "marker": False,
                    "content": "Second.",
                    "role": "system",
                },
            ],
            order=[
                {"identifier": "a", "enabled": True},
                {"identifier": "b", "enabled": True},
            ],
        )
        result = assemble_prompt(preset, _CARD)
        assert result == "First.\n\nSecond."

    def test_unknown_identifier_skipped(self) -> None:
        preset = _make_minimal_preset(
            prompts=[],
            order=[{"identifier": "not-in-prompts", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert not result


# ---------------------------------------------------------------------------
# assemble_prompt — marker substitution
# ---------------------------------------------------------------------------


class TestMarkerSubstitution:
    def test_char_description_marker(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "charDescription",
                    "system_prompt": True,
                    "marker": True,
                    "role": "system",
                    "content": "",
                }
            ],
            order=[{"identifier": "charDescription", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert result == _CARD.description

    def test_char_personality_marker(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "charPersonality",
                    "system_prompt": True,
                    "marker": True,
                    "role": "system",
                    "content": "",
                }
            ],
            order=[{"identifier": "charPersonality", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert result == _CARD.personality

    def test_scenario_marker(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "scenario",
                    "system_prompt": True,
                    "marker": True,
                    "role": "system",
                    "content": "",
                }
            ],
            order=[{"identifier": "scenario", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert result == _CARD.scenario

    def test_dialogue_examples_marker(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "dialogueExamples",
                    "system_prompt": True,
                    "marker": True,
                    "role": "system",
                    "content": "",
                }
            ],
            order=[{"identifier": "dialogueExamples", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        # mes_example macros resolved with char name
        assert "Greetings!" in result

    def test_chat_history_marker_skipped(self) -> None:
        """The chatHistory marker is skipped — caller provides history as messages."""
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "chatHistory",
                    "system_prompt": True,
                    "marker": True,
                    "role": "system",
                    "content": "",
                }
            ],
            order=[{"identifier": "chatHistory", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert not result

    def test_world_info_markers_produce_empty(self) -> None:
        for ident in ("worldInfoBefore", "worldInfoAfter"):
            preset = _make_minimal_preset(
                prompts=[
                    {
                        "identifier": ident,
                        "system_prompt": True,
                        "marker": True,
                        "role": "system",
                        "content": "",
                    }
                ],
                order=[{"identifier": ident, "enabled": True}],
            )
            result = assemble_prompt(preset, _CARD)
            assert not result, f"Expected empty for {ident}, got {result!r}"

    def test_empty_marker_value_not_included(self) -> None:
        card = CharacterCard(name="X", scenario="")  # empty scenario
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "scenario",
                    "system_prompt": True,
                    "marker": True,
                    "role": "system",
                    "content": "",
                }
            ],
            order=[{"identifier": "scenario", "enabled": True}],
        )
        result = assemble_prompt(preset, card)
        assert not result


# ---------------------------------------------------------------------------
# assemble_prompt — macro resolution
# ---------------------------------------------------------------------------


class TestMacroResolution:
    def test_char_macro_resolved_in_content(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "x",
                    "system_prompt": False,
                    "marker": False,
                    "content": "You are {{char}}.",
                    "role": "system",
                }
            ],
            order=[{"identifier": "x", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert result == "You are Aria."

    def test_comment_stripped_and_trim_applied(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "x",
                    "system_prompt": False,
                    "marker": False,
                    "content": "{{// comment }}Hello{{trim}}",
                    "role": "system",
                }
            ],
            order=[{"identifier": "x", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert result == "Hello"

    def test_comment_only_content_excluded(self) -> None:
        preset = _make_minimal_preset(
            prompts=[
                {
                    "identifier": "x",
                    "system_prompt": False,
                    "marker": False,
                    "content": "{{// Empty for card override. }}{{trim}}",
                    "role": "system",
                }
            ],
            order=[{"identifier": "x", "enabled": True}],
        )
        result = assemble_prompt(preset, _CARD)
        assert not result


# ---------------------------------------------------------------------------
# Character order fallback
# ---------------------------------------------------------------------------


class TestCharacterOrderSelection:
    def test_uses_specified_character_id(self) -> None:
        preset = {
            "prompts": [
                {
                    "identifier": "a",
                    "system_prompt": False,
                    "marker": False,
                    "content": "From 99999.",
                    "role": "system",
                }
            ],
            "prompt_order": [
                {
                    "character_id": 99999,
                    "order": [{"identifier": "a", "enabled": True}],
                },
                {"character_id": 100001, "order": []},
            ],
        }
        result = assemble_prompt(preset, _CARD, character_id=99999)
        assert result == "From 99999."

    def test_falls_back_to_first_order_if_id_not_found(self) -> None:
        preset = {
            "prompts": [
                {
                    "identifier": "a",
                    "system_prompt": False,
                    "marker": False,
                    "content": "Fallback.",
                    "role": "system",
                }
            ],
            "prompt_order": [
                {
                    "character_id": 100000,
                    "order": [{"identifier": "a", "enabled": True}],
                },
            ],
        }
        result = assemble_prompt(preset, _CARD, character_id=999)
        assert result == "Fallback."


# ---------------------------------------------------------------------------
# Integration test against the real marinara preset + Sapphire card
# ---------------------------------------------------------------------------

_PRESET_PATH = (
    r"C:/Users/User/OneDrive/Documents/Writing/SillyTavern"
    r"/presets/marinara_spaghetti_recipe_safe.json"
)
_CARD_PATH = (
    r"C:/Users/User/OneDrive/Documents/Writing/SillyTavern"
    r"/characters/sapphire/Sapphire_0.1.0.card.v3.png"
)


@pytest.mark.skipif(
    not (Path(_PRESET_PATH).exists() and Path(_CARD_PATH).exists()),
    reason="Real preset/card files not available on this machine",
)
def test_marinara_with_sapphire_card() -> None:
    card = load_card(_CARD_PATH)
    preset = load_preset(_PRESET_PATH)
    result = assemble_prompt(preset, card)

    assert isinstance(result, str)
    assert result
    # The card's name should appear somewhere (via {{char}} macro or description)
    assert card.name in result or card.description[:20] in result
