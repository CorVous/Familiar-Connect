"""Tests for the simplified CharacterConfig / TOML loader.

The re-arch demolition pass cut most fields and all three enums
(ChannelMode, Interjection, InterruptTolerance). What remains: TOML
deep-merge over the default profile, a single LLM slot (main_prose),
and the TTS config table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import (
    LLM_SLOT_NAMES,
    ChannelOverrides,
    CharacterConfig,
    ConfigError,
    LLMSlotConfig,
    TTSConfig,
    load_character_config,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLLMSlotNames:
    def test_only_main_prose_slot(self) -> None:
        assert frozenset({"main_prose"}) == LLM_SLOT_NAMES


class TestCharacterConfigDefaults:
    def test_fields(self) -> None:
        cfg = CharacterConfig()
        assert cfg.display_tz == "UTC"
        assert cfg.aliases == []
        assert cfg.history_window_size == 20
        assert cfg.llm == {}
        assert isinstance(cfg.tts, TTSConfig)


class TestLoadCharacterConfig:
    def test_missing_defaults_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text("display_tz = 'UTC'\n")
        with pytest.raises(ConfigError, match="default character profile"):
            load_character_config(path, defaults_path=tmp_path / "missing.toml")

    def test_defaults_only_roundtrip(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.display_tz == "UTC"
        assert "main_prose" in cfg.llm
        assert isinstance(cfg.llm["main_prose"], LLMSlotConfig)

    def test_user_overrides_default(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('display_tz = "America/New_York"\naliases = ["m"]\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.display_tz == "America/New_York"
        assert cfg.aliases == ["m"]

    def test_unknown_llm_slot_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.mystery]\nmodel = "foo"\n')
        with pytest.raises(ConfigError, match="unknown LLM slot"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_temperature_out_of_range(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.main_prose]\nmodel = "m"\ntemperature = 3.0\n')
        with pytest.raises(ConfigError, match="temperature must be in"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_provider_order_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """``provider_order`` pins OpenRouter routing for cache stability.

        Stopgap until OpenRouter routing improves or model swaps; the
        config plumbing is decoupled so the pin can be dropped by
        deleting the line.
        """
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.main_prose]\nmodel = "z-ai/glm-5.1"\n'
            'provider_order = ["z-ai", "deepinfra"]\n'
            "provider_allow_fallbacks = false\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        slot = cfg.llm["main_prose"]
        assert slot.provider_order == ("z-ai", "deepinfra")
        assert slot.provider_allow_fallbacks is False

    def test_provider_order_omitted_means_none(self, tmp_path: Path) -> None:
        """Slot without ``provider_order`` parses to ``None`` (default routing).

        Uses a custom default profile so the shipped default's pin
        (currently ``["z-ai"]``) doesn't bleed in via TOML merge.
        """
        defaults = tmp_path / "defaults.toml"
        defaults.write_text('[llm.main_prose]\nmodel = "x"\n')
        path = tmp_path / "character.toml"
        path.write_text('[llm.main_prose]\nmodel = "m"\n')
        cfg = load_character_config(path, defaults_path=defaults)
        slot = cfg.llm["main_prose"]
        assert slot.provider_order is None
        assert slot.provider_allow_fallbacks is True

    def test_provider_order_must_be_list_of_strings(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.main_prose]\nmodel = "m"\nprovider_order = [1, 2]\n')
        with pytest.raises(ConfigError, match="provider_order"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_tts_provider_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[tts]\nprovider = "mysterybox"\n')
        with pytest.raises(ConfigError, match=r"\[tts\]\.provider"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_history_window_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.history]\nwindow_size = 50\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.history_window_size == 50


class TestChannelOverrides:
    def test_no_channels_section_is_empty_map(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.channels == {}

    def test_channel_overrides_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[channels.12345]\n"
            "history_window_size = 8\n"
            'message_rendering = "name_only"\n'
            'prompt_layers = ["core_instructions", "character_card"]\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        over = cfg.channels[12345]
        assert over.history_window_size == 8
        assert over.message_rendering == "name_only"
        assert over.prompt_layers == ("core_instructions", "character_card")

    def test_window_size_for_falls_back_to_default(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.history]\nwindow_size = 20\n\n"
            "[channels.12345]\nhistory_window_size = 8\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.window_size_for(12345) == 8  # override
        assert cfg.window_size_for(99999) == 20  # default
        assert cfg.window_size_for(None) == 20  # no channel → default

    def test_invalid_window_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[channels.12345]\nhistory_window_size = 0\n")
        with pytest.raises(ConfigError, match="must be positive"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_invalid_message_rendering_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[channels.12345]\nmessage_rendering = "garble"\n')
        with pytest.raises(ConfigError, match="message_rendering"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_channel_overrides_dataclass_default(self) -> None:
        over = ChannelOverrides()
        assert over.history_window_size is None
        assert over.prompt_layers is None
        assert over.message_rendering is None
