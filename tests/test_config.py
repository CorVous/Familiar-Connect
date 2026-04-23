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
