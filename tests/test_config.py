"""Tests for the simplified CharacterConfig / TOML loader.

Covers TOML deep-merge over the default profile, the tiered LLM
slots (``fast`` / ``prose`` / ``background``), and the TTS config
table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import (
    LLM_SLOT_NAMES,
    ChannelOverrides,
    CharacterConfig,
    ConfigError,
    DeepgramSTTConfig,
    LLMSlotConfig,
    STTConfig,
    TTSConfig,
    TurnDetectionConfig,
    load_character_config,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLLMSlotNames:
    def test_tiered_slots(self) -> None:
        assert frozenset({"fast", "prose", "background"}) == LLM_SLOT_NAMES


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
        for slot_name in ("fast", "prose", "background"):
            assert slot_name in cfg.llm
            assert isinstance(cfg.llm[slot_name], LLMSlotConfig)

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

    def test_legacy_main_prose_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """The retired ``main_prose`` slot fails loudly post-split."""
        path = tmp_path / "character.toml"
        path.write_text('[llm.main_prose]\nmodel = "m"\n')
        with pytest.raises(ConfigError, match="unknown LLM slot 'main_prose'"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_temperature_out_of_range(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\ntemperature = 3.0\n')
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
            '[llm.prose]\nmodel = "z-ai/glm-5.1"\n'
            'provider_order = ["z-ai", "deepinfra"]\n'
            "provider_allow_fallbacks = false\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        slot = cfg.llm["prose"]
        assert slot.provider_order == ("z-ai", "deepinfra")
        assert slot.provider_allow_fallbacks is False

    def test_provider_order_omitted_means_none(self, tmp_path: Path) -> None:
        """Slot without ``provider_order`` parses to ``None`` (default routing).

        Uses a custom default profile so the shipped default's pin
        doesn't bleed in via TOML merge.
        """
        defaults = tmp_path / "defaults.toml"
        defaults.write_text(
            '[llm.fast]\nmodel = "x"\n'
            '[llm.prose]\nmodel = "x"\n'
            '[llm.background]\nmodel = "x"\n'
        )
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\n')
        cfg = load_character_config(path, defaults_path=defaults)
        slot = cfg.llm["prose"]
        assert slot.provider_order is None
        assert slot.provider_allow_fallbacks is True

    def test_provider_order_must_be_list_of_strings(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\nprovider_order = [1, 2]\n')
        with pytest.raises(ConfigError, match="provider_order"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_reasoning_levels_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.fast]\nmodel = "m"\nreasoning = "off"\n'
            '[llm.prose]\nmodel = "m"\nreasoning = "medium"\n'
            '[llm.background]\nmodel = "m"\nreasoning = "high"\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.llm["fast"].reasoning == "off"
        assert cfg.llm["prose"].reasoning == "medium"
        assert cfg.llm["background"].reasoning == "high"

    def test_reasoning_omitted_means_none(self, tmp_path: Path) -> None:
        defaults = tmp_path / "defaults.toml"
        defaults.write_text(
            '[llm.fast]\nmodel = "x"\n'
            '[llm.prose]\nmodel = "x"\n'
            '[llm.background]\nmodel = "x"\n'
        )
        cfg = load_character_config(tmp_path / "missing.toml", defaults_path=defaults)
        assert cfg.llm["prose"].reasoning is None

    def test_invalid_reasoning_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\nreasoning = "ultra"\n')
        with pytest.raises(ConfigError, match="reasoning"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_reasoning_must_be_string(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\nreasoning = true\n')
        with pytest.raises(ConfigError, match="reasoning"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_tool_calling_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.background]\nmodel = "m"\ntool_calling = true\n'
            '[llm.fast]\nmodel = "m"\ntool_calling = false\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.llm["background"].tool_calling is True
        assert cfg.llm["fast"].tool_calling is False

    def test_tool_calling_omitted_defaults_false(self, tmp_path: Path) -> None:
        defaults = tmp_path / "defaults.toml"
        defaults.write_text(
            '[llm.fast]\nmodel = "x"\n'
            '[llm.prose]\nmodel = "x"\n'
            '[llm.background]\nmodel = "x"\n'
        )
        cfg = load_character_config(tmp_path / "missing.toml", defaults_path=defaults)
        assert cfg.llm["prose"].tool_calling is False

    def test_tool_calling_must_be_bool(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\ntool_calling = "yes"\n')
        with pytest.raises(ConfigError, match="tool_calling"):
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


class TestTurnDetectionConfig:
    def test_default_strategy_is_deepgram(self) -> None:
        cfg = TurnDetectionConfig()
        assert cfg.strategy == "deepgram"

    def test_omitted_section_uses_default(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.turn_detection.strategy == "deepgram"

    def test_ten_plus_smart_turn_strategy_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.turn_detection]\nstrategy = "ten+smart_turn"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.turn_detection.strategy == "ten+smart_turn"

    def test_deepgram_strategy_explicit(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.turn_detection]\nstrategy = "deepgram"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.turn_detection.strategy == "deepgram"

    def test_unknown_strategy_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.turn_detection]\nstrategy = "mystery"\n')
        match = r"\[providers\.turn_detection\]\.strategy"
        with pytest.raises(ConfigError, match=match):
            load_character_config(path, defaults_path=default_profile_path)

    def test_strategy_must_be_string(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.turn_detection]\nstrategy = 42\n")
        match = r"\[providers\.turn_detection\]\.strategy"
        with pytest.raises(ConfigError, match=match):
            load_character_config(path, defaults_path=default_profile_path)


class TestSTTConfig:
    def test_default_backend_is_deepgram(self) -> None:
        cfg = STTConfig()
        assert cfg.backend == "deepgram"
        assert isinstance(cfg.deepgram, DeepgramSTTConfig)

    def test_deepgram_defaults_match_shipped_values(self) -> None:
        """Defaults mirror the previous DEEPGRAM_* env var defaults."""
        cfg = DeepgramSTTConfig()
        assert cfg.model == "nova-3"
        assert cfg.language == "en"
        assert cfg.endpointing_ms == 500
        assert cfg.utterance_end_ms == 1500
        assert cfg.smart_format is True
        assert cfg.punctuate is True
        assert cfg.keyterms == ()
        assert cfg.replay_buffer_s == pytest.approx(5.0)
        assert cfg.keepalive_interval_s == pytest.approx(3.0)
        assert cfg.reconnect_max_attempts == 5
        assert cfg.reconnect_backoff_cap_s == pytest.approx(16.0)
        assert cfg.idle_close_s == pytest.approx(30.0)

    def test_omitted_section_uses_defaults(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.stt.backend == "deepgram"
        assert cfg.stt.deepgram.endpointing_ms == 500
        assert cfg.stt.deepgram.utterance_end_ms == 1500
        assert cfg.stt.deepgram.keyterms == ()

    def test_deepgram_knobs_overridden(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.stt.deepgram]\n"
            'model = "nova-2"\n'
            'language = "es"\n'
            "endpointing_ms = 300\n"
            "utterance_end_ms = 1200\n"
            "smart_format = false\n"
            "punctuate = false\n"
            'keyterms = ["lifecycle mesh", "Tam"]\n'
            "replay_buffer_s = 7.5\n"
            "keepalive_interval_s = 2.0\n"
            "reconnect_max_attempts = 8\n"
            "reconnect_backoff_cap_s = 32.0\n"
            "idle_close_s = 45.0\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        dg = cfg.stt.deepgram
        assert dg.model == "nova-2"
        assert dg.language == "es"
        assert dg.endpointing_ms == 300
        assert dg.utterance_end_ms == 1200
        assert dg.smart_format is False
        assert dg.punctuate is False
        assert dg.keyterms == ("lifecycle mesh", "Tam")
        assert dg.replay_buffer_s == pytest.approx(7.5)
        assert dg.keepalive_interval_s == pytest.approx(2.0)
        assert dg.reconnect_max_attempts == 8
        assert dg.reconnect_backoff_cap_s == pytest.approx(32.0)
        assert dg.idle_close_s == pytest.approx(45.0)

    def test_unknown_backend_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.stt]\nbackend = "mystery"\n')
        match = r"\[providers\.stt\]\.backend"
        with pytest.raises(ConfigError, match=match):
            load_character_config(path, defaults_path=default_profile_path)

    def test_backend_must_be_string(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.stt]\nbackend = 42\n")
        match = r"\[providers\.stt\]\.backend"
        with pytest.raises(ConfigError, match=match):
            load_character_config(path, defaults_path=default_profile_path)

    def test_endpointing_ms_must_be_int(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.stt.deepgram]\nendpointing_ms = 1.5\n")
        match = r"endpointing_ms"
        with pytest.raises(ConfigError, match=match):
            load_character_config(path, defaults_path=default_profile_path)

    def test_keyterms_must_be_list_of_strings(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.stt.deepgram]\nkeyterms = [1, 2]\n")
        match = r"keyterms"
        with pytest.raises(ConfigError, match=match):
            load_character_config(path, defaults_path=default_profile_path)
