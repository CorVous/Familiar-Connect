"""Red-first tests for the per-character / per-channel config loader.

Step 7 of docs/architecture/context-pipeline.md. The bot reads a
character's TOML sidecar at startup and a per-channel TOML sidecar
lazily on first use. Both are optional — missing files produce sane
defaults so a brand-new install with an empty ``data/`` works.

Covers familiar_connect.config, which doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import (
    DEFAULT_CHANNEL_MODE,
    ChannelConfig,
    ChannelMode,
    CharacterConfig,
    ConfigError,
    channel_config_for_mode,
    load_channel_config,
    load_character_config,
)
from familiar_connect.context.types import Layer

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# ChannelMode enum
# ---------------------------------------------------------------------------


class TestChannelMode:
    def test_has_full_rp(self) -> None:
        assert ChannelMode.full_rp.value == "full_rp"

    def test_has_text_conversation_rp(self) -> None:
        assert ChannelMode.text_conversation_rp.value == "text_conversation_rp"

    def test_has_imitate_voice(self) -> None:
        assert ChannelMode.imitate_voice.value == "imitate_voice"

    def test_from_string_recognises_all_modes(self) -> None:
        assert ChannelMode("full_rp") is ChannelMode.full_rp
        assert ChannelMode("text_conversation_rp") is ChannelMode.text_conversation_rp
        assert ChannelMode("imitate_voice") is ChannelMode.imitate_voice


# ---------------------------------------------------------------------------
# Mode defaults
# ---------------------------------------------------------------------------


class TestModeDefaults:
    def test_full_rp_enables_all_providers(self) -> None:
        cfg = channel_config_for_mode(ChannelMode.full_rp)
        assert "character" in cfg.providers_enabled
        assert "history" in cfg.providers_enabled
        assert "content_search" in cfg.providers_enabled

    def test_full_rp_enables_both_processors(self) -> None:
        cfg = channel_config_for_mode(ChannelMode.full_rp)
        assert "stepped_thinking" in cfg.preprocessors_enabled
        assert "recast" in cfg.postprocessors_enabled

    def test_text_conversation_rp_omits_content_search(self) -> None:
        cfg = channel_config_for_mode(ChannelMode.text_conversation_rp)
        # Content search IS on in text_conversation_rp so lore files work
        # in casual text chat; only imitate_voice drops it for TTFB.
        assert "content_search" in cfg.providers_enabled
        assert "character" in cfg.providers_enabled
        assert "history" in cfg.providers_enabled
        assert cfg.budget_by_layer.get(Layer.content, 0) == 2000

    def test_imitate_voice_drops_content_search_for_latency(self) -> None:
        """Voice TTFB is too tight to carry the content-search agent.

        The extra 2-5 side-model calls per turn would blow the voice
        reply budget, so ``imitate_voice`` drops the provider entirely.
        """
        cfg = channel_config_for_mode(ChannelMode.imitate_voice)
        assert "content_search" not in cfg.providers_enabled
        assert cfg.budget_by_layer.get(Layer.content, 0) == 0

    def test_imitate_voice_drops_stepped_thinking_for_latency(self) -> None:
        cfg = channel_config_for_mode(ChannelMode.imitate_voice)
        assert "stepped_thinking" not in cfg.preprocessors_enabled
        # Recast stays on for the voice-flavour rewrite.
        assert "recast" in cfg.postprocessors_enabled

    def test_imitate_voice_has_tighter_budget_than_full_rp(self) -> None:
        voice = channel_config_for_mode(ChannelMode.imitate_voice)
        full = channel_config_for_mode(ChannelMode.full_rp)
        assert voice.budget_tokens < full.budget_tokens
        assert voice.deadline_s <= full.deadline_s

    def test_every_mode_enables_mode_instructions_provider(self) -> None:
        """Every built-in mode ships the per-mode instruction knob."""
        for mode in ChannelMode:
            cfg = channel_config_for_mode(mode)
            assert "mode_instructions" in cfg.providers_enabled, (
                f"{mode.value} must enable mode_instructions"
            )

    def test_every_mode_budgets_the_author_note_layer(self) -> None:
        """Guard against a mode that forgets its own instruction budget.

        ``author_note`` is where :class:`ModeInstructionProvider`
        lands — a mode that doesn't budget for the layer would
        silently drop its own instruction file.
        """
        for mode in ChannelMode:
            cfg = channel_config_for_mode(mode)
            assert cfg.budget_by_layer.get(Layer.author_note, 0) > 0, (
                f"{mode.value} must allocate budget for Layer.author_note"
            )


# ---------------------------------------------------------------------------
# load_character_config
# ---------------------------------------------------------------------------


class TestLoadCharacterConfig:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_character_config(tmp_path / "does-not-exist.toml")
        assert isinstance(cfg, CharacterConfig)
        assert cfg.default_mode is DEFAULT_CHANNEL_MODE

    def test_reads_default_mode_from_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text('default_mode = "full_rp"\n')
        cfg = load_character_config(path)
        assert cfg.default_mode is ChannelMode.full_rp

    def test_reads_history_window_size(self, tmp_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.history]\nwindow_size = 42\n",
        )
        cfg = load_character_config(path)
        assert cfg.history_window_size == 42

    def test_reads_depth_inject_position(self, tmp_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[layers.depth_inject]\nposition = 4\nrole = "user"\n',
        )
        cfg = load_character_config(path)
        assert cfg.depth_inject_position == 4
        assert cfg.depth_inject_role == "user"

    def test_unknown_mode_string_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text('default_mode = "chaos"\n')
        with pytest.raises(ConfigError, match="chaos"):
            load_character_config(path)

    def test_malformed_toml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text("this is = = not toml\n")
        with pytest.raises(ConfigError):
            load_character_config(path)


# ---------------------------------------------------------------------------
# load_channel_config
# ---------------------------------------------------------------------------


class TestLoadChannelConfig:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_channel_config(tmp_path / "missing.toml") is None

    def test_reads_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "channel.toml"
        path.write_text('mode = "imitate_voice"\n')
        cfg = load_channel_config(path)
        assert isinstance(cfg, ChannelConfig)
        assert cfg.mode is ChannelMode.imitate_voice

    def test_mode_drives_provider_set(self, tmp_path: Path) -> None:
        path = tmp_path / "channel.toml"
        path.write_text('mode = "full_rp"\n')
        cfg = load_channel_config(path)
        assert cfg is not None
        assert "content_search" in cfg.providers_enabled

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "channel.toml"
        path.write_text('mode = "nonsense"\n')
        with pytest.raises(ConfigError):
            load_channel_config(path)
