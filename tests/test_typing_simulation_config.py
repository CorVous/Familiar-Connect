"""Tests for ``TypingSimulationConfig`` defaults, TOML parse, and layering.

Layering precedence (low → high):
    channel_config_for_mode(mode).typing_simulation   # per-mode default
    → character.toml [typing_simulation]              # familiar-wide override
    → channels/<id>.toml [typing_simulation]          # per-channel override
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.channel_config import ChannelConfigStore
from familiar_connect.config import (
    ChannelMode,
    ConfigError,
    TypingSimulationConfig,
    channel_config_for_mode,
    load_channel_config,
    load_character_config,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestTypingSimulationDefaults:
    def test_default_is_disabled(self) -> None:
        cfg = TypingSimulationConfig()
        assert cfg.enabled is False

    def test_has_chars_per_second(self) -> None:
        cfg = TypingSimulationConfig()
        assert cfg.chars_per_second > 0

    def test_min_less_than_or_equal_max(self) -> None:
        cfg = TypingSimulationConfig()
        assert cfg.min_delay_s <= cfg.max_delay_s

    def test_frozen(self) -> None:
        cfg = TypingSimulationConfig()
        with pytest.raises(Exception):  # noqa: B017, PT011 — FrozenInstanceError
            cfg.enabled = True  # type: ignore[misc]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Per-mode defaults in channel_config_for_mode
# ---------------------------------------------------------------------------


class TestPerModeDefaults:
    def test_text_conversation_rp_enables_typing_simulation(self) -> None:
        cfg = channel_config_for_mode(ChannelMode.text_conversation_rp)
        assert cfg.typing_simulation.enabled is True

    def test_full_rp_disables_typing_simulation(self) -> None:
        cfg = channel_config_for_mode(ChannelMode.full_rp)
        assert cfg.typing_simulation.enabled is False

    def test_imitate_voice_disables_typing_simulation(self) -> None:
        cfg = channel_config_for_mode(ChannelMode.imitate_voice)
        assert cfg.typing_simulation.enabled is False


# ---------------------------------------------------------------------------
# Character-level overrides
# ---------------------------------------------------------------------------


class TestCharacterLevelOverrides:
    def test_absent_section_yields_empty_overrides(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        # empty dict — no overrides
        assert cfg.typing_simulation_overrides == {}

    def test_reads_enabled_override(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[typing_simulation]\nenabled = false\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.typing_simulation_overrides == {"enabled": False}

    def test_reads_all_fields(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[typing_simulation]\n"
            "enabled = true\n"
            "chars_per_second = 25.0\n"
            "min_delay_s = 1.0\n"
            "max_delay_s = 10.0\n"
            "inter_line_pause_s = 1.5\n"
            "sentence_split_threshold = 300\n",
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.typing_simulation_overrides == {
            "enabled": True,
            "chars_per_second": 25.0,
            "min_delay_s": 1.0,
            "max_delay_s": 10.0,
            "inter_line_pause_s": 1.5,
            "sentence_split_threshold": 300,
        }

    def test_wrong_type_for_enabled_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[typing_simulation]\nenabled = "yes"\n')

        with pytest.raises(ConfigError, match="enabled"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_negative_chars_per_second_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[typing_simulation]\nchars_per_second = -5\n")

        with pytest.raises(ConfigError, match="chars_per_second"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_min_greater_than_max_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[typing_simulation]\nmin_delay_s = 5.0\nmax_delay_s = 1.0\n",
        )

        with pytest.raises(ConfigError, match="min_delay_s"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_negative_pause_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[typing_simulation]\ninter_line_pause_s = -1.0\n")

        with pytest.raises(ConfigError, match="inter_line_pause_s"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_non_table_section_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('typing_simulation = "oops"\n')

        with pytest.raises(ConfigError, match="typing_simulation"):
            load_character_config(path, defaults_path=default_profile_path)


# ---------------------------------------------------------------------------
# Channel-level overrides (load_channel_config)
# ---------------------------------------------------------------------------


class TestChannelLevelOverrides:
    def test_mode_only_sidecar_keeps_mode_default(self, tmp_path: Path) -> None:
        path = tmp_path / "42.toml"
        path.write_text('mode = "text_conversation_rp"\n')
        cfg = load_channel_config(path)
        assert cfg is not None
        assert cfg.typing_simulation.enabled is True  # text-rp default

    def test_channel_override_disables(self, tmp_path: Path) -> None:
        path = tmp_path / "42.toml"
        path.write_text(
            'mode = "text_conversation_rp"\n\n[typing_simulation]\nenabled = false\n',
        )
        cfg = load_channel_config(path)
        assert cfg is not None
        assert cfg.typing_simulation.enabled is False

    def test_channel_override_enables_in_full_rp(self, tmp_path: Path) -> None:
        path = tmp_path / "42.toml"
        path.write_text(
            'mode = "full_rp"\n\n[typing_simulation]\nenabled = true\n',
        )
        cfg = load_channel_config(path)
        assert cfg is not None
        # full_rp default is disabled; channel opt-in overrides
        assert cfg.typing_simulation.enabled is True

    def test_channel_partial_override_preserves_other_fields(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "42.toml"
        path.write_text(
            'mode = "text_conversation_rp"\n\n'
            "[typing_simulation]\nchars_per_second = 20.0\n",
        )
        cfg = load_channel_config(path)
        assert cfg is not None
        # still enabled (mode default preserved)
        assert cfg.typing_simulation.enabled is True
        assert cfg.typing_simulation.chars_per_second == pytest.approx(20.0)

    def test_channel_invalid_cps_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "42.toml"
        path.write_text(
            'mode = "text_conversation_rp"\n\n'
            "[typing_simulation]\nchars_per_second = 0\n",
        )

        with pytest.raises(ConfigError, match="chars_per_second"):
            load_channel_config(path)


# ---------------------------------------------------------------------------
# Full layering via ChannelConfigStore
# ---------------------------------------------------------------------------


class TestStoreLayering:
    def test_store_applies_character_overrides_for_mode_default_channel(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        # character.toml disables typing sim family-wide
        char_path = tmp_path / "character.toml"
        char_path.write_text(
            "[typing_simulation]\nenabled = false\n",
        )
        character = load_character_config(
            char_path,
            defaults_path=default_profile_path,
        )
        # no channel sidecar → fall through to character.default_mode
        channels_dir = tmp_path / "channels"
        channels_dir.mkdir()
        store = ChannelConfigStore(root=channels_dir, character=character)

        cfg = store.get(channel_id=123)
        assert cfg.typing_simulation.enabled is False

    def test_store_layers_character_then_channel(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        # character sets cps=20, channel sets min_delay=2.0 → both present
        char_path = tmp_path / "character.toml"
        char_path.write_text(
            "[typing_simulation]\nchars_per_second = 20.0\n",
        )
        character = load_character_config(
            char_path,
            defaults_path=default_profile_path,
        )

        channels_dir = tmp_path / "channels"
        channels_dir.mkdir()
        (channels_dir / "42.toml").write_text(
            'mode = "text_conversation_rp"\n\n[typing_simulation]\nmin_delay_s = 2.0\n',
        )
        store = ChannelConfigStore(root=channels_dir, character=character)

        cfg = store.get(channel_id=42)
        ts = cfg.typing_simulation
        assert ts.enabled is True  # text-rp mode default
        assert ts.chars_per_second == pytest.approx(20.0)  # character override
        assert ts.min_delay_s == pytest.approx(2.0)  # channel override

    def test_channel_override_beats_character_override(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        # character says disabled; channel re-enables
        char_path = tmp_path / "character.toml"
        char_path.write_text("[typing_simulation]\nenabled = false\n")
        character = load_character_config(
            char_path,
            defaults_path=default_profile_path,
        )

        channels_dir = tmp_path / "channels"
        channels_dir.mkdir()
        (channels_dir / "7.toml").write_text(
            'mode = "text_conversation_rp"\n\n[typing_simulation]\nenabled = true\n',
        )
        store = ChannelConfigStore(root=channels_dir, character=character)

        cfg = store.get(channel_id=7)
        assert cfg.typing_simulation.enabled is True
