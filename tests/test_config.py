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
    LLM_SLOT_NAMES,
    ChannelConfig,
    ChannelMode,
    CharacterConfig,
    ConfigError,
    DynamicLullStrategy,
    Interjection,
    InterruptTolerance,
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
    def test_missing_file_returns_defaults(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        cfg = load_character_config(
            tmp_path / "does-not-exist.toml",
            defaults_path=default_profile_path,
        )
        assert isinstance(cfg, CharacterConfig)
        assert cfg.default_mode is DEFAULT_CHANNEL_MODE

    def test_reads_default_mode_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('default_mode = "full_rp"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.default_mode is ChannelMode.full_rp

    def test_reads_history_window_size(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.history]\nwindow_size = 42\n",
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.history_window_size == 42

    def test_reads_depth_inject_position(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[layers.depth_inject]\nposition = 4\nrole = "user"\n',
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.depth_inject_position == 4
        assert cfg.depth_inject_role == "user"

    def test_unknown_mode_string_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('default_mode = "chaos"\n')
        with pytest.raises(ConfigError, match="chaos"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_malformed_toml_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("this is = = not toml\n")
        with pytest.raises(ConfigError):
            load_character_config(path, defaults_path=default_profile_path)

    def test_missing_defaults_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "character.toml"
        target.write_text("")
        with pytest.raises(ConfigError, match="default character profile"):
            load_character_config(
                target,
                defaults_path=tmp_path / "missing-defaults.toml",
            )

    def test_user_slot_override_wins_over_defaults(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.main_prose]\nmodel = "user/custom-model"\ntemperature = 0.9\n',
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        # Overridden slot reflects the user value.
        assert cfg.llm["main_prose"].model == "user/custom-model"
        assert cfg.llm["main_prose"].temperature == 0.9  # noqa: RUF069
        # Untouched slots still come from the default profile.
        assert "reasoning_context" in cfg.llm
        assert cfg.llm["reasoning_context"].model

    def test_unknown_llm_slot_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.nonexistent]\nmodel = "foo/bar"\n')
        with pytest.raises(ConfigError, match="unknown LLM slot"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_llm_slot_requires_non_empty_model(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.main_prose]\nmodel = ""\n')
        with pytest.raises(ConfigError, match="must be a non-empty string"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_llm_slot_temperature_out_of_range_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.main_prose]\nmodel = "user/m"\ntemperature = 5.0\n',
        )
        with pytest.raises(ConfigError, match=r"temperature must be in"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_tts_section_parsed(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[tts]\nvoice_id = "user-voice"\nmodel = "sonic-4"\n',
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.tts.voice_id == "user-voice"
        assert cfg.tts.model == "sonic-4"

    def test_defaults_populate_every_llm_slot(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        """An empty user file must still produce a fully-populated llm dict."""
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert set(cfg.llm.keys()) == set(LLM_SLOT_NAMES)
        for slot in LLM_SLOT_NAMES:
            assert cfg.llm[slot].model  # non-empty


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


# ---------------------------------------------------------------------------
# Interjection enum
# ---------------------------------------------------------------------------


class TestInterjectionEnum:
    def test_has_five_tiers(self) -> None:
        assert len(list(Interjection)) == 5

    def test_tier_values(self) -> None:
        assert Interjection("very_quiet") is Interjection.very_quiet
        assert Interjection("quiet") is Interjection.quiet
        assert Interjection("average") is Interjection.average
        assert Interjection("eager") is Interjection.eager
        assert Interjection("very_eager") is Interjection.very_eager

    def test_starting_intervals(self) -> None:
        assert Interjection.very_quiet.starting_interval == 15
        assert Interjection.quiet.starting_interval == 12
        assert Interjection.average.starting_interval == 9
        assert Interjection.eager.starting_interval == 6
        assert Interjection.very_eager.starting_interval == 3


# ---------------------------------------------------------------------------
# CharacterConfig conversation-flow fields
# ---------------------------------------------------------------------------


class TestCharacterConfigConversationFields:
    def test_defaults_when_file_absent(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        cfg = load_character_config(
            tmp_path / "no-such-file.toml",
            defaults_path=default_profile_path,
        )
        assert cfg.aliases == []
        assert cfg.chattiness == "Balanced — responds when the conversation is relevant"
        assert cfg.interjection is Interjection.average
        assert cfg.text_lull_timeout == 10.0  # noqa: RUF069

    def test_reads_aliases_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('aliases = ["aria", "ari"]\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.aliases == ["aria", "ari"]

    def test_empty_aliases_list(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("aliases = []\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.aliases == []

    def test_reads_chattiness_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('chattiness = "Shy and reserved"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.chattiness == "Shy and reserved"

    def test_reads_interjection_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('interjection = "eager"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.interjection is Interjection.eager

    def test_reads_text_lull_timeout_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("text_lull_timeout = 5.0\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.text_lull_timeout == 5.0  # noqa: RUF069

    def test_unknown_interjection_value_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('interjection = "obnoxious"\n')
        with pytest.raises(ConfigError, match="obnoxious"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_all_interjection_tiers_load_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        for tier in Interjection:
            path = tmp_path / f"character_{tier.value}.toml"
            path.write_text(f'interjection = "{tier.value}"\n')
            cfg = load_character_config(path, defaults_path=default_profile_path)
            assert cfg.interjection is tier


# InterruptTolerance enum and [voice.interruption] TOML section
# ---------------------------------------------------------------------------


class TestInterruptToleranceEnum:
    def test_has_five_tiers(self) -> None:
        assert len(list(InterruptTolerance)) == 5

    def test_tier_values(self) -> None:
        assert InterruptTolerance("very_meek") is InterruptTolerance.very_meek
        assert InterruptTolerance("meek") is InterruptTolerance.meek
        assert InterruptTolerance("average") is InterruptTolerance.average
        assert InterruptTolerance("stubborn") is InterruptTolerance.stubborn
        assert InterruptTolerance("very_stubborn") is InterruptTolerance.very_stubborn

    def test_base_probabilities(self) -> None:
        assert InterruptTolerance.very_meek.base_probability == 0.10  # noqa: RUF069
        assert InterruptTolerance.meek.base_probability == 0.20  # noqa: RUF069
        assert InterruptTolerance.average.base_probability == 0.30  # noqa: RUF069
        assert InterruptTolerance.stubborn.base_probability == 0.45  # noqa: RUF069
        assert InterruptTolerance.very_stubborn.base_probability == 0.60  # noqa: RUF069


class TestVoiceInterruptionConfig:
    def test_defaults_when_section_absent(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        cfg = load_character_config(
            tmp_path / "no-such-file.toml",
            defaults_path=default_profile_path,
        )
        assert cfg.interrupt_tolerance is InterruptTolerance.average
        assert cfg.min_interruption_s == 2.0  # noqa: RUF069
        assert cfg.short_long_boundary_s == 30.0  # noqa: RUF069

    def test_reads_tolerance_tier(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[voice.interruption]\ninterrupt_tolerance = "stubborn"\n',
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.interrupt_tolerance is InterruptTolerance.stubborn

    def test_reads_min_interruption(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[voice.interruption]\nmin_interruption_s = 2.25\n",
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.min_interruption_s == 2.25  # noqa: RUF069

    def test_reads_short_long_boundary(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[voice.interruption]\nshort_long_boundary_s = 6.0\n",
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.short_long_boundary_s == 6.0  # noqa: RUF069

    def test_unknown_tolerance_tier_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[voice.interruption]\ninterrupt_tolerance = "defiant"\n',
        )
        with pytest.raises(ConfigError, match="defiant"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_all_tolerance_tiers_load_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        for tier in InterruptTolerance:
            path = tmp_path / f"character_{tier.value}.toml"
            path.write_text(
                f'[voice.interruption]\ninterrupt_tolerance = "{tier.value}"\n',
            )
            cfg = load_character_config(path, defaults_path=default_profile_path)
            assert cfg.interrupt_tolerance is tier

    def test_negative_min_interruption_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[voice.interruption]\nmin_interruption_s = -1.0\n",
        )
        with pytest.raises(ConfigError, match="min_interruption_s"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_boundary_must_exceed_min(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[voice.interruption]\n"
            "min_interruption_s = 3.0\n"
            "short_long_boundary_s = 2.0\n",
        )
        with pytest.raises(ConfigError, match="short_long_boundary_s"):
            load_character_config(path, defaults_path=default_profile_path)


# ---------------------------------------------------------------------------
# CharacterConfig memory-writer fields
# ---------------------------------------------------------------------------


class TestCharacterConfigMemoryWriterFields:
    def test_defaults_when_file_absent(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        cfg = load_character_config(
            tmp_path / "no-such-file.toml",
            defaults_path=default_profile_path,
        )
        assert cfg.memory_writer_turn_threshold == 50
        assert cfg.memory_writer_idle_timeout == 1800.0  # noqa: RUF069

    def test_reads_memory_writer_section_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[memory_writer]\nturn_threshold = 30\nidle_timeout = 900.0\n",
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.memory_writer_turn_threshold == 30
        assert cfg.memory_writer_idle_timeout == 900.0  # noqa: RUF069

    def test_partial_memory_writer_section(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[memory_writer]\nturn_threshold = 100\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.memory_writer_turn_threshold == 100
        assert cfg.memory_writer_idle_timeout == 1800.0  # noqa: RUF069


# ---------------------------------------------------------------------------
# DynamicLullStrategy enum and dynamic lull fields
# ---------------------------------------------------------------------------


class TestDynamicLullStrategyEnum:
    def test_has_four_values(self) -> None:
        assert len(list(DynamicLullStrategy)) == 4

    def test_value_strings(self) -> None:
        assert DynamicLullStrategy("none") is DynamicLullStrategy.none
        assert (
            DynamicLullStrategy("reply_activity") is DynamicLullStrategy.reply_activity
        )
        assert (
            DynamicLullStrategy("llm_engagement") is DynamicLullStrategy.llm_engagement
        )
        assert DynamicLullStrategy("combined") is DynamicLullStrategy.combined


class TestCharacterConfigDynamicLullFields:
    def test_defaults_when_file_absent(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        cfg = load_character_config(
            tmp_path / "no-such-file.toml",
            defaults_path=default_profile_path,
        )
        assert cfg.dynamic_lull is DynamicLullStrategy.reply_activity
        assert cfg.lull_timeout_min == 3.0  # noqa: RUF069
        assert cfg.lull_timeout_max == 30.0  # noqa: RUF069

    def test_reads_dynamic_lull_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('dynamic_lull = "llm_engagement"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.dynamic_lull is DynamicLullStrategy.llm_engagement

    def test_reads_none_strategy_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('dynamic_lull = "none"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.dynamic_lull is DynamicLullStrategy.none

    def test_reads_lull_timeout_min_max_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("lull_timeout_min = 2.5\nlull_timeout_max = 45.0\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.lull_timeout_min == 2.5  # noqa: RUF069
        assert cfg.lull_timeout_max == 45.0  # noqa: RUF069

    def test_unknown_dynamic_lull_raises(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('dynamic_lull = "turbo"\n')
        with pytest.raises(ConfigError, match="turbo"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_all_strategies_load_from_toml(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        for strategy in DynamicLullStrategy:
            path = tmp_path / f"character_{strategy.value}.toml"
            path.write_text(f'dynamic_lull = "{strategy.value}"\n')
            cfg = load_character_config(path, defaults_path=default_profile_path)
            assert cfg.dynamic_lull is strategy

    def test_engagement_check_slot_in_defaults(
        self,
        tmp_path: Path,
        default_profile_path: Path,
    ) -> None:
        cfg = load_character_config(
            tmp_path / "no-such-file.toml",
            defaults_path=default_profile_path,
        )
        assert "engagement_check" in cfg.llm
        assert cfg.llm["engagement_check"].model

    def test_engagement_check_in_llm_slot_names(self) -> None:
        assert "engagement_check" in LLM_SLOT_NAMES
