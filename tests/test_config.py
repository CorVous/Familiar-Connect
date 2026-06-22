"""Tests for the simplified CharacterConfig / TOML loader.

Covers TOML deep-merge over the default profile, the tiered LLM
slots (``fast`` / ``prose`` / ``background``), and the TTS config
table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.budget import TierBudget
from familiar_connect.config import (
    LLM_SLOT_NAMES,
    ChannelOverrides,
    CharacterConfig,
    ConfigError,
    DeepgramSTTConfig,
    DiscordTextConfig,
    EmbeddingConfig,
    FactSupersedeConfig,
    FocusConfig,
    LLMSlotConfig,
    MemoryProvidersConfig,
    MemoryRetrievalConfig,
    PeopleDossierConfig,
    ReflectionConfig,
    RichNoteConfig,
    RollingSummaryConfig,
    STTConfig,
    ToolsConfig,
    TTSConfig,
    TurnDetectionConfig,
    load_character_config,
)
from familiar_connect.processors.projectors import DEFAULT_PROJECTORS

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
        assert cfg.voice_window_size == 100
        assert cfg.text_window_size == 200
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

    def test_invalid_display_tz_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('display_tz = "PST"\n')
        with pytest.raises(ConfigError, match="display_tz"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_valid_display_tz_accepted(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('display_tz = "America/Los_Angeles"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.display_tz == "America/Los_Angeles"

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

    def test_reasoning_none_level_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """``"none"`` (disable thinking, OpenRouter effort=none) is valid."""
        path = tmp_path / "character.toml"
        path.write_text('[llm.fast]\nmodel = "m"\nreasoning = "none"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.llm["fast"].reasoning == "none"

    def test_reasoning_omitted_means_none(self, tmp_path: Path) -> None:
        defaults = tmp_path / "defaults.toml"
        defaults.write_text(
            '[llm.fast]\nmodel = "x"\n'
            '[llm.prose]\nmodel = "x"\n'
            '[llm.background]\nmodel = "x"\n'
        )
        cfg = load_character_config(tmp_path / "missing.toml", defaults_path=defaults)
        assert cfg.llm["prose"].reasoning is None

    def test_sampling_params_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """``top_p``/``top_k``/``presence_penalty``/``think_prepend`` parse."""
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.fast]\nmodel = "m"\ntop_p = 0.8\ntop_k = 20\n'
            "presence_penalty = 1.5\nthink_prepend = true\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        slot = cfg.llm["fast"]
        assert slot.top_p == pytest.approx(0.8)
        assert slot.top_k == 20
        assert slot.presence_penalty == pytest.approx(1.5)
        assert slot.think_prepend is True

    def test_sampling_params_omitted_mean_provider_default(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        cfg = load_character_config(
            tmp_path / "missing.toml", defaults_path=default_profile_path
        )
        slot = cfg.llm["prose"]
        assert slot.top_p is None
        assert slot.top_k is None
        assert slot.presence_penalty is None
        assert slot.think_prepend is False

    def test_top_p_out_of_range(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\ntop_p = 1.5\n')
        with pytest.raises(ConfigError, match="top_p must be in"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_top_k_must_be_positive_int(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\ntop_k = 0\n')
        with pytest.raises(ConfigError, match="top_k must be"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_presence_penalty_out_of_range(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "m"\npresence_penalty = 3.0\n')
        with pytest.raises(ConfigError, match="presence_penalty must be in"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_think_prepend_must_be_bool(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.fast]\nmodel = "m"\nthink_prepend = "yes"\n')
        with pytest.raises(ConfigError, match="think_prepend must be a bool"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_reasoning_default_sentinel_overrides_merged_value(
        self, tmp_path: Path
    ) -> None:
        # TOML has no null — "default" is the only way to reclaim model
        # default when _default/character.toml merges a reasoning level in
        defaults = tmp_path / "defaults.toml"
        defaults.write_text(
            '[llm.fast]\nmodel = "x"\nreasoning = "off"\n'
            '[llm.prose]\nmodel = "x"\nreasoning = "medium"\n'
            '[llm.background]\nmodel = "x"\n'
        )
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.fast]\nmodel = "m"\nreasoning = "default"\n'
            '[llm.prose]\nmodel = "m"\nreasoning = "default"\n'
        )
        cfg = load_character_config(path, defaults_path=defaults)
        assert cfg.llm["fast"].reasoning is None
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

    def test_post_history_instructions_default_from_profile(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """Shipped default profile carries the etiquette post-history block."""
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert "<silent>" in cfg.post_history_instructions
        assert cfg.post_history_instructions.strip()

    def test_post_history_instructions_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[prompt]\npost_history_instructions = "be terse"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.post_history_instructions == "be terse"

    def test_post_history_instructions_absent_defaults_empty(
        self, tmp_path: Path
    ) -> None:
        """No [prompt] anywhere → empty (no Python-side default text)."""
        defaults = tmp_path / "defaults.toml"
        defaults.write_text(
            '[llm.fast]\nmodel = "x"\n'
            '[llm.prose]\nmodel = "x"\n'
            '[llm.background]\nmodel = "x"\n'
        )
        cfg = load_character_config(tmp_path / "missing.toml", defaults_path=defaults)
        assert not cfg.post_history_instructions

    def test_post_history_instructions_must_be_string(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[prompt]\npost_history_instructions = 42\n")
        with pytest.raises(ConfigError, match="post_history_instructions"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_prompt_unknown_key_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[prompt]\nmystery = "x"\n')
        with pytest.raises(ConfigError, match=r"\[prompt\] has unknown keys"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_image_description_constraints_absent_defaults_empty(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """Shipped default profile leaves the constraint empty (neutral)."""
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert not cfg.image_description_constraints

    def test_image_description_constraints_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[prompt]\nimage_description_constraints = "no brands"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.image_description_constraints == "no brands"

    def test_image_description_constraints_must_be_string(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[prompt]\nimage_description_constraints = 42\n")
        with pytest.raises(ConfigError, match="image_description_constraints"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_sleep_prompts_default_from_profile(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """Shipped default profile carries the sleep-pass prompt text."""
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.sleep_consolidation_system.strip()
        assert "{self_name}" in cfg.sleep_stance_system
        assert "{self_name}" in cfg.sleep_synthesis_system
        assert "{self_name}" in cfg.dream_extraction_clause
        assert "{self_key}" in cfg.dream_extraction_clause

    def test_sleep_consolidation_system_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[prompt]\nsleep_consolidation_system = "custom tidy pass"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.sleep_consolidation_system == "custom tidy pass"

    def test_sleep_stance_system_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[prompt]\nsleep_stance_system = "stances for {self_name}"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.sleep_stance_system == "stances for {self_name}"

    def test_dream_extraction_clause_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[prompt]\ndream_extraction_clause = "dream {self_name} {self_key} {ids}"\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.dream_extraction_clause == "dream {self_name} {self_key} {ids}"

    def test_sleep_prompt_must_be_string(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[prompt]\nsleep_consolidation_system = 42\n")
        with pytest.raises(ConfigError, match="sleep_consolidation_system"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_history_window_split_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.history]\nvoice_window_size = 25\ntext_window_size = 60\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.voice_window_size == 25
        assert cfg.text_window_size == 60

    def test_legacy_window_size_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """The retired ``[providers.history].window_size`` fails loudly."""
        path = tmp_path / "character.toml"
        path.write_text("[providers.history]\nwindow_size = 50\n")
        with pytest.raises(ConfigError, match="window_size"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_voice_window_must_be_positive_int(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.history]\nvoice_window_size = 0\n")
        with pytest.raises(ConfigError, match="voice_window_size"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_text_window_must_be_positive_int(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.history]\ntext_window_size = -1\n")
        with pytest.raises(ConfigError, match="text_window_size"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_text_silence_gap_fold_defaults_to_zero(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.text_silence_gap_fold_seconds == pytest.approx(0.0)

    def test_text_silence_gap_fold_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.history]\ntext_silence_gap_fold_seconds = 1800\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.text_silence_gap_fold_seconds == pytest.approx(1800.0)

    def test_text_silence_gap_fold_rejects_negative(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.history]\ntext_silence_gap_fold_seconds = -1\n")
        with pytest.raises(ConfigError, match="text_silence_gap_fold_seconds"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_text_silence_gap_fold_rejects_non_numeric(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.history]\ntext_silence_gap_fold_seconds = "big"\n')
        with pytest.raises(ConfigError, match="text_silence_gap_fold_seconds"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_text_silence_gap_fold_accepts_zero(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.history]\ntext_silence_gap_fold_seconds = 0\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.text_silence_gap_fold_seconds == pytest.approx(0.0)


class TestBudgets:
    def test_shipped_default_voice_budget(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """``_default/character.toml`` is the source of truth for voice."""
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        v = cfg.budgets["voice"]
        assert v.total_tokens == 3000
        assert v.recent_history_tokens == 1500
        assert v.rag_tokens == 450
        assert v.dossier_tokens == 450
        assert v.summary_tokens == 300
        assert v.cross_channel_tokens == 300
        assert v.max_history_turns == 100
        assert v.max_rag_turns == 5
        assert v.max_rag_facts == 3
        assert v.max_dossier_people == 8

    def test_shipped_default_text_and_background(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.budgets["text"].total_tokens == 8000
        assert cfg.budgets["background"].total_tokens == 24000

    def test_partial_override_keeps_other_subcaps(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """Override one knob; the rest inherit from ``_default``."""
        path = tmp_path / "character.toml"
        path.write_text("[budget.voice]\ntotal_tokens = 5000\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        v = cfg.budgets["voice"]
        assert v.total_tokens == 5000
        # Untouched — TOML deep-merge over the default.
        assert v.recent_history_tokens == 1500
        assert v.rag_tokens == 450
        assert v.max_dossier_people == 8
        # Other tiers untouched.
        assert cfg.budgets["text"].total_tokens == 8000

    def test_subcap_overrides_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[budget.text]\n"
            "total_tokens = 10000\n"
            "recent_history_tokens = 4000\n"
            "rag_tokens = 1000\n"
            "max_dossier_people = 12\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        b = cfg.budgets["text"]
        assert b.total_tokens == 10000
        assert b.recent_history_tokens == 4000
        assert b.rag_tokens == 1000
        assert b.max_dossier_people == 12

    def test_unknown_tier_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[budget.mystery]\ntotal_tokens = 100\n")
        with pytest.raises(ConfigError, match="unknown budget tier"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_key_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[budget.voice]\nwobble = 5\n")
        with pytest.raises(ConfigError, match="unknown keys"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_negative_total_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[budget.voice]\ntotal_tokens = -1\n")
        with pytest.raises(ConfigError, match="total_tokens"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_dataclass_default_is_voice_tier(self) -> None:
        """Programmatic ``TierBudget()`` matches the voice envelope."""
        b = TierBudget()
        assert b.total_tokens == 3000
        assert b.recent_history_tokens == 1500


class TestMemoryRetrieval:
    """[memory.retrieval] — M2 importance-weighted retrieval weights."""

    def test_shipped_default_weights(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        r = cfg.memory_retrieval
        assert r.bm25_weight == pytest.approx(1.0)
        assert r.recency_weight == pytest.approx(0.0)
        assert r.importance_weight == pytest.approx(0.6)
        assert r.embedding_weight == pytest.approx(0.0)

    def test_partial_override(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[memory.retrieval]\nimportance_weight = 1.5\nrecency_weight = 0.4\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        r = cfg.memory_retrieval
        assert r.importance_weight == pytest.approx(1.5)
        assert r.recency_weight == pytest.approx(0.4)
        assert r.bm25_weight == pytest.approx(1.0)  # untouched

    def test_unknown_key_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[memory.retrieval]\nmagic_weight = 1\n")
        with pytest.raises(ConfigError, match="unknown keys"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_negative_weight_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[memory.retrieval]\nimportance_weight = -1\n")
        with pytest.raises(ConfigError, match="non-negative"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_non_numeric_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[memory.retrieval]\nimportance_weight = "high"\n')
        with pytest.raises(ConfigError, match="non-negative number"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_dataclass_default_is_pre_m2_bm25_only(self) -> None:
        r = MemoryRetrievalConfig()
        assert r.bm25_weight == pytest.approx(1.0)
        assert r.recency_weight == pytest.approx(0.0)
        assert r.importance_weight == pytest.approx(0.0)
        assert r.embedding_weight == pytest.approx(0.0)


class TestEmbeddingConfig:
    """[providers.embedding] — M6 embedder backend selection."""

    def test_dataclass_default_is_off(self) -> None:
        e = EmbeddingConfig()
        assert e.backend == "off"
        assert e.dim == 256

    def test_shipped_default_is_off(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.embedding.backend == "off"

    def test_override_to_hash(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.embedding]\nbackend = "hash"\ndim = 128\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.embedding.backend == "hash"
        assert cfg.embedding.dim == 128

    def test_unknown_backend_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.embedding]\nbackend = "magic"\n')
        with pytest.raises(ConfigError, match="is unknown"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_key_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.embedding]\nweird = 1\n")
        with pytest.raises(ConfigError, match="unknown keys"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_non_positive_dim_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.embedding]\ndim = 0\n")
        with pytest.raises(ConfigError, match="must be > 0"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_non_int_dim_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.embedding]\ndim = "wide"\n')
        with pytest.raises(ConfigError, match="must be a positive integer"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_default_fastembed_model_is_bge_small(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.embedding.fastembed_model == "BAAI/bge-small-en-v1.5"
        assert cfg.embedding.fastembed_cache_dir is None

    def test_fastembed_model_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.embedding]\n"
            'backend = "fastembed"\n'
            'fastembed_model = "BAAI/bge-base-en-v1.5"\n'
            'fastembed_cache_dir = "/var/cache/fastembed"\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.embedding.backend == "fastembed"
        assert cfg.embedding.fastembed_model == "BAAI/bge-base-en-v1.5"
        assert cfg.embedding.fastembed_cache_dir == "/var/cache/fastembed"

    def test_empty_fastembed_model_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[providers.embedding]\nfastembed_model = ""\n')
        with pytest.raises(ConfigError, match="non-empty string"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_non_string_fastembed_cache_dir_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.embedding]\nfastembed_cache_dir = 42\n")
        with pytest.raises(ConfigError, match="must be a string"):
            load_character_config(path, defaults_path=default_profile_path)


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
            'prompt_layers = ["character_card", "operating_mode"]\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        over = cfg.channels[12345]
        assert over.history_window_size == 8
        assert over.message_rendering == "name_only"
        assert over.prompt_layers == ("character_card", "operating_mode")

    def test_voice_window_for_falls_back_to_default(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.history]\nvoice_window_size = 20\n\n"
            "[channels.12345]\nhistory_window_size = 8\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.voice_window_for(12345) == 8  # override
        assert cfg.voice_window_for(99999) == 20  # default
        assert cfg.voice_window_for(None) == 20

    def test_text_window_for_falls_back_to_default(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.history]\ntext_window_size = 50\n\n"
            "[channels.12345]\nhistory_window_size = 8\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.text_window_for(12345) == 8  # override
        assert cfg.text_window_for(99999) == 50  # default
        assert cfg.text_window_for(None) == 50

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
        assert over.total_tokens is None

    def test_channel_total_tokens_parsed(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[channels.12345]\ntotal_tokens = 2000\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.channels[12345].total_tokens == 2000

    def test_channel_total_tokens_must_be_positive(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[channels.12345]\ntotal_tokens = 0\n")
        with pytest.raises(ConfigError, match="total_tokens"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_channel_total_tokens_must_be_integer(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[channels.12345]\ntotal_tokens = 1.5\n")
        with pytest.raises(ConfigError, match="total_tokens"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_budget_for_returns_base_when_no_channel_override(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        base = cfg.budgets["text"]
        assert cfg.budget_for("text", 99999).total_tokens == base.total_tokens

    def test_budget_for_applies_channel_total_tokens(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[channels.12345]\ntotal_tokens = 1234\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        b = cfg.budget_for("text", 12345)
        assert b.total_tokens == 1234
        # other caps unchanged from tier default
        assert b.rag_tokens == cfg.budgets["text"].rag_tokens

    def test_budget_for_ignores_other_channel(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[channels.12345]\ntotal_tokens = 1234\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        b = cfg.budget_for("text", 99999)
        assert b.total_tokens == cfg.budgets["text"].total_tokens

    def test_budget_for_none_channel_returns_base(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[channels.12345]\ntotal_tokens = 1234\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        b = cfg.budget_for("text", None)
        assert b.total_tokens == cfg.budgets["text"].total_tokens


class TestBudgetCurves:
    def test_no_curves_section_empty_dict(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.budget_curves == {}

    def test_curve_parsed(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[budget.model_curves."claude-opus-4-7"]\n'
            "total_tokens = 2.0\n"
            "rag_tokens = 1.5\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        curve = cfg.budget_curves["claude-opus-4-7"]
        assert curve.total_tokens == pytest.approx(2.0)
        assert curve.rag_tokens == pytest.approx(1.5)
        assert curve.recent_history_tokens == pytest.approx(1.0)  # default

    def test_unknown_curve_field_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[budget.model_curves."claude-opus-4-7"]\nno_such_field = 1.5\n'
        )
        with pytest.raises(ConfigError, match="no_such_field"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_non_positive_multiplier_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[budget.model_curves."claude-opus-4-7"]\ntotal_tokens = 0.0\n')
        with pytest.raises(ConfigError, match="total_tokens"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_non_numeric_multiplier_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[budget.model_curves."claude-opus-4-7"]\ntotal_tokens = "big"\n'
        )
        with pytest.raises(ConfigError, match="total_tokens"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_budget_for_applies_curve_when_model_matches(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        # Wire fast slot to a model that has a curve.
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.fast]\nmodel = "claude-opus-4-7"\napi_key_env = "X"\n\n'
            '[budget.model_curves."claude-opus-4-7"]\ntotal_tokens = 2.0\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        base = cfg.budgets["voice"]
        b = cfg.budget_for("voice", None)
        assert b.total_tokens == round(base.total_tokens * 2.0)
        # Sub-caps unaffected by a curve that only sets total_tokens.
        assert b.rag_tokens == base.rag_tokens

    def test_budget_for_no_curve_returns_base(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        base = cfg.budgets["voice"]
        b = cfg.budget_for("voice", None)
        assert b.total_tokens == base.total_tokens

    def test_budget_for_channel_override_wins_over_curve(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.fast]\nmodel = "claude-opus-4-7"\napi_key_env = "X"\n\n'
            '[budget.model_curves."claude-opus-4-7"]\ntotal_tokens = 2.0\n\n'
            "[channels.99]\ntotal_tokens = 1234\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        b = cfg.budget_for("voice", 99)
        # Explicit channel value wins over the scaled value.
        assert b.total_tokens == 1234

    def test_budget_for_curve_not_applied_for_different_model(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.fast]\nmodel = "claude-sonnet-4-6"\napi_key_env = "X"\n\n'
            '[budget.model_curves."claude-opus-4-7"]\ntotal_tokens = 2.0\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        base = cfg.budgets["voice"]
        b = cfg.budget_for("voice", None)
        # Sonnet is active; opus curve should not be applied.
        assert b.total_tokens == base.total_tokens


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


class TestDiscordTextConfig:
    """``[discord.text]`` knobs for the typing-indicator + interruption path."""

    def test_defaults(self) -> None:
        cfg = DiscordTextConfig()
        # Default: respect other users typing — treat as interruption.
        assert cfg.respond_to_typing is True
        # Backoff envelope for "another familiar-connect bot is typing"
        # — initial pause doubles up to the cap to avoid pingpong.
        assert cfg.typing_backoff_initial_s > 0
        assert cfg.typing_backoff_max_s >= cfg.typing_backoff_initial_s

    def test_present_on_character_config(self) -> None:
        cfg = CharacterConfig()
        assert isinstance(cfg.discord_text, DiscordTextConfig)

    def test_loads_from_toml(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[discord.text]\n"
            "respond_to_typing = false\n"
            "typing_backoff_initial_s = 2.5\n"
            "typing_backoff_max_s = 60.0\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.discord_text.respond_to_typing is False
        assert cfg.discord_text.typing_backoff_initial_s == pytest.approx(2.5)
        assert cfg.discord_text.typing_backoff_max_s == pytest.approx(60.0)

    def test_respond_to_typing_must_be_bool(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[discord.text]\nrespond_to_typing = "yes"\n')
        with pytest.raises(ConfigError, match="respond_to_typing"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_key_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[discord.text]\nunknown_knob = 1\n")
        with pytest.raises(ConfigError, match="unknown"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_backoff_max_must_not_be_below_initial(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[discord.text]\n"
            "typing_backoff_initial_s = 5.0\n"
            "typing_backoff_max_s = 1.0\n"
        )
        with pytest.raises(ConfigError, match="typing_backoff_max_s"):
            load_character_config(path, defaults_path=default_profile_path)


class TestImageToolConfig:
    def test_llm_slot_parses_image_tools_and_multimodal(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            '[llm.prose]\nmodel = "x/y"\nimage_tools = true\nmultimodal = true\n'
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        slot = cfg.llm["prose"]
        assert slot.image_tools is True
        assert slot.multimodal is True

    def test_image_tools_defaults_false(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm.prose]\nmodel = "x/y"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.llm["prose"].image_tools is False
        assert cfg.llm["prose"].multimodal is False

    def test_image_description_model_parsed_at_llm_level(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text('[llm]\nimage_description_model = "openai/gpt-4o"\n')
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.image_description_model == "openai/gpt-4o"

    def test_image_description_model_not_treated_as_slot(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        """image_description_model at [llm] level must not trigger unknown-slot."""
        path = tmp_path / "character.toml"
        path.write_text('[llm]\nimage_description_model = "openai/gpt-4o"\n')
        # should not raise
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.image_description_model == "openai/gpt-4o"

    def test_image_description_model_defaults_empty(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert not cfg.image_description_model


class TestFocusConfig:
    """``[focus]`` knobs — attentional idle-nudge timing."""

    def test_defaults(self) -> None:
        cfg = FocusConfig()
        assert cfg.idle_wake_seconds == pytest.approx(120.0)
        assert cfg.nudge_debounce_seconds == pytest.approx(30.0)

    def test_present_on_character_config(self) -> None:
        cfg = CharacterConfig()
        assert isinstance(cfg.focus, FocusConfig)

    def test_loads_from_toml(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[focus]\nidle_wake_seconds = 45.0\nnudge_debounce_seconds = 10\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.focus.idle_wake_seconds == pytest.approx(45.0)
        assert cfg.focus.nudge_debounce_seconds == pytest.approx(10.0)

    def test_must_be_positive(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[focus]\nidle_wake_seconds = -1\n")
        with pytest.raises(ConfigError, match="idle_wake_seconds"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_key_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[focus]\nbogus = 1\n")
        with pytest.raises(ConfigError, match="unknown"):
            load_character_config(path, defaults_path=default_profile_path)


class TestToolsConfig:
    """``[tools]`` knobs — agentic loop bounds."""

    def test_defaults(self) -> None:
        assert ToolsConfig().loop_max_iterations == 5

    def test_present_on_character_config(self) -> None:
        cfg = CharacterConfig()
        assert isinstance(cfg.tools, ToolsConfig)

    def test_loads_from_toml(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[tools]\nloop_max_iterations = 9\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.tools.loop_max_iterations == 9

    def test_must_be_positive_int(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[tools]\nloop_max_iterations = 0\n")
        with pytest.raises(ConfigError, match="loop_max_iterations"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_key_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[tools]\nbogus = 1\n")
        with pytest.raises(ConfigError, match="unknown"):
            load_character_config(path, defaults_path=default_profile_path)


class TestLLMMaxConcurrentRequests:
    """Shared ``[llm].max_concurrent_requests`` scalar."""

    def test_defaults_to_four(self) -> None:
        assert CharacterConfig().llm_max_concurrent_requests == 4

    def test_loads_from_toml(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[llm]\nmax_concurrent_requests = 8\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.llm_max_concurrent_requests == 8

    def test_not_treated_as_slot(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[llm]\nmax_concurrent_requests = 8\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert set(cfg.llm) == {"fast", "prose", "background"}

    def test_must_be_positive_int(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[llm]\nmax_concurrent_requests = 0\n")
        with pytest.raises(ConfigError, match="max_concurrent_requests"):
            load_character_config(path, defaults_path=default_profile_path)


class TestMemoryWorkerConfigs:
    """Per-projector tuning tables under ``[providers.memory.<name>]``."""

    def test_dataclass_defaults_match_legacy_hardcodes(self) -> None:
        assert RollingSummaryConfig() == RollingSummaryConfig(
            turns_threshold=10, cross_k=5, tick_interval_s=5.0
        )
        assert RichNoteConfig() == RichNoteConfig(
            batch_size=10, tick_interval_s=15.0, participants_max=30
        )
        assert PeopleDossierConfig() == PeopleDossierConfig(tick_interval_s=20.0)
        assert ReflectionConfig() == ReflectionConfig(
            turns_threshold=20,
            max_reflections_per_tick=3,
            max_turns_per_tick=50,
            recent_facts_limit=20,
            tick_interval_s=60.0,
        )
        assert FactSupersedeConfig() == FactSupersedeConfig(
            batch_size=5, tick_interval_s=60.0, priors_max=20
        )

    def test_present_on_memory_providers(self) -> None:
        cfg = MemoryProvidersConfig()
        assert isinstance(cfg.rolling_summary, RollingSummaryConfig)
        assert isinstance(cfg.rich_note, RichNoteConfig)
        assert isinstance(cfg.people_dossier, PeopleDossierConfig)
        assert isinstance(cfg.reflection, ReflectionConfig)
        assert isinstance(cfg.fact_supersede, FactSupersedeConfig)

    def test_loads_from_toml(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text(
            "[providers.memory.rolling_summary]\n"
            "turns_threshold = 4\n"
            "cross_k = 2\n"
            "tick_interval_s = 1.5\n"
            "[providers.memory.rich_note]\n"
            "batch_size = 3\n"
            "tick_interval_s = 7\n"
            "participants_max = 12\n"
            "[providers.memory.people_dossier]\n"
            "tick_interval_s = 11.0\n"
            "[providers.memory.reflection]\n"
            "turns_threshold = 8\n"
            "max_reflections_per_tick = 1\n"
            "max_turns_per_tick = 25\n"
            "recent_facts_limit = 5\n"
            "tick_interval_s = 90.0\n"
            "[providers.memory.fact_supersede]\n"
            "batch_size = 2\n"
            "tick_interval_s = 120.0\n"
            "priors_max = 6\n"
        )
        cfg = load_character_config(path, defaults_path=default_profile_path)
        mem = cfg.memory_providers
        assert mem.rolling_summary == RollingSummaryConfig(
            turns_threshold=4, cross_k=2, tick_interval_s=1.5
        )
        assert mem.rich_note == RichNoteConfig(
            batch_size=3, tick_interval_s=7.0, participants_max=12
        )
        assert mem.people_dossier == PeopleDossierConfig(tick_interval_s=11.0)
        assert mem.reflection == ReflectionConfig(
            turns_threshold=8,
            max_reflections_per_tick=1,
            max_turns_per_tick=25,
            recent_facts_limit=5,
            tick_interval_s=90.0,
        )
        assert mem.fact_supersede == FactSupersedeConfig(
            batch_size=2, tick_interval_s=120.0, priors_max=6
        )

    def test_partial_override_keeps_other_knobs(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.memory.rich_note]\nbatch_size = 3\n")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.memory_providers.rich_note.batch_size == 3
        assert cfg.memory_providers.rich_note.tick_interval_s == pytest.approx(15.0)

    def test_must_be_positive(self, tmp_path: Path, default_profile_path: Path) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.memory.rich_note]\nbatch_size = 0\n")
        with pytest.raises(ConfigError, match="batch_size"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_knob_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.memory.rolling_summary]\nbogus = 1\n")
        with pytest.raises(ConfigError, match="unknown"):
            load_character_config(path, defaults_path=default_profile_path)

    def test_unknown_subtable_rejected(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("[providers.memory.bogus_worker]\ntick_interval_s = 1\n")
        with pytest.raises(ConfigError, match="bogus_worker"):
            load_character_config(path, defaults_path=default_profile_path)


class TestDefaultProfileDrift:
    """Shipped ``_default/character.toml`` must agree with code defaults."""

    def test_default_profile_enables_every_default_projector(
        self, tmp_path: Path, default_profile_path: Path
    ) -> None:
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=default_profile_path)
        assert cfg.memory_providers.projectors == DEFAULT_PROJECTORS

    def test_dataclass_projectors_match_registry_default(self) -> None:
        assert MemoryProvidersConfig().projectors == DEFAULT_PROJECTORS

    def test_history_window_fallbacks_match_dataclass_defaults(
        self, tmp_path: Path
    ) -> None:
        """Parser fallbacks (defaults file omits keys) match dataclass."""
        defaults = tmp_path / "defaults.toml"
        defaults.write_text("")
        path = tmp_path / "character.toml"
        path.write_text("")
        cfg = load_character_config(path, defaults_path=defaults)
        assert cfg.voice_window_size == CharacterConfig().voice_window_size
        assert cfg.text_window_size == CharacterConfig().text_window_size
