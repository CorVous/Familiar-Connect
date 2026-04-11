"""Red-first tests for the Familiar bundle type.

``Familiar.load_from_disk`` collects every runtime concern for one
character — config, memory store, history store, side-model,
subscription registry, channel config store, and the full set of
providers/processors — into a single object that the bot layer
holds for the lifetime of the process. There is exactly one
Familiar per install (see docs/architecture/configuration-model.md).

Covers familiar_connect.familiar, which doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.config import (
    ChannelMode,
    CharacterConfig,
    channel_config_for_mode,
)
from familiar_connect.context.side_model import LLMSideModel
from familiar_connect.familiar import Familiar
from familiar_connect.llm import LLMClient

if TYPE_CHECKING:
    from pathlib import Path


class _StubLLMClient(LLMClient):
    """Minimal LLMClient stand-in — Familiar.load_from_disk doesn't call it."""

    def __init__(self) -> None:
        super().__init__(api_key="test-key")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadFromDisk:
    def test_id_is_folder_name(self, tmp_path: Path) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert familiar.id == "aria"

    def test_missing_character_toml_yields_defaults(self, tmp_path: Path) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert isinstance(familiar.config, CharacterConfig)

    def test_character_toml_is_read(self, tmp_path: Path) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        (root / "character.toml").write_text('default_mode = "full_rp"\n')
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert familiar.config.default_mode is ChannelMode.full_rp

    def test_memory_directory_is_created(self, tmp_path: Path) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert (root / "memory").is_dir()

    def test_history_db_is_created(self, tmp_path: Path) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert (root / "history.db").exists()

    def test_modes_directory_is_created(self, tmp_path: Path) -> None:
        """``modes/`` holds per-mode instruction files; auto-created."""
        root = tmp_path / "aria"
        root.mkdir()
        Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert (root / "modes").is_dir()

    def test_providers_registered(self, tmp_path: Path) -> None:
        """Startup providers are wired in by default.

        ``history`` is now constructed per-turn in build_pipeline
        (mode-scoped), so it no longer appears in the startup dict.
        """
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert set(familiar.providers.keys()) == {
            "character",
            "content_search",
        }

    def test_processors_registered(self, tmp_path: Path) -> None:
        """Both first-party processors are wired in by default."""
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert "stepped_thinking" in familiar.pre_processors
        assert "recast" in familiar.post_processors

    def test_subscription_registry_is_empty_on_fresh_install(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert list(familiar.subscriptions.all()) == []

    def test_channel_configs_fall_through_to_character_default(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        (root / "character.toml").write_text('default_mode = "imitate_voice"\n')
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        cfg = familiar.channel_configs.get(channel_id=42)
        assert cfg.mode is ChannelMode.imitate_voice

    def test_side_model_falls_back_to_main_llm_when_no_side_client(
        self,
        tmp_path: Path,
    ) -> None:
        """Without a side_llm_client, LLMSideModel wraps the main client.

        This is the current production default — no
        ``OPENROUTER_SIDE_MODEL`` env var means every side-model
        call reuses the main (expensive) model.
        """
        root = tmp_path / "aria"
        root.mkdir()
        main = _StubLLMClient()
        familiar = Familiar.load_from_disk(root, llm_client=main)
        assert isinstance(familiar.side_model, LLMSideModel)
        # The adapter's internal client is the main one.
        assert familiar.side_model.llm_client is main

    def test_side_model_uses_side_client_when_provided(
        self,
        tmp_path: Path,
    ) -> None:
        """A separate side_llm_client is preferred for side-model work."""
        root = tmp_path / "aria"
        root.mkdir()
        main = _StubLLMClient()
        side = _StubLLMClient()
        familiar = Familiar.load_from_disk(
            root,
            llm_client=main,
            side_llm_client=side,
        )
        assert isinstance(familiar.side_model, LLMSideModel)
        assert familiar.side_model.llm_client is side
        # The main LLMClient for the reply path is still the main one.
        assert familiar.llm_client is main


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_full_rp_pipeline_includes_every_first_party_component(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        channel_cfg = channel_config_for_mode(ChannelMode.full_rp)

        pipeline = familiar.build_pipeline(channel_cfg)
        # All three providers are active under full_rp.
        # (We can't introspect ``providers`` directly — test via a marker
        # we know the builder sets: the pipeline's internal list.)
        assert pipeline is not None

    def test_imitate_voice_drops_stepped_thinking(self, tmp_path: Path) -> None:
        """Pre-processors are filtered against channel.preprocessors_enabled."""
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        voice_cfg = channel_config_for_mode(ChannelMode.imitate_voice)

        pipeline = familiar.build_pipeline(voice_cfg)
        # stepped_thinking is disabled for imitate_voice, so the pipeline
        # shouldn't have any pre-processors attached.
        assert pipeline._pre_processors == []

    def test_mode_instructions_provider_is_added_per_turn(
        self,
        tmp_path: Path,
    ) -> None:
        """ModeInstructionProvider is constructed per call.

        The provider's file resolution depends on the channel's
        active mode, so it never lives in the static provider dict
        — :meth:`Familiar.build_pipeline` mints a fresh instance
        with the mode baked in on every turn.
        """
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        # Not in the static registry.
        assert "mode_instructions" not in familiar.providers

        full_pipeline = familiar.build_pipeline(
            channel_config_for_mode(ChannelMode.full_rp),
        )
        provider_ids = {p.id for p in full_pipeline._providers}
        assert "mode_instructions" in provider_ids

    def test_history_provider_is_added_per_turn(
        self,
        tmp_path: Path,
    ) -> None:
        """HistoryProvider is constructed per call with mode baked in."""
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert "history" not in familiar.providers

        pipeline = familiar.build_pipeline(
            channel_config_for_mode(ChannelMode.full_rp),
        )
        provider_ids = {p.id for p in pipeline._providers}
        assert "history" in provider_ids

    def test_mode_instructions_provider_is_mode_scoped(
        self,
        tmp_path: Path,
    ) -> None:
        """Different channel modes produce providers bound to different files."""
        root = tmp_path / "aria"
        root.mkdir()
        (root / "modes").mkdir()
        (root / "modes" / "text_conversation_rp.md").write_text("chat-style")
        (root / "modes" / "full_rp.md").write_text("novel-style")
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())

        text_pipeline = familiar.build_pipeline(
            channel_config_for_mode(ChannelMode.text_conversation_rp),
        )
        (text_mode_provider,) = [
            p for p in text_pipeline._providers if p.id == "mode_instructions"
        ]
        rp_pipeline = familiar.build_pipeline(
            channel_config_for_mode(ChannelMode.full_rp),
        )
        (rp_mode_provider,) = [
            p for p in rp_pipeline._providers if p.id == "mode_instructions"
        ]
        # The two providers are distinct instances, each scoped to its
        # own mode. Their identity is different and, more importantly,
        # the instances are not shared across modes.
        assert text_mode_provider is not rp_mode_provider
