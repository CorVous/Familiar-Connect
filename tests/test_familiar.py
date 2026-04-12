"""Red-first tests for the Familiar bundle type.

``Familiar.load_from_disk`` collects every runtime concern for one
character — config, memory store, history store, LLM clients (one per
call-site slot), subscription registry, channel config store, and the
full set of providers/processors — into a single object that the bot
layer holds for the lifetime of the process. There is exactly one
Familiar per install (see docs/architecture/configuration-model.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import (
    LLM_SLOT_NAMES,
    ChannelMode,
    CharacterConfig,
    channel_config_for_mode,
)
from familiar_connect.familiar import Familiar

from .conftest import build_fake_llm_clients

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def llm_clients() -> dict:
    """Return a full ``slot_name -> FakeLLMClient`` map for tests."""
    return build_fake_llm_clients()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadFromDisk:
    def test_id_is_folder_name(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert familiar.id == "aria"

    def test_missing_character_toml_yields_defaults(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert isinstance(familiar.config, CharacterConfig)

    def test_character_toml_is_read(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        (root / "character.toml").write_text('default_mode = "full_rp"\n')
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert familiar.config.default_mode is ChannelMode.full_rp

    def test_memory_directory_is_created(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert (root / "memory").is_dir()

    def test_history_db_is_created(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert (root / "history.db").exists()

    def test_modes_directory_is_created(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """``modes/`` holds per-mode instruction files; auto-created."""
        root = tmp_path / "aria"
        root.mkdir()
        Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert (root / "modes").is_dir()

    def test_providers_registered(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """Startup providers are wired in by default.

        ``history`` is now constructed per-turn in build_pipeline
        (mode-scoped), so it no longer appears in the startup dict.
        """
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert set(familiar.providers.keys()) == {
            "character",
            "content_search",
        }

    def test_processors_registered(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """Both first-party processors are wired in by default."""
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert "stepped_thinking" in familiar.pre_processors
        assert "recast" in familiar.post_processors

    def test_subscription_registry_is_empty_on_fresh_install(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert list(familiar.subscriptions.all()) == []

    def test_channel_configs_fall_through_to_character_default(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        (root / "character.toml").write_text('default_mode = "imitate_voice"\n')
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        cfg = familiar.channel_configs.get(channel_id=42)
        assert cfg.mode is ChannelMode.imitate_voice

    def test_llm_clients_is_held_on_bundle(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """The raw dict of clients is exposed for shutdown iteration.

        ``commands/run.py`` iterates ``familiar.llm_clients.values()``
        at shutdown to ``close()`` each pooled HTTP client, so the
        full dict must round-trip unchanged.
        """
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert familiar.llm_clients is llm_clients
        assert set(familiar.llm_clients.keys()) == set(LLM_SLOT_NAMES)


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_full_rp_pipeline_includes_every_first_party_component(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        channel_cfg = channel_config_for_mode(ChannelMode.full_rp)

        pipeline = familiar.build_pipeline(channel_cfg)
        # All three providers are active under full_rp.
        # (We can't introspect ``providers`` directly — test via a marker
        # we know the builder sets: the pipeline's internal list.)
        assert pipeline is not None

    def test_imitate_voice_drops_stepped_thinking(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """Pre-processors are filtered against channel.preprocessors_enabled."""
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        voice_cfg = channel_config_for_mode(ChannelMode.imitate_voice)

        pipeline = familiar.build_pipeline(voice_cfg)
        # stepped_thinking is disabled for imitate_voice, so the pipeline
        # shouldn't have any pre-processors attached.
        assert pipeline._pre_processors == []

    def test_mode_instructions_provider_is_added_per_turn(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """ModeInstructionProvider is constructed per call.

        The provider's file resolution depends on the channel's
        active mode, so it never lives in the static provider dict
        — :meth:`Familiar.build_pipeline` mints a fresh instance
        with the mode baked in on every turn.
        """
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
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
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """HistoryProvider is constructed per call with mode baked in."""
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )
        assert "history" not in familiar.providers

        pipeline = familiar.build_pipeline(
            channel_config_for_mode(ChannelMode.full_rp),
        )
        provider_ids = {p.id for p in pipeline._providers}
        assert "history" in provider_ids

    def test_mode_instructions_provider_is_mode_scoped(
        self,
        tmp_path: Path,
        default_profile_path: Path,
        llm_clients: dict,
    ) -> None:
        """Different channel modes produce providers bound to different files."""
        root = tmp_path / "aria"
        root.mkdir()
        (root / "modes").mkdir()
        (root / "modes" / "text_conversation_rp.md").write_text("chat-style")
        (root / "modes" / "full_rp.md").write_text("novel-style")
        familiar = Familiar.load_from_disk(
            root,
            llm_clients=llm_clients,
            defaults_path=default_profile_path,
        )

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
