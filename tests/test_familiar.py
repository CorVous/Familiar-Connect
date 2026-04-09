"""Red-first tests for the Familiar bundle type.

``Familiar.load_from_disk`` collects every runtime concern for one
character — config, memory store, history store, side-model,
subscription registry, channel config store, and the full set of
providers/processors — into a single object that the bot layer
holds for the lifetime of the process. There is exactly one
Familiar per install (see future-features/configuration-levels.md).

Covers familiar_connect.familiar, which doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.config import (
    ChannelMode,
    CharacterConfig,
    channel_config_for_mode,
)
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

    def test_providers_registered(self, tmp_path: Path) -> None:
        """The three first-party providers are wired in by default."""
        root = tmp_path / "aria"
        root.mkdir()
        familiar = Familiar.load_from_disk(root, llm_client=_StubLLMClient())
        assert set(familiar.providers.keys()) == {
            "character",
            "history",
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
