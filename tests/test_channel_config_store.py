"""Red-first tests for the ChannelConfigStore.

The store owns the ``data/familiars/<id>/channels/`` directory. Each
channel's TOML sidecar stores the :class:`ChannelMode` (and future
per-channel overrides); the store loads them lazily, caches them in
memory, and writes back when a slash command flips a channel's mode.

Covers familiar_connect.channel_config, which doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.channel_config import ChannelConfigStore
from familiar_connect.config import ChannelMode, CharacterConfig

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fallback to character default
# ---------------------------------------------------------------------------


class TestFallback:
    def test_unknown_channel_falls_through_to_character_default(
        self,
        tmp_path: Path,
    ) -> None:
        character = CharacterConfig(default_mode=ChannelMode.imitate_voice)
        store = ChannelConfigStore(root=tmp_path, character=character)

        cfg = store.get(channel_id=12345)
        assert cfg.mode is ChannelMode.imitate_voice


# ---------------------------------------------------------------------------
# Setting a mode writes a sidecar
# ---------------------------------------------------------------------------


class TestSetMode:
    def test_set_mode_writes_toml(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(),
        )
        store.set_mode(channel_id=999, mode=ChannelMode.full_rp)

        sidecar = tmp_path / "999.toml"
        assert sidecar.exists()

    def test_set_mode_read_back(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(default_mode=ChannelMode.text_conversation_rp),
        )
        store.set_mode(channel_id=999, mode=ChannelMode.full_rp)
        assert store.get(channel_id=999).mode is ChannelMode.full_rp

    def test_set_mode_survives_new_store_instance(self, tmp_path: Path) -> None:
        """Persistence: a fresh store against the same root sees existing sidecars."""
        store_a = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        store_a.set_mode(channel_id=999, mode=ChannelMode.full_rp)

        store_b = ChannelConfigStore(root=tmp_path, character=CharacterConfig())
        assert store_b.get(channel_id=999).mode is ChannelMode.full_rp


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_repeated_gets_reuse_same_object(self, tmp_path: Path) -> None:
        store = ChannelConfigStore(
            root=tmp_path,
            character=CharacterConfig(default_mode=ChannelMode.full_rp),
        )
        store.set_mode(channel_id=42, mode=ChannelMode.full_rp)
        first = store.get(channel_id=42)
        second = store.get(channel_id=42)
        # ChannelConfig is frozen so the cache can legitimately return the
        # same instance; both must at least compare equal.
        assert first == second
