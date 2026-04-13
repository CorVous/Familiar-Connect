"""Per-channel configuration store.

Owns ``data/familiars/<id>/channels/`` TOML sidecars. Lazy-loaded,
cached (at most tens of channels). Unknown channels fall through to
``CharacterConfig.default_mode``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.config import (
    channel_config_for_mode,
    load_channel_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.config import (
        ChannelConfig,
        ChannelMode,
        CharacterConfig,
    )


class ChannelConfigStore:
    """Lazy loader + writer for per-channel TOML sidecars."""

    def __init__(self, *, root: Path, character: CharacterConfig) -> None:
        self._root = root
        self._character = character
        self._cache: dict[int, ChannelConfig] = {}

    def get(self, *, channel_id: int) -> ChannelConfig:
        """Resolve config: cache → sidecar → character default."""
        cached = self._cache.get(channel_id)
        if cached is not None:
            return cached

        sidecar = self._sidecar_path(channel_id)
        loaded = load_channel_config(sidecar) if sidecar.exists() else None
        if loaded is None:
            loaded = channel_config_for_mode(self._character.default_mode)

        self._cache[channel_id] = loaded
        return loaded

    def set_mode(self, *, channel_id: int, mode: ChannelMode) -> ChannelConfig:
        """Write minimal sidecar and return resulting config.

        Overwrites any hand-edited overrides in existing sidecar.
        """
        sidecar = self._sidecar_path(channel_id)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f'mode = "{mode.value}"\n')

        cfg = channel_config_for_mode(mode)
        self._cache[channel_id] = cfg
        return cfg

    def _sidecar_path(self, channel_id: int) -> Path:
        return self._root / f"{channel_id}.toml"
