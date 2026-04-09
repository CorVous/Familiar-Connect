"""Per-channel configuration store.

Owns ``data/familiars/<id>/channels/`` and the ``/channel-*`` slash
commands that flip a channel's :class:`ChannelMode`. Sidecars are
loaded lazily on first reference and cached; the bot calls
:meth:`ChannelConfigStore.get` on every incoming message, so the
cache is important — but there are at most tens of channels, so the
map can stay resident.

Unknown channels fall through to the character-level default from
:class:`CharacterConfig.default_mode`, so a fresh channel the user
``/subscribe-text``s in just works with sane defaults.
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
    """Lazy loader + writer for per-channel TOML sidecars.

    :param root: Directory under which ``<channel_id>.toml`` files
        live. Typically ``data/familiars/<id>/channels/``.
    :param character: The loaded :class:`CharacterConfig`, used as
        the fallback for channels that don't have a sidecar of
        their own.
    """

    def __init__(self, *, root: Path, character: CharacterConfig) -> None:
        self._root = root
        self._character = character
        self._cache: dict[int, ChannelConfig] = {}

    def get(self, *, channel_id: int) -> ChannelConfig:
        """Return the :class:`ChannelConfig` for *channel_id*.

        Order of resolution:

        1. In-memory cache hit → return it.
        2. Sidecar on disk → load, cache, return.
        3. No sidecar → fall back to ``character.default_mode`` via
           :func:`channel_config_for_mode`; cache and return.
        """
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
        """Persist *mode* for *channel_id* and return the resulting config.

        Writes the sidecar as a minimal ``mode = "..."`` file. Any
        hand-edited overrides present in the existing sidecar are
        **overwritten** — if that becomes a problem, a later
        iteration can round-trip unknown keys.
        """
        sidecar = self._sidecar_path(channel_id)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f'mode = "{mode.value}"\n')

        cfg = channel_config_for_mode(mode)
        self._cache[channel_id] = cfg
        return cfg

    def _sidecar_path(self, channel_id: int) -> Path:
        return self._root / f"{channel_id}.toml"
