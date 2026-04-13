"""Per-channel configuration store.

Owns ``data/familiars/<id>/channels/`` and the ``/channel-*`` slash
commands that flip a channel's :class:`ChannelMode`. Sidecars are
loaded lazily on first reference and cached; the bot calls
:meth:`ChannelConfigStore.get` on every incoming message, so the
cache is important — but there are at most tens of channels, so the
map can stay resident.

Unknown channels fall through to a **kind-aware** default. Text
subscriptions inherit :class:`CharacterConfig.default_mode`; voice
subscriptions default to :attr:`ChannelMode.imitate_voice` so a fresh
``/subscribe-my-voice`` lands on the low-latency voice tuning profile
without the admin having to run ``/channel-imitate-voice`` afterward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.config import (
    ChannelMode,
    channel_config_for_mode,
    load_channel_config,
)
from familiar_connect.subscriptions import SubscriptionKind

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.config import (
        ChannelConfig,
        CharacterConfig,
    )


class ChannelConfigStore:
    """Lazy loader + writer for per-channel TOML sidecars.

    :param root: Directory under which ``<channel_id>.toml`` files
        live. Typically ``data/familiars/<id>/channels/``.
    :param character: The loaded :class:`CharacterConfig`, used as
        the fallback for *text* channels that don't have a sidecar
        of their own. Voice channels bypass this and fall back to
        :attr:`ChannelMode.imitate_voice`.
    """

    def __init__(self, *, root: Path, character: CharacterConfig) -> None:
        self._root = root
        self._character = character
        self._cache: dict[tuple[int, SubscriptionKind], ChannelConfig] = {}

    def get(
        self,
        *,
        channel_id: int,
        kind: SubscriptionKind = SubscriptionKind.text,
    ) -> ChannelConfig:
        """Return the :class:`ChannelConfig` for *channel_id*.

        Order of resolution:

        1. In-memory cache hit for ``(channel_id, kind)`` → return it.
        2. Sidecar on disk → load, cache, return.
        3. No sidecar → fall back by *kind*:
           - :attr:`SubscriptionKind.voice` →
             :attr:`ChannelMode.imitate_voice`.
           - :attr:`SubscriptionKind.text` →
             ``character.default_mode``.

           ``channel_config_for_mode`` materializes the resulting
           config; it's cached and returned.
        """
        key = (channel_id, kind)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        sidecar = self._sidecar_path(channel_id)
        loaded = load_channel_config(sidecar) if sidecar.exists() else None
        if loaded is None:
            fallback_mode = (
                ChannelMode.imitate_voice
                if kind is SubscriptionKind.voice
                else self._character.default_mode
            )
            loaded = channel_config_for_mode(fallback_mode)

        self._cache[key] = loaded
        return loaded

    def set_mode(
        self,
        *,
        channel_id: int,
        mode: ChannelMode,
        kind: SubscriptionKind = SubscriptionKind.text,
    ) -> ChannelConfig:
        """Persist *mode* for *channel_id* and return the resulting config.

        Writes the sidecar as a minimal ``mode = "..."`` file. Any
        hand-edited overrides present in the existing sidecar are
        **overwritten** — if that becomes a problem, a later
        iteration can round-trip unknown keys.

        *kind* selects which cache slot to update so a later
        :meth:`get` with the same kind returns the freshly written
        value without re-reading the sidecar.
        """
        sidecar = self._sidecar_path(channel_id)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f'mode = "{mode.value}"\n')

        cfg = channel_config_for_mode(mode)
        self._cache[channel_id, kind] = cfg
        return cfg

    def _sidecar_path(self, channel_id: int) -> Path:
        return self._root / f"{channel_id}.toml"
