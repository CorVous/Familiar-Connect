"""Per-channel configuration store.

Owns ``data/familiars/<id>/channels/`` TOML sidecars. Lazy-loaded,
cached (at most tens of channels). Unknown channels fall through to
``CharacterConfig.default_mode``.

Sidecar schema::

    channel_name = "general"    # informational; written by /channel-backdrop
    mode = "full_rp"
    backdrop = \"\"\"            # optional; replaces modes/<mode>.md
    Custom author-note text.
    \"\"\"

    [typing_simulation]
    enabled = false
"""

from __future__ import annotations

import tomllib
from dataclasses import replace
from typing import TYPE_CHECKING

import tomli_w

from familiar_connect.config import (
    _resolve_typing_simulation,
    channel_config_for_mode,
    load_channel_config,
)
from familiar_connect.llm import Message

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
        """Resolve config: cache → sidecar → character default.

        Applies character-level ``typing_simulation`` overrides on the
        per-mode default. Channel TOML overrides (if present) stack on top.
        """
        cached = self._cache.get(channel_id)
        if cached is not None:
            return cached

        sidecar = self._sidecar_path(channel_id)
        loaded = (
            load_channel_config(
                sidecar,
                character_overrides=self._character.typing_simulation_overrides,
            )
            if sidecar.exists()
            else None
        )
        if loaded is None:
            loaded = self._fallback_for_mode(self._character.default_mode)

        self._cache[channel_id] = loaded
        return loaded

    def set_mode(self, *, channel_id: int, mode: ChannelMode) -> ChannelConfig:
        """Write updated sidecar and return resulting config.

        Preserves existing ``backdrop``, ``channel_name``, and
        ``[typing_simulation]`` fields so that switching modes does not
        erase other per-channel settings.
        """
        sidecar = self._sidecar_path(channel_id)
        sidecar.parent.mkdir(parents=True, exist_ok=True)

        existing = self._read_raw(sidecar)
        existing["mode"] = mode.value
        self._write_sidecar(sidecar, existing)

        cfg = self._load_from_sidecar(sidecar)
        self._cache[channel_id] = cfg
        return cfg

    def set_backdrop(
        self,
        *,
        channel_id: int,
        backdrop: str,
        channel_name: str | None = None,
    ) -> ChannelConfig:
        """Persist *backdrop* for *channel_id*; seed sidecar if absent.

        :param backdrop: author-note text; empty/whitespace → clears override.
        :param channel_name: Discord channel name; written for operator
            reference but never used for config resolution.
        """
        sidecar = self._sidecar_path(channel_id)
        sidecar.parent.mkdir(parents=True, exist_ok=True)

        existing = self._read_raw(sidecar)
        if "mode" not in existing:
            existing["mode"] = self._character.default_mode.value

        stripped = backdrop.strip()
        if stripped:
            existing["backdrop"] = stripped
        else:
            existing.pop("backdrop", None)

        if channel_name is not None:
            existing["channel_name"] = channel_name

        self._write_sidecar(sidecar, existing)

        cfg = self._load_from_sidecar(sidecar)
        self._cache[channel_id] = cfg
        return cfg

    def get_backdrop(self, *, channel_id: int) -> str | None:
        """Return current ``backdrop_override`` or ``None`` if unset."""
        return self.get(channel_id=channel_id).backdrop_override

    def set_last_context(
        self,
        *,
        channel_id: int,
        messages: list[Message],
    ) -> None:
        """Persist *messages* as the channel's last assembled context."""
        sidecar = self._sidecar_path(channel_id)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read_raw(sidecar)
        if "mode" not in existing:
            existing["mode"] = self._character.default_mode.value
        existing["last_context"] = [
            {
                "role": m.role,
                **({"name": m.name} if m.name else {}),
                "content": m.content,
            }
            for m in messages
        ]
        self._write_sidecar(sidecar, existing)

    def get_last_context(self, *, channel_id: int) -> list[Message] | None:
        """Return cached messages, or ``None`` if no cache exists."""
        sidecar = self._sidecar_path(channel_id)
        if not sidecar.exists():
            return None
        raw = self._read_raw(sidecar)
        entries = raw.get("last_context")
        if not entries:
            return None
        return [
            Message(role=e["role"], content=e["content"], name=e.get("name"))
            for e in entries
        ]

    def _fallback_for_mode(self, mode: ChannelMode) -> ChannelConfig:
        """Per-mode baseline with character-level ``typing_simulation`` applied."""
        base = channel_config_for_mode(mode)
        resolved_ts = _resolve_typing_simulation(
            base.typing_simulation,
            character_overrides=self._character.typing_simulation_overrides,
            channel_overrides={},
        )
        return replace(base, typing_simulation=resolved_ts)

    def _load_from_sidecar(self, sidecar: Path) -> ChannelConfig:
        """Load sidecar with character-level typing_simulation applied."""
        cfg = load_channel_config(
            sidecar,
            character_overrides=self._character.typing_simulation_overrides,
        )
        if cfg is None:
            # sidecar was just written; should not happen
            return self._fallback_for_mode(self._character.default_mode)
        return cfg

    def _read_raw(self, sidecar: Path) -> dict:
        """Return raw TOML dict from *sidecar*, or empty dict if absent."""
        if not sidecar.exists():
            return {}
        with sidecar.open("rb") as f:
            return dict(tomllib.load(f))

    def _write_sidecar(self, sidecar: Path, data: dict) -> None:
        """Serialize *data* with canonical top-level key order.

        ``channel_name`` first (human-readable header), then ``mode``,
        then ``backdrop``, then any other scalars/tables, then
        ``last_context`` (large array-of-tables) always last.
        """
        preferred = ("channel_name", "mode", "backdrop")
        ordered: dict = {k: data[k] for k in preferred if k in data}
        for k, v in data.items():
            if k not in ordered and k != "last_context":
                ordered[k] = v
        if "last_context" in data:
            ordered["last_context"] = data["last_context"]
        sidecar.write_bytes(tomli_w.dumps(ordered).encode())

    def _sidecar_path(self, channel_id: int) -> Path:
        return self._root / f"{channel_id}.toml"
