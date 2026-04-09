"""Per-character and per-channel configuration loaded from TOML sidecars.

Step 7 of future-features/context-management.md. The bot reads two
kinds of config at runtime:

- ``character.toml`` — one per install, loaded once on startup.
  Tuning knobs the user sets *for the familiar* (default channel
  mode, history window size, depth-inject placement). Missing file
  is fine and falls back to :data:`DEFAULT_CHANNEL_MODE` and sensible
  per-field defaults.

- ``channels/<channel_id>.toml`` — one per subscribed channel, loaded
  lazily when the bot first needs it. Stores the channel's
  :class:`ChannelMode` plus (eventually) per-channel budget
  overrides. Missing file is also fine — the store falls back to the
  character's ``default_mode``.

A :class:`ChannelMode` is the human-selected tuning profile:

- ``full_rp`` — everything on, high budget. Good for roleplay
  channels where latency doesn't dominate.
- ``text_conversation_rp`` — character + history + stepped thinking,
  no content-search agent, no recast. Balanced defaults for a
  normal text channel.
- ``imitate_voice`` — tight budget, no stepped thinking, recast with
  a voice flavour. Tuned for TTFB on voice channels.

The three modes' concrete provider / processor sets live in
:func:`channel_config_for_mode`. Editing those defaults is a one-
line change; adding a fourth mode is a new enum member plus a table
entry.

Per-channel TOML files are currently expected to contain just a
single ``mode = "…"`` key. Future work can add per-key overrides on
top of the mode defaults without breaking existing files — the
loader ignores unknown top-level keys so hand-edited overrides are
forward-compatible.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from familiar_connect.context.types import Layer

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(Exception):
    """Raised when a config file is malformed or references an unknown value."""


class ChannelMode(Enum):
    """Tuning profile for a single Discord channel."""

    full_rp = "full_rp"
    text_conversation_rp = "text_conversation_rp"
    imitate_voice = "imitate_voice"


DEFAULT_CHANNEL_MODE = ChannelMode.text_conversation_rp
"""Mode used when neither the character nor the channel sidecar names one."""


@dataclass(frozen=True)
class CharacterConfig:
    """Config loaded once per install from ``character.toml``.

    Carries the familiar-wide defaults that the pipeline and renderer
    need regardless of which channel a turn arrives in.

    :param default_mode: Fallback :class:`ChannelMode` used when a
        specific channel has no sidecar of its own.
    :param history_window_size: How many recent turns
        :class:`HistoryProvider` surfaces verbatim.
    :param depth_inject_position: Position (from the end of the
        message list) at which the renderer inserts
        :class:`Layer.depth_inject` content. ``0`` means immediately
        before the final user turn — SillyTavern's ``@D 0``.
    :param depth_inject_role: Role the renderer assigns to the
        depth-injected message. Either ``"system"`` (default) or
        ``"user"``.
    """

    default_mode: ChannelMode = DEFAULT_CHANNEL_MODE
    history_window_size: int = 20
    depth_inject_position: int = 0
    depth_inject_role: str = "system"


@dataclass(frozen=True)
class ChannelConfig:
    """Concrete tuning for one channel, derived from a :class:`ChannelMode`.

    The dataclass is frozen so it can be cached by
    :class:`ChannelConfigStore` and passed around without copying.

    :param mode: The :class:`ChannelMode` this config was derived from.
    :param budget_tokens: Total token budget for the assembled
        system prompt, handed to :class:`ContextPipeline` on every
        request.
    :param deadline_s: Hard wall-clock deadline the pipeline enforces
        on itself per request.
    :param budget_by_layer: Per-layer budget map handed to the
        :class:`Budgeter`. Layers not present here get zero budget
        and are silently dropped.
    :param providers_enabled: Set of provider ``id``s allowed to run
        in this channel. The bot layer filters its registered
        providers against this set before constructing the pipeline.
    :param preprocessors_enabled: Set of pre-processor ``id``s
        allowed to run in this channel.
    :param postprocessors_enabled: Set of post-processor ``id``s
        allowed to run in this channel.
    """

    mode: ChannelMode
    budget_tokens: int
    deadline_s: float
    budget_by_layer: dict[Layer, int] = field(default_factory=dict)
    providers_enabled: frozenset[str] = field(default_factory=frozenset)
    preprocessors_enabled: frozenset[str] = field(default_factory=frozenset)
    postprocessors_enabled: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Mode defaults
# ---------------------------------------------------------------------------


def channel_config_for_mode(mode: ChannelMode) -> ChannelConfig:
    """Return a :class:`ChannelConfig` with the baked-in defaults for *mode*.

    Called by :class:`ChannelConfigStore.get` whenever a channel's
    TOML sidecar is absent or only names a mode. Hand-editing the
    tables below is the intended way to reshape a mode's behaviour;
    adding a fourth mode is a single new enum member plus a branch
    here.
    """
    if mode is ChannelMode.full_rp:
        return ChannelConfig(
            mode=mode,
            budget_tokens=8000,
            deadline_s=30.0,
            budget_by_layer={
                Layer.core: 400,
                Layer.character: 2000,
                Layer.content: 2000,
                Layer.history_summary: 1000,
                Layer.recent_history: 2000,
                Layer.depth_inject: 400,
                Layer.author_note: 400,
            },
            providers_enabled=frozenset(
                {"character", "history", "content_search", "mode_instructions"},
            ),
            preprocessors_enabled=frozenset({"stepped_thinking"}),
            postprocessors_enabled=frozenset({"recast"}),
        )
    if mode is ChannelMode.text_conversation_rp:
        return ChannelConfig(
            mode=mode,
            budget_tokens=6000,
            deadline_s=24.0,
            budget_by_layer={
                Layer.core: 400,
                Layer.character: 1200,
                Layer.content: 2000,
                Layer.history_summary: 600,
                Layer.recent_history: 1400,
                Layer.author_note: 300,
                Layer.depth_inject: 400,
            },
            providers_enabled=frozenset(
                {"character", "history", "content_search", "mode_instructions"},
            ),
            preprocessors_enabled=frozenset({"stepped_thinking"}),
            postprocessors_enabled=frozenset(),
        )
    if mode is ChannelMode.imitate_voice:
        return ChannelConfig(
            mode=mode,
            budget_tokens=2000,
            deadline_s=9.0,
            budget_by_layer={
                Layer.core: 300,
                Layer.character: 700,
                Layer.recent_history: 900,
                Layer.author_note: 150,
                Layer.depth_inject: 100,
            },
            providers_enabled=frozenset({"character", "history", "mode_instructions"}),
            preprocessors_enabled=frozenset(),
            postprocessors_enabled=frozenset({"recast"}),
        )
    # Exhaustive match — new enum member → failing test here.
    msg = f"Unhandled ChannelMode: {mode!r}"
    raise ConfigError(msg)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_character_config(path: Path) -> CharacterConfig:
    """Load a :class:`CharacterConfig` from *path*, or defaults if missing.

    :raises ConfigError: If the file exists but is not valid TOML,
        or references an unknown :class:`ChannelMode`.
    """
    data = _read_toml(path)
    if data is None:
        return CharacterConfig()

    default_mode = _parse_mode(data.get("default_mode"), default=DEFAULT_CHANNEL_MODE)

    history_section = data.get("providers", {}).get("history", {})
    history_window_size = int(history_section.get("window_size", 20))

    depth_section = data.get("layers", {}).get("depth_inject", {})
    depth_inject_position = int(depth_section.get("position", 0))
    depth_inject_role = str(depth_section.get("role", "system"))
    if depth_inject_role not in {"system", "user"}:
        msg = (
            f"layers.depth_inject.role must be 'system' or 'user', "
            f"got {depth_inject_role!r}"
        )
        raise ConfigError(msg)

    return CharacterConfig(
        default_mode=default_mode,
        history_window_size=history_window_size,
        depth_inject_position=depth_inject_position,
        depth_inject_role=depth_inject_role,
    )


def load_channel_config(path: Path) -> ChannelConfig | None:
    """Load a :class:`ChannelConfig` from *path*, or ``None`` if missing.

    Currently only the ``mode`` key is read; the rest of the config
    comes from :func:`channel_config_for_mode`. Unknown top-level
    keys are ignored so future per-channel overrides can be added
    without breaking forward compatibility of already-written files.

    :raises ConfigError: If the file exists but is not valid TOML,
        or references an unknown mode.
    """
    data = _read_toml(path)
    if data is None:
        return None

    mode_raw = data.get("mode")
    if mode_raw is None:
        msg = f"channel config at {path} is missing required 'mode' key"
        raise ConfigError(msg)
    mode = _parse_mode(mode_raw, default=None)
    return channel_config_for_mode(mode)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        msg = f"failed to parse TOML config at {path}: {exc}"
        raise ConfigError(msg) from exc


def _parse_mode(raw: object, *, default: ChannelMode | None) -> ChannelMode:
    if raw is None:
        if default is None:
            msg = "missing required mode value"
            raise ConfigError(msg)
        return default
    if not isinstance(raw, str):
        msg = f"mode must be a string, got {type(raw).__name__}"
        raise ConfigError(msg)
    try:
        return ChannelMode(raw)
    except ValueError as exc:
        valid = ", ".join(m.value for m in ChannelMode)
        msg = f"unknown channel mode {raw!r}; valid options: {valid}"
        raise ConfigError(msg) from exc
