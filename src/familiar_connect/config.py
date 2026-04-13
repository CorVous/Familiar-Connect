"""Per-character and per-channel configuration from TOML sidecars.

Two config levels at runtime:

- ``character.toml`` — loaded once on startup; familiar-wide defaults
  (mode, history window, depth-inject, LLM slots, TTS)
- ``channels/<channel_id>.toml`` — loaded lazily per channel; stores
  :class:`ChannelMode` and eventual per-channel budget overrides

Three :class:`ChannelMode` profiles: ``full_rp``, ``text_conversation_rp``,
``imitate_voice``. Concrete provider/processor sets live in
:func:`channel_config_for_mode`.
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


class Interjection(Enum):
    """Controls message-count threshold for interjection checks.

    Higher tiers evaluate sooner (shorter starting interval). After
    each declined check, interval shrinks by 3, flooring at 3.

    | Value        | Starting interval (messages) |
    |--------------|------------------------------|
    | ``very_quiet`` | 15                         |
    | ``quiet``      | 12                         |
    | ``average``    | 9                          |
    | ``eager``      | 6                          |
    | ``very_eager`` | 3                          |
    """

    very_quiet = "very_quiet"
    quiet = "quiet"
    average = "average"
    eager = "eager"
    very_eager = "very_eager"

    @property
    def starting_interval(self) -> int:
        """Messages between the last response and the first interjection check."""
        return {
            Interjection.very_quiet: 15,
            Interjection.quiet: 12,
            Interjection.average: 9,
            Interjection.eager: 6,
            Interjection.very_eager: 3,
        }[self]


_DEFAULT_CHATTINESS = "Balanced — responds when the conversation is relevant"


LLM_SLOT_NAMES: frozenset[str] = frozenset(
    {
        "main_prose",
        "post_process_style",
        "reasoning_context",
        "history_summary",
        "memory_search",
        "interjection_decision",
    },
)
"""Canonical LLM call-site slot names.

Missing slots in ``character.toml`` inherit from
``data/familiars/_default/character.toml``.
"""


@dataclass(frozen=True)
class LLMSlotConfig:
    """Per-call-site LLM config loaded from a ``[llm.<slot>]`` TOML section."""

    model: str
    temperature: float | None = None


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech config loaded from the ``[tts]`` TOML section."""

    voice_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class CharacterConfig:
    """Config loaded once per install from ``character.toml``.

    :param depth_inject_position: position from end of message list
        for :class:`Layer.depth_inject`; ``0`` = SillyTavern's ``@D 0``.
    :param chattiness: free-text personality trait fed to interjection
        evaluation prompt.
    :param text_lull_timeout: text-only; voice uses ``voice_lull_timeout``.
    :param voice_lull_timeout: debounce for Deepgram final-transcript stream.
    """

    default_mode: ChannelMode = DEFAULT_CHANNEL_MODE
    history_window_size: int = 20
    depth_inject_position: int = 0
    depth_inject_role: str = "system"
    display_tz: str = "UTC"
    """IANA timezone name for rendering timestamps in ``text_conversation_rp``
    mode (e.g. ``"America/New_York"``). Defaults to UTC. Loaded from
    ``display_tz`` in ``character.toml``."""
    aliases: list[str] = field(default_factory=list)
    chattiness: str = _DEFAULT_CHATTINESS
    interjection: Interjection = Interjection.average
    text_lull_timeout: float = 10.0
    voice_lull_timeout: float = 5.0
    llm: dict[str, LLMSlotConfig] = field(default_factory=dict)
    tts: TTSConfig = field(default_factory=TTSConfig)


@dataclass(frozen=True)
class ChannelConfig:
    """Concrete tuning for one channel, derived from :class:`ChannelMode`.

    Frozen for caching by :class:`ChannelConfigStore`.
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
    """Return a :class:`ChannelConfig` with baked-in defaults for *mode*."""
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
                Layer.history_summary: 200,
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


def load_character_config(
    path: Path,
    *,
    defaults_path: Path,
) -> CharacterConfig:
    """Load :class:`CharacterConfig` from *path*, merged over *defaults_path*.

    Deep-merges user's TOML over default profile — missing slots
    inherit from defaults.

    :raises ConfigError: if default profile missing, invalid TOML,
        unknown ``[llm.<slot>]``, or validation failure.
    """
    defaults_data = _read_toml(defaults_path)
    if defaults_data is None:
        msg = (
            f"default character profile not found at {defaults_path}. "
            "This file is a required repo asset — check your install."
        )
        raise ConfigError(msg)

    target_data = _read_toml(path) or {}
    merged = _deep_merge(defaults_data, target_data)
    return _parse_character_config(merged)


def _parse_character_config(data: dict) -> CharacterConfig:
    """Build a :class:`CharacterConfig` from a merged TOML dict."""
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

    display_tz = str(data.get("display_tz", "UTC"))

    aliases_raw = data.get("aliases", [])
    if not isinstance(aliases_raw, list):
        msg = f"aliases must be a list of strings, got {type(aliases_raw).__name__}"
        raise ConfigError(msg)
    aliases = [str(a) for a in aliases_raw]

    chattiness = str(data.get("chattiness", _DEFAULT_CHATTINESS))

    interjection = _parse_interjection(
        data.get("interjection"), default=Interjection.average
    )

    text_lull_timeout = float(data.get("text_lull_timeout", 10.0))
    voice_lull_timeout = float(data.get("voice_lull_timeout", 5.0))

    llm_raw = data.get("llm", {})
    if not isinstance(llm_raw, dict):
        msg = f"[llm] must be a table, got {type(llm_raw).__name__}"
        raise ConfigError(msg)
    llm = _parse_llm_slots(llm_raw)

    tts_raw = data.get("tts", {})
    if not isinstance(tts_raw, dict):
        msg = f"[tts] must be a table, got {type(tts_raw).__name__}"
        raise ConfigError(msg)
    tts = _parse_tts_config(tts_raw)

    return CharacterConfig(
        default_mode=default_mode,
        history_window_size=history_window_size,
        depth_inject_position=depth_inject_position,
        depth_inject_role=depth_inject_role,
        display_tz=display_tz,
        aliases=aliases,
        chattiness=chattiness,
        interjection=interjection,
        text_lull_timeout=text_lull_timeout,
        voice_lull_timeout=voice_lull_timeout,
        llm=llm,
        tts=tts,
    )


def load_channel_config(path: Path) -> ChannelConfig | None:
    """Load :class:`ChannelConfig` from *path*, or ``None`` if missing.

    Only ``mode`` key is read; remainder from :func:`channel_config_for_mode`.

    :raises ConfigError: if file exists but is invalid TOML or unknown mode.
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* on top of *base*; neither input mutated.

    Nested dicts merge key-by-key; non-dict values replaced wholesale.
    """
    result: dict = {}
    for key, base_value in base.items():
        if key in override:
            override_value = override[key]
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                result[key] = _deep_merge(base_value, override_value)
            else:
                result[key] = override_value
        else:
            result[key] = base_value
    result.update({k: v for k, v in override.items() if k not in base})
    return result


def _parse_llm_slots(raw: dict) -> dict[str, LLMSlotConfig]:
    """Parse and validate the ``[llm.*]`` section into typed slot configs."""
    slots: dict[str, LLMSlotConfig] = {}
    for name, section in raw.items():
        if name not in LLM_SLOT_NAMES:
            valid = ", ".join(sorted(LLM_SLOT_NAMES))
            msg = f"unknown LLM slot {name!r}; valid slots: {valid}"
            raise ConfigError(msg)
        if not isinstance(section, dict):
            msg = f"[llm.{name}] must be a table, got {type(section).__name__}"
            raise ConfigError(msg)
        model = section.get("model")
        if not isinstance(model, str) or not model:
            msg = f"[llm.{name}].model must be a non-empty string"
            raise ConfigError(msg)
        temperature_raw = section.get("temperature")
        temperature: float | None
        if temperature_raw is None:
            temperature = None
        elif isinstance(temperature_raw, (int, float)) and not isinstance(
            temperature_raw,
            bool,
        ):
            temperature = float(temperature_raw)
            if not 0.0 <= temperature <= 2.0:
                msg = f"[llm.{name}].temperature must be in [0, 2], got {temperature}"
                raise ConfigError(msg)
        else:
            msg = (
                f"[llm.{name}].temperature must be a number, "
                f"got {type(temperature_raw).__name__}"
            )
            raise ConfigError(msg)
        slots[name] = LLMSlotConfig(model=model, temperature=temperature)
    return slots


def _parse_tts_config(raw: dict) -> TTSConfig:
    """Parse and validate the ``[tts]`` section into a typed config."""
    voice_id = raw.get("voice_id")
    if voice_id is not None and not isinstance(voice_id, str):
        msg = f"[tts].voice_id must be a string, got {type(voice_id).__name__}"
        raise ConfigError(msg)
    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        msg = f"[tts].model must be a string, got {type(model).__name__}"
        raise ConfigError(msg)
    return TTSConfig(voice_id=voice_id, model=model)


def _parse_interjection(raw: object, *, default: Interjection) -> Interjection:
    if raw is None:
        return default
    if not isinstance(raw, str):
        msg = f"interjection must be a string, got {type(raw).__name__}"
        raise ConfigError(msg)
    try:
        return Interjection(raw)
    except ValueError as exc:
        valid = ", ".join(m.value for m in Interjection)
        msg = f"unknown interjection tier {raw!r}; valid options: {valid}"
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
