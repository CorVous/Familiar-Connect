"""Per-character configuration from TOML.

Loads ``character.toml`` once on startup, deep-merged over the
``_default/character.toml`` defaults. The reply-orchestration layers
that shaped the prototype have been ripped out; what remains is the
minimum a Discord bot shell still needs plus the seams the next
reply-path design will build on (LLM slot table, TTS config).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(Exception):
    """Raised when a config file is malformed or references an unknown value."""


LLM_SLOT_NAMES: frozenset[str] = frozenset({"main_prose"})
"""Canonical LLM call-site slot names.

The prototype had eight; demolition pass trimmed to one. Slot shape
is kept (rather than collapsed to a single client) so the cache-reuse
redesign can reintroduce slots without churn.
"""


@dataclass(frozen=True)
class LLMSlotConfig:
    """Per-call-site LLM config loaded from a ``[llm.<slot>]`` TOML section."""

    model: str
    temperature: float | None = None


_TTS_PROVIDERS: frozenset[str] = frozenset({"azure", "cartesia", "gemini"})

DEFAULT_AZURE_TTS_VOICE = "en-US-AmberNeural"
DEFAULT_GEMINI_TTS_VOICE = "Kore"
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech config loaded from the ``[tts]`` TOML section."""

    provider: str = "azure"
    cartesia_voice_id: str | None = None
    cartesia_model: str | None = None
    azure_voice: str = DEFAULT_AZURE_TTS_VOICE
    gemini_voice: str = DEFAULT_GEMINI_TTS_VOICE
    gemini_model: str = DEFAULT_GEMINI_TTS_MODEL
    gemini_scene: str | None = None
    gemini_context: str | None = None
    gemini_audio_profile: str | None = None
    gemini_style: str | None = None
    gemini_pace: str | None = None
    gemini_accent: str | None = None
    greetings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CharacterConfig:
    """Config loaded once per install from ``character.toml``."""

    history_window_size: int = 20
    display_tz: str = "UTC"
    aliases: list[str] = field(default_factory=list)
    llm: dict[str, LLMSlotConfig] = field(default_factory=dict)
    tts: TTSConfig = field(default_factory=TTSConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_character_config(
    path: Path,
    *,
    defaults_path: Path,
) -> CharacterConfig:
    """Load :class:`CharacterConfig` from *path*, merged over *defaults_path*.

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
    history_section = data.get("providers", {}).get("history", {})
    history_window_size = int(history_section.get("window_size", 20))

    display_tz = str(data.get("display_tz", "UTC"))

    aliases_raw = data.get("aliases", [])
    if not isinstance(aliases_raw, list):
        msg = f"aliases must be a list of strings, got {type(aliases_raw).__name__}"
        raise ConfigError(msg)
    aliases = [str(a) for a in aliases_raw]

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
        history_window_size=history_window_size,
        display_tz=display_tz,
        aliases=aliases,
        llm=llm,
        tts=tts,
    )


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
    """Deep-merge *override* on top of *base*; neither input mutated."""
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
    provider_raw = raw.get("provider", "azure")
    if not isinstance(provider_raw, str):
        msg = f"[tts].provider must be a string, got {type(provider_raw).__name__}"
        raise ConfigError(msg)
    if provider_raw not in _TTS_PROVIDERS:
        valid = ", ".join(sorted(_TTS_PROVIDERS))
        msg = f"[tts].provider {provider_raw!r} unknown; valid options: {valid}"
        raise ConfigError(msg)

    cartesia_voice_id = raw.get("cartesia_voice_id")
    if cartesia_voice_id is not None and not isinstance(cartesia_voice_id, str):
        msg = "[tts].cartesia_voice_id must be a string"
        raise ConfigError(msg)

    cartesia_model = raw.get("cartesia_model")
    if cartesia_model is not None and not isinstance(cartesia_model, str):
        msg = "[tts].cartesia_model must be a string"
        raise ConfigError(msg)

    azure_voice_raw = raw.get("azure_voice", DEFAULT_AZURE_TTS_VOICE)
    if not isinstance(azure_voice_raw, str) or not azure_voice_raw:
        msg = "[tts].azure_voice must be a non-empty string"
        raise ConfigError(msg)

    gemini_voice_raw = raw.get("gemini_voice", DEFAULT_GEMINI_TTS_VOICE)
    if not isinstance(gemini_voice_raw, str) or not gemini_voice_raw:
        msg = "[tts].gemini_voice must be a non-empty string"
        raise ConfigError(msg)

    gemini_model_raw = raw.get("gemini_model", DEFAULT_GEMINI_TTS_MODEL)
    if not isinstance(gemini_model_raw, str) or not gemini_model_raw:
        msg = "[tts].gemini_model must be a non-empty string"
        raise ConfigError(msg)

    def _opt_str(key: str) -> str | None:
        val = raw.get(key)
        if val is not None and not isinstance(val, str):
            msg = f"[tts].{key} must be a string"
            raise ConfigError(msg)
        return val or None

    greetings_raw = raw.get("greetings", [])
    if not isinstance(greetings_raw, list):
        msg = (
            "[tts].greetings must be a list of strings, "
            f"got {type(greetings_raw).__name__}"
        )
        raise ConfigError(msg)
    greetings = [str(g) for g in greetings_raw]

    return TTSConfig(
        provider=provider_raw,
        cartesia_voice_id=cartesia_voice_id,
        cartesia_model=cartesia_model,
        azure_voice=azure_voice_raw,
        gemini_voice=gemini_voice_raw,
        gemini_model=gemini_model_raw,
        gemini_scene=_opt_str("gemini_scene"),
        gemini_context=_opt_str("gemini_context"),
        gemini_audio_profile=_opt_str("gemini_audio_profile"),
        gemini_style=_opt_str("gemini_style"),
        gemini_pace=_opt_str("gemini_pace"),
        gemini_accent=_opt_str("gemini_accent"),
        greetings=greetings,
    )
