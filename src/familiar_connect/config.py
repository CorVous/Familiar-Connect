"""Per-character configuration from TOML.

Loads ``character.toml`` once on startup, deep-merged over the
``_default/character.toml`` defaults.
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
"""Canonical LLM call-site slot names."""


@dataclass(frozen=True)
class LLMSlotConfig:
    """Per-call-site LLM config loaded from a ``[llm.<slot>]`` TOML section."""

    model: str
    temperature: float | None = None
    # OpenRouter provider routing override. ``None`` = default (per-call).
    # Pinned order stabilises prompt caching, costs some availability.
    # stopgap — see docs/architecture/tuning.md § provider pinning.
    provider_order: tuple[str, ...] | None = None
    provider_allow_fallbacks: bool = True


_TTS_PROVIDERS: frozenset[str] = frozenset({"azure", "cartesia", "gemini"})

_TURN_DETECTION_STRATEGIES: frozenset[str] = frozenset({"deepgram", "ten+smart_turn"})


@dataclass(frozen=True)
class TurnDetectionConfig:
    """Turn-detection strategy from ``[providers.turn_detection]``."""

    strategy: str = "deepgram"


# V3 phase 2 added "parakeet"; phase 3 will add "faster_whisper".
_STT_BACKENDS: frozenset[str] = frozenset({"deepgram", "parakeet"})


@dataclass(frozen=True)
class DeepgramSTTConfig:
    """Deepgram knobs from ``[providers.stt.deepgram]``.

    Non-secret. ``DEEPGRAM_API_KEY`` stays in env. ``DEEPGRAM_*`` env
    vars override per-knob (same precedence as ``LOCAL_TURN_DETECTION``).
    Defaults match prior env-only defaults — unconfigured installs
    unaffected.
    """

    model: str = "nova-3"
    language: str = "en"
    endpointing_ms: int = 500
    utterance_end_ms: int = 1500
    smart_format: bool = True
    punctuate: bool = True
    keyterms: tuple[str, ...] = ()
    replay_buffer_s: float = 5.0
    keepalive_interval_s: float = 3.0
    reconnect_max_attempts: int = 5
    reconnect_backoff_cap_s: float = 16.0
    idle_close_s: float = 30.0


@dataclass(frozen=True)
class ParakeetSTTConfig:
    """Parakeet knobs from ``[providers.stt.parakeet]``.

    Local NeMo model — no API key. Pair with
    ``[providers.turn_detection].strategy = "ten+smart_turn"`` so
    ``finalize()`` actually fires (no internal silence detector).
    """

    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    idle_close_s: float = 30.0


@dataclass(frozen=True)
class STTConfig:
    """STT backend selector + per-backend knobs from ``[providers.stt]``.

    V3 phase 1 lifted the selector behind ``Transcriber`` Protocol;
    phase 2 added the ``parakeet`` arm. Phase 3 will add
    ``faster_whisper`` plus a parallel ``[providers.stt.faster_whisper]``
    table.
    """

    backend: str = "deepgram"
    deepgram: DeepgramSTTConfig = field(default_factory=DeepgramSTTConfig)
    parakeet: ParakeetSTTConfig = field(default_factory=ParakeetSTTConfig)


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
class ChannelOverrides:
    """Per-channel overrides for latency-sensitive knobs.

    Layered over global defaults — test channels tunable independently.
    See plan § Design.4 *Channel-aware composition order*.

    history_window_size: overrides :attr:`CharacterConfig.history_window_size`.
    prompt_layers: ordered layer names; ``None`` inherits default order.
    message_rendering: ``"prefixed"`` (always include ``[display_name]``
    prefix) or ``"name_only"`` (rely on OpenAI ``name`` field — saves
    tokens in DMs).
    """

    history_window_size: int | None = None
    prompt_layers: tuple[str, ...] | None = None
    message_rendering: str | None = None


@dataclass(frozen=True)
class CharacterConfig:
    """Config loaded once per install from ``character.toml``."""

    history_window_size: int = 20
    display_tz: str = "UTC"
    aliases: list[str] = field(default_factory=list)
    llm: dict[str, LLMSlotConfig] = field(default_factory=dict)
    tts: TTSConfig = field(default_factory=TTSConfig)
    channels: dict[int, ChannelOverrides] = field(default_factory=dict)
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)
    stt: STTConfig = field(default_factory=STTConfig)

    def for_channel(self, channel_id: int | None) -> ChannelOverrides:
        """Return overrides for ``channel_id``; empty overrides if none."""
        if channel_id is None:
            return ChannelOverrides()
        return self.channels.get(channel_id, ChannelOverrides())

    def window_size_for(self, channel_id: int | None) -> int:
        """Resolve the history window size for ``channel_id``.

        Returns the per-channel override if present; otherwise the
        global default.
        """
        override = self.for_channel(channel_id).history_window_size
        return override if override is not None else self.history_window_size


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
    providers_raw = data.get("providers", {})
    history_section = providers_raw.get("history", {})
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

    channels_raw = data.get("channels", {})
    if not isinstance(channels_raw, dict):
        msg = f"[channels] must be a table, got {type(channels_raw).__name__}"
        raise ConfigError(msg)
    channels = _parse_channel_overrides(channels_raw)

    turn_detection_raw = providers_raw.get("turn_detection", {})
    if not isinstance(turn_detection_raw, dict):
        msg = (
            "[providers.turn_detection] must be a table, "
            f"got {type(turn_detection_raw).__name__}"
        )
        raise ConfigError(msg)
    turn_detection = _parse_turn_detection_config(turn_detection_raw)

    stt_raw = providers_raw.get("stt", {})
    if not isinstance(stt_raw, dict):
        msg = f"[providers.stt] must be a table, got {type(stt_raw).__name__}"
        raise ConfigError(msg)
    stt = _parse_stt_config(stt_raw)

    return CharacterConfig(
        history_window_size=history_window_size,
        display_tz=display_tz,
        aliases=aliases,
        llm=llm,
        tts=tts,
        channels=channels,
        turn_detection=turn_detection,
        stt=stt,
    )


_VALID_MESSAGE_RENDERING: frozenset[str] = frozenset({"prefixed", "name_only"})


def _parse_channel_overrides(raw: dict) -> dict[int, ChannelOverrides]:
    """Parse ``[channels.<id>]`` TOML blocks into typed overrides.

    Channel keys are TOML strings (TOML table keys are strings); we
    coerce to ``int`` here since Discord channel IDs are snowflakes.
    """
    out: dict[int, ChannelOverrides] = {}
    for key, section in raw.items():
        try:
            channel_id = int(key)
        except (TypeError, ValueError) as exc:
            msg = f"[channels.{key}] key must be an integer channel id"
            raise ConfigError(msg) from exc
        if not isinstance(section, dict):
            msg = f"[channels.{key}] must be a table, got {type(section).__name__}"
            raise ConfigError(msg)

        window_raw = section.get("history_window_size")
        window: int | None
        if window_raw is None:
            window = None
        elif isinstance(window_raw, int) and not isinstance(window_raw, bool):
            if window_raw <= 0:
                msg = (
                    f"[channels.{key}].history_window_size must be positive, "
                    f"got {window_raw}"
                )
                raise ConfigError(msg)
            window = window_raw
        else:
            msg = (
                f"[channels.{key}].history_window_size must be an integer, "
                f"got {type(window_raw).__name__}"
            )
            raise ConfigError(msg)

        layers_raw = section.get("prompt_layers")
        layers: tuple[str, ...] | None
        if layers_raw is None:
            layers = None
        elif isinstance(layers_raw, list) and all(
            isinstance(x, str) for x in layers_raw
        ):
            layers = tuple(layers_raw)
        else:
            msg = f"[channels.{key}].prompt_layers must be a list of strings"
            raise ConfigError(msg)

        rendering = section.get("message_rendering")
        if rendering is not None:
            if not isinstance(rendering, str):
                msg = f"[channels.{key}].message_rendering must be a string"
                raise ConfigError(msg)
            if rendering not in _VALID_MESSAGE_RENDERING:
                valid = ", ".join(sorted(_VALID_MESSAGE_RENDERING))
                msg = (
                    f"[channels.{key}].message_rendering={rendering!r} "
                    f"unknown; valid options: {valid}"
                )
                raise ConfigError(msg)

        out[channel_id] = ChannelOverrides(
            history_window_size=window,
            prompt_layers=layers,
            message_rendering=rendering,
        )
    return out


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
        provider_order = _parse_provider_order(name, section.get("provider_order"))
        allow_fallbacks_raw = section.get("provider_allow_fallbacks", True)
        if not isinstance(allow_fallbacks_raw, bool):
            msg = (
                f"[llm.{name}].provider_allow_fallbacks must be a bool, "
                f"got {type(allow_fallbacks_raw).__name__}"
            )
            raise ConfigError(msg)
        slots[name] = LLMSlotConfig(
            model=model,
            temperature=temperature,
            provider_order=provider_order,
            provider_allow_fallbacks=allow_fallbacks_raw,
        )
    return slots


def _parse_provider_order(slot_name: str, raw: object) -> tuple[str, ...] | None:
    """Validate ``[llm.<slot>].provider_order`` as a list of non-empty strings."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        msg = (
            f"[llm.{slot_name}].provider_order must be a list of strings, "
            f"got {type(raw).__name__}"
        )
        raise ConfigError(msg)
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            msg = (
                f"[llm.{slot_name}].provider_order entries must be "
                f"non-empty strings, got {item!r}"
            )
            raise ConfigError(msg)
        out.append(item)
    return tuple(out)


def _parse_turn_detection_config(raw: dict) -> TurnDetectionConfig:
    """Parse and validate the ``[providers.turn_detection]`` section."""
    strategy_raw = raw.get("strategy", "deepgram")
    if not isinstance(strategy_raw, str):
        msg = (
            f"[providers.turn_detection].strategy must be a string, "
            f"got {type(strategy_raw).__name__}"
        )
        raise ConfigError(msg)
    if strategy_raw not in _TURN_DETECTION_STRATEGIES:
        valid = ", ".join(sorted(_TURN_DETECTION_STRATEGIES))
        msg = (
            f"[providers.turn_detection].strategy {strategy_raw!r} unknown; "
            f"valid options: {valid}"
        )
        raise ConfigError(msg)
    return TurnDetectionConfig(strategy=strategy_raw)


def _parse_stt_config(raw: dict) -> STTConfig:
    """Parse and validate the ``[providers.stt]`` section."""
    backend_raw = raw.get("backend", "deepgram")
    if not isinstance(backend_raw, str):
        msg = (
            f"[providers.stt].backend must be a string, "
            f"got {type(backend_raw).__name__}"
        )
        raise ConfigError(msg)
    if backend_raw not in _STT_BACKENDS:
        valid = ", ".join(sorted(_STT_BACKENDS))
        msg = f"[providers.stt].backend {backend_raw!r} unknown; valid options: {valid}"
        raise ConfigError(msg)

    deepgram_raw = raw.get("deepgram", {})
    if not isinstance(deepgram_raw, dict):
        msg = (
            f"[providers.stt.deepgram] must be a table, "
            f"got {type(deepgram_raw).__name__}"
        )
        raise ConfigError(msg)
    deepgram = _parse_deepgram_stt_config(deepgram_raw)

    parakeet_raw = raw.get("parakeet", {})
    if not isinstance(parakeet_raw, dict):
        msg = (
            f"[providers.stt.parakeet] must be a table, "
            f"got {type(parakeet_raw).__name__}"
        )
        raise ConfigError(msg)
    parakeet = _parse_parakeet_stt_config(parakeet_raw)
    return STTConfig(backend=backend_raw, deepgram=deepgram, parakeet=parakeet)


def _parse_deepgram_stt_config(raw: dict) -> DeepgramSTTConfig:
    """Parse ``[providers.stt.deepgram]`` knobs; validate types."""
    defaults = DeepgramSTTConfig()

    def _str(key: str, default: str) -> str:
        v = raw.get(key, default)
        if not isinstance(v, str) or not v:
            msg = f"[providers.stt.deepgram].{key} must be a non-empty string"
            raise ConfigError(msg)
        return v

    def _int(key: str, default: int) -> int:
        v = raw.get(key, default)
        if not isinstance(v, int) or isinstance(v, bool):
            msg = (
                f"[providers.stt.deepgram].{key} must be an integer, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        return v

    def _float(key: str, default: float) -> float:
        v = raw.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            msg = (
                f"[providers.stt.deepgram].{key} must be a number, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        return float(v)

    def _bool(key: str, *, default: bool) -> bool:
        v = raw.get(key, default)
        if not isinstance(v, bool):
            msg = (
                f"[providers.stt.deepgram].{key} must be a bool, got {type(v).__name__}"
            )
            raise ConfigError(msg)
        return v

    keyterms_raw = raw.get("keyterms", list(defaults.keyterms))
    if not isinstance(keyterms_raw, list) or not all(
        isinstance(x, str) for x in keyterms_raw
    ):
        msg = "[providers.stt.deepgram].keyterms must be a list of strings"
        raise ConfigError(msg)

    return DeepgramSTTConfig(
        model=_str("model", defaults.model),
        language=_str("language", defaults.language),
        endpointing_ms=_int("endpointing_ms", defaults.endpointing_ms),
        utterance_end_ms=_int("utterance_end_ms", defaults.utterance_end_ms),
        smart_format=_bool("smart_format", default=defaults.smart_format),
        punctuate=_bool("punctuate", default=defaults.punctuate),
        keyterms=tuple(keyterms_raw),
        replay_buffer_s=_float("replay_buffer_s", defaults.replay_buffer_s),
        keepalive_interval_s=_float(
            "keepalive_interval_s", defaults.keepalive_interval_s
        ),
        reconnect_max_attempts=_int(
            "reconnect_max_attempts", defaults.reconnect_max_attempts
        ),
        reconnect_backoff_cap_s=_float(
            "reconnect_backoff_cap_s", defaults.reconnect_backoff_cap_s
        ),
        idle_close_s=_float("idle_close_s", defaults.idle_close_s),
    )


def _parse_parakeet_stt_config(raw: dict) -> ParakeetSTTConfig:
    """Parse ``[providers.stt.parakeet]`` knobs; validate types."""
    defaults = ParakeetSTTConfig()

    def _str(key: str, default: str) -> str:
        v = raw.get(key, default)
        if not isinstance(v, str) or not v:
            msg = f"[providers.stt.parakeet].{key} must be a non-empty string"
            raise ConfigError(msg)
        return v

    def _float(key: str, default: float) -> float:
        v = raw.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            msg = (
                f"[providers.stt.parakeet].{key} must be a number, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        return float(v)

    return ParakeetSTTConfig(
        model_name=_str("model_name", defaults.model_name),
        device=_str("device", defaults.device),
        idle_close_s=_float("idle_close_s", defaults.idle_close_s),
    )


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
