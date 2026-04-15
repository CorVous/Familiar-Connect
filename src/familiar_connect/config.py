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
from dataclasses import dataclass, field, replace
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


class InterruptTolerance(Enum):
    """Base probability of pushing through a mid-speech interruption (voice).

    Mood modifier and unsolicited bias added on top at runtime.
    """

    very_meek = "very_meek"
    meek = "meek"
    average = "average"
    stubborn = "stubborn"
    very_stubborn = "very_stubborn"

    @property
    def base_probability(self) -> float:
        """Probability of continuing to talk when interrupted, before bias."""
        return {
            InterruptTolerance.very_meek: 0.10,
            InterruptTolerance.meek: 0.20,
            InterruptTolerance.average: 0.30,
            InterruptTolerance.stubborn: 0.45,
            InterruptTolerance.very_stubborn: 0.60,
        }[self]


_DEFAULT_CHATTINESS = "Balanced — responds when the conversation is relevant"


LLM_SLOT_NAMES: frozenset[str] = frozenset(
    {
        "main_prose",
        "post_process_style",
        "reasoning_context",
        "history_summary",
        "memory_search",
        "memory_writer",
        "interjection_decision",
        "mood_eval",
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


_TTS_PROVIDERS: frozenset[str] = frozenset({"azure", "cartesia"})
"""Valid ``[tts].provider`` values."""

DEFAULT_AZURE_TTS_VOICE = "en-US-AmberNeural"
"""Azure Neural voice used when ``[tts].azure_voice`` is unset."""


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech config loaded from the ``[tts]`` TOML section."""

    provider: str = "azure"
    """Active TTS provider. ``"azure"`` or ``"cartesia"``."""
    voice_id: str | None = None
    """Cartesia voice UUID (``provider = "cartesia"`` only)."""
    model: str | None = None
    """Cartesia model name (``provider = "cartesia"`` only)."""
    azure_voice: str = DEFAULT_AZURE_TTS_VOICE
    """Azure Neural voice name (``provider = "azure"`` only)."""
    greetings: tuple[str, ...] = ()
    """Pool of spoken greetings for voice-channel join.

    One picked at random when the familiar joins a voice channel.
    Empty tuple → fallback to hardcoded ``"Hello!"``.
    """


@dataclass(frozen=True)
class TypingSimulationConfig:
    """Per-chunk typing+delay simulation for text channels.

    Splits the LLM reply into paragraphs (with a sentence-split fallback
    for long paragraphs), then for each chunk: shows ``typing…`` for a
    length-proportional delay, sends the chunk, pauses briefly, repeats.

    A new user message on the channel cancels the remaining chunks; see
    :mod:`familiar_connect.text.delivery` for the tracker.

    :param enabled: master on/off for chunked delivery on this channel.
    :param chars_per_second: typing speed; ``~40`` ≈ 240 wpm.
    :param min_delay_s: floor per-chunk typing delay.
    :param max_delay_s: ceiling per-chunk typing delay.
    :param inter_line_pause_s: gap after each send before next ``typing…``.
    :param sentence_split_threshold: paragraph longer than this (chars)
        splits by sentence instead of going out as one chunk.
    """

    enabled: bool = False
    chars_per_second: float = 40.0
    min_delay_s: float = 0.8
    max_delay_s: float = 6.0
    inter_line_pause_s: float = 0.7
    sentence_split_threshold: int = 400


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
    interrupt_tolerance: InterruptTolerance = InterruptTolerance.average
    min_interruption_s: float = 2.0
    short_long_boundary_s: float = 30.0
    memory_writer_turn_threshold: int = 50
    """Run the memory writer pass after this many new turns."""
    memory_writer_idle_timeout: float = 1800.0
    """Run the memory writer pass after this many seconds of silence (30 min)."""
    llm: dict[str, LLMSlotConfig] = field(default_factory=dict)
    tts: TTSConfig = field(default_factory=TTSConfig)
    typing_simulation_overrides: dict[str, object] = field(default_factory=dict)
    """Partial ``[typing_simulation]`` overrides from ``character.toml``.

    Layered over :attr:`ChannelConfig.typing_simulation` (the per-mode
    default) before any per-channel TOML overrides apply. Empty dict
    when the character.toml omits the section entirely.
    """


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
    typing_simulation: TypingSimulationConfig = field(
        default_factory=TypingSimulationConfig,
    )


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
            # text-rp channels simulate natural typing by default;
            # operators can disable per familiar or per channel.
            typing_simulation=TypingSimulationConfig(enabled=True),
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

    voice_raw = data.get("voice", {})
    if not isinstance(voice_raw, dict):
        msg = f"[voice] must be a table, got {type(voice_raw).__name__}"
        raise ConfigError(msg)
    interruption_raw = voice_raw.get("interruption", {})
    if not isinstance(interruption_raw, dict):
        msg = (
            f"[voice.interruption] must be a table, "
            f"got {type(interruption_raw).__name__}"
        )
        raise ConfigError(msg)
    interrupt_tolerance, min_interruption_s, short_long_boundary_s = (
        _parse_voice_interruption(interruption_raw)
    )

    mw_section = data.get("memory_writer", {})
    if not isinstance(mw_section, dict):
        msg = f"[memory_writer] must be a table, got {type(mw_section).__name__}"
        raise ConfigError(msg)
    memory_writer_turn_threshold = int(mw_section.get("turn_threshold", 50))
    memory_writer_idle_timeout = float(mw_section.get("idle_timeout", 1800.0))

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

    ts_raw = data.get("typing_simulation", {})
    if not isinstance(ts_raw, dict):
        msg = f"[typing_simulation] must be a table, got {type(ts_raw).__name__}"
        raise ConfigError(msg)
    typing_simulation_overrides = _parse_typing_simulation_overrides(ts_raw)

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
        interrupt_tolerance=interrupt_tolerance,
        min_interruption_s=min_interruption_s,
        short_long_boundary_s=short_long_boundary_s,
        memory_writer_turn_threshold=memory_writer_turn_threshold,
        memory_writer_idle_timeout=memory_writer_idle_timeout,
        llm=llm,
        tts=tts,
        typing_simulation_overrides=typing_simulation_overrides,
    )


def load_channel_config(
    path: Path,
    *,
    character_overrides: dict[str, object] | None = None,
) -> ChannelConfig | None:
    """Load :class:`ChannelConfig` from *path*, or ``None`` if missing.

    Resolves ``typing_simulation`` by layering (low → high):
    per-mode default → *character_overrides* → channel ``[typing_simulation]``.

    :param character_overrides: partial overrides from ``character.toml``
        (typically :attr:`CharacterConfig.typing_simulation_overrides`).
    :raises ConfigError: if file exists but is invalid TOML, has an
        unknown mode, or the ``[typing_simulation]`` table fails validation.
    """
    data = _read_toml(path)
    if data is None:
        return None

    mode_raw = data.get("mode")
    if mode_raw is None:
        msg = f"channel config at {path} is missing required 'mode' key"
        raise ConfigError(msg)
    mode = _parse_mode(mode_raw, default=None)
    base = channel_config_for_mode(mode)

    ts_raw = data.get("typing_simulation", {})
    if not isinstance(ts_raw, dict):
        msg = f"[typing_simulation] must be a table, got {type(ts_raw).__name__}"
        raise ConfigError(msg)
    channel_overrides = _parse_typing_simulation_overrides(ts_raw)

    # resolve: per-mode default → character overrides → channel overrides.
    resolved_ts = _resolve_typing_simulation(
        base.typing_simulation,
        character_overrides=character_overrides or {},
        channel_overrides=channel_overrides,
    )
    return replace(base, typing_simulation=resolved_ts)


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


def _parse_typing_simulation_overrides(raw: dict) -> dict[str, object]:
    """Parse and validate the ``[typing_simulation]`` TOML section.

    Returns a dict of only the fields the user set — suitable for
    :func:`dataclasses.replace` against a :class:`TypingSimulationConfig`
    baseline. Missing keys are omitted (not set to defaults) so the
    baseline's values survive.

    :raises ConfigError: on unknown keys, wrong types, or out-of-range
        numeric values. ``min_delay_s > max_delay_s`` is checked only
        when both are present in *raw*.
    """
    if not raw:
        return {}

    valid_keys = {
        "enabled",
        "chars_per_second",
        "min_delay_s",
        "max_delay_s",
        "inter_line_pause_s",
        "sentence_split_threshold",
    }
    unknown = set(raw) - valid_keys
    if unknown:
        valid = ", ".join(sorted(valid_keys))
        msg = (
            f"unknown [typing_simulation] keys: {sorted(unknown)}; valid keys: {valid}"
        )
        raise ConfigError(msg)

    overrides: dict[str, object] = {}

    if "enabled" in raw:
        enabled_raw = raw["enabled"]
        if not isinstance(enabled_raw, bool):
            msg = (
                "[typing_simulation].enabled must be a boolean, "
                f"got {type(enabled_raw).__name__}"
            )
            raise ConfigError(msg)
        overrides["enabled"] = enabled_raw

    for numeric_key in (
        "chars_per_second",
        "min_delay_s",
        "max_delay_s",
        "inter_line_pause_s",
    ):
        if numeric_key not in raw:
            continue
        val = raw[numeric_key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            msg = (
                f"[typing_simulation].{numeric_key} must be a number, "
                f"got {type(val).__name__}"
            )
            raise ConfigError(msg)
        val = float(val)
        if numeric_key == "chars_per_second" and val <= 0:
            msg = f"[typing_simulation].chars_per_second must be > 0, got {val}"
            raise ConfigError(msg)
        if numeric_key != "chars_per_second" and val < 0:
            msg = f"[typing_simulation].{numeric_key} must be >= 0, got {val}"
            raise ConfigError(msg)
        overrides[numeric_key] = val

    if "sentence_split_threshold" in raw:
        thresh = raw["sentence_split_threshold"]
        if isinstance(thresh, bool) or not isinstance(thresh, int):
            msg = (
                "[typing_simulation].sentence_split_threshold must be an int, "
                f"got {type(thresh).__name__}"
            )
            raise ConfigError(msg)
        if thresh <= 0:
            msg = (
                "[typing_simulation].sentence_split_threshold must be > 0, "
                f"got {thresh}"
            )
            raise ConfigError(msg)
        overrides["sentence_split_threshold"] = thresh

    min_val = overrides.get("min_delay_s")
    max_val = overrides.get("max_delay_s")
    if isinstance(min_val, float) and isinstance(max_val, float) and min_val > max_val:
        msg = (
            "[typing_simulation].min_delay_s "
            f"({min_val}) must be <= max_delay_s ({max_val})"
        )
        raise ConfigError(msg)

    return overrides


def _resolve_typing_simulation(
    base: TypingSimulationConfig,
    *,
    character_overrides: dict[str, object],
    channel_overrides: dict[str, object],
) -> TypingSimulationConfig:
    """Apply character- then channel-level overrides on top of *base*.

    Also catches cross-layer ``min_delay_s > max_delay_s`` violations
    produced when one layer sets min and another sets max.
    """
    merged: dict[str, object] = {**character_overrides, **channel_overrides}
    result = replace(base, **merged)  # type: ignore[arg-type]
    if result.min_delay_s > result.max_delay_s:
        msg = (
            f"typing_simulation.min_delay_s ({result.min_delay_s}) must be "
            f"<= max_delay_s ({result.max_delay_s}) after layered overrides"
        )
        raise ConfigError(msg)
    return result


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

    voice_id = raw.get("voice_id")
    if voice_id is not None and not isinstance(voice_id, str):
        msg = f"[tts].voice_id must be a string, got {type(voice_id).__name__}"
        raise ConfigError(msg)

    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        msg = f"[tts].model must be a string, got {type(model).__name__}"
        raise ConfigError(msg)

    azure_voice_raw = raw.get("azure_voice", DEFAULT_AZURE_TTS_VOICE)
    if not isinstance(azure_voice_raw, str) or not azure_voice_raw:
        msg = "[tts].azure_voice must be a non-empty string"
        raise ConfigError(msg)

    greetings_raw = raw.get("greetings", [])
    if not isinstance(greetings_raw, list):
        msg = (
            f"[tts].greetings must be a list of strings, "
            f"got {type(greetings_raw).__name__}"
        )
        raise ConfigError(msg)
    greetings: list[str] = []
    for i, entry in enumerate(greetings_raw):
        if not isinstance(entry, str):
            msg = f"[tts].greetings[{i}] must be a string, got {type(entry).__name__}"
            raise ConfigError(msg)
        if not entry:
            msg = f"[tts].greetings[{i}] must be non-empty"
            raise ConfigError(msg)
        greetings.append(entry)

    return TTSConfig(
        provider=provider_raw,
        voice_id=voice_id,
        model=model,
        azure_voice=azure_voice_raw,
        greetings=tuple(greetings),
    )


def _parse_voice_interruption(
    raw: dict,
) -> tuple[InterruptTolerance, float, float]:
    """Parse the ``[voice.interruption]`` TOML table.

    Returns ``(tolerance, min_interruption_s, short_long_boundary_s)``. Any
    missing key falls back to the :class:`CharacterConfig` defaults
    (``average`` / 1.5 / 4.0).
    """
    tolerance_raw = raw.get("interrupt_tolerance")
    if tolerance_raw is None:
        tolerance = InterruptTolerance.average
    elif not isinstance(tolerance_raw, str):
        msg = (
            "[voice.interruption].interrupt_tolerance must be a string, "
            f"got {type(tolerance_raw).__name__}"
        )
        raise ConfigError(msg)
    else:
        try:
            tolerance = InterruptTolerance(tolerance_raw)
        except ValueError as exc:
            valid = ", ".join(t.value for t in InterruptTolerance)
            msg = (
                f"unknown interrupt_tolerance tier {tolerance_raw!r}; "
                f"valid options: {valid}"
            )
            raise ConfigError(msg) from exc

    min_raw = raw.get("min_interruption_s", 2.0)
    if not isinstance(min_raw, (int, float)) or isinstance(min_raw, bool):
        msg = (
            "[voice.interruption].min_interruption_s must be a number, "
            f"got {type(min_raw).__name__}"
        )
        raise ConfigError(msg)
    min_interruption_s = float(min_raw)
    if min_interruption_s < 0:
        msg = (
            "[voice.interruption].min_interruption_s must be non-negative, "
            f"got {min_interruption_s}"
        )
        raise ConfigError(msg)

    boundary_raw = raw.get("short_long_boundary_s", 30.0)
    if not isinstance(boundary_raw, (int, float)) or isinstance(boundary_raw, bool):
        msg = (
            "[voice.interruption].short_long_boundary_s must be a number, "
            f"got {type(boundary_raw).__name__}"
        )
        raise ConfigError(msg)
    short_long_boundary_s = float(boundary_raw)
    if short_long_boundary_s <= min_interruption_s:
        msg = (
            "[voice.interruption].short_long_boundary_s "
            f"({short_long_boundary_s}) must exceed min_interruption_s "
            f"({min_interruption_s})"
        )
        raise ConfigError(msg)

    return tolerance, min_interruption_s, short_long_boundary_s


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
