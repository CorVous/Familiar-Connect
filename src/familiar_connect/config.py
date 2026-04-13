"""Per-character and per-channel configuration loaded from TOML sidecars.

Step 7 of docs/architecture/context-pipeline.md. The bot reads two
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


class Interjection(Enum):
    """How patient the familiar is before the side model is consulted.

    Controls the message-count threshold at which the monitor asks the
    side model whether to interject. Higher tiers evaluate sooner and
    more frequently (shorter starting interval). After each declined
    check the interval shrinks by 3, flooring at 3.

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
        "memory_writer",
        "interjection_decision",
    },
)
"""Canonical set of LLM call-site slot names.

Every LLM call site in the codebase resolves its ``LLMClient`` from
:attr:`CharacterConfig.llm` by one of these names. A user's
``character.toml`` may omit individual slots — missing slots inherit
from ``data/familiars/_default/character.toml``. Adding a new call
site means adding a name here, wiring it in the familiar bundle,
and adding the default in ``_default/character.toml``.
"""


@dataclass(frozen=True)
class LLMSlotConfig:
    """Per-call-site LLM config loaded from a ``[llm.<slot>]`` TOML section.

    :param model: OpenRouter-style model id (e.g. ``"z-ai/glm-5.1"``).
    :param temperature: Sampling temperature. May be ``None`` if the
        call site should rely on the provider's own default.
    """

    model: str
    temperature: float | None = None


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech config loaded from the ``[tts]`` TOML section.

    :param voice_id: Cartesia voice id. ``None`` means no TTS at all.
    :param model: Cartesia model name (e.g. ``"sonic-3"``).
    """

    voice_id: str | None = None
    model: str | None = None


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
    :param aliases: Additional names the familiar responds to (beyond
        the ``familiar_id``). Used for direct-address detection.
    :param chattiness: Free-text personality trait describing the
        familiar's conversational disposition. Fed to the interjection
        monitor's evaluation prompt.
    :param interjection: Controls how long the familiar waits before
        the interjection monitor is consulted during an active
        conversation.
    :param lull_timeout: Seconds of silence before the lull evaluation
        fires.
    :param llm: ``slot_name -> LLMSlotConfig`` map for every LLM call
        site. Populated by the loader; slots missing from the user's
        ``character.toml`` fall back to ``_default/character.toml``.
    :param tts: :class:`TTSConfig` for Cartesia voice output.
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
    lull_timeout: float = 2.0
    memory_writer_turn_threshold: int = 50
    """Run the memory writer pass after this many new turns."""
    memory_writer_idle_timeout: float = 1800.0
    """Run the memory writer pass after this many seconds of silence (30 min)."""
    llm: dict[str, LLMSlotConfig] = field(default_factory=dict)
    tts: TTSConfig = field(default_factory=TTSConfig)


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
    """Load a :class:`CharacterConfig` from *path*, merged over *defaults_path*.

    The loader reads two TOML files and deep-merges the user's on top
    of the default profile:

    1. ``defaults_path`` — the repo's reference profile at
       ``data/familiars/_default/character.toml``. **Required**:
       missing or malformed raises :class:`ConfigError` because it's
       a checked-in repo asset, not user data.
    2. ``path`` — the active familiar's ``character.toml``. Missing
       is fine (the defaults alone produce a valid config). Present
       values override the defaults nested per-key, so a user who
       only wants to tweak ``[llm.main_prose].model`` doesn't have
       to restate the other five LLM slots.

    :raises ConfigError: If the default profile is missing, either
        file is invalid TOML, ``[llm.<slot>]`` names an unknown slot,
        or any parsed value fails validation.
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

    lull_timeout = float(data.get("lull_timeout", 2.0))

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

    return CharacterConfig(
        default_mode=default_mode,
        history_window_size=history_window_size,
        depth_inject_position=depth_inject_position,
        depth_inject_role=depth_inject_role,
        display_tz=display_tz,
        aliases=aliases,
        chattiness=chattiness,
        interjection=interjection,
        lull_timeout=lull_timeout,
        memory_writer_turn_threshold=memory_writer_turn_threshold,
        memory_writer_idle_timeout=memory_writer_idle_timeout,
        llm=llm,
        tts=tts,
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* layered on top of *base*.

    Nested dicts are merged key-by-key; non-dict values (including
    lists) are replaced wholesale. Neither input is mutated.
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
