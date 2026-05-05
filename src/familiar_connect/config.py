"""Per-character configuration from TOML.

Loads ``character.toml`` once on startup, deep-merged over the
``_default/character.toml`` defaults.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.budget import TierBudget

if TYPE_CHECKING:
    from pathlib import Path


BUDGET_TIER_NAMES: frozenset[str] = frozenset({"voice", "text", "background"})
"""Canonical assembly-tier names matching :class:`LLMSlotConfig` slots.

Voice and text are reply tiers (consumed by responders); background
is for offline workers (summary, fact extraction, dossier). Each
tier has its own :class:`TierBudget` envelope.
"""


class ConfigError(Exception):
    """Raised when a config file is malformed or references an unknown value."""


LLM_SLOT_NAMES: frozenset[str] = frozenset({"fast", "prose", "background"})
"""Canonical LLM call-site slot names.

Tiered by latency / quality:

* ``fast`` — voice replies; reasoning off, tools off.
* ``prose`` — text-channel replies; reasoning on, tools off.
* ``background`` — summaries, fact extraction, dossier; reasoning
  on, tools on (surface only — no tool wiring yet).
"""

REASONING_LEVELS: frozenset[str] = frozenset({"off", "low", "medium", "high"})
"""Allowed values for ``[llm.<slot>].reasoning``.

* ``"off"`` — explicitly suppress (OpenRouter ``reasoning.exclude``).
* ``"low"`` / ``"medium"`` / ``"high"`` — effort levels.
* omitted (``None``) — defer to the model's default.
"""


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
    # OpenRouter reasoning effort. ``None`` = model default.
    # see ``REASONING_LEVELS`` for allowed strings.
    reasoning: str | None = None
    # surface-only flag for now — call sites haven't wired tools yet.
    tool_calling: bool = False


_TTS_PROVIDERS: frozenset[str] = frozenset({"azure", "cartesia", "gemini"})

_TURN_DETECTION_STRATEGIES: frozenset[str] = frozenset({"deepgram", "ten+smart_turn"})


@dataclass(frozen=True)
class LocalTurnConfig:
    """V1 local-turn-detection knobs from ``[providers.turn_detection.local]``.

    Read only when ``[providers.turn_detection].strategy = "ten+smart_turn"``.
    Defaults match the V1 field-tested values; rarely need tweaking.

    Smart Turn weights are pulled from HuggingFace on first use (cached
    under ``~/.cache/huggingface``). Default filename is the CPU ONNX
    export — switch to ``smart-turn-v3.2-gpu.onnx`` if ``onnxruntime-gpu``
    is installed.
    """

    smart_turn_repo_id: str = "pipecat-ai/smart-turn-v3"
    smart_turn_filename: str = "smart-turn-v3.2-cpu.onnx"
    silence_ms: int = 200
    speech_start_ms: int = 100
    vad_threshold: float = 0.5
    smart_turn_threshold: float = 0.5
    vad_hop_size: int = 256


@dataclass(frozen=True)
class TurnDetectionConfig:
    """Turn-detection strategy from ``[providers.turn_detection]``."""

    strategy: str = "deepgram"
    local: LocalTurnConfig = field(default_factory=LocalTurnConfig)


# V3 phase 2 added "parakeet"; phase 3 added "faster_whisper".
_STT_BACKENDS: frozenset[str] = frozenset({"deepgram", "parakeet", "faster_whisper"})


@dataclass(frozen=True)
class DeepgramSTTConfig:
    """Deepgram knobs from ``[providers.stt.deepgram]``.

    Non-secret. ``DEEPGRAM_API_KEY`` is the only env input.
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
class FasterWhisperSTTConfig:
    """FasterWhisper (CTranslate2) knobs from ``[providers.stt.faster_whisper]``.

    Local CTranslate2-backed Whisper. Same pairing requirement as
    Parakeet — no internal silence detector, so ``finalize()`` is
    driven by the local turn detector.
    """

    model_size: str = "small"  # "tiny" | "base" | "small" | "medium" | "large-v3"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    compute_type: str = "auto"  # "auto" | "int8" | "float16" | "float32"
    language: str = "en"
    idle_close_s: float = 30.0


@dataclass(frozen=True)
class STTConfig:
    """STT backend selector + per-backend knobs from ``[providers.stt]``.

    V3 phase 1 lifted the selector behind ``Transcriber`` Protocol;
    phase 2 added ``parakeet``; phase 3 added ``faster_whisper``.
    """

    backend: str = "deepgram"
    deepgram: DeepgramSTTConfig = field(default_factory=DeepgramSTTConfig)
    parakeet: ParakeetSTTConfig = field(default_factory=ParakeetSTTConfig)
    faster_whisper: FasterWhisperSTTConfig = field(
        default_factory=FasterWhisperSTTConfig
    )


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
class DiscordTextConfig:
    """``[discord.text]`` knobs — typing-indicator + interruption behavior.

    ``respond_to_typing``: treat other users' typing-start events as
    interruptions of the bot's in-flight reply (cancels the active
    :class:`TurnScope`). Default ``True`` — respect users typing.
    ``typing_backoff_initial_s`` / ``typing_backoff_max_s``: when
    another bot (e.g. another familiar-connect instance) is typing in
    the same channel, hold off responding for an exponentially
    increasing window — initial doubles toward max — to avoid two
    bots pingponging on each other's typing events.
    """

    respond_to_typing: bool = True
    typing_backoff_initial_s: float = 1.0
    typing_backoff_max_s: float = 30.0


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    """Ranking weights for :class:`RagContextLayer` (M2).

    Default keeps pre-M2 ordering (BM25-only). Operators opt into
    importance / recency by raising the matching weight in
    ``[memory.retrieval]``.

    :param bm25_weight: ranking weight on BM25 quality (best=1.0,
        worst=0.0 within the candidate batch).
    :param recency_weight: ranking weight on fact-id rank within
        the candidate batch (newest=1.0).
    :param importance_weight: ranking weight on the extractor's 1-10
        importance hint (``importance/10``). Legacy / unscored facts
        get the neutral midpoint.
    :param embedding_weight: ranking weight on cosine similarity to
        the cue embedding (M6). Requires
        ``[providers.embedding].backend != "off"``; without an
        embedder the layer skips this signal even when the weight is
        positive (warned once at startup).
    """

    bm25_weight: float = 1.0
    recency_weight: float = 0.0
    importance_weight: float = 0.0
    embedding_weight: float = 0.0


@dataclass(frozen=True)
class MemoryProvidersConfig:
    """Memory-projector selector from ``[providers.memory]`` (M5).

    Lifts the existing writers (`SummaryWorker`, `FactExtractor`,
    `PeopleDossierWorker`, `ReflectionWorker`) behind a
    :class:`MemoryProjector` Protocol so operators swap the strategy
    via TOML. Default keeps every shipped projector.

    :param projectors: ordered tuple of projector names. Each must be
        registered in
        :mod:`familiar_connect.processors.projectors`. Empty tuple
        disables all memory projection.
    """

    projectors: tuple[str, ...] = (
        "rolling_summary",
        "rich_note",
        "people_dossier",
        "reflection",
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedder backend selector from ``[providers.embedding]`` (M6).

    Picks the text → vector backend used by the ``fact_embedding``
    projector and (when ``[memory.retrieval].embedding_weight > 0``)
    by :class:`RagContextLayer` at rerank time.

    :param backend: name registered in
        :mod:`familiar_connect.embedding.factory`. ``"off"`` (default)
        disables the seam end to end — projector raises if listed,
        RAG layer skips embedding signal.
    :param dim: dimensionality hint for backends that accept one
        (``hash``). Real backends with a fixed model dim ignore this.
    """

    backend: str = "off"
    dim: int = 256


@dataclass(frozen=True)
class ChannelOverrides:
    """Per-channel overrides for latency-sensitive knobs.

    Layered over global defaults — test channels tunable independently.
    See plan § Design.4 *Channel-aware composition order*.

    history_window_size: overrides the tier default
    (:attr:`CharacterConfig.voice_window_size` /
    :attr:`CharacterConfig.text_window_size`).
    prompt_layers: ordered layer names; ``None`` inherits default order.
    message_rendering: ``"prefixed"`` (always include ``[display_name]``
    prefix) or ``"name_only"`` (rely on OpenAI ``name`` field — saves
    tokens in DMs).
    """

    history_window_size: int | None = None
    prompt_layers: tuple[str, ...] | None = None
    message_rendering: str | None = None


def _default_budgets() -> dict[str, TierBudget]:
    """Programmatic fallback for ``CharacterConfig()`` without TOML.

    The shipped per-tier values live in
    ``data/familiars/_default/character.toml`` and overlay these at
    load time. Tests that bypass TOML loading get the dataclass
    defaults — voice-tier sized; pass an explicit ``TierBudget`` if
    you need other shapes.
    """
    return {tier: TierBudget() for tier in BUDGET_TIER_NAMES}


@dataclass(frozen=True)
class CharacterConfig:
    """Config loaded once per install from ``character.toml``."""

    # Hard cap on history turns. The token-aware budget below usually
    # bites first; keep this as a safety net (and a way to force the
    # absolute upper bound on prompt size).
    voice_window_size: int = 100
    text_window_size: int = 200
    # Consecutive same-speaker voice fragments within this gap (seconds)
    # collapse into one rendered history message. 0 disables.
    recent_history_coalesce_max_gap_seconds: float = 45.0
    display_tz: str = "UTC"
    aliases: list[str] = field(default_factory=list)
    llm: dict[str, LLMSlotConfig] = field(default_factory=dict)
    tts: TTSConfig = field(default_factory=TTSConfig)
    channels: dict[int, ChannelOverrides] = field(default_factory=dict)
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    # Per-tier token envelopes for the Budgeter. Operators tune
    # ``total_tokens`` per tier; sub-caps derive from it unless
    # explicitly overridden.
    budgets: dict[str, TierBudget] = field(default_factory=_default_budgets)
    discord_text: DiscordTextConfig = field(default_factory=DiscordTextConfig)
    # Retrieval ranking weights — see :class:`MemoryRetrievalConfig`.
    memory_retrieval: MemoryRetrievalConfig = field(
        default_factory=MemoryRetrievalConfig
    )
    # M5 — memory-projector selection from [providers.memory].
    memory_providers: MemoryProvidersConfig = field(
        default_factory=MemoryProvidersConfig
    )
    # M6 — embedder backend selection from [providers.embedding].
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def for_channel(self, channel_id: int | None) -> ChannelOverrides:
        """Return overrides for ``channel_id``; empty overrides if none."""
        if channel_id is None:
            return ChannelOverrides()
        return self.channels.get(channel_id, ChannelOverrides())

    def voice_window_for(self, channel_id: int | None) -> int:
        """Voice-tier window for ``channel_id``; channel override wins."""
        override = self.for_channel(channel_id).history_window_size
        return override if override is not None else self.voice_window_size

    def text_window_for(self, channel_id: int | None) -> int:
        """Text-tier window for ``channel_id``; channel override wins."""
        override = self.for_channel(channel_id).history_window_size
        return override if override is not None else self.text_window_size


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
    voice_window_size, text_window_size = _parse_history_windows(history_section)
    recent_history_coalesce_max_gap_seconds = _parse_recent_history_coalesce_gap(
        history_section
    )

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

    memory_providers_raw = providers_raw.get("memory", {})
    if not isinstance(memory_providers_raw, dict):
        msg = (
            "[providers.memory] must be a table, "
            f"got {type(memory_providers_raw).__name__}"
        )
        raise ConfigError(msg)
    memory_providers = _parse_memory_providers(memory_providers_raw)

    embedding_raw = providers_raw.get("embedding", {})
    if not isinstance(embedding_raw, dict):
        msg = (
            f"[providers.embedding] must be a table, got {type(embedding_raw).__name__}"
        )
        raise ConfigError(msg)
    embedding = _parse_embedding_config(embedding_raw)

    budget_raw = data.get("budget", {})
    if not isinstance(budget_raw, dict):
        msg = f"[budget] must be a table, got {type(budget_raw).__name__}"
        raise ConfigError(msg)
    budgets = _parse_budgets(budget_raw)

    discord_raw = data.get("discord", {})
    if not isinstance(discord_raw, dict):
        msg = f"[discord] must be a table, got {type(discord_raw).__name__}"
        raise ConfigError(msg)
    discord_text_raw = discord_raw.get("text", {})
    if not isinstance(discord_text_raw, dict):
        msg = f"[discord.text] must be a table, got {type(discord_text_raw).__name__}"
        raise ConfigError(msg)
    discord_text = _parse_discord_text_config(discord_text_raw)

    memory_raw = data.get("memory", {})
    if not isinstance(memory_raw, dict):
        msg = f"[memory] must be a table, got {type(memory_raw).__name__}"
        raise ConfigError(msg)
    retrieval_raw = memory_raw.get("retrieval", {})
    if not isinstance(retrieval_raw, dict):
        msg = f"[memory.retrieval] must be a table, got {type(retrieval_raw).__name__}"
        raise ConfigError(msg)
    memory_retrieval = _parse_memory_retrieval(retrieval_raw)

    return CharacterConfig(
        voice_window_size=voice_window_size,
        text_window_size=text_window_size,
        recent_history_coalesce_max_gap_seconds=(
            recent_history_coalesce_max_gap_seconds
        ),
        display_tz=display_tz,
        aliases=aliases,
        llm=llm,
        tts=tts,
        channels=channels,
        turn_detection=turn_detection,
        stt=stt,
        budgets=budgets,
        discord_text=discord_text,
        memory_retrieval=memory_retrieval,
        memory_providers=memory_providers,
        embedding=embedding,
    )


_DISCORD_TEXT_FIELDS: frozenset[str] = frozenset({
    "respond_to_typing",
    "typing_backoff_initial_s",
    "typing_backoff_max_s",
})


def _parse_discord_text_config(raw: dict) -> DiscordTextConfig:
    """Parse and validate ``[discord.text]``."""
    unknown = set(raw) - _DISCORD_TEXT_FIELDS
    if unknown:
        bad = ", ".join(sorted(unknown))
        msg = f"[discord.text] has unknown keys: {bad}"
        raise ConfigError(msg)
    defaults = DiscordTextConfig()

    rt_raw = raw.get("respond_to_typing", defaults.respond_to_typing)
    if not isinstance(rt_raw, bool):
        msg = (
            "[discord.text].respond_to_typing must be a bool, "
            f"got {type(rt_raw).__name__}"
        )
        raise ConfigError(msg)

    def _positive_float(key: str, fallback: float) -> float:
        v = raw.get(key, fallback)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            msg = f"[discord.text].{key} must be a number, got {type(v).__name__}"
            raise ConfigError(msg)
        if v <= 0:
            msg = f"[discord.text].{key} must be positive, got {v}"
            raise ConfigError(msg)
        return float(v)

    initial_s = _positive_float(
        "typing_backoff_initial_s", defaults.typing_backoff_initial_s
    )
    max_s = _positive_float("typing_backoff_max_s", defaults.typing_backoff_max_s)
    if max_s < initial_s:
        msg = (
            "[discord.text].typing_backoff_max_s must be >= "
            f"typing_backoff_initial_s, got {max_s} < {initial_s}"
        )
        raise ConfigError(msg)
    return DiscordTextConfig(
        respond_to_typing=rt_raw,
        typing_backoff_initial_s=initial_s,
        typing_backoff_max_s=max_s,
    )


def _parse_budgets(raw: dict) -> dict[str, TierBudget]:
    """Parse ``[budget.<tier>]`` blocks; tiers must be canonical."""
    out: dict[str, TierBudget] = dict(_default_budgets())
    for tier, section in raw.items():
        if tier not in BUDGET_TIER_NAMES:
            valid = ", ".join(sorted(BUDGET_TIER_NAMES))
            msg = f"unknown budget tier {tier!r}; valid tiers: {valid}"
            raise ConfigError(msg)
        if not isinstance(section, dict):
            msg = f"[budget.{tier}] must be a table, got {type(section).__name__}"
            raise ConfigError(msg)
        out[tier] = _parse_tier_budget(tier, section, default=out[tier])
    return out


_BUDGET_FIELDS: tuple[str, ...] = (
    "total_tokens",
    "recent_history_tokens",
    "rag_tokens",
    "dossier_tokens",
    "summary_tokens",
    "cross_channel_tokens",
    "reflection_tokens",
    "lorebook_tokens",
    "max_history_turns",
    "max_rag_turns",
    "max_rag_facts",
    "max_dossier_people",
    "max_reflections",
    "max_lorebook_entries",
)


def _parse_tier_budget(tier: str, raw: dict, *, default: TierBudget) -> TierBudget:
    """Validate and merge ``[budget.<tier>]`` over the default envelope.

    Each field is a hard positive int. Keys present in ``raw`` win;
    keys absent inherit ``default`` (which itself comes from the
    deep-merged ``_default/character.toml``).
    """
    unknown = set(raw) - set(_BUDGET_FIELDS)
    if unknown:
        bad = ", ".join(sorted(unknown))
        msg = f"[budget.{tier}] has unknown keys: {bad}"
        raise ConfigError(msg)

    def _positive_int(key: str, fallback: int) -> int:
        if key not in raw:
            return fallback
        v = raw[key]
        if isinstance(v, bool) or not isinstance(v, int):
            msg = (
                f"[budget.{tier}].{key} must be a positive integer, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        if v <= 0:
            msg = f"[budget.{tier}].{key} must be positive, got {v}"
            raise ConfigError(msg)
        return v

    return TierBudget(**{
        key: _positive_int(key, getattr(default, key)) for key in _BUDGET_FIELDS
    })


_RETRIEVAL_FIELDS: tuple[str, ...] = (
    "bm25_weight",
    "recency_weight",
    "importance_weight",
    "embedding_weight",
)


def _parse_memory_retrieval(raw: dict) -> MemoryRetrievalConfig:
    """Validate ``[memory.retrieval]``; non-negative floats only."""
    unknown = set(raw) - set(_RETRIEVAL_FIELDS)
    if unknown:
        bad = ", ".join(sorted(unknown))
        msg = f"[memory.retrieval] has unknown keys: {bad}"
        raise ConfigError(msg)
    defaults = MemoryRetrievalConfig()
    out: dict[str, float] = {f: float(getattr(defaults, f)) for f in _RETRIEVAL_FIELDS}
    for key in _RETRIEVAL_FIELDS:
        if key not in raw:
            continue
        v = raw[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            msg = (
                f"[memory.retrieval].{key} must be a non-negative number, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        if v < 0:
            msg = f"[memory.retrieval].{key} must be non-negative, got {v}"
            raise ConfigError(msg)
        out[key] = float(v)
    return MemoryRetrievalConfig(**out)


_MEMORY_PROVIDERS_FIELDS: frozenset[str] = frozenset({"projectors"})


def _parse_memory_providers(raw: dict) -> MemoryProvidersConfig:
    """Validate ``[providers.memory]``; reject unknown projector names.

    Names are checked against
    :func:`familiar_connect.processors.projectors.known_projectors` so
    a typo fails loudly at config load rather than silently dropping a
    writer.
    """
    unknown = set(raw) - _MEMORY_PROVIDERS_FIELDS
    if unknown:
        bad = ", ".join(sorted(unknown))
        msg = f"[providers.memory] has unknown keys: {bad}"
        raise ConfigError(msg)
    if "projectors" not in raw:
        return MemoryProvidersConfig()
    raw_projectors = raw["projectors"]
    if not isinstance(raw_projectors, list) or not all(
        isinstance(p, str) for p in raw_projectors
    ):
        msg = "[providers.memory].projectors must be a list of strings"
        raise ConfigError(msg)
    # Deferred to avoid pulling the registry into module import time —
    # config.py is loaded by lots of test helpers.
    from familiar_connect.processors.projectors import known_projectors  # noqa: PLC0415

    valid_names = known_projectors()
    for name in raw_projectors:
        if name not in valid_names:
            valid = ", ".join(sorted(valid_names)) or "(none)"
            msg = (
                f"[providers.memory].projectors lists unknown memory "
                f"projector {name!r}; valid: {valid}"
            )
            raise ConfigError(msg)
    return MemoryProvidersConfig(projectors=tuple(raw_projectors))


_EMBEDDING_FIELDS: frozenset[str] = frozenset({"backend", "dim"})


def _parse_embedding_config(raw: dict) -> EmbeddingConfig:
    """Validate ``[providers.embedding]``.

    ``backend`` is checked against
    :func:`familiar_connect.embedding.factory.known_embedders` so a
    typo fails loudly at config load. ``dim`` must be a positive int.
    """
    unknown = set(raw) - _EMBEDDING_FIELDS
    if unknown:
        bad = ", ".join(sorted(unknown))
        msg = f"[providers.embedding] has unknown keys: {bad}"
        raise ConfigError(msg)
    defaults = EmbeddingConfig()
    backend = raw.get("backend", defaults.backend)
    if not isinstance(backend, str):
        msg = (
            f"[providers.embedding].backend must be a string, "
            f"got {type(backend).__name__}"
        )
        raise ConfigError(msg)
    # Deferred to keep config.py importable without the embedding
    # registry — same shape as the projector validator.
    from familiar_connect.embedding.factory import known_embedders  # noqa: PLC0415

    valid = known_embedders()
    if backend not in valid:
        valid_list = ", ".join(sorted(valid)) or "(none)"
        msg = (
            f"[providers.embedding].backend = {backend!r} is unknown; "
            f"valid: {valid_list}"
        )
        raise ConfigError(msg)
    dim_raw = raw.get("dim", defaults.dim)
    if isinstance(dim_raw, bool) or not isinstance(dim_raw, int):
        msg = (
            f"[providers.embedding].dim must be a positive integer, "
            f"got {type(dim_raw).__name__}"
        )
        raise ConfigError(msg)
    if dim_raw <= 0:
        msg = f"[providers.embedding].dim must be > 0, got {dim_raw}"
        raise ConfigError(msg)
    return EmbeddingConfig(backend=backend, dim=int(dim_raw))


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
        reasoning_raw = section.get("reasoning")
        reasoning: str | None
        if reasoning_raw is None:
            reasoning = None
        elif isinstance(reasoning_raw, str):
            if reasoning_raw not in REASONING_LEVELS:
                valid = ", ".join(sorted(REASONING_LEVELS))
                msg = (
                    f"[llm.{name}].reasoning {reasoning_raw!r} unknown; "
                    f"valid options: {valid}"
                )
                raise ConfigError(msg)
            reasoning = reasoning_raw
        else:
            msg = (
                f"[llm.{name}].reasoning must be a string, "
                f"got {type(reasoning_raw).__name__}"
            )
            raise ConfigError(msg)
        tool_calling_raw = section.get("tool_calling", False)
        if not isinstance(tool_calling_raw, bool):
            msg = (
                f"[llm.{name}].tool_calling must be a bool, "
                f"got {type(tool_calling_raw).__name__}"
            )
            raise ConfigError(msg)
        slots[name] = LLMSlotConfig(
            model=model,
            temperature=temperature,
            provider_order=provider_order,
            provider_allow_fallbacks=allow_fallbacks_raw,
            reasoning=reasoning,
            tool_calling=tool_calling_raw,
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


def _parse_history_windows(raw: dict) -> tuple[int, int]:
    """Parse split history-window keys; reject the retired ``window_size``."""
    if "window_size" in raw:
        msg = (
            "[providers.history].window_size has been split into "
            "voice_window_size and text_window_size. Replace the legacy "
            "key with both. See docs/architecture/tuning.md § History windows."
        )
        raise ConfigError(msg)

    def _positive_int(key: str, default: int) -> int:
        v = raw.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int):
            msg = (
                f"[providers.history].{key} must be a positive integer, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        if v <= 0:
            msg = f"[providers.history].{key} must be positive, got {v}"
            raise ConfigError(msg)
        return v

    return _positive_int("voice_window_size", 20), _positive_int("text_window_size", 30)


def _parse_recent_history_coalesce_gap(raw: dict) -> float:
    """Parse ``[providers.history].coalesce_max_gap_seconds`` (≥ 0)."""
    default = 45.0
    if "coalesce_max_gap_seconds" not in raw:
        return default
    v = raw["coalesce_max_gap_seconds"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        msg = (
            "[providers.history].coalesce_max_gap_seconds must be a number, "
            f"got {type(v).__name__}"
        )
        raise ConfigError(msg)
    if v < 0:
        msg = f"[providers.history].coalesce_max_gap_seconds must be >= 0, got {v}"
        raise ConfigError(msg)
    return float(v)


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

    local_raw = raw.get("local", {})
    if not isinstance(local_raw, dict):
        msg = (
            f"[providers.turn_detection.local] must be a table, "
            f"got {type(local_raw).__name__}"
        )
        raise ConfigError(msg)
    local = _parse_local_turn_config(local_raw)
    return TurnDetectionConfig(strategy=strategy_raw, local=local)


def _parse_local_turn_config(raw: dict) -> LocalTurnConfig:
    """Parse ``[providers.turn_detection.local]`` knobs; validate types."""
    defaults = LocalTurnConfig()

    def _str(key: str, default: str) -> str:
        v = raw.get(key, default)
        if not isinstance(v, str) or not v:
            msg = f"[providers.turn_detection.local].{key} must be a non-empty string"
            raise ConfigError(msg)
        return v

    def _int(key: str, default: int) -> int:
        v = raw.get(key, default)
        if not isinstance(v, int) or isinstance(v, bool):
            msg = (
                f"[providers.turn_detection.local].{key} must be an integer, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        return v

    def _float(key: str, default: float) -> float:
        v = raw.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            msg = (
                f"[providers.turn_detection.local].{key} must be a number, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        return float(v)

    return LocalTurnConfig(
        smart_turn_repo_id=_str("smart_turn_repo_id", defaults.smart_turn_repo_id),
        smart_turn_filename=_str("smart_turn_filename", defaults.smart_turn_filename),
        silence_ms=_int("silence_ms", defaults.silence_ms),
        speech_start_ms=_int("speech_start_ms", defaults.speech_start_ms),
        vad_threshold=_float("vad_threshold", defaults.vad_threshold),
        smart_turn_threshold=_float(
            "smart_turn_threshold", defaults.smart_turn_threshold
        ),
        vad_hop_size=_int("vad_hop_size", defaults.vad_hop_size),
    )


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

    fw_raw = raw.get("faster_whisper", {})
    if not isinstance(fw_raw, dict):
        msg = (
            f"[providers.stt.faster_whisper] must be a table, "
            f"got {type(fw_raw).__name__}"
        )
        raise ConfigError(msg)
    faster_whisper = _parse_faster_whisper_stt_config(fw_raw)
    return STTConfig(
        backend=backend_raw,
        deepgram=deepgram,
        parakeet=parakeet,
        faster_whisper=faster_whisper,
    )


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


def _parse_faster_whisper_stt_config(raw: dict) -> FasterWhisperSTTConfig:
    """Parse ``[providers.stt.faster_whisper]`` knobs; validate types."""
    defaults = FasterWhisperSTTConfig()

    def _str(key: str, default: str) -> str:
        v = raw.get(key, default)
        if not isinstance(v, str) or not v:
            msg = f"[providers.stt.faster_whisper].{key} must be a non-empty string"
            raise ConfigError(msg)
        return v

    def _float(key: str, default: float) -> float:
        v = raw.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            msg = (
                f"[providers.stt.faster_whisper].{key} must be a number, "
                f"got {type(v).__name__}"
            )
            raise ConfigError(msg)
        return float(v)

    return FasterWhisperSTTConfig(
        model_size=_str("model_size", defaults.model_size),
        device=_str("device", defaults.device),
        compute_type=_str("compute_type", defaults.compute_type),
        language=_str("language", defaults.language),
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
