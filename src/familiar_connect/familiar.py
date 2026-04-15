"""Runtime bundle for the single active character.

One process runs one character (selected by ``FAMILIAR_ID``).
:class:`Familiar` carries config, memory store, history store,
registered providers/processors, subscriptions, and channel configs.

- :meth:`load_from_disk` — sole constructor; walks ``data/familiars/<id>/``
- :meth:`build_pipeline` — per-turn :class:`ContextPipeline` filtered by channel mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect.channel_config import ChannelConfigStore
from familiar_connect.chattiness import ConversationMonitor
from familiar_connect.config import load_character_config
from familiar_connect.context.pipeline import ContextPipeline
from familiar_connect.context.processors.recast import RecastPostProcessor
from familiar_connect.context.processors.stepped_thinking import (
    SteppedThinkingPreProcessor,
)
from familiar_connect.context.providers.character import CharacterProvider
from familiar_connect.context.providers.content_search import ContentSearchProvider
from familiar_connect.context.providers.history import HistoryProvider
from familiar_connect.context.providers.mode_instructions import (
    ModeInstructionProvider,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.memory.scheduler import MemoryWriterScheduler
from familiar_connect.memory.store import MemoryStore
from familiar_connect.memory.writer import MemoryWriter
from familiar_connect.metrics import MetricsCollector, NullCollector
from familiar_connect.mood import MoodEvaluator
from familiar_connect.subscriptions import SubscriptionRegistry
from familiar_connect.voice.interruption import ResponseTrackerRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.config import ChannelConfig, CharacterConfig
    from familiar_connect.context.protocols import (
        ContextProvider,
        PostProcessor,
        PreProcessor,
    )
    from familiar_connect.llm import LLMClient
    from familiar_connect.transcription import DeepgramTranscriber
    from familiar_connect.tts import CartesiaTTSClient


@dataclass
class Familiar:
    """Runtime bundle for one character.

    :param transcriber: when ``None``, voice subscription joins for TTS
        playback only — no incoming audio is transcribed.
    """

    id: str
    root: Path
    config: CharacterConfig
    memory_store: MemoryStore
    history_store: HistoryStore
    llm_clients: dict[str, LLMClient]
    tts_client: CartesiaTTSClient | None
    transcriber: DeepgramTranscriber | None
    providers: dict[str, ContextProvider]
    pre_processors: dict[str, PreProcessor]
    post_processors: dict[str, PostProcessor]
    subscriptions: SubscriptionRegistry
    channel_configs: ChannelConfigStore
    monitor: ConversationMonitor
    memory_writer_scheduler: MemoryWriterScheduler
    tracker_registry: ResponseTrackerRegistry = field(
        default_factory=ResponseTrackerRegistry,
    )
    """Per-guild :class:`ResponseTracker` lookup; lazy-created."""
    mood_evaluator: MoodEvaluator = field(default_factory=MoodEvaluator)
    """Per-response mood modifier source. Real LLM call when wired
    with ``llm_client`` + ``history_store``; stub (0.0) otherwise."""
    metrics_collector: MetricsCollector = field(default_factory=NullCollector)
    """sink for per-turn ``TurnTrace`` records; ``NullCollector`` by default
    so tests don't need to opt out. Replaced by ``SQLiteCollector`` in ``run``."""
    extras: dict[str, object] = field(default_factory=dict)
    """scratch space for later additions (e.g. Twitch client) that don't
    justify a dedicated field yet"""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_from_disk(
        cls,
        root: Path,
        *,
        llm_clients: dict[str, LLMClient],
        tts_client: CartesiaTTSClient | None = None,
        transcriber: DeepgramTranscriber | None = None,
        defaults_path: Path | None = None,
    ) -> Familiar:
        """Build Familiar bundle from on-disk ``data/familiars/<id>/`` layout.

        Loads ``character.toml`` (merged over defaults), opens stores,
        registers providers/processors, builds :class:`ConversationMonitor`.

        :param defaults_path: override for default profile path; tests
            pass this to avoid staging a sibling ``_default/`` folder.
        """
        familiar_id = root.name
        if defaults_path is None:
            defaults_path = root.parent / "_default" / "character.toml"
        character_config = load_character_config(
            root / "character.toml",
            defaults_path=defaults_path,
        )

        memory_root = root / "memory"
        memory_root.mkdir(parents=True, exist_ok=True)
        memory_store = MemoryStore(memory_root)

        history_store = HistoryStore(root / "history.db")

        providers: dict[str, ContextProvider] = {
            "character": CharacterProvider(memory_store),
            "content_search": ContentSearchProvider(
                store=memory_store,
                llm_client=llm_clients["memory_search"],
            ),
        }
        pre_processors: dict[str, PreProcessor] = {
            "stepped_thinking": SteppedThinkingPreProcessor(
                llm_client=llm_clients["reasoning_context"],
            ),
        }
        post_processors: dict[str, PostProcessor] = {
            "recast": RecastPostProcessor(
                llm_client=llm_clients["post_process_style"],
            ),
        }

        subscriptions = SubscriptionRegistry(root / "subscriptions.toml")
        channels_root = root / "channels"
        channels_root.mkdir(parents=True, exist_ok=True)
        channel_configs = ChannelConfigStore(
            root=channels_root,
            character=character_config,
        )

        # modes/ holds per-mode static instruction files; created on
        # first boot so users can drop <mode>.md files without mkdir.
        # ModeInstructionProvider resolves files here per turn.
        modes_root = root / "modes"
        modes_root.mkdir(parents=True, exist_ok=True)

        # pre-load character card text from memory/self/*.md so the
        # monitor has it ready without hitting disk per evaluation
        self_dir = memory_root / "self"
        character_card = ""
        if self_dir.exists():
            parts = [p.read_text() for p in sorted(self_dir.glob("*.md"))]
            character_card = "\n\n".join(parts)

        async def _noop_respond(
            channel_id: int,
            buffer: object,
            trigger: object,
        ) -> None:
            """Act as a no-op until create_bot wires the real callback."""

        monitor = ConversationMonitor(
            familiar_name=familiar_id,
            aliases=character_config.aliases,
            chattiness=character_config.chattiness,
            interjection=character_config.interjection,
            lull_timeout=character_config.text_lull_timeout,
            llm_client=llm_clients["interjection_decision"],
            character_card=character_card,
            on_respond=_noop_respond,
            dynamic_lull=character_config.dynamic_lull,
            lull_timeout_min=character_config.lull_timeout_min,
            lull_timeout_max=character_config.lull_timeout_max,
            engagement_client=llm_clients["engagement_check"],
        )

        mood_evaluator = MoodEvaluator(
            llm_client=llm_clients["mood_eval"],
            history_store=history_store,
        )
        memory_writer = MemoryWriter(
            memory_store=memory_store,
            history_store=history_store,
            llm_client=llm_clients["memory_writer"],
            familiar_id=familiar_id,
        )
        memory_writer_scheduler = MemoryWriterScheduler(
            writer=memory_writer,
            history_store=history_store,
            familiar_id=familiar_id,
            turn_threshold=character_config.memory_writer_turn_threshold,
            idle_timeout=character_config.memory_writer_idle_timeout,
        )

        return cls(
            id=familiar_id,
            root=root,
            config=character_config,
            memory_store=memory_store,
            history_store=history_store,
            llm_clients=llm_clients,
            tts_client=tts_client,
            transcriber=transcriber,
            providers=providers,
            pre_processors=pre_processors,
            post_processors=post_processors,
            subscriptions=subscriptions,
            channel_configs=channel_configs,
            monitor=monitor,
            mood_evaluator=mood_evaluator,
            memory_writer_scheduler=memory_writer_scheduler,
        )

    # ------------------------------------------------------------------
    # Per-turn pipeline assembly
    # ------------------------------------------------------------------

    def build_pipeline(self, channel_config: ChannelConfig) -> ContextPipeline:
        """Return :class:`ContextPipeline` filtered for *channel_config*.

        Per-turn providers (e.g. :class:`ModeInstructionProvider`) are
        constructed fresh with the channel's mode baked in.
        """
        active_providers: list[ContextProvider] = [
            self.providers[pid]
            for pid in channel_config.providers_enabled
            if pid in self.providers
        ]
        if "history" in channel_config.providers_enabled:
            active_providers.append(
                HistoryProvider(
                    store=self.history_store,
                    llm_client=self.llm_clients["history_summary"],
                    window_size=self.config.history_window_size,
                    mode=channel_config.mode,
                ),
            )
        if "mode_instructions" in channel_config.providers_enabled:
            active_providers.append(
                ModeInstructionProvider(
                    modes_root=self.root / "modes",
                    mode=channel_config.mode,
                ),
            )
        active_pre = [
            self.pre_processors[pid]
            for pid in channel_config.preprocessors_enabled
            if pid in self.pre_processors
        ]
        active_post = [
            self.post_processors[pid]
            for pid in channel_config.postprocessors_enabled
            if pid in self.post_processors
        ]
        return ContextPipeline(
            providers=active_providers,
            pre_processors=active_pre,
            post_processors=active_post,
        )
