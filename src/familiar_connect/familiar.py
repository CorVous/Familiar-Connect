"""Runtime bundle for the single active character.

Per ``docs/architecture/configuration-model.md``, a Familiar-Connect
process runs exactly one character at a time. Multiple character
folders may coexist under ``data/familiars/``; ``FAMILIAR_ID`` at
startup selects which one this process actually loads. The
:class:`Familiar` dataclass is the runtime singleton that carries
every concern tied to that active character: its config, its
memory store, its history store, its registered providers /
processors, its subscription registry, and its per-channel config
store.

:meth:`Familiar.load_from_disk` walks a character's folder under
``data/familiars/<id>/`` and builds the whole bundle in one pass.
It's the only constructor callers should use; the dataclass is
mostly transparent so tests can still mint minimal fakes when they
need to.

The bundle also exposes :meth:`build_pipeline`, which constructs a
per-turn :class:`ContextPipeline` from the channel's active
:class:`ChannelConfig`. Building the pipeline per turn (rather than
holding a single pipeline across turns) lets the set of providers
and processors vary per channel mode without fancy filtering inside
the pipeline itself.
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
from familiar_connect.subscriptions import SubscriptionRegistry

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
    """Runtime bundle for the one character this install is running.

    :param id: Folder name under ``data/familiars/``. Used as the
        ``familiar_id`` on :class:`ContextRequest`.
    :param root: The character's root directory on disk.
    :param config: Parsed :class:`CharacterConfig` (defaults if the
        sidecar is missing).
    :param memory_store: The :class:`MemoryStore` rooted at
        ``<root>/memory``.
    :param history_store: The :class:`HistoryStore` backed by
        ``<root>/history.db``.
    :param llm_clients: ``slot_name -> LLMClient`` map, one entry
        per :data:`familiar_connect.config.LLM_SLOT_NAMES`. Holds
        every client the bundle needs so shutdown can iterate and
        ``close()`` all of them.
    :param tts_client: Optional TTS client for voice output.
    :param transcriber: Optional Deepgram transcriber template used
        by the voice-subscription path to drive per-user speech-to-
        text streams. When ``None``, ``/subscribe-my-voice`` still
        joins the voice channel and plays TTS, but no incoming
        audio is transcribed.
    :param providers: ``id -> ContextProvider`` map of every
        registered provider, filtered per-turn via
        :meth:`build_pipeline`.
    :param pre_processors: ``id -> PreProcessor`` map.
    :param post_processors: ``id -> PostProcessor`` map.
    :param subscriptions: Persistent subscription registry.
    :param channel_configs: Per-channel config store.
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
    extras: dict[str, object] = field(default_factory=dict)
    """Scratch space for later additions (e.g. Twitch client) that don't
    justify a dedicated field yet."""

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
        """Build a Familiar bundle from the on-disk ``data/familiars/<id>/`` layout.

        - ``character.toml`` is loaded (merged over
          ``data/familiars/_default/character.toml``).
        - ``memory/`` is created if missing.
        - ``history.db`` is opened (and created if missing).
        - ``subscriptions.toml`` is loaded if present.
        - ``channels/`` is created if missing.

        :param llm_clients: ``slot_name -> LLMClient`` map built by
            :func:`familiar_connect.llm.create_llm_clients`. Must
            contain one client per
            :data:`familiar_connect.config.LLM_SLOT_NAMES`.
        :param tts_client: Optional Cartesia TTS client for voice
            output.
        :param transcriber: Optional Deepgram transcriber template
            used by ``/subscribe-my-voice`` to drive per-user
            speech-to-text streams. When ``None``, the voice
            subscription still joins the channel and plays TTS but
            no audio is transcribed.
        :param defaults_path: Override for the default character
            profile path. Production leaves this ``None``, which
            resolves ``root.parent / "_default" / "character.toml"``.
            Tests pass the checked-in default profile directly so
            they don't have to stage a sibling ``_default/`` folder
            in ``tmp_path``.
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

        # modes/ holds per-mode static instruction files; it's created
        # on first boot so users can drop <mode>.md files into it
        # without having to mkdir themselves. ModeInstructionProvider
        # resolves files relative to this directory per turn.
        modes_root = root / "modes"
        modes_root.mkdir(parents=True, exist_ok=True)

        # Pre-load character card text from memory/self/*.md files so the
        # monitor has it ready for evaluation prompts without hitting disk
        # per evaluation.
        self_dir = memory_root / "self"
        character_card = ""
        if self_dir.exists():
            parts = [p.read_text() for p in sorted(self_dir.glob("*.md"))]
            character_card = "\n\n".join(parts)

        async def _noop_respond(
            channel_id: int,
            buffer: object,
        ) -> None:
            """Act as a no-op until create_bot wires the real callback."""

        monitor = ConversationMonitor(
            familiar_name=familiar_id,
            aliases=character_config.aliases,
            chattiness=character_config.chattiness,
            interjection=character_config.interjection,
            lull_timeout=character_config.lull_timeout,
            llm_client=llm_clients["interjection_decision"],
            character_card=character_card,
            on_respond=_noop_respond,
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
            memory_writer_scheduler=memory_writer_scheduler,
        )

    # ------------------------------------------------------------------
    # Per-turn pipeline assembly
    # ------------------------------------------------------------------

    def build_pipeline(self, channel_config: ChannelConfig) -> ContextPipeline:
        """Return a :class:`ContextPipeline` filtered for the given channel.

        Most providers are registered once on the Familiar at
        startup and filtered per turn by id. A small set of providers
        depends on the channel's :class:`ChannelMode` and is
        constructed fresh per turn with the mode baked in —
        :class:`ModeInstructionProvider` is the only one today, but
        the branch here is the sanctioned place to add future
        mode-scoped providers without leaking ``ChannelMode`` into
        :class:`ContextRequest`.
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
