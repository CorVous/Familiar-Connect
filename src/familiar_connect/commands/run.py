"""Run subcommand — start the Discord bot under asyncio."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes.util
import logging
import os
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

import discord

from familiar_connect import log_style as ls
from familiar_connect.bot import BotHandle, create_bot
from familiar_connect.bus.topics import (
    TOPIC_DISCORD_TEXT,
    TOPIC_TWITCH_EVENT,
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)
from familiar_connect.config import ConfigError, load_character_config
from familiar_connect.context import (
    Assembler,
    CharacterCardLayer,
    ConversationSummaryLayer,
    CoreInstructionsLayer,
    CrossChannelContextLayer,
    OperatingModeLayer,
    PeopleDossierLayer,
    RagContextLayer,
    RecentHistoryLayer,
)
from familiar_connect.familiar import Familiar
from familiar_connect.llm import create_llm_clients
from familiar_connect.processors import (
    DebugLoggerProcessor,
    TextResponder,
    VoiceResponder,
)
from familiar_connect.processors.fact_extractor import FactExtractor
from familiar_connect.processors.people_dossier_worker import PeopleDossierWorker
from familiar_connect.processors.summary_worker import SummaryWorker
from familiar_connect.stt import create_transcriber
from familiar_connect.tts import create_tts_client
from familiar_connect.tts_player import (
    DiscordVoicePlayer,
    LoggingTTSPlayer,
    TTSPlayer,
)
from familiar_connect.voice.turn_detection import (
    create_local_turn_detector,
)

_logger = logging.getLogger(__name__)

_DEFAULT_FAMILIARS_ROOT = Path("data") / "familiars"


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the run subcommand.

    :param subparsers: Subparser action from main parser
    :param common_parser: Parent parser with common arguments
    :return: The created subparser
    """
    parser = subparsers.add_parser(
        "run",
        parents=[common_parser],
        help="Start the Discord bot",
        description="Start the familiar and connect to Discord",
    )
    parser.add_argument(
        "--familiar",
        metavar="ID",
        default=None,
        help=(
            "Folder name of the character to run "
            "(under data/familiars/). Overrides FAMILIAR_ID."
        ),
    )
    parser.set_defaults(func=run)
    return parser


def _resolve_familiar_root(args: argparse.Namespace) -> Path:
    """Return the root directory of the active familiar.

    Resolution order: ``--familiar`` CLI flag → ``FAMILIAR_ID``
    environment variable. Raises :class:`ValueError` if neither is
    set or the resulting directory is missing.
    """
    familiar_id = args.familiar or os.environ.get("FAMILIAR_ID")
    if not familiar_id:
        msg = (
            "No familiar selected. Set FAMILIAR_ID, pass --familiar, "
            "or create data/familiars/<id>/."
        )
        raise ValueError(msg)

    root = _DEFAULT_FAMILIARS_ROOT / familiar_id
    if not root.exists():
        msg = f"Familiar folder does not exist: {root}"
        raise ValueError(msg)
    return root


_DEBUG_TOPICS: tuple[str, ...] = (
    TOPIC_DISCORD_TEXT,
    TOPIC_TWITCH_EVENT,
    TOPIC_VOICE_ACTIVITY_START,
    TOPIC_VOICE_TRANSCRIPT_FINAL,
)


async def _run_debug_processor(familiar: Familiar) -> None:
    """Drain subscribed topics into the debug logger.

    Proves the bus is carrying traffic. Responders handle their own
    topics below; the debug logger is a passive observer.
    """
    proc = DebugLoggerProcessor(topics=_DEBUG_TOPICS)
    async for event in familiar.bus.subscribe(proc.topics):
        await proc.handle(event, familiar.bus)


def _default_assembler(familiar: Familiar) -> Assembler:
    """Build the Phase-3 full layer stack.

    Order is **stability descending** so OpenAI's prompt cache keeps the
    longest matching prefix across consecutive turns. Static layers
    (file/mode-keyed) come first; dynamic layers come next, ordered by
    refresh rate (slowest first):

    * ``ConversationSummaryLayer`` — every ``turns_threshold`` turns
      (default 10), the slowest of the dynamic block.
    * ``CrossChannelContextLayer`` — when *any* other channel's summary
      is rewritten; per-channel rate is bounded by the same threshold,
      but multiple sources fan in.
    * ``PeopleDossierLayer`` — when any active-channel subject's facts
      tick the watermark; effectively per-fact write.
    * ``RagContextLayer`` — every turn (cue is the inbound user text).

    A change in any layer cache-invalidates everything *after* it in
    the system message, so reordering gives the prefix the longest
    stable run before the per-turn ``RagContextLayer`` flips. See
    [Voice pipeline § Prompt cache friendliness](
    ../../docs/architecture/voice-pipeline.md#prompt-cache-friendliness).
    """
    root = familiar.root
    core_path = root.parent / "_default" / "core_instructions.md"
    card_path = root / "character.md"
    store = familiar.history_store
    return Assembler(
        layers=[
            CoreInstructionsLayer(path=core_path),
            CharacterCardLayer(card_path=card_path),
            OperatingModeLayer(
                modes={
                    "voice": (
                        "You are speaking aloud. Keep replies short "
                        "(one or two sentences). Avoid markdown."
                    ),
                    "text": (
                        "You are chatting in a text channel. Markdown "
                        "and multi-line replies are fine."
                    ),
                }
            ),
            # slowest dynamic layer — per-channel summary, every N turns
            ConversationSummaryLayer(store=store),
            # other-channel summaries — fans in but per-source rate matches above
            CrossChannelContextLayer(
                store=store,
                viewer_map={},  # populated by per-channel config when present
                ttl_seconds=600,
            ),
            # per-fact watermark — refreshes ahead of every active-channel turn
            PeopleDossierLayer(
                store=store,
                window_size=familiar.config.history_window_size,
            ),
            # per-turn cue — always changes, so it sits last in the system msg
            RagContextLayer(
                store=store,
                max_results=5,
                # Match RecentHistoryLayer's window so RAG only surfaces
                # turns *older* than what's already shown verbatim.
                recent_window_size=familiar.config.history_window_size,
            ),
            RecentHistoryLayer(
                store=store,
                window_size=familiar.config.history_window_size,
            ),
        ]
    )


async def _run_voice_responder(familiar: Familiar, responder: VoiceResponder) -> None:
    """Drain voice-topic events into the :class:`VoiceResponder`."""
    async for event in familiar.bus.subscribe(responder.topics):
        await responder.handle(event, familiar.bus)


async def _run_text_responder(familiar: Familiar, responder: TextResponder) -> None:
    """Drain ``discord.text`` events into the :class:`TextResponder`."""
    async for event in familiar.bus.subscribe(responder.topics):
        await responder.handle(event, familiar.bus)


def _first_voice_client(handle: BotHandle) -> discord.VoiceClient | None:
    """Pick any active voice client from the runtime map.

    v1 supports one voice channel at a time, so the first runtime
    entry is unambiguous. Returns ``None`` if no voice subscription
    is active.
    """
    for rt in handle.voice_runtime.values():
        return rt.voice_client
    return None


async def _async_main(token: str, familiar: Familiar) -> None:
    """Asyncio entry point: bring up bus, responders, bot.

    :param token: Discord bot token.
    :param familiar: The loaded :class:`Familiar` bundle.
    """
    handle = create_bot(familiar)
    await familiar.bus.start()

    assembler = _default_assembler(familiar)
    tts_player: TTSPlayer
    if familiar.tts_client is not None:
        tts_player = DiscordVoicePlayer(
            tts_client=familiar.tts_client,
            get_voice_client=lambda: _first_voice_client(handle),
        )
    else:
        tts_player = LoggingTTSPlayer()
    voice_responder = VoiceResponder(
        assembler=assembler,
        llm_client=familiar.llm_clients["main_prose"],
        tts_player=tts_player,
        history_store=familiar.history_store,
        router=familiar.router,
        familiar_id=familiar.id,
        member_resolver=handle.resolve_member,
    )
    text_responder = TextResponder(
        assembler=assembler,
        llm_client=familiar.llm_clients["main_prose"],
        send_text=handle.send_text,
        history_store=familiar.history_store,
        router=familiar.router,
        familiar_id=familiar.id,
    )
    summary_worker = SummaryWorker(
        store=familiar.history_store,
        llm_client=familiar.llm_clients["main_prose"],
        familiar_id=familiar.id,
        turns_threshold=10,
    )
    fact_extractor = FactExtractor(
        store=familiar.history_store,
        llm_client=familiar.llm_clients["main_prose"],
        familiar_id=familiar.id,
        batch_size=10,
    )
    people_dossier_worker = PeopleDossierWorker(
        store=familiar.history_store,
        llm_client=familiar.llm_clients["main_prose"],
        familiar_id=familiar.id,
    )

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_run_debug_processor(familiar), name="debug-logger")
            tg.create_task(
                _run_voice_responder(familiar, voice_responder),
                name="voice-responder",
            )
            tg.create_task(
                _run_text_responder(familiar, text_responder),
                name="text-responder",
            )
            tg.create_task(summary_worker.run(), name="summary-worker")
            tg.create_task(fact_extractor.run(), name="fact-extractor")
            tg.create_task(people_dossier_worker.run(), name="people-dossier-worker")
            tg.create_task(handle.bot.start(token), name="discord-bot")
    finally:
        # close py-cord first so its aiohttp ClientSession doesn't leak
        # past loop shutdown; suppress so a close error can't mask the
        # original exception that triggered the finally
        with contextlib.suppress(Exception):
            await handle.bot.close()
        if familiar.transcriber is not None:
            with contextlib.suppress(Exception):
                await familiar.transcriber.stop()
        familiar.router.shutdown()
        await familiar.bus.shutdown()
        for client in familiar.llm_clients.values():
            await client.close()


_OPUS_FALLBACK_PATHS = [
    # macOS — Homebrew
    "/opt/homebrew/lib/libopus.dylib",  # Apple Silicon
    "/usr/local/lib/libopus.dylib",  # Intel
    # macOS — MacPorts
    "/opt/local/lib/libopus.dylib",
    # Linux — common distro paths
    "/usr/lib/x86_64-linux-gnu/libopus.so.0",  # Debian/Ubuntu amd64
    "/usr/lib/aarch64-linux-gnu/libopus.so.0",  # Debian/Ubuntu arm64
    "/usr/lib64/libopus.so.0",  # Fedora/RHEL/CentOS
    "/usr/lib/libopus.so.0",  # Arch/Alpine
    "/usr/lib/libopus.so",
    # Windows — common install locations
    "C:\\Windows\\System32\\opus.dll",
    "C:\\Tools\\opus\\opus.dll",
]


def load_opus() -> None:
    """Load the system Opus shared library for Discord voice."""
    if discord.opus.is_loaded():
        return
    lib = ctypes.util.find_library("opus")
    if not lib:
        for path in _OPUS_FALLBACK_PATHS:
            if pathlib.Path(path).exists():
                lib = path
                break
    if lib:
        discord.opus.load_opus(lib)
        _logger.debug("Loaded Opus from: %s", lib)
    else:
        _logger.warning(
            "Opus library not found — voice playback will not work. "
            "Install it with: brew install opus (macOS), "
            "apt install libopus0 (Debian/Ubuntu), "
            "dnf install opus (Fedora), or pacman -S opus (Arch)"
        )


def run(args: argparse.Namespace) -> int:
    """Start the Discord bot.

    Reads the bot token from the DISCORD_BOT environment variable,
    selects the active familiar via ``--familiar`` / ``FAMILIAR_ID``,
    builds the :class:`Familiar` bundle, then launches the bot under
    asyncio.

    :param args: Parsed command-line arguments.
    :return: Exit code (0 for success, 1 for missing token or config).
    """
    token = os.environ.get("DISCORD_BOT")
    if not token:
        _logger.error("DISCORD_BOT environment variable is not set")
        return 1

    try:
        familiar_root = _resolve_familiar_root(args)
    except ValueError as exc:
        _logger.error("%s", exc)
        return 1

    # Load the merged character config first so both the LLM client
    # factory and the TTS client factory have access to per-slot
    # model selections and the [tts] voice id / model.
    defaults_path = familiar_root.parent / "_default" / "character.toml"
    try:
        character_config = load_character_config(
            familiar_root / "character.toml",
            defaults_path=defaults_path,
        )
    except ConfigError as exc:
        _logger.error("Failed to load familiar config: %s", exc)
        return 1

    _logger.info(
        f"{ls.tag('Config', ls.W)} "
        f"{ls.kv('familiar', familiar_root.name)} "
        f"{ls.kv('tts', character_config.tts.provider)}"
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        _logger.error("OPENROUTER_API_KEY environment variable is required")
        return 1

    try:
        llm_clients = create_llm_clients(api_key, character_config)
    except KeyError as exc:
        _logger.error(
            "Character config is missing LLM slot %s — check "
            "data/familiars/_default/character.toml",
            exc,
        )
        return 1

    try:
        tts_client = create_tts_client(character_config.tts)
    except ValueError as exc:
        _logger.warning("TTS client unavailable: %s", exc)
        tts_client = None

    # ``ValueError`` on missing API key or unknown backend → degrade
    # gracefully (text path still works).
    try:
        transcriber = create_transcriber(character_config.stt)
    except ValueError as exc:
        _logger.warning("Transcriber unavailable: %s", exc)
        transcriber = None

    try:
        if character_config.turn_detection.strategy == "ten+smart_turn":
            local_turn_detector = create_local_turn_detector(
                character_config.turn_detection.local
            )
        else:
            local_turn_detector = None
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Local turn detector unavailable: %s", exc)
        local_turn_detector = None

    try:
        familiar = Familiar.load_from_disk(
            familiar_root,
            llm_clients=llm_clients,
            tts_client=tts_client,
            transcriber=transcriber,
            local_turn_detector=local_turn_detector,
        )
    except ConfigError as exc:
        _logger.error("Failed to load familiar config: %s", exc)
        return 1

    _logger.info(
        f"{ls.tag('❇️ Loaded', ls.W)} "
        f"{ls.kv('familiar', familiar.id)} "
        f"{ls.kv('from', str(familiar_root))}"
    )

    load_opus()
    asyncio.run(_async_main(token, familiar))
    return 0
