"""Run subcommand — start the Discord bot under asyncio."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes.util
import logging
import os
import pathlib
import signal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from familiar_connect.embedding.protocol import Embedder

import discord

from familiar_connect import log_style as ls
from familiar_connect.bot import BotHandle, create_bot
from familiar_connect.budget import Budgeter, TierBudget
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
    CrossChannelContextLayer,
    LorebookLayer,
    OperatingModeLayer,
    PeopleDossierLayer,
    RagContextLayer,
    RecentHistoryLayer,
    ReflectionLayer,
)
from familiar_connect.embedding import create_embedder
from familiar_connect.familiar import Familiar
from familiar_connect.focus import FocusManager
from familiar_connect.llm import create_llm_clients
from familiar_connect.processors import (
    DebugLoggerProcessor,
    TextResponder,
    VoiceResponder,
)
from familiar_connect.processors.projectors import (
    ProjectorContext,
    create_projectors,
)
from familiar_connect.stt import create_transcriber
from familiar_connect.subscriptions import SubscriptionKind
from familiar_connect.tools.alarm import build_alarm_tool, build_cancel_alarm_tool
from familiar_connect.tools.image import build_view_image_tool
from familiar_connect.tools.read_channel import build_read_channel_tool
from familiar_connect.tools.registry import ToolContext, ToolRegistry
from familiar_connect.tools.scheduler import AlarmScheduler
from familiar_connect.tools.shift_focus import build_shift_focus_tool
from familiar_connect.tools.silent import build_silent_tool
from familiar_connect.tools.waker import AlarmWaker
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

    :param subparsers: subparser action from main parser
    :param common_parser: parent parser with common arguments
    :return: created subparser
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
    """Return root directory of active familiar.

    Resolution order: ``--familiar`` CLI flag → ``FAMILIAR_ID`` env.
    Raises :class:`ValueError` if neither is set or resulting
    directory is missing.
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
    """Drain subscribed topics into debug logger.

    Proves bus is carrying traffic. Responders handle their own
    topics below; debug logger is a passive observer.
    """
    proc = DebugLoggerProcessor(topics=_DEBUG_TOPICS)
    async for event in familiar.bus.subscribe(proc.topics):
        await proc.handle(event, familiar.bus)


def _default_assembler(
    familiar: Familiar,
    *,
    window_size: int,
    budget: TierBudget,
    channel_total_tokens: dict[int, int] | None = None,
    silence_gap_fold_seconds: float = 0.0,
    embedder: Embedder | None = None,
) -> Assembler:
    """Build full layer stack with token-aware per-section caps.

    Order is **stability descending** so OpenAI's prompt cache keeps
    the longest matching prefix across consecutive turns. Static
    layers (file/mode-keyed) come first; dynamic layers next, ordered
    by refresh rate (slowest first):

    * ``LorebookLayer`` — file-sourced authored canon, keyword-
      activated against recent window. Flips when topic shifts cross
      an activation key; otherwise stable.
    * ``ConversationSummaryLayer`` — every ``turns_threshold`` turns
      (default 10), slowest of the dynamic block.
    * ``CrossChannelContextLayer`` — when *any* other channel's
      summary is rewritten; per-channel rate bounded by same
      threshold, but multiple sources fan in.
    * ``ReflectionLayer`` — when :class:`ReflectionWorker` writes a
      new reflection (default every ~20 turns); slower than dossiers,
      faster than rolling summaries.
    * ``PeopleDossierLayer`` — when any active-channel subject's
      facts tick the watermark; effectively per-fact write.
    * ``RagContextLayer`` — every turn (cue is inbound user text).

    Change in any layer cache-invalidates everything *after* it in
    the system message, so reordering gives the prefix the longest
    stable run before the per-turn ``RagContextLayer`` flips. See
    [Voice pipeline § Prompt cache friendliness](
    ../../docs/architecture/voice-pipeline.md#prompt-cache-friendliness).

    ``window_size`` is hard upper bound on history turns;
    :class:`Budgeter`'s token caps usually bite first.
    """
    root = familiar.root
    card_path = root / "character.md"
    lorebook_path = root / "lorebook.toml"
    store = familiar.history_store
    retrieval = familiar.config.memory_retrieval
    return Assembler(
        layers=[
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
            LorebookLayer(
                store=store,
                path=lorebook_path,
                recent_window=window_size,
                max_entries=budget.max_lorebook_entries,
                max_tokens=budget.lorebook_tokens,
            ),
            ConversationSummaryLayer(
                store=store,
                max_tokens=budget.summary_tokens,
            ),
            CrossChannelContextLayer(
                store=store,
                viewer_map={},  # populated by per-channel config when present
                ttl_seconds=600,
                max_tokens=budget.cross_channel_tokens,
            ),
            ReflectionLayer(
                store=store,
                max_reflections=budget.max_reflections,
                max_tokens=budget.reflection_tokens,
            ),
            PeopleDossierLayer(
                store=store,
                window_size=window_size,
                max_people=budget.max_dossier_people,
                max_tokens=budget.dossier_tokens,
            ),
            RagContextLayer(
                store=store,
                max_results=budget.max_rag_turns,
                max_facts=budget.max_rag_facts,
                # match RecentHistoryLayer's window so RAG only surfaces
                # turns *older* than what's already shown verbatim
                recent_window_size=window_size,
                max_tokens=budget.rag_tokens,
                bm25_weight=retrieval.bm25_weight,
                recency_weight=retrieval.recency_weight,
                importance_weight=retrieval.importance_weight,
                embedding_weight=retrieval.embedding_weight,
                embedder=embedder,
                display_tz=familiar.config.display_tz,
            ),
            RecentHistoryLayer(
                store=store,
                window_size=window_size,
                max_tokens=budget.recent_history_tokens,
                coalesce_max_gap_seconds=(
                    familiar.config.recent_history_coalesce_max_gap_seconds
                ),
                silence_gap_fold_seconds=silence_gap_fold_seconds,
                display_tz=familiar.config.display_tz,
            ),
        ],
        budgeter=Budgeter(budget, channel_total_tokens=channel_total_tokens),
    )


async def _run_voice_responder(familiar: Familiar, responder: VoiceResponder) -> None:
    """Drain voice-topic events into :class:`VoiceResponder`."""
    async for event in familiar.bus.subscribe(responder.topics):
        await responder.handle(event, familiar.bus)


async def _run_text_responder(familiar: Familiar, responder: TextResponder) -> None:
    """Drain ``discord.text`` events into :class:`TextResponder`."""
    async for event in familiar.bus.subscribe(responder.topics):
        await responder.handle(event, familiar.bus)


async def _run_alarm_waker(familiar: Familiar, waker: AlarmWaker) -> None:
    """Drain ``alarm.fired`` events into :class:`AlarmWaker`."""
    async for event in familiar.bus.subscribe(waker.topics):
        await waker.handle(event, familiar.bus)


def _first_voice_client(handle: BotHandle) -> discord.VoiceClient | None:
    """Pick any active voice client from runtime map.

    v1 supports one voice channel at a time, so the first runtime
    entry is unambiguous. Returns ``None`` if no voice subscription
    active.
    """
    for rt in handle.voice_runtime.values():
        return rt.voice_client
    return None


class _GracefulShutdown(Exception):  # noqa: N818
    """Sentinel raised inside the TaskGroup to unwind on SIGINT/SIGTERM."""


# SIGINT (Ctrl-C, frequent in dev) + SIGTERM (container/systemd stop)
_SHUTDOWN_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM)


async def _wait_for_shutdown(stop: asyncio.Event) -> None:
    """Park until *stop* set, then raise to unwind the run-loop TaskGroup."""
    await stop.wait()
    raise _GracefulShutdown


def _install_shutdown_handlers(stop: asyncio.Event) -> Callable[[], None]:
    """Route SIGINT/SIGTERM into a cooperative shutdown.

    First signal sets *stop*; the supervisor task then raises
    :class:`_GracefulShutdown`, so the TaskGroup cancels its siblings
    and the ``finally`` teardown runs in normal (non-cancelling) task
    state — no re-raised ``KeyboardInterrupt`` traceback, no half-closed
    aiohttp session. A second signal restores the OS default handler so
    a wedged shutdown stays force-killable.

    No-op where ``add_signal_handler`` is unsupported (Windows Proactor
    loop, non-main thread); :func:`run` keeps a ``KeyboardInterrupt``
    fallback for those.

    :return: callable removing the installed handlers.
    """
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []

    def _remove() -> None:
        for sig in installed:
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.remove_signal_handler(sig)
        installed.clear()

    def _on_signal(signame: str) -> None:
        if not stop.is_set():
            _logger.info(
                f"{ls.tag('Shutdown', ls.Y)} "
                f"{ls.kv('signal', signame, vc=ls.LY)} "
                f"draining — {ls.word('signal again to force', ls.LY)}"
            )
            stop.set()
        else:
            _logger.warning(
                f"{ls.tag('Shutdown', ls.Y)} forced — OS default handler restored"
            )
            _remove()

    for sig in _SHUTDOWN_SIGNALS:
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
        except (NotImplementedError, RuntimeError, ValueError):
            # no asyncio signal support here — KeyboardInterrupt covers it
            _logger.debug("signal handler unavailable for %s", sig.name)
        else:
            installed.append(sig)
    return _remove


async def _async_main(token: str, familiar: Familiar) -> None:
    """Asyncio entry point: bring up bus, responders, bot.

    :param token: Discord bot token.
    :param familiar: loaded :class:`Familiar` bundle.
    """
    await familiar.bus.start()

    focus_manager = FocusManager(
        familiar_id=familiar.id,
        store=familiar.history_store,
        subscriptions=familiar.subscriptions,
    )
    await focus_manager.initialize()

    # startup default: if no focus pointer persisted, use first subscribed
    # channel per modality so attentional stream is always live.
    for sub in familiar.subscriptions.all():
        if (
            sub.kind == SubscriptionKind.text
            and focus_manager.get_focus("text") is None
        ):
            focus_manager.set_focus_immediately(sub.channel_id, "text")
        if (
            sub.kind == SubscriptionKind.voice
            and focus_manager.get_focus("voice") is None
        ):
            focus_manager.set_focus_immediately(sub.channel_id, "voice")

    handle = create_bot(familiar, focus_manager=focus_manager)

    embedder = create_embedder(familiar.config.embedding)
    channel_total_tokens: dict[int, int] = {
        ch_id: over.total_tokens
        for ch_id, over in familiar.config.channels.items()
        if over.total_tokens is not None
    }
    voice_assembler = _default_assembler(
        familiar,
        window_size=familiar.config.voice_window_size,
        budget=familiar.config.budget_for("voice", None),
        channel_total_tokens=channel_total_tokens or None,
        embedder=embedder,
    )
    text_assembler = _default_assembler(
        familiar,
        window_size=familiar.config.text_window_size,
        budget=familiar.config.budget_for("text", None),
        channel_total_tokens=channel_total_tokens or None,
        silence_gap_fold_seconds=familiar.config.text_silence_gap_fold_seconds,
        embedder=embedder,
    )
    tts_player: TTSPlayer
    if familiar.tts_client is not None:
        tts_player = DiscordVoicePlayer(
            tts_client=familiar.tts_client,
            get_voice_client=lambda: _first_voice_client(handle),
        )
    else:
        tts_player = LoggingTTSPlayer()

    # tool-calling: scheduler + split registries + per-turn context factory.
    # voice registry: alarm + cancel only (view_image NEVER in voice).
    # text registry: alarm + cancel + view_image (when image_tools enabled).
    # familiar's ``llm_clients`` already carry ``tool_calling_enabled`` (see
    # :func:`create_llm_clients`); responders check that flag before
    # entering the agentic loop, so wiring is always safe.
    alarm_scheduler = AlarmScheduler(
        history=familiar.history_store,
        bus=familiar.bus,
        familiar_id=familiar.id,
    )
    voice_tool_registry = ToolRegistry()
    voice_tool_registry.register(build_alarm_tool(alarm_scheduler))
    voice_tool_registry.register(build_cancel_alarm_tool(alarm_scheduler))
    voice_tool_registry.register(build_silent_tool())
    voice_tool_registry.register(build_shift_focus_tool())

    text_tool_registry = ToolRegistry()
    text_tool_registry.register(build_alarm_tool(alarm_scheduler))
    text_tool_registry.register(build_cancel_alarm_tool(alarm_scheduler))
    text_tool_registry.register(build_silent_tool())
    text_tool_registry.register(build_shift_focus_tool())
    text_tool_registry.register(build_read_channel_tool())

    prose_slot = familiar.config.llm.get("prose")
    if prose_slot is not None and prose_slot.image_tools:
        text_tool_registry.register(
            build_view_image_tool(
                describe_constraints=familiar.config.image_description_constraints
            )
        )

    # description client: used by view_image handler via ToolContext
    description_llm = familiar.llm_clients.get("__image_description__")

    def _make_tool_context(
        channel_kind: str,
    ) -> Callable[[int, str, dict[str, str]], ToolContext]:
        def _build(
            channel_id: int,
            turn_id: str,
            images: dict[str, str] | None = None,
        ) -> ToolContext:
            return ToolContext(
                familiar_id=familiar.id,
                channel_id=channel_id,
                channel_kind=channel_kind,
                turn_id=turn_id,
                history=familiar.history_store,
                bus=familiar.bus,
                scheduler=alarm_scheduler,
                images=images or {},
                description_llm=description_llm if channel_kind == "text" else None,
                focus_manager=focus_manager,
                store=familiar.history_store,
            )

        return _build

    alarm_waker = AlarmWaker(familiar_id=familiar.id)

    voice_responder = VoiceResponder(
        assembler=voice_assembler,
        llm_client=familiar.llm_clients["fast"],
        tts_player=tts_player,
        history_store=familiar.history_store,
        router=familiar.router,
        familiar_id=familiar.id,
        member_resolver=handle.resolve_member,
        tool_registry=voice_tool_registry,
        tool_context_factory=_make_tool_context("voice"),
        post_history_instructions=familiar.config.post_history_instructions,
        focus_manager=focus_manager,
    )
    text_responder = TextResponder(
        assembler=text_assembler,
        llm_client=familiar.llm_clients["prose"],
        send_text=handle.send_text,
        history_store=familiar.history_store,
        router=familiar.router,
        familiar_id=familiar.id,
        trigger_typing=handle.trigger_typing,
        typing_handler=handle.typing_interrupt,
        tool_registry=text_tool_registry,
        tool_context_factory=_make_tool_context("text"),
        post_history_instructions=familiar.config.post_history_instructions,
        focus_manager=focus_manager,
    )
    projector_context = ProjectorContext(
        store=familiar.history_store,
        llm_clients=familiar.llm_clients,
        familiar_id=familiar.id,
        embedder=embedder,
    )
    projectors = create_projectors(
        names=list(familiar.config.memory_providers.projectors),
        context=projector_context,
    )

    # load pending alarms now so they start counting down before
    # bot accepts new traffic
    await alarm_scheduler.start()

    # cooperative shutdown: SIGINT/SIGTERM set the event, the supervisor
    # task unwinds the group, and the finally below tears down in order
    # (see _install_shutdown_handlers)
    stop = asyncio.Event()
    remove_signal_handlers = _install_shutdown_handlers(stop)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_wait_for_shutdown(stop), name="shutdown-supervisor")
            tg.create_task(_run_debug_processor(familiar), name="debug-logger")
            tg.create_task(
                _run_voice_responder(familiar, voice_responder),
                name="voice-responder",
            )
            tg.create_task(
                _run_text_responder(familiar, text_responder),
                name="text-responder",
            )
            tg.create_task(
                _run_alarm_waker(familiar, alarm_waker),
                name="alarm-waker",
            )
            for proj in projectors:
                tg.create_task(proj.run(), name=proj.name)
            tg.create_task(handle.bot.start(token), name="discord-bot")
    except* _GracefulShutdown:
        # signal-initiated: siblings already cancelled by the group;
        # fall through to finally for orderly teardown
        _logger.info(f"{ls.tag('Shutdown', ls.G)} {ls.word('clean', ls.LG)}")
    finally:
        remove_signal_handlers()
        # close py-cord first so its aiohttp ClientSession doesn't leak
        # past loop shutdown; suppress so close error can't mask the
        # original exception that triggered the finally
        with contextlib.suppress(Exception):
            await handle.bot.close()
        if familiar.transcriber is not None:
            with contextlib.suppress(Exception):
                await familiar.transcriber.stop()
        with contextlib.suppress(Exception):
            await alarm_scheduler.shutdown()
        familiar.router.shutdown()
        await familiar.bus.shutdown()
        for client in familiar.llm_clients.values():
            await client.close()


_OPUS_FALLBACK_PATHS = [
    # macOS — Homebrew
    "/opt/homebrew/lib/libopus.dylib",  # apple silicon
    "/usr/local/lib/libopus.dylib",  # intel
    # macOS — MacPorts
    "/opt/local/lib/libopus.dylib",
    # Linux — common distro paths
    "/usr/lib/x86_64-linux-gnu/libopus.so.0",  # debian/ubuntu amd64
    "/usr/lib/aarch64-linux-gnu/libopus.so.0",  # debian/ubuntu arm64
    "/usr/lib64/libopus.so.0",  # fedora/RHEL/centos
    "/usr/lib/libopus.so.0",  # arch/alpine
    "/usr/lib/libopus.so",
    # Windows — common install locations
    "C:\\Windows\\System32\\opus.dll",
    "C:\\Tools\\opus\\opus.dll",
]


def load_opus() -> None:
    """Load system Opus shared library for Discord voice."""
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

    Reads bot token from ``DISCORD_BOT`` env, selects active familiar
    via ``--familiar`` / ``FAMILIAR_ID``, builds :class:`Familiar`
    bundle, then launches the bot under asyncio.

    :param args: parsed command-line arguments.
    :return: exit code (0 success, 1 missing token or config).
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

    # load merged character config first so both LLM client and TTS
    # client factories have access to per-slot model selections and
    # [tts] voice id/model
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
    # gracefully (text path still works)
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
    login_failed = False
    try:
        try:
            asyncio.run(_async_main(token, familiar))
        except* discord.errors.LoginFailure:
            login_failed = True
    except KeyboardInterrupt:
        # SIGINT landed before the asyncio signal handler armed, or on a
        # platform without add_signal_handler. asyncio.run already
        # cancelled tasks and ran cleanup — exit quietly, no traceback.
        _logger.info(
            f"{ls.tag('Shutdown', ls.G)} interrupted — {ls.word('bye', ls.LG)}"
        )
        return 0
    if login_failed:
        _logger.error(
            "Discord login failed — DISCORD_BOT token is invalid or expired. "
            "Generate a new token at https://discord.com/developers/applications "
            "and set it in your environment: DISCORD_BOT=<token>"
        )
        return 1
    return 0
