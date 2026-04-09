"""Run subcommand — start the Discord bot under asyncio."""

from __future__ import annotations

import asyncio
import ctypes.util
import logging
import os
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

import discord

from familiar_connect.bot import create_bot
from familiar_connect.config import ConfigError
from familiar_connect.familiar import Familiar
from familiar_connect.llm import create_client_from_env, create_side_client_from_env
from familiar_connect.transcription import create_transcriber_from_env
from familiar_connect.tts import create_tts_client_from_env

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


async def _async_main(token: str, familiar: Familiar) -> None:
    """Asyncio entry point: build the bot and start it.

    :param token: Discord bot token.
    :param familiar: The loaded :class:`Familiar` bundle.
    """
    bot = create_bot(familiar)
    await bot.start(token)


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

    try:
        llm_client = create_client_from_env()
    except ValueError as exc:
        _logger.error("LLM client unavailable: %s", exc)
        return 1

    side_llm_client = create_side_client_from_env()
    if side_llm_client is None:
        _logger.info(
            "No OPENROUTER_SIDE_MODEL set — side-model work (stepped "
            "thinking, recast, history summary, content search) will "
            "reuse the main model. Set OPENROUTER_SIDE_MODEL to a "
            "cheaper/faster model to avoid paying main-model price "
            "for sub-tasks.",
        )
    else:
        _logger.info(
            "Side-model calls will use %s (separate from main model %s).",
            side_llm_client.model,
            llm_client.model,
        )

    try:
        tts_client = create_tts_client_from_env()
    except ValueError as exc:
        _logger.warning("TTS client unavailable: %s", exc)
        tts_client = None

    try:
        transcriber = create_transcriber_from_env()
    except ValueError as exc:
        _logger.warning("Transcriber unavailable: %s", exc)
        transcriber = None

    try:
        familiar = Familiar.load_from_disk(
            familiar_root,
            llm_client=llm_client,
            tts_client=tts_client,
            side_llm_client=side_llm_client,
            transcriber=transcriber,
        )
    except ConfigError as exc:
        _logger.error("Failed to load familiar config: %s", exc)
        return 1

    _logger.info(
        "Loaded familiar %s from %s (default_mode=%s)",
        familiar.id,
        familiar_root,
        familiar.config.default_mode.value,
    )

    load_opus()
    asyncio.run(_async_main(token, familiar))
    return 0
