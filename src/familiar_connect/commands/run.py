"""Run subcommand — start the Discord bot under asyncio."""

from __future__ import annotations

import asyncio
import ctypes.util
import logging
import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

import discord

from familiar_connect.bot import create_bot
from familiar_connect.character import CharacterCardError, load_card
from familiar_connect.llm import create_client_from_env
from familiar_connect.preset import PresetError, assemble_prompt, load_preset
from familiar_connect.tts import create_tts_client_from_env

_logger = logging.getLogger(__name__)


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
        "--character",
        metavar="PATH",
        default=None,
        help=(
            "Path to a Character Card V3 PNG file. "
            "Overrides the FAMILIAR_CHARACTER environment variable."
        ),
    )
    parser.add_argument(
        "--preset",
        metavar="PATH",
        default=None,
        help=(
            "Path to a SillyTavern preset JSON file. "
            "Overrides the FAMILIAR_PRESET environment variable."
        ),
    )
    parser.set_defaults(func=run)
    return parser


def build_system_prompt(args: argparse.Namespace) -> str:
    """Load the character card and preset, then assemble a system prompt.

    Falls back gracefully: if no card or preset is configured, returns "".

    :param args: Parsed command-line arguments.
    :return: Assembled system prompt string (may be empty).
    """
    card_path = args.character or os.environ.get("FAMILIAR_CHARACTER")
    preset_path = args.preset or os.environ.get("FAMILIAR_PRESET")

    if not card_path:
        _logger.info("No character card configured — running without persona.")
        return ""

    try:
        card = load_card(card_path)
    except CharacterCardError as exc:
        _logger.error("Failed to load character card: %s", exc)
        return ""

    _logger.info("Loaded character card: %s", card.name)

    if not preset_path:
        _logger.info("No preset configured — using card description as system prompt.")
        return card.description

    try:
        preset = load_preset(preset_path)
    except PresetError as exc:
        _logger.error("Failed to load preset: %s", exc)
        return card.description

    prompt = assemble_prompt(preset, card)
    _logger.info("Assembled system prompt (%d chars).", len(prompt))
    return prompt


async def _async_main(token: str, system_prompt: str) -> None:
    """Asyncio entry point: build the bot and start it.

    :param token: Discord bot token.
    :param system_prompt: Pre-assembled system prompt for the familiar.
    """
    try:
        llm_client = create_client_from_env()
    except ValueError as exc:
        _logger.warning("LLM client unavailable: %s", exc)
        llm_client = None

    try:
        tts_client = create_tts_client_from_env()
    except ValueError as exc:
        _logger.warning("TTS client unavailable: %s", exc)
        tts_client = None

    bot = create_bot(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tts_client=tts_client,
    )
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

    Reads the bot token from the DISCORD_BOT environment variable, loads the
    character card and preset (if configured), then launches the bot under
    asyncio.

    :param args: Parsed command-line arguments.
    :return: Exit code (0 for success, 1 for missing token).
    """
    token = os.environ.get("DISCORD_BOT")
    if not token:
        _logger.error("DISCORD_BOT environment variable is not set")
        return 1

    load_opus()
    system_prompt = build_system_prompt(args)
    asyncio.run(_async_main(token, system_prompt))
    return 0
