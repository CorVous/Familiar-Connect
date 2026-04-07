"""Run subcommand — start the Discord bot under trio."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

import trio

from familiar_connect.bot import create_bot
from familiar_connect.character import CharacterCardError, load_card
from familiar_connect.llm import create_client_from_env
from familiar_connect.preset import PresetError, assemble_prompt, load_preset

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


def _run_bot(token: str, system_prompt: str) -> None:
    """Run the Discord bot in a dedicated asyncio event loop.

    Called from a trio worker thread via trio.to_thread.run_sync, so
    py-cord gets an uncontested asyncio event loop while trio remains the
    top-level runtime on the main thread. Future pipeline tasks (transcription,
    TTS) will communicate with this thread via trio.from_thread / asyncio
    futures.

    create_bot() is called inside asyncio.run() because discord.Bot.__init__
    calls asyncio.get_event_loop(), which requires a running loop to exist.

    :param token: Discord bot token.
    :param system_prompt: Pre-assembled system prompt for the familiar.
    """

    async def _start() -> None:
        try:
            llm_client = create_client_from_env()
        except ValueError as exc:
            _logger.warning("LLM client unavailable: %s", exc)
            llm_client = None

        bot = create_bot(llm_client=llm_client, system_prompt=system_prompt)
        await bot.start(token)

    asyncio.run(_start())


async def _trio_main(token: str, system_prompt: str) -> None:
    """Trio entry point: run the bot in a dedicated asyncio worker thread.

    :param token: Discord bot token.
    :param system_prompt: Pre-assembled system prompt for the familiar.
    """
    await trio.to_thread.run_sync(_run_bot, token, system_prompt)


def run(args: argparse.Namespace) -> int:
    """Start the Discord bot.

    Reads the bot token from the DISCORD_BOT environment variable, loads the
    character card and preset (if configured), then launches the bot in a
    worker thread under the trio runtime.

    :param args: Parsed command-line arguments.
    :return: Exit code (0 for success, 1 for missing token).
    """
    token = os.environ.get("DISCORD_BOT")
    if not token:
        _logger.error("DISCORD_BOT environment variable is not set")
        return 1

    system_prompt = build_system_prompt(args)
    trio.run(_trio_main, token, system_prompt)
    return 0
