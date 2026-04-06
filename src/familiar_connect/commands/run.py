"""Run subcommand — start the Discord bot under trio."""

import argparse
import asyncio
import logging
import os

import trio

from familiar_connect.bot import create_bot

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
    parser.set_defaults(func=run)
    return parser


def _run_bot(token: str) -> None:
    """Run the Discord bot in a dedicated asyncio event loop.

    Called from a trio worker thread via trio.to_thread.run_sync, so
    py-cord gets an uncontested asyncio event loop while trio remains the
    top-level runtime on the main thread. Future pipeline tasks (transcription,
    TTS) will communicate with this thread via trio.from_thread / asyncio
    futures.

    create_bot() is called inside asyncio.run() because discord.Bot.__init__
    calls asyncio.get_event_loop(), which requires a running loop to exist.

    :param token: Discord bot token
    """

    async def _start() -> None:
        bot = create_bot()
        await bot.start(token)

    asyncio.run(_start())


async def _trio_main(token: str) -> None:
    """Trio entry point: run the bot in a dedicated asyncio worker thread.

    :param token: Discord bot token
    """
    await trio.to_thread.run_sync(_run_bot, token)


def run(_args: argparse.Namespace) -> int:
    """Start the Discord bot.

    Reads the bot token from the DISCORD_BOT environment variable, then
    launches the bot in a worker thread under the trio runtime.

    :param _args: Parsed command-line arguments (unused)
    :return: Exit code (0 for success, 1 for missing token)
    """
    token = os.environ.get("DISCORD_BOT")
    if not token:
        _logger.error("DISCORD_BOT environment variable is not set")
        return 1

    trio.run(_trio_main, token)
    return 0
