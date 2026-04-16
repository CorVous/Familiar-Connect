"""Version subcommand - display package version."""

import argparse
import logging

from familiar_connect import __version__
from familiar_connect import log_style as ls

_logger = logging.getLogger(__name__)


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the version subcommand.

    :param subparsers: Subparser action from main parser
    :param common_parser: Parent parser with common arguments
    :return: The created subparser
    """
    parser = subparsers.add_parser(
        "version",
        parents=[common_parser],
        help="Display package version",
        description="Display the installed version of familiar-connect",
    )
    parser.set_defaults(func=run)
    return parser


def run(_args: argparse.Namespace) -> int:
    """Execute the version command.

    :param _args: Parsed command-line arguments (unused)
    :return: Exit code (0 for success)
    """
    print(  # noqa: T201
        f"{ls.tag('✨ Version', ls.C)} "
        f"{ls.word('familiar-connect', ls.W)} "
        f"{ls.word(__version__, ls.LC)}"
    )
    return 0
