"""Version subcommand - display package version."""

import argparse
import logging

from python_template import __version__

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
        description="Display the installed version of python-template",
    )
    parser.set_defaults(func=run)
    return parser


def run(_args: argparse.Namespace) -> int:
    """Execute the version command.

    :param _args: Parsed command-line arguments (unused)
    :return: Exit code (0 for success)
    """
    _logger.info("Displaying version information")
    print(f"python-template {__version__}")  # noqa: T201
    return 0
