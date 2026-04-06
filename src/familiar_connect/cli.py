"""Command-line interface.

This module provides the main CLI entry point and argument parsing.
Subcommands are defined in separate files under the commands/ directory.
"""

import argparse
import importlib.metadata
import logging
import sys

from dotenv import load_dotenv

from familiar_connect import __version__
from familiar_connect.commands import run_cmd, version_cmd

# Get package name dynamically from installed metadata
try:
    _PACKAGE_METADATA = importlib.metadata.metadata(__package__ or "familiar_connect")
    _CLI_NAME = _PACKAGE_METADATA["Name"]
except (importlib.metadata.PackageNotFoundError, KeyError):
    # Fallback if metadata not available (e.g., editable install issues)
    _CLI_NAME = "familiar-connect"

_logger = logging.getLogger(__name__)


def setup_logging(verbose: int = 0, level: str | None = None) -> None:
    """Configure logging based on verbosity level.

    Supports both count-based verbosity (-v, -vv, -vvv) and explicit
    log level specification (-v DEBUG).

    :param verbose: Verbosity count (0=WARNING, 1=INFO, 2+=DEBUG)
    :param level: Explicit log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :raises ValueError: If level is provided but invalid
    """
    # If explicit level provided, use it
    if level is not None:
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            msg = f"Invalid log level: {level}"
            raise ValueError(msg)
        log_level = numeric_level
    else:
        # Count-based: 0=WARNING, 1=INFO, 2+=DEBUG
        levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        log_level = levels[min(verbose, len(levels) - 1)]

    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        force=True,  # Reconfigure if already configured
    )
    _logger.debug("Logging configured: level=%s", logging.getLevelName(log_level))


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the main argument parser.

    :return: Configured ArgumentParser with subcommands
    """
    # Main parser with global arguments
    parser = argparse.ArgumentParser(
        prog=_CLI_NAME,
        description=f"{_CLI_NAME} CLI tool",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated: -v, -vv, -vvv)",
    )

    # Common arguments shared across all subcommands (empty for now, but available)
    common_parser = argparse.ArgumentParser(add_help=False)

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=False,  # Allow running without subcommand to show help
    )

    # Register subcommands
    run_cmd.add_parser(subparsers, common_parser)
    version_cmd.add_parser(subparsers, common_parser)

    return parser


def main() -> int:
    """Execute the main CLI entry point.

    :return: Exit code (0 for success, non-zero for error)
    """
    load_dotenv()
    parser = create_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        return 0

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Execute command
    try:
        return args.func(args)
    except Exception:
        _logger.exception("Command failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
