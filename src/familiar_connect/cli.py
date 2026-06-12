"""CLI entry + argument parsing. Subcommands in ``commands/``."""

import argparse
import importlib.metadata
import logging
import sys

from dotenv import load_dotenv

from familiar_connect import __version__, log_style
from familiar_connect.commands import diagnose_cmd, run_cmd, version_cmd
from familiar_connect.log_style import StyledFormatter

# Dynamic package name from installed metadata
try:
    _PACKAGE_METADATA = importlib.metadata.metadata(__package__ or "familiar_connect")
    _CLI_NAME = _PACKAGE_METADATA["Name"]
except (importlib.metadata.PackageNotFoundError, KeyError):
    # Fallback for editable-install
    _CLI_NAME = "familiar-connect"

_logger = logging.getLogger(__name__)


def setup_logging(verbose: int = 0, level: str | None = None) -> None:
    """Configure logging.

    verbose: 0=WARNING, 1=INFO, 2+=DEBUG; ignored when ``level`` set.
    level: explicit name (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    Raises ``ValueError`` on unknown ``level``.
    """
    if level is not None:
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            msg = f"Invalid log level: {level}"
            raise ValueError(msg)
        log_level = numeric_level
    else:
        levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        log_level = levels[min(verbose, len(levels) - 1)]

    log_style.init()
    handler = logging.StreamHandler()
    handler.setFormatter(StyledFormatter())
    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        force=True,  # Reconfigure if already configured
    )

    # Keep INFO visible even if root is WARNING; -vv still flips DEBUG
    pkg_logger = logging.getLogger("familiar_connect")
    pkg_logger.setLevel(min(log_level, logging.INFO))

    _logger.debug("Logging configured: level=%s", logging.getLevelName(log_level))


def create_parser() -> argparse.ArgumentParser:
    """Top-level parser; subcommands wired in."""
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

    # Shared subcommand args (empty placeholder)
    common_parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=False,  # Allow bare invocation to print help
    )

    run_cmd.add_parser(subparsers, common_parser)
    diagnose_cmd.add_parser(subparsers, common_parser)
    version_cmd.add_parser(subparsers, common_parser)

    return parser


def main() -> int:
    """Return exit code (0 ok, non-zero error)."""
    load_dotenv()
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    setup_logging(verbose=args.verbose)

    try:
        return args.func(args)
    except Exception:
        _logger.exception("Command failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
