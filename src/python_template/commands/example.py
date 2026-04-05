"""Example subcommand template.

This file demonstrates the pattern for creating new subcommands.
To add a new subcommand:

1. Copy this file to a new name (e.g., fetch_data.py)
2. Update the command name in add_parser()
3. Implement your logic in run()
4. Import and register in cli.py's create_parser()

Example registration in cli.py:
    from python_template.commands import example
    example.add_parser(subparsers, common_parser)
"""

import argparse
import logging

_logger = logging.getLogger(__name__)


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register the example subcommand.

    :param subparsers: Subparser action from main parser
    :param common_parser: Parent parser with common arguments
    :return: The created subparser
    """
    parser = subparsers.add_parser(
        "example",  # Change this to your command name
        parents=[common_parser],
        help="Example subcommand (replace with your description)",
        description="Detailed description of what this command does",
    )

    # Add command-specific arguments
    parser.add_argument(
        "name",
        help="Example positional argument",
    )
    parser.add_argument(
        "--greeting",
        default="Hello",
        help="Example optional argument (default: %(default)s)",
    )

    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute the example command.

    :param args: Parsed command-line arguments
    :return: Exit code (0 for success, non-zero for error)
    """
    _logger.info("Running example command with name=%s", args.name)
    _logger.debug("Greeting: %s", args.greeting)

    # Your command logic here
    print(f"{args.greeting}, {args.name}!")  # noqa: T201

    return 0
