"""Example subcommand template.

Pattern for new subcommands:

1. Copy to new name (e.g. fetch_data.py)
2. Update command name in ``add_parser()``
3. Implement ``run()``
4. Import + register in ``cli.py``'s ``create_parser()``

Example registration in cli.py::

    from python_template.commands import example
    example.add_parser(subparsers, common_parser)
"""

import argparse
import logging

from familiar_connect import log_style as ls

_logger = logging.getLogger(__name__)


def add_parser(
    subparsers: argparse._SubParsersAction,
    common_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "example",  # Change to your command name
        parents=[common_parser],
        help="Example subcommand (replace with your description)",
        description="Detailed description of what this command does",
    )

    # Command-specific args
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
    _logger.debug("Greeting: %s", args.greeting)

    print(  # noqa: T201
        f"{ls.tag('👋 Example', ls.W)} {args.greeting}, {args.name}!"
    )

    return 0
