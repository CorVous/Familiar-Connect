"""CLI subcommands for python-template.

Each subcommand is defined in its own module and exports:
- add_parser(subparsers, common_parser): Register the subcommand
- run(args): Execute the subcommand logic
"""

from python_template.commands import version as version_cmd

__all__ = ["version_cmd"]
