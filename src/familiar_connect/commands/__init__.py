"""CLI subcommands for familiar-connect.

Each subcommand is defined in its own module and exports:
- add_parser(subparsers, common_parser): Register the subcommand
- run(args): Execute the subcommand logic
"""

from familiar_connect.commands import metrics as metrics_cmd
from familiar_connect.commands import run as run_cmd
from familiar_connect.commands import version as version_cmd

__all__ = ["metrics_cmd", "run_cmd", "version_cmd"]
