"""CLI subcommands.

Each submodule exports ``add_parser(subparsers, common_parser)`` to
register and ``run(args)`` to execute.
"""

from familiar_connect.commands import diagnose as diagnose_cmd
from familiar_connect.commands import run as run_cmd
from familiar_connect.commands import version as version_cmd

__all__ = ["diagnose_cmd", "run_cmd", "version_cmd"]
