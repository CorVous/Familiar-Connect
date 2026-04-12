"""Mkdocs hook that inlines CLI help and slash-command tables.

Lets docs pages reference live strings from source instead of
maintaining a second copy. The hook rewrites two kinds of HTML-comment
tokens during ``on_page_markdown``:

- ``<!-- @slash-commands-table -->`` — expands to a markdown table
  of every ``bot.slash_command(name=..., description=...)`` call
  registered in ``src/familiar_connect/bot.py``. Order matches the
  registration order in source (grouped by related commands).

- ``<!-- @cli-help: familiar-connect [subcommand] -->`` — expands to
  a fenced code block containing the output of
  ``familiar-connect [subcommand] --help``. The top-level parser is
  referenced as ``familiar-connect``; a subcommand is
  ``familiar-connect <name>``.

Unknown tokens and unknown subcommands raise ``PluginError`` so typos
break ``mkdocs build --strict`` instead of silently surviving into
the published site.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.exceptions import PluginError

from familiar_connect.cli import create_parser

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BOT_MODULE = _REPO_ROOT / "src" / "familiar_connect" / "bot.py"

_TOKEN_RE = re.compile(r"<!--\s*@(?P<name>[a-z-]+)(?::\s*(?P<arg>[^>]*?))?\s*-->")


def _collect_slash_commands() -> list[tuple[str, str]]:
    """Return ``(name, description)`` for every slash command in bot.py.

    Walks the AST for ``bot.slash_command(name=..., description=...)``
    calls and preserves registration order (which groups related
    commands like ``/subscribe-text`` / ``/unsubscribe-text``).
    """
    tree = ast.parse(_BOT_MODULE.read_text(encoding="utf-8"))
    commands: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "slash_command"):
            continue
        name: str | None = None
        description: str | None = None
        for kw in node.keywords:
            if (
                kw.arg == "name"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                name = kw.value.value
            elif (
                kw.arg == "description"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                description = kw.value.value
        if name is not None and description is not None:
            commands.append((name, description))
    return commands


def _slash_commands_table() -> str:
    """Render all registered slash commands as a markdown table."""
    commands = _collect_slash_commands()
    if not commands:
        msg = (
            f"cli_reference hook: no bot.slash_command(...) calls found in "
            f"{_BOT_MODULE} — either the registration pattern changed or "
            f"the file moved."
        )
        raise PluginError(msg)
    lines = ["| Command | What it does |", "|---|---|"]
    lines.extend(f"| `/{name}` | {description} |" for name, description in commands)
    return "\n".join(lines)


def _find_subparser(
    parser: argparse.ArgumentParser,
    subcommand: str,
) -> argparse.ArgumentParser:
    """Return the argparse subparser for ``subcommand``.

    Raises :class:`PluginError` if the top-level parser has no
    subparsers or the requested subcommand is not registered.
    """
    for action in parser._actions:  # noqa: SLF001 — argparse exposes no public API
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            if subcommand in action.choices:
                return action.choices[subcommand]
            known = sorted(action.choices)
            msg = (
                f"cli_reference hook: unknown subcommand '{subcommand}'. "
                f"Registered subcommands: {known}."
            )
            raise PluginError(msg)
    msg = (
        f"cli_reference hook: top-level parser has no subparsers, cannot "
        f"resolve '{subcommand}'."
    )
    raise PluginError(msg)


def _cli_help_block(command: str) -> str:
    """Render ``<command> --help`` output as a fenced code block.

    ``command`` is the full invocation string, e.g.
    ``"familiar-connect"`` or ``"familiar-connect run"``. Anything
    before the first token must match the CLI program name; anything
    after selects the subparser.
    """
    parser = create_parser()
    parts = command.split()
    if not parts or parts[0] != parser.prog:
        msg = (
            f"cli_reference hook: @cli-help target must start with "
            f"'{parser.prog}', got '{command}'."
        )
        raise PluginError(msg)
    top_level_only = 1
    single_subcommand = 2
    if len(parts) == top_level_only:
        help_text = parser.format_help()
    elif len(parts) == single_subcommand:
        subparser = _find_subparser(parser, parts[1])
        help_text = subparser.format_help()
    else:
        msg = (
            f"cli_reference hook: @cli-help only supports the top-level "
            f"parser or a single subcommand, got '{command}'."
        )
        raise PluginError(msg)
    return f"```text\n{help_text.rstrip()}\n```"


def _resolve_token(match: re.Match[str]) -> str:
    name = match.group("name")
    arg = (match.group("arg") or "").strip()
    if name == "slash-commands-table":
        if arg:
            msg = (
                f"cli_reference hook: @slash-commands-table takes no "
                f"argument, got '{arg}'."
            )
            raise PluginError(msg)
        return _slash_commands_table()
    if name == "cli-help":
        if not arg:
            msg = (
                "cli_reference hook: @cli-help requires a command, e.g. "
                "'<!-- @cli-help: familiar-connect run -->'."
            )
            raise PluginError(msg)
        return _cli_help_block(arg)
    msg = (
        f"cli_reference hook: unknown token '@{name}'. Supported tokens: "
        f"@slash-commands-table, @cli-help."
    )
    raise PluginError(msg)


def substitute(markdown: str) -> str:
    """Replace every ``<!-- @... -->`` token in ``markdown``.

    Split out from :func:`on_page_markdown` so tests can call it
    directly without constructing mkdocs ``Page`` / ``Config`` objects.
    """
    return _TOKEN_RE.sub(_resolve_token, markdown)


def on_page_markdown(
    markdown: str,
    *,
    page: Page,  # noqa: ARG001 — mkdocs hook signature
    config: MkDocsConfig,  # noqa: ARG001
    files: Files,  # noqa: ARG001
) -> str:
    """Mkdocs hook entry point — rewrites tokens before rendering."""
    return substitute(markdown)
