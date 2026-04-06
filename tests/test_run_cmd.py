"""Tests for the 'run' CLI subcommand."""

import argparse
from unittest.mock import patch

from familiar_connect.cli import create_parser
from familiar_connect.commands.run import run


def test_run_subcommand_registered() -> None:
    """The 'run' subcommand is recognized by the CLI parser."""
    parser = create_parser()
    args = parser.parse_args(["run"])
    assert args.command == "run"


def test_run_missing_token_returns_error() -> None:
    """When DISCORD_BOT env var is missing, run returns 1."""
    args = argparse.Namespace()
    with patch.dict("os.environ", {}, clear=True):
        result = run(args)
    assert result == 1


def test_run_starts_trio_with_token() -> None:
    """run() launches trio.run with the token from the environment."""
    args = argparse.Namespace()
    captured: list[tuple[object, ...]] = []

    def fake_trio_run(async_fn: object, *args: object) -> None:
        captured.append((async_fn, args))

    with (
        patch.dict("os.environ", {"DISCORD_BOT": "fake-token"}),
        patch("familiar_connect.commands.run.trio") as mock_trio,
    ):
        mock_trio.run.side_effect = fake_trio_run
        result = run(args)

    assert result == 0
    mock_trio.run.assert_called_once()
    _, call_args, _ = mock_trio.run.mock_calls[0]
    assert call_args[1] == "fake-token"
