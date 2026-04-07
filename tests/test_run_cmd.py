"""Tests for the 'run' CLI subcommand."""

import argparse
from unittest.mock import MagicMock, patch

from familiar_connect.character import CharacterCardError
from familiar_connect.cli import create_parser
from familiar_connect.commands.run import build_system_prompt, run
from familiar_connect.preset import PresetError


def test_run_subcommand_registered() -> None:
    """The 'run' subcommand is recognized by the CLI parser."""
    parser = create_parser()
    args = parser.parse_args(["run"])
    assert args.command == "run"


def test_run_subcommand_has_character_flag() -> None:
    """The 'run' subcommand accepts --character."""
    parser = create_parser()
    args = parser.parse_args(["run", "--character", "/some/card.png"])
    assert args.character == "/some/card.png"


def test_run_subcommand_has_preset_flag() -> None:
    """The 'run' subcommand accepts --preset."""
    parser = create_parser()
    args = parser.parse_args(["run", "--preset", "/some/preset.json"])
    assert args.preset == "/some/preset.json"


def test_run_missing_token_returns_error() -> None:
    """When DISCORD_BOT env var is missing, run returns 1."""
    args = argparse.Namespace(character=None, preset=None)
    with patch.dict("os.environ", {}, clear=True):
        result = run(args)
    assert result == 1


def test_run_starts_trio_with_token() -> None:
    """run() launches trio.run with the token from the environment."""
    args = argparse.Namespace(character=None, preset=None)
    captured: list[tuple[object, ...]] = []

    def fake_trio_run(async_fn: object, *call_args: object) -> None:
        captured.append((async_fn, call_args))

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


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def _args(
        self,
        character: str | None = None,
        preset: str | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(character=character, preset=preset)

    def test_no_card_returns_empty(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = build_system_prompt(self._args())
        assert not result

    def test_card_env_var_used_when_no_flag(self) -> None:
        fake_card = MagicMock()
        fake_card.name = "TestChar"
        fake_card.description = "A test character."

        env = {"FAMILIAR_CHARACTER": "/fake/card.png"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "familiar_connect.commands.run.load_card",
                return_value=fake_card,
            ) as mock_load,
        ):
            result = build_system_prompt(self._args(preset=None))

        mock_load.assert_called_once_with("/fake/card.png")
        assert result == "A test character."

    def test_character_flag_overrides_env(self) -> None:
        fake_card = MagicMock()
        fake_card.name = "FlagChar"
        fake_card.description = "From flag."

        env = {"FAMILIAR_CHARACTER": "/env/card.png"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "familiar_connect.commands.run.load_card",
                return_value=fake_card,
            ) as mock_load,
        ):
            result = build_system_prompt(self._args(character="/flag/card.png"))

        mock_load.assert_called_once_with("/flag/card.png")
        assert result == "From flag."

    def test_card_load_error_returns_empty(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "familiar_connect.commands.run.load_card",
                side_effect=CharacterCardError("bad"),
            ),
        ):
            result = build_system_prompt(self._args(character="/bad/card.png"))

        assert not result

    def test_preset_assembled_with_card(self) -> None:
        fake_card = MagicMock()
        fake_card.name = "Aria"
        fake_card.description = "A spirit."
        fake_preset = {"prompts": [], "prompt_order": []}

        with (
            patch("familiar_connect.commands.run.load_card", return_value=fake_card),
            patch(
                "familiar_connect.commands.run.load_preset",
                return_value=fake_preset,
            ),
            patch(
                "familiar_connect.commands.run.assemble_prompt",
                return_value="Full system prompt.",
            ) as mock_assemble,
        ):
            result = build_system_prompt(
                self._args(character="/card.png", preset="/preset.json")
            )

        mock_assemble.assert_called_once_with(fake_preset, fake_card)
        assert result == "Full system prompt."

    def test_preset_load_error_falls_back_to_description(self) -> None:
        fake_card = MagicMock()
        fake_card.name = "Aria"
        fake_card.description = "A spirit."

        with (
            patch("familiar_connect.commands.run.load_card", return_value=fake_card),
            patch(
                "familiar_connect.commands.run.load_preset",
                side_effect=PresetError("bad"),
            ),
        ):
            result = build_system_prompt(
                self._args(character="/card.png", preset="/bad/preset.json")
            )

        assert result == "A spirit."
