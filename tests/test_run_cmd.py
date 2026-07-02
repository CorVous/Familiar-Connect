"""Tests for the 'run' CLI subcommand."""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import signal
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_connect.activities.engine import ActivityEngine
from familiar_connect.budget import TierBudget
from familiar_connect.cli import create_parser
from familiar_connect.commands.run import (
    _async_main,
    _build_activity_engine,
    _default_assembler,
    _GracefulShutdown,
    _install_shutdown_handlers,
    _resolve_familiar_root,
    _wait_for_shutdown,
    load_opus,
    run,
)
from familiar_connect.context.layers import (
    ConversationSummaryLayer,
    PeopleDossierLayer,
    RagContextLayer,
    RecentHistoryLayer,
    ReflectionLayer,
)

if TYPE_CHECKING:
    from pathlib import Path


def _fake_character_config() -> MagicMock:
    """Return a stand-in CharacterConfig for run() plumbing tests.

    ``run()`` passes this object through to :func:`create_llm_clients`,
    :func:`create_tts_client`, and :meth:`Familiar.load_from_disk`, all
    of which are themselves patched in these tests — so the attribute
    surface the mock needs only covers the one accessor ``run()``
    itself uses (``tts.voice_id`` / ``tts.model``).
    """
    config = MagicMock(name="character_config")
    config.tts.provider = "azure"
    config.tts.voice_id = "test-voice-id"
    config.tts.model = "test-model"
    return config


def test_run_subcommand_registered() -> None:
    """The 'run' subcommand is recognized by the CLI parser."""
    parser = create_parser()
    args = parser.parse_args(["run"])
    assert args.command == "run"


def test_run_subcommand_has_familiar_flag() -> None:
    """The 'run' subcommand accepts --familiar."""
    parser = create_parser()
    args = parser.parse_args(["run", "--familiar", "aria"])
    assert args.familiar == "aria"


def test_run_missing_token_returns_error() -> None:
    """When DISCORD_BOT env var is missing, run returns 1."""
    args = argparse.Namespace(familiar="aria")
    with patch.dict("os.environ", {}, clear=True):
        result = run(args)
    assert result == 1


def test_run_missing_familiar_returns_error() -> None:
    """When neither --familiar nor FAMILIAR_ID is set, run returns 1."""
    args = argparse.Namespace(familiar=None)
    with patch.dict("os.environ", {"DISCORD_BOT": "fake-token"}, clear=True):
        result = run(args)
    assert result == 1


def test_run_missing_familiar_folder_returns_error(tmp_path: Path) -> None:
    """When the selected familiar folder does not exist, run returns 1."""
    args = argparse.Namespace(familiar="does-not-exist")
    with (
        patch.dict("os.environ", {"DISCORD_BOT": "fake-token"}, clear=True),
        patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
    ):
        result = run(args)
    assert result == 1


def test_run_missing_api_key_returns_error(tmp_path: Path) -> None:
    """When OPENROUTER_API_KEY is missing, run returns 1 after loading config."""
    (tmp_path / "aria").mkdir()
    args = argparse.Namespace(familiar="aria")

    with (
        patch.dict("os.environ", {"DISCORD_BOT": "fake-token"}, clear=True),
        patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
        patch(
            "familiar_connect.commands.run.load_character_config",
            return_value=_fake_character_config(),
        ),
    ):
        result = run(args)

    assert result == 1


# ---------------------------------------------------------------------------
# _resolve_familiar_root
# ---------------------------------------------------------------------------


class TestResolveFamiliarRoot:
    def test_flag_overrides_env(self, tmp_path: Path) -> None:
        (tmp_path / "flag").mkdir()
        (tmp_path / "env").mkdir()
        args = argparse.Namespace(familiar="flag")

        with (
            patch.dict("os.environ", {"FAMILIAR_ID": "env"}, clear=True),
            patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
        ):
            root = _resolve_familiar_root(args)

        assert root == tmp_path / "flag"

    def test_env_used_when_no_flag(self, tmp_path: Path) -> None:
        (tmp_path / "env-chosen").mkdir()
        args = argparse.Namespace(familiar=None)

        with (
            patch.dict("os.environ", {"FAMILIAR_ID": "env-chosen"}, clear=True),
            patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
        ):
            root = _resolve_familiar_root(args)

        assert root == tmp_path / "env-chosen"


# ---------------------------------------------------------------------------
# load_opus
# ---------------------------------------------------------------------------


class TestLoadOpus:
    def test_skips_if_already_loaded(self) -> None:
        """load_opus returns immediately when opus is already loaded."""
        with (
            patch(
                "familiar_connect.commands.run.discord.opus.is_loaded",
                return_value=True,
            ),
            patch("familiar_connect.commands.run.discord.opus.load_opus") as mock_load,
            patch(
                "familiar_connect.commands.run.ctypes.util.find_library"
            ) as mock_find,
        ):
            load_opus()

        mock_find.assert_not_called()
        mock_load.assert_not_called()

    def test_loads_from_find_library(self) -> None:
        """load_opus uses ctypes.util.find_library result when available."""
        with (
            patch(
                "familiar_connect.commands.run.discord.opus.is_loaded",
                return_value=False,
            ),
            patch(
                "familiar_connect.commands.run.ctypes.util.find_library",
                return_value="libopus.so",
            ),
            patch("familiar_connect.commands.run.discord.opus.load_opus") as mock_load,
        ):
            load_opus()

        mock_load.assert_called_once_with("libopus.so")

    def test_falls_back_to_known_paths(self) -> None:
        """load_opus tries fallback paths when find_library returns None."""
        with (
            patch(
                "familiar_connect.commands.run.discord.opus.is_loaded",
                return_value=False,
            ),
            patch(
                "familiar_connect.commands.run.ctypes.util.find_library",
                return_value=None,
            ),
            patch(
                "familiar_connect.commands.run.pathlib.Path.exists",
                autospec=True,
                side_effect=lambda self: str(self) == "/usr/lib/libopus.so.0",
            ),
            patch("familiar_connect.commands.run.discord.opus.load_opus") as mock_load,
        ):
            load_opus()

        mock_load.assert_called_once_with("/usr/lib/libopus.so.0")

    def test_warns_when_not_found(self) -> None:
        """load_opus logs a warning when no opus library is found."""
        with (
            patch(
                "familiar_connect.commands.run.discord.opus.is_loaded",
                return_value=False,
            ),
            patch(
                "familiar_connect.commands.run.ctypes.util.find_library",
                return_value=None,
            ),
            patch(
                "familiar_connect.commands.run.pathlib.Path.exists",
                return_value=False,
            ),
            patch("familiar_connect.commands.run.discord.opus.load_opus") as mock_load,
            patch("familiar_connect.commands.run._logger") as mock_logger,
        ):
            load_opus()

        mock_load.assert_not_called()
        mock_logger.warning.assert_called_once()
        assert "not found" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# run happy path (mocked asyncio)
# ---------------------------------------------------------------------------


def test_run_starts_asyncio_with_familiar(tmp_path: Path) -> None:
    """run() builds a Familiar from disk and hands it to _async_main.

    The new load order is: config → llm_clients → tts_client →
    transcriber → Familiar.load_from_disk. Every disk / network
    dependency is patched, so the test only pins the plumbing.
    """
    (tmp_path / "aria").mkdir()
    args = argparse.Namespace(familiar="aria")
    sentinel_coro = MagicMock(name="coroutine")

    env = {
        "DISCORD_BOT": "fake-token",
        "OPENROUTER_API_KEY": "sk-test",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
        patch(
            "familiar_connect.commands.run.load_character_config",
            return_value=_fake_character_config(),
        ),
        patch(
            "familiar_connect.commands.run.create_llm_clients",
            return_value={
                "fast": MagicMock(),
                "prose": MagicMock(),
                "background": MagicMock(),
            },
        ),
        patch(
            "familiar_connect.commands.run.create_tts_client",
            return_value=None,
        ),
        patch(
            "familiar_connect.commands.run.create_transcriber",
            return_value=None,
        ),
        patch(
            "familiar_connect.commands.run.Familiar.load_from_disk",
            return_value=MagicMock(
                id="aria",
                config=MagicMock(),
            ),
        ),
        patch("familiar_connect.commands.run.load_opus"),
        patch(
            "familiar_connect.commands.run._async_main",
            new_callable=MagicMock,
            return_value=sentinel_coro,
        ) as mock_async_main,
        patch("familiar_connect.commands.run.asyncio.run") as mock_run,
    ):
        result = run(args)

    assert result == 0
    mock_async_main.assert_called_once()
    # First arg is the token, second is a Familiar bundle.
    call_args = mock_async_main.call_args
    assert call_args[0][0] == "fake-token"
    mock_run.assert_called_once_with(sentinel_coro)


def test_run_loads_config_before_building_clients(tmp_path: Path) -> None:
    """The config must be loaded first so ``create_llm_clients`` can see it.

    Pins the "config → clients" order in :func:`run` — flipping
    them would mean building clients with no character config in
    hand, and the failure would only surface at runtime.
    """
    (tmp_path / "aria").mkdir()
    args = argparse.Namespace(familiar="aria")
    fake_config = _fake_character_config()

    env = {
        "DISCORD_BOT": "fake-token",
        "OPENROUTER_API_KEY": "sk-test",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
        patch(
            "familiar_connect.commands.run.load_character_config",
            return_value=fake_config,
        ) as mock_load_config,
        patch(
            "familiar_connect.commands.run.create_llm_clients",
            return_value={
                "fast": MagicMock(),
                "prose": MagicMock(),
                "background": MagicMock(),
            },
        ) as mock_create_llm,
        patch(
            "familiar_connect.commands.run.create_tts_client",
            return_value=None,
        ),
        patch(
            "familiar_connect.commands.run.create_transcriber",
            return_value=None,
        ),
        patch(
            "familiar_connect.commands.run.Familiar.load_from_disk",
            return_value=MagicMock(
                id="aria",
                config=MagicMock(),
            ),
        ),
        patch("familiar_connect.commands.run.load_opus"),
        patch(
            "familiar_connect.commands.run._async_main",
            new_callable=MagicMock,
            return_value=MagicMock(name="coroutine"),
        ),
        patch("familiar_connect.commands.run.asyncio.run"),
    ):
        run(args)

    # ``load_character_config`` is called; its return value is passed
    # into ``create_llm_clients`` as the second positional argument.
    mock_load_config.assert_called_once()
    mock_create_llm.assert_called_once()
    llm_call = mock_create_llm.call_args
    assert llm_call.args[0] == "sk-test"
    assert llm_call.args[1] is fake_config


# ---------------------------------------------------------------------------
# run — transcriber integration
# ---------------------------------------------------------------------------


class TestRunTranscriberIntegration:
    """Cover the Deepgram transcriber factory plumbing added in PR #17.

    The bot-level wiring shifted away from ``_async_main`` creating the
    clients directly — it now happens in :func:`run`, which passes the
    transcriber to :meth:`Familiar.load_from_disk`. These tests pin
    that new plumbing so the PR #17 coverage intent survives the merge.
    """

    def test_run_passes_transcriber_to_familiar_when_configured(
        self,
        tmp_path: Path,
    ) -> None:
        """A successful create_transcriber reaches load_from_disk."""
        (tmp_path / "aria").mkdir()
        args = argparse.Namespace(familiar="aria")
        mock_transcriber = MagicMock(name="transcriber")

        env = {
            "DISCORD_BOT": "fake-token",
            "OPENROUTER_API_KEY": "sk-test",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
            patch(
                "familiar_connect.commands.run.load_character_config",
                return_value=_fake_character_config(),
            ),
            patch(
                "familiar_connect.commands.run.create_llm_clients",
                return_value={
                    "fast": MagicMock(),
                    "prose": MagicMock(),
                    "background": MagicMock(),
                },
            ),
            patch(
                "familiar_connect.commands.run.create_tts_client",
                return_value=None,
            ),
            patch(
                "familiar_connect.commands.run.create_transcriber",
                return_value=mock_transcriber,
            ) as mock_create,
            patch(
                "familiar_connect.commands.run.Familiar.load_from_disk",
                return_value=MagicMock(
                    id="aria",
                    config=MagicMock(),
                ),
            ) as mock_load,
            patch("familiar_connect.commands.run.load_opus"),
            patch(
                "familiar_connect.commands.run._async_main",
                new_callable=MagicMock,
                return_value=MagicMock(name="coroutine"),
            ),
            patch("familiar_connect.commands.run.asyncio.run"),
        ):
            run(args)

        mock_create.assert_called_once()
        assert mock_load.call_args.kwargs.get("transcriber") is mock_transcriber

    def test_run_threads_none_when_transcriber_unavailable(
        self,
        tmp_path: Path,
    ) -> None:
        """A ValueError from the factory becomes transcriber=None (warn + continue)."""
        (tmp_path / "aria").mkdir()
        args = argparse.Namespace(familiar="aria")

        env = {
            "DISCORD_BOT": "fake-token",
            "OPENROUTER_API_KEY": "sk-test",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
            patch(
                "familiar_connect.commands.run.load_character_config",
                return_value=_fake_character_config(),
            ),
            patch(
                "familiar_connect.commands.run.create_llm_clients",
                return_value={
                    "fast": MagicMock(),
                    "prose": MagicMock(),
                    "background": MagicMock(),
                },
            ),
            patch(
                "familiar_connect.commands.run.create_tts_client",
                return_value=None,
            ),
            patch(
                "familiar_connect.commands.run.create_transcriber",
                side_effect=ValueError("DEEPGRAM_API_KEY not set"),
            ),
            patch(
                "familiar_connect.commands.run.Familiar.load_from_disk",
                return_value=MagicMock(
                    id="aria",
                    config=MagicMock(),
                ),
            ) as mock_load,
            patch("familiar_connect.commands.run.load_opus"),
            patch(
                "familiar_connect.commands.run._async_main",
                new_callable=MagicMock,
                return_value=MagicMock(name="coroutine"),
            ),
            patch("familiar_connect.commands.run.asyncio.run"),
        ):
            result = run(args)

        assert result == 0
        assert mock_load.call_args.kwargs.get("transcriber") is None


# ---------------------------------------------------------------------------
# run — turn detection TOML selector (A1)
# ---------------------------------------------------------------------------


def _fake_character_config_with_turn_strategy(strategy: str) -> MagicMock:
    config = _fake_character_config()
    config.turn_detection.strategy = strategy
    return config


def _run_with_strategy(tmp_path: Path, strategy: str):
    """Run :func:`run` with all deps patched; return mocks for assertions."""
    args = argparse.Namespace(familiar="aria")
    env = {"DISCORD_BOT": "fake-token", "OPENROUTER_API_KEY": "sk-test"}

    mock_load = MagicMock(
        return_value=MagicMock(id="aria", config=MagicMock()),
    )
    mock_create_local = MagicMock(return_value=None)

    with (
        patch.dict("os.environ", env, clear=True),
        patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
        patch(
            "familiar_connect.commands.run.load_character_config",
            return_value=_fake_character_config_with_turn_strategy(strategy),
        ),
        patch(
            "familiar_connect.commands.run.create_llm_clients",
            return_value={
                "fast": MagicMock(),
                "prose": MagicMock(),
                "background": MagicMock(),
            },
        ),
        patch("familiar_connect.commands.run.create_tts_client", return_value=None),
        patch(
            "familiar_connect.commands.run.create_transcriber",
            return_value=None,
        ),
        patch(
            "familiar_connect.commands.run.create_local_turn_detector",
            mock_create_local,
        ),
        patch("familiar_connect.commands.run.Familiar.load_from_disk", mock_load),
        patch("familiar_connect.commands.run.load_opus"),
        patch(
            "familiar_connect.commands.run._async_main",
            new_callable=MagicMock,
            return_value=MagicMock(name="coroutine"),
        ),
        patch("familiar_connect.commands.run.asyncio.run"),
    ):
        result = run(args)

    return result, mock_load, mock_create_local


class TestRunTurnDetectionTomlSelector:
    """TOML ``[providers.turn_detection] strategy`` drives detector creation."""

    def test_deepgram_strategy_skips_local_detector(self, tmp_path: Path) -> None:
        """strategy='deepgram' → create_local_turn_detector not called."""
        (tmp_path / "aria").mkdir()
        _result, mock_load, mock_create_local = _run_with_strategy(tmp_path, "deepgram")
        mock_create_local.assert_not_called()
        assert mock_load.call_args.kwargs.get("local_turn_detector") is None

    def test_ten_smart_turn_strategy_creates_local_detector(
        self, tmp_path: Path
    ) -> None:
        """strategy='ten+smart_turn' → create_local_turn_detector called."""
        (tmp_path / "aria").mkdir()
        mock_detector = MagicMock(name="detector")
        args = argparse.Namespace(familiar="aria")
        env = {"DISCORD_BOT": "fake-token", "OPENROUTER_API_KEY": "sk-test"}
        mock_load = MagicMock(return_value=MagicMock(id="aria", config=MagicMock()))

        with (
            patch.dict("os.environ", env, clear=True),
            patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
            patch(
                "familiar_connect.commands.run.load_character_config",
                return_value=_fake_character_config_with_turn_strategy(
                    "ten+smart_turn"
                ),
            ),
            patch(
                "familiar_connect.commands.run.create_llm_clients",
                return_value={
                    "fast": MagicMock(),
                    "prose": MagicMock(),
                    "background": MagicMock(),
                },
            ),
            patch("familiar_connect.commands.run.create_tts_client", return_value=None),
            patch(
                "familiar_connect.commands.run.create_transcriber",
                return_value=None,
            ),
            patch(
                "familiar_connect.commands.run.create_local_turn_detector",
                return_value=mock_detector,
            ) as mock_create,
            patch("familiar_connect.commands.run.Familiar.load_from_disk", mock_load),
            patch("familiar_connect.commands.run.load_opus"),
            patch(
                "familiar_connect.commands.run._async_main",
                new_callable=MagicMock,
                return_value=MagicMock(name="coroutine"),
            ),
            patch("familiar_connect.commands.run.asyncio.run"),
        ):
            result = run(args)

        assert result == 0
        mock_create.assert_called_once()
        assert mock_load.call_args.kwargs.get("local_turn_detector") is mock_detector


# ---------------------------------------------------------------------------
# _async_main — shutdown cleanup
# ---------------------------------------------------------------------------


def _fake_familiar_for_async_main() -> MagicMock:
    """Build a Familiar mock shaped just enough for ``_async_main`` plumbing."""
    fam = MagicMock(name="familiar")
    fam.id = "test"
    fam.tts_client = None
    fam.bus = MagicMock()
    fam.bus.start = AsyncMock()
    fam.bus.shutdown = AsyncMock()
    fam.router = MagicMock()
    fam.router.shutdown = MagicMock()
    llm = MagicMock(name="llm")
    llm.close = AsyncMock()
    fam.llm_clients = {"fast": llm, "prose": llm, "background": llm}
    fam.history_store = MagicMock()
    fam.config = MagicMock(voice_window_size=10, text_window_size=10)
    # real-but-missing path: activities.toml absent ⇒ engine disabled,
    # so _async_main runs the engine-None branch (zero behavior change)
    fam.root = pathlib.Path("data") / "familiars" / "_nonexistent-test"
    return fam


async def _hang() -> None:
    await asyncio.sleep(60)


@pytest.mark.filterwarnings("ignore:coroutine '_hang' was never awaited:RuntimeWarning")
class TestAsyncMainCleanup:
    """Pin the ``finally``-block cleanup so leaked aiohttp sessions don't return."""

    @pytest.mark.asyncio
    async def test_closes_bot_and_transcriber_on_bot_start_failure(self) -> None:
        """``bot.start`` failure must still trigger ``bot.close`` and ``transcriber.stop``."""  # noqa: E501
        familiar = _fake_familiar_for_async_main()
        familiar.transcriber = MagicMock(name="transcriber")
        familiar.transcriber.stop = AsyncMock()

        bot = MagicMock(name="bot")
        bot.start = AsyncMock(side_effect=RuntimeError("login failed"))
        bot.close = AsyncMock()
        handle = MagicMock(bot=bot)

        proj = MagicMock(name="projector")
        proj.run = AsyncMock(side_effect=_hang)
        proj.name = "stub-projector"

        scheduler_mock = MagicMock(name="alarm_scheduler")
        scheduler_mock.start = AsyncMock()
        scheduler_mock.shutdown = AsyncMock()

        fm_mock = MagicMock(name="focus_manager")
        fm_mock.initialize = AsyncMock()
        fm_mock.get_focus = MagicMock(return_value=None)

        with (
            patch(
                "familiar_connect.commands.run.FocusManager",
                return_value=fm_mock,
            ),
            patch("familiar_connect.commands.run.create_bot", return_value=handle),
            patch("familiar_connect.commands.run._default_assembler"),
            patch(
                "familiar_connect.commands.run._run_debug_processor",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_voice_responder",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_text_responder",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_alarm_waker",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch("familiar_connect.commands.run.VoiceResponder"),
            patch("familiar_connect.commands.run.TextResponder"),
            patch(
                "familiar_connect.commands.run.AlarmScheduler",
                return_value=scheduler_mock,
            ),
            patch("familiar_connect.commands.run.AlarmWaker"),
            patch(
                "familiar_connect.commands.run.create_projectors",
                return_value=[proj],
            ),
            patch("familiar_connect.commands.run.create_embedder", return_value=None),
            patch(
                "familiar_connect.commands.run._install_shutdown_handlers",
                return_value=lambda: None,
            ),
            pytest.raises(BaseExceptionGroup),  # TaskGroup wraps the inner raise
        ):
            await _async_main("fake-token", familiar)

        bot.close.assert_awaited_once()
        familiar.transcriber.stop.assert_awaited_once()
        familiar.bus.shutdown.assert_awaited_once()
        familiar.router.shutdown.assert_called_once()
        # one shared mock backs all three slots — close fires per slot
        assert familiar.llm_clients["prose"].close.await_count == 3

    @pytest.mark.asyncio
    async def test_skips_transcriber_when_none(self) -> None:
        """No transcriber → cleanup must still close the bot, no AttributeError."""
        familiar = _fake_familiar_for_async_main()
        familiar.transcriber = None

        bot = MagicMock(name="bot")
        bot.start = AsyncMock(side_effect=RuntimeError("nope"))
        bot.close = AsyncMock()
        handle = MagicMock(bot=bot)

        proj = MagicMock(name="projector")
        proj.run = AsyncMock(side_effect=_hang)
        proj.name = "stub-projector"

        scheduler_mock = MagicMock(name="alarm_scheduler")
        scheduler_mock.start = AsyncMock()
        scheduler_mock.shutdown = AsyncMock()

        fm_mock = MagicMock(name="focus_manager")
        fm_mock.initialize = AsyncMock()
        fm_mock.get_focus = MagicMock(return_value=None)

        with (
            patch(
                "familiar_connect.commands.run.FocusManager",
                return_value=fm_mock,
            ),
            patch("familiar_connect.commands.run.create_bot", return_value=handle),
            patch("familiar_connect.commands.run._default_assembler"),
            patch(
                "familiar_connect.commands.run._run_debug_processor",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_voice_responder",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_text_responder",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_alarm_waker",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch("familiar_connect.commands.run.VoiceResponder"),
            patch("familiar_connect.commands.run.TextResponder"),
            patch(
                "familiar_connect.commands.run.AlarmScheduler",
                return_value=scheduler_mock,
            ),
            patch("familiar_connect.commands.run.AlarmWaker"),
            patch(
                "familiar_connect.commands.run.create_projectors",
                return_value=[proj],
            ),
            patch("familiar_connect.commands.run.create_embedder", return_value=None),
            patch(
                "familiar_connect.commands.run._install_shutdown_handlers",
                return_value=lambda: None,
            ),
            pytest.raises(BaseExceptionGroup),
        ):
            await _async_main("fake-token", familiar)

        bot.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Graceful SIGINT / SIGTERM shutdown
# ---------------------------------------------------------------------------


class TestWaitForShutdown:
    """``_wait_for_shutdown`` blocks until the event fires, then unwinds."""

    @pytest.mark.asyncio
    async def test_blocks_until_event_then_raises(self) -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(_wait_for_shutdown(stop))
        await asyncio.sleep(0)
        assert not task.done()  # still parked on the event

        stop.set()
        with pytest.raises(_GracefulShutdown):
            await task


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="add_signal_handler unsupported on Windows ProactorEventLoop",
)
class TestInstallShutdownHandlers:
    """POSIX signal handlers translate SIGINT/SIGTERM into the stop event."""

    @pytest.mark.asyncio
    async def test_sigint_sets_stop_event(self) -> None:
        stop = asyncio.Event()
        remove = _install_shutdown_handlers(stop)
        try:
            signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(0.05)  # let the loop run the callback
            assert stop.is_set()
        finally:
            remove()

    @pytest.mark.asyncio
    async def test_remove_restores_handlers(self) -> None:
        stop = asyncio.Event()
        remove = _install_shutdown_handlers(stop)
        remove()  # idempotent + must not leave loop handlers behind
        remove()
        # after removal a fresh event is untouched by a new signal handler
        assert not stop.is_set()


class TestGracefulShutdown:
    """A signal during the run loop drains the TaskGroup and runs cleanup."""

    @pytest.mark.asyncio
    @pytest.mark.filterwarnings(
        "ignore:coroutine '_hang' was never awaited:RuntimeWarning"
    )
    async def test_signal_runs_cleanup_without_raising(self) -> None:
        """Stop event set → orderly teardown, no exception bubbles out."""
        familiar = _fake_familiar_for_async_main()
        familiar.transcriber = None

        async def _hang_args(*_a: object, **_kw: object) -> None:
            await asyncio.sleep(60)  # bot.start(token) passes one positional arg

        bot = MagicMock(name="bot")
        bot.start = AsyncMock(side_effect=_hang_args)  # hangs until cancelled
        bot.close = AsyncMock()
        handle = MagicMock(bot=bot)

        proj = MagicMock(name="projector")
        proj.run = AsyncMock(side_effect=_hang)
        proj.name = "stub-projector"

        scheduler_mock = MagicMock(name="alarm_scheduler")
        scheduler_mock.start = AsyncMock()
        scheduler_mock.shutdown = AsyncMock()

        fm_mock = MagicMock(name="focus_manager")
        fm_mock.initialize = AsyncMock()
        fm_mock.get_focus = MagicMock(return_value=None)

        def _fire_shutdown(stop: asyncio.Event):
            # mimic a SIGINT landing just after the group spins up
            asyncio.get_running_loop().call_soon(stop.set)
            return lambda: None

        with (
            patch(
                "familiar_connect.commands.run.FocusManager",
                return_value=fm_mock,
            ),
            patch("familiar_connect.commands.run.create_bot", return_value=handle),
            patch("familiar_connect.commands.run._default_assembler"),
            patch(
                "familiar_connect.commands.run._run_debug_processor",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_voice_responder",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_text_responder",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch(
                "familiar_connect.commands.run._run_alarm_waker",
                side_effect=lambda *_a, **_kw: _hang(),
            ),
            patch("familiar_connect.commands.run.VoiceResponder"),
            patch("familiar_connect.commands.run.TextResponder"),
            patch(
                "familiar_connect.commands.run.AlarmScheduler",
                return_value=scheduler_mock,
            ),
            patch("familiar_connect.commands.run.AlarmWaker"),
            patch(
                "familiar_connect.commands.run.create_projectors",
                return_value=[proj],
            ),
            patch("familiar_connect.commands.run.create_embedder", return_value=None),
            patch(
                "familiar_connect.commands.run._install_shutdown_handlers",
                _fire_shutdown,
            ),
        ):
            # must return cleanly — no KeyboardInterrupt, no ExceptionGroup
            await _async_main("fake-token", familiar)

        bot.close.assert_awaited_once()
        familiar.bus.shutdown.assert_awaited_once()
        scheduler_mock.shutdown.assert_awaited_once()
        familiar.router.shutdown.assert_called_once()


class TestRunKeyboardInterruptFallback:
    """KeyboardInterrupt escaping asyncio.run is swallowed, not surfaced."""

    def test_keyboard_interrupt_returns_quietly(self, tmp_path: Path) -> None:
        (tmp_path / "aria").mkdir()
        args = argparse.Namespace(familiar="aria")
        env = {"DISCORD_BOT": "fake-token", "OPENROUTER_API_KEY": "sk-test"}

        with (
            patch.dict("os.environ", env, clear=True),
            patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
            patch(
                "familiar_connect.commands.run.load_character_config",
                return_value=_fake_character_config(),
            ),
            patch(
                "familiar_connect.commands.run.create_llm_clients",
                return_value={
                    "fast": MagicMock(),
                    "prose": MagicMock(),
                    "background": MagicMock(),
                },
            ),
            patch("familiar_connect.commands.run.create_tts_client", return_value=None),
            patch(
                "familiar_connect.commands.run.create_transcriber",
                return_value=None,
            ),
            patch(
                "familiar_connect.commands.run.Familiar.load_from_disk",
                return_value=MagicMock(id="aria", config=MagicMock()),
            ),
            patch("familiar_connect.commands.run.load_opus"),
            patch(
                "familiar_connect.commands.run._async_main",
                new_callable=MagicMock,
                return_value=MagicMock(name="coroutine"),
            ),
            patch(
                "familiar_connect.commands.run.asyncio.run",
                side_effect=KeyboardInterrupt,
            ),
        ):
            # no traceback escapes — run() returns an int exit code
            result = run(args)

        assert result == 0


class TestDefaultAssemblerLayerOrder:
    """Pin the system-prompt layer order for OpenAI prompt-cache friendliness.

    OpenAI caches the longest matching prompt prefix; any layer that
    changes between turns invalidates everything *after* it. The fix is
    to place the slowest-refreshing dynamic layer first and the
    per-turn-changing one last.
    """

    def _layer_order(self, tmp_path: Path) -> list[str]:
        familiar = MagicMock(name="familiar")
        familiar.root = tmp_path
        familiar.history_store = MagicMock(name="history_store")
        familiar.config.display_tz = "UTC"  # real IANA name; layers resolve ZoneInfo
        asm = _default_assembler(familiar, window_size=20, budget=TierBudget())
        return [type(layer).__name__ for layer in asm._layers]

    def test_conversation_summary_precedes_reflection(self, tmp_path: Path) -> None:
        """ConvSummary refreshes every N turns; reflections refresh faster."""
        order = self._layer_order(tmp_path)
        assert order.index(ConversationSummaryLayer.__name__) < order.index(
            ReflectionLayer.__name__
        )

    def test_people_dossier_precedes_rag(self, tmp_path: Path) -> None:
        """RAG flips every turn; dossier only on per-fact watermark advance."""
        order = self._layer_order(tmp_path)
        assert order.index(PeopleDossierLayer.__name__) < order.index(
            RagContextLayer.__name__
        )

    def test_rag_is_last_system_prompt_layer(self, tmp_path: Path) -> None:
        """RAG sits at system-prompt tail; only RecentHistory follows (in messages)."""
        order = self._layer_order(tmp_path)
        rag = order.index(RagContextLayer.__name__)
        recent = order.index(RecentHistoryLayer.__name__)
        assert rag == recent - 1
        assert recent == len(order) - 1


# ---------------------------------------------------------------------------
# activities wiring
# ---------------------------------------------------------------------------

_ACTIVITIES_TOML = """\
[[catalog]]
id = "walk"
label = "creek walk"
duration_minutes = [20, 40]
seed = "A walk along the creek."
"""


def _activity_familiar(tmp_path: Path, *, with_catalog: bool) -> MagicMock:
    root = tmp_path / "aria"
    root.mkdir()
    if with_catalog:
        (root / "activities.toml").write_text(_ACTIVITIES_TOML)
    fam = MagicMock(name="familiar")
    fam.id = "aria"
    fam.root = root
    fam.bot_user_id = None
    fam.config.display_tz = "UTC"
    return fam


class TestBuildActivityEngine:
    """activities.toml drives engine construction — disabled ⇒ None."""

    def test_missing_sidecar_disables_engine(self, tmp_path: Path) -> None:
        fam = _activity_familiar(tmp_path, with_catalog=False)
        engine = _build_activity_engine(
            fam, focus_manager=MagicMock(), handle=MagicMock(voice_runtime={})
        )
        assert engine is None

    def test_catalog_enables_engine(self, tmp_path: Path) -> None:
        fam = _activity_familiar(tmp_path, with_catalog=True)
        engine = _build_activity_engine(
            fam, focus_manager=MagicMock(), handle=MagicMock(voice_runtime={})
        )
        assert isinstance(engine, ActivityEngine)

    def test_voice_active_fn_tracks_runtime_map(self, tmp_path: Path) -> None:
        fam = _activity_familiar(tmp_path, with_catalog=True)
        handle = MagicMock(voice_runtime={})
        engine = _build_activity_engine(fam, focus_manager=MagicMock(), handle=handle)
        assert engine is not None
        assert engine.defer_start("walk").get("ack") == "ok"
        # second engine sees the live runtime entry → start refused
        handle.voice_runtime[1] = MagicMock(name="voice_runtime")
        other = _build_activity_engine(fam, focus_manager=MagicMock(), handle=handle)
        assert other is not None
        assert "voice" in other.defer_start("walk")["error"]
