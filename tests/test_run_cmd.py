"""Tests for the 'run' CLI subcommand."""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_connect.cli import create_parser
from familiar_connect.commands.run import (
    _async_main,
    _default_assembler,
    _resolve_familiar_root,
    load_opus,
    run,
)
from familiar_connect.context.layers import (
    ConversationSummaryLayer,
    CrossChannelContextLayer,
    PeopleDossierLayer,
    RagContextLayer,
    RecentHistoryLayer,
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
            return_value={"main_prose": MagicMock()},
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
            return_value={"main_prose": MagicMock()},
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
                return_value={"main_prose": MagicMock()},
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
                return_value={"main_prose": MagicMock()},
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


def _run_with_strategy(tmp_path: Path, strategy: str, extra_env: dict | None = None):
    """Run :func:`run` with all deps patched; return mocks for assertions."""
    args = argparse.Namespace(familiar="aria")
    env = {"DISCORD_BOT": "fake-token", "OPENROUTER_API_KEY": "sk-test"}
    if extra_env:
        env.update(extra_env)

    mock_load = MagicMock(
        return_value=MagicMock(id="aria", config=MagicMock()),
    )
    mock_create_local = MagicMock(return_value=None)
    mock_create_local_env = MagicMock(return_value=None)

    with (
        patch.dict("os.environ", env, clear=True),
        patch("familiar_connect.commands.run._DEFAULT_FAMILIARS_ROOT", tmp_path),
        patch(
            "familiar_connect.commands.run.load_character_config",
            return_value=_fake_character_config_with_turn_strategy(strategy),
        ),
        patch(
            "familiar_connect.commands.run.create_llm_clients",
            return_value={"main_prose": MagicMock()},
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
        patch(
            "familiar_connect.commands.run.create_local_turn_detector_from_env",
            mock_create_local_env,
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

    return result, mock_load, mock_create_local, mock_create_local_env


class TestRunTurnDetectionTomlSelector:
    """TOML ``[providers.turn_detection] strategy`` drives detector creation (A1)."""

    def test_deepgram_strategy_skips_local_detector(self, tmp_path: Path) -> None:
        """strategy='deepgram' → create_local_turn_detector not called."""
        (tmp_path / "aria").mkdir()
        _result, mock_load, mock_create_local, _ = _run_with_strategy(
            tmp_path, "deepgram"
        )
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
                return_value={"main_prose": MagicMock()},
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

    def test_env_var_overrides_deepgram_toml_strategy(self, tmp_path: Path) -> None:
        """LOCAL_TURN_DETECTION=1 activates local detector even when TOML='deepgram'."""
        (tmp_path / "aria").mkdir()
        _result, _load, mock_create_local, mock_create_env = _run_with_strategy(
            tmp_path,
            "deepgram",
            extra_env={"LOCAL_TURN_DETECTION": "1"},
        )
        mock_create_env.assert_called_once()
        mock_create_local.assert_not_called()


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
    fam.llm_clients = {"main_prose": llm}
    fam.history_store = MagicMock()
    fam.config = MagicMock(history_window_size=10)
    fam.root = MagicMock()
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

        sw = MagicMock()
        sw.run = AsyncMock(side_effect=_hang)
        fe = MagicMock()
        fe.run = AsyncMock(side_effect=_hang)

        with (
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
            patch("familiar_connect.commands.run.VoiceResponder"),
            patch("familiar_connect.commands.run.TextResponder"),
            patch("familiar_connect.commands.run.SummaryWorker", return_value=sw),
            patch("familiar_connect.commands.run.FactExtractor", return_value=fe),
            pytest.raises(BaseExceptionGroup),  # TaskGroup wraps the inner raise
        ):
            await _async_main("fake-token", familiar)

        bot.close.assert_awaited_once()
        familiar.transcriber.stop.assert_awaited_once()
        familiar.bus.shutdown.assert_awaited_once()
        familiar.router.shutdown.assert_called_once()
        familiar.llm_clients["main_prose"].close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_transcriber_when_none(self) -> None:
        """No transcriber → cleanup must still close the bot, no AttributeError."""
        familiar = _fake_familiar_for_async_main()
        familiar.transcriber = None

        bot = MagicMock(name="bot")
        bot.start = AsyncMock(side_effect=RuntimeError("nope"))
        bot.close = AsyncMock()
        handle = MagicMock(bot=bot)

        sw = MagicMock()
        sw.run = AsyncMock(side_effect=_hang)
        fe = MagicMock()
        fe.run = AsyncMock(side_effect=_hang)

        with (
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
            patch("familiar_connect.commands.run.VoiceResponder"),
            patch("familiar_connect.commands.run.TextResponder"),
            patch("familiar_connect.commands.run.SummaryWorker", return_value=sw),
            patch("familiar_connect.commands.run.FactExtractor", return_value=fe),
            pytest.raises(BaseExceptionGroup),
        ):
            await _async_main("fake-token", familiar)

        bot.close.assert_awaited_once()


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
        familiar.config.history_window_size = 20
        familiar.history_store = MagicMock(name="history_store")
        asm = _default_assembler(familiar)
        return [type(layer).__name__ for layer in asm._layers]

    def test_conversation_summary_precedes_cross_channel(self, tmp_path: Path) -> None:
        """ConvSummary refreshes every N turns; cross-channel fans across sources."""
        order = self._layer_order(tmp_path)
        assert order.index(ConversationSummaryLayer.__name__) < order.index(
            CrossChannelContextLayer.__name__
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
