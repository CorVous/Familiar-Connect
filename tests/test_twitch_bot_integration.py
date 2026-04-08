"""Integration tests for the /twitch command group lifecycle."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

from familiar_connect.bot import create_bot
from familiar_connect.twitch_commands import (
    connect_cmd,
    disconnect_cmd,
    events_cmd,
    redemptions_add_cmd,
    redemptions_list_cmd,
    status_cmd,
)
from familiar_connect.twitch_registry import clear_guild_twitch, get_guild_twitch


def _make_ctx(guild_id: int = 100) -> MagicMock:
    ctx = MagicMock()
    ctx.guild_id = guild_id
    ctx.respond = AsyncMock()
    return ctx


def _respond_text(ctx: MagicMock) -> str:
    call_args = ctx.respond.call_args
    if call_args is None:
        return ""
    return call_args.args[0] if call_args.args else ""


# ---------------------------------------------------------------------------
# Step 12 — /twitch group registered in bot
# ---------------------------------------------------------------------------


def test_create_bot_has_twitch_command_group() -> None:
    """create_bot registers a /twitch slash command group."""
    bot = create_bot()
    names = [cmd.name for cmd in bot.pending_application_commands]
    assert "twitch" in names


def test_twitch_group_requires_manage_guild() -> None:
    """The /twitch command group enforces Manage Server permission."""
    bot = create_bot()
    twitch_cmd = next(
        cmd for cmd in bot.pending_application_commands if cmd.name == "twitch"
    )
    perms = twitch_cmd.default_member_permissions
    assert perms is not None
    assert perms.manage_guild


# ---------------------------------------------------------------------------
# Step 13 — End-to-end lifecycle
# ---------------------------------------------------------------------------


class TestTwitchLifecycle:
    """Full connect → configure → disconnect lifecycle via command handlers."""

    GUILD = 500

    def setup_method(self) -> None:
        clear_guild_twitch(self.GUILD)

    def teardown_method(self) -> None:
        clear_guild_twitch(self.GUILD)

    def _run(self, coro: Coroutine[Any, Any, None]) -> None:
        asyncio.run(coro)

    def _ctx(self) -> MagicMock:
        return _make_ctx(guild_id=self.GUILD)

    def test_full_lifecycle(self) -> None:
        # 1. connect
        ctx = self._ctx()
        with (
            patch(
                "familiar_connect.twitch_commands.os.environ.get",
                side_effect=lambda k, d=None: (
                    "dummy" if k in {"TWITCH_CLIENT_ID", "TWITCH_ACCESS_TOKEN"} else d
                ),
            ),
            patch(
                "familiar_connect.twitch_commands._resolve_broadcaster_id",
                new=AsyncMock(return_value="77"),
            ),
            patch(
                "familiar_connect.twitch_commands._run_watcher",
                new=AsyncMock(),
            ),
        ):
            self._run(connect_cmd(ctx, channel="coolstreamer"))
        assert "coolstreamer" in _respond_text(ctx)
        assert get_guild_twitch(self.GUILD) is not None

        # 2. status shows channel + defaults
        ctx = self._ctx()
        self._run(status_cmd(ctx))
        text = _respond_text(ctx)
        assert "coolstreamer" in text
        assert "enabled" in text.lower()

        # 3. events — disable subscriptions
        ctx = self._ctx()
        self._run(events_cmd(ctx, subscriptions=False))
        state = get_guild_twitch(self.GUILD)
        assert state is not None
        assert state.config.subscriptions_enabled is False

        # 4. redemptions add
        ctx = self._ctx()
        self._run(redemptions_add_cmd(ctx, name="Hydrate"))
        state = get_guild_twitch(self.GUILD)
        assert state is not None
        assert "Hydrate" in state.config.redemption_names

        # 5. redemptions list shows Hydrate
        ctx = self._ctx()
        self._run(redemptions_list_cmd(ctx))
        assert "Hydrate" in _respond_text(ctx)

        # 6. disconnect
        ctx = self._ctx()
        self._run(disconnect_cmd(ctx))
        assert "disconnect" in _respond_text(ctx).lower()
        assert get_guild_twitch(self.GUILD) is None

        # 7. status after disconnect → not connected
        ctx = self._ctx()
        self._run(status_cmd(ctx))
        assert "not connected" in _respond_text(ctx).lower()
