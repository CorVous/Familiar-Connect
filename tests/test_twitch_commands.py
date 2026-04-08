"""Tests for /twitch slash command handlers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_connect.twitch import TwitchWatcherConfig
from familiar_connect.twitch_commands import (
    ads_immediate_cmd,
    connect_cmd,
    disconnect_cmd,
    events_cmd,
    redemptions_add_cmd,
    redemptions_clear_cmd,
    redemptions_list_cmd,
    redemptions_remove_cmd,
    status_cmd,
)
from familiar_connect.twitch_registry import (
    GuildTwitchState,
    clear_guild_twitch,
    get_guild_twitch,
    set_guild_twitch,
)
from familiar_connect.twitch_watcher import TwitchWatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(guild_id: int = 100) -> MagicMock:
    ctx = MagicMock()
    ctx.guild_id = guild_id
    ctx.respond = AsyncMock()
    return ctx


def _make_state(
    guild_id: int = 100,
    channel: str = "coolstreamer",
    config: TwitchWatcherConfig | None = None,
) -> GuildTwitchState:
    cfg = config or TwitchWatcherConfig()
    watcher = TwitchWatcher(config=cfg, broadcaster_id="999", channel=channel)
    task = MagicMock()
    return GuildTwitchState(
        guild_id=guild_id,
        channel=channel,
        broadcaster_id="999",
        config=cfg,
        watcher=watcher,
        task=task,
    )


def _respond_text(ctx: MagicMock) -> str:
    """Extract the first positional arg from the last respond() call."""
    call_args = ctx.respond.call_args
    if call_args is None:
        return ""
    return call_args.args[0] if call_args.args else ""


@pytest.fixture(autouse=True)
def _clear():
    clear_guild_twitch(100)
    clear_guild_twitch(200)
    yield
    clear_guild_twitch(100)
    clear_guild_twitch(200)


# ---------------------------------------------------------------------------
# /twitch connect
# ---------------------------------------------------------------------------


class TestConnectCmd:
    def test_connect_registers_guild(self) -> None:
        ctx = _make_ctx(guild_id=100)

        with (
            patch(
                "familiar_connect.twitch_commands.os.environ.get",
                side_effect=lambda k, d=None: (
                    "dummy" if k in {"TWITCH_CLIENT_ID", "TWITCH_ACCESS_TOKEN"} else d
                ),
            ),
            patch(
                "familiar_connect.twitch_commands._resolve_broadcaster_id",
                new=AsyncMock(return_value="42"),
            ),
        ):
            asyncio.run(connect_cmd(ctx, channel="coolstreamer"))

        state = get_guild_twitch(100)
        assert state is not None
        assert state.channel == "coolstreamer"

    def test_connect_response_includes_channel_name(self) -> None:
        ctx = _make_ctx(guild_id=100)

        with (
            patch(
                "familiar_connect.twitch_commands.os.environ.get",
                side_effect=lambda k, d=None: (
                    "dummy" if k in {"TWITCH_CLIENT_ID", "TWITCH_ACCESS_TOKEN"} else d
                ),
            ),
            patch(
                "familiar_connect.twitch_commands._resolve_broadcaster_id",
                new=AsyncMock(return_value="42"),
            ),
        ):
            asyncio.run(connect_cmd(ctx, channel="coolstreamer"))

        assert "coolstreamer" in _respond_text(ctx)

    def test_connect_already_connected_returns_error(self) -> None:
        set_guild_twitch(100, _make_state(guild_id=100, channel="oldchan"))
        ctx = _make_ctx(guild_id=100)

        with patch(
            "familiar_connect.twitch_commands.os.environ.get",
            side_effect=lambda k, d=None: (
                "dummy" if k in {"TWITCH_CLIENT_ID", "TWITCH_ACCESS_TOKEN"} else d
            ),
        ):
            asyncio.run(connect_cmd(ctx, channel="newchan"))

        text = _respond_text(ctx).lower()
        assert "already" in text or "oldchan" in text

    def test_connect_missing_credentials_returns_error(self) -> None:
        ctx = _make_ctx(guild_id=100)

        with patch(
            "familiar_connect.twitch_commands.os.environ.get",
            return_value=None,
        ):
            asyncio.run(connect_cmd(ctx, channel="coolstreamer"))

        text = _respond_text(ctx).lower()
        cred_words = {"credential", "token", "client", "env", "missing"}
        assert any(word in text for word in cred_words)
        assert get_guild_twitch(100) is None

    def test_connect_creates_background_task(self) -> None:
        ctx = _make_ctx(guild_id=100)

        with (
            patch(
                "familiar_connect.twitch_commands.os.environ.get",
                side_effect=lambda k, d=None: (
                    "dummy" if k in {"TWITCH_CLIENT_ID", "TWITCH_ACCESS_TOKEN"} else d
                ),
            ),
            patch(
                "familiar_connect.twitch_commands._resolve_broadcaster_id",
                new=AsyncMock(return_value="42"),
            ),
        ):
            asyncio.run(connect_cmd(ctx, channel="coolstreamer"))

        state = get_guild_twitch(100)
        assert state is not None
        assert isinstance(state.task, asyncio.Task)


# ---------------------------------------------------------------------------
# /twitch disconnect
# ---------------------------------------------------------------------------


class TestDisconnectCmd:
    def test_disconnect_clears_registry(self) -> None:
        state = _make_state(guild_id=100)
        set_guild_twitch(100, state)
        ctx = _make_ctx(guild_id=100)

        asyncio.run(disconnect_cmd(ctx))

        assert get_guild_twitch(100) is None

    def test_disconnect_cancels_task(self) -> None:
        state = _make_state(guild_id=100)
        set_guild_twitch(100, state)
        ctx = _make_ctx(guild_id=100)

        asyncio.run(disconnect_cmd(ctx))

        state.task.cancel.assert_called_once()

    def test_disconnect_responds_disconnected(self) -> None:
        state = _make_state(guild_id=100)
        set_guild_twitch(100, state)
        ctx = _make_ctx(guild_id=100)

        asyncio.run(disconnect_cmd(ctx))

        text = _respond_text(ctx).lower()
        assert "disconnect" in text

    def test_disconnect_not_connected_error(self) -> None:
        ctx = _make_ctx(guild_id=100)

        asyncio.run(disconnect_cmd(ctx))

        text = _respond_text(ctx).lower()
        assert "not connected" in text


# ---------------------------------------------------------------------------
# /twitch status
# ---------------------------------------------------------------------------


class TestStatusCmd:
    def test_status_not_connected(self) -> None:
        ctx = _make_ctx(guild_id=100)

        asyncio.run(status_cmd(ctx))

        text = _respond_text(ctx).lower()
        assert "not connected" in text

    def test_status_shows_channel_name(self) -> None:
        set_guild_twitch(100, _make_state(channel="coolstreamer"))
        ctx = _make_ctx(guild_id=100)

        asyncio.run(status_cmd(ctx))

        assert "coolstreamer" in _respond_text(ctx)

    def test_status_shows_enabled_disabled_flags(self) -> None:
        cfg = TwitchWatcherConfig(
            subscriptions_enabled=True,
            cheers_enabled=False,
            follows_enabled=True,
            ads_enabled=False,
        )
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx(guild_id=100)

        asyncio.run(status_cmd(ctx))

        text = _respond_text(ctx).lower()
        assert "enabled" in text
        assert "disabled" in text

    def test_status_shows_redemption_names(self) -> None:
        cfg = TwitchWatcherConfig(redemption_names=["Hydrate", "Talk to Familiar"])
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx(guild_id=100)

        asyncio.run(status_cmd(ctx))

        text = _respond_text(ctx)
        assert "Hydrate" in text
        assert "Talk to Familiar" in text

    def test_status_no_redemptions_shows_none(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx(guild_id=100)

        asyncio.run(status_cmd(ctx))

        text = _respond_text(ctx).lower()
        assert "no redemption" in text or "none" in text


# ---------------------------------------------------------------------------
# /twitch events
# ---------------------------------------------------------------------------


class TestEventsCmd:
    def test_events_not_connected(self) -> None:
        ctx = _make_ctx()
        asyncio.run(events_cmd(ctx))
        assert "not connected" in _respond_text(ctx).lower()

    def test_events_sets_subscriptions_false(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(events_cmd(ctx, subscriptions=False))
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.subscriptions_enabled is False

    def test_events_sets_multiple_flags(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(events_cmd(ctx, cheers=False, follows=True))
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.cheers_enabled is False
        assert state.config.follows_enabled is True

    def test_events_omitted_flag_unchanged(self) -> None:
        cfg = TwitchWatcherConfig(subscriptions_enabled=True, cheers_enabled=False)
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(events_cmd(ctx, follows=False))
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.subscriptions_enabled is True
        assert state.config.cheers_enabled is False

    def test_events_responds_with_new_state(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(events_cmd(ctx, subscriptions=False))
        assert "subscription" in _respond_text(ctx).lower()


# ---------------------------------------------------------------------------
# /twitch ads-immediate
# ---------------------------------------------------------------------------


class TestAdsImmediateCmd:
    def test_ads_immediate_not_connected(self) -> None:
        ctx = _make_ctx()
        asyncio.run(ads_immediate_cmd(ctx, enabled=True))
        assert "not connected" in _respond_text(ctx).lower()

    def test_ads_immediate_sets_true(self) -> None:
        cfg = TwitchWatcherConfig(ads_immediate=False)
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(ads_immediate_cmd(ctx, enabled=True))
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.ads_immediate is True

    def test_ads_immediate_sets_false(self) -> None:
        cfg = TwitchWatcherConfig(ads_immediate=True)
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(ads_immediate_cmd(ctx, enabled=False))
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.ads_immediate is False

    def test_ads_immediate_confirms_new_value(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(ads_immediate_cmd(ctx, enabled=False))
        assert "ads" in _respond_text(ctx).lower()


# ---------------------------------------------------------------------------
# /twitch redemptions add
# ---------------------------------------------------------------------------


class TestRedemptionsAddCmd:
    def test_add_not_connected(self) -> None:
        ctx = _make_ctx()
        asyncio.run(redemptions_add_cmd(ctx, name="Hydrate"))
        assert "not connected" in _respond_text(ctx).lower()

    def test_add_appends_to_list(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(redemptions_add_cmd(ctx, name="Hydrate"))
        state = get_guild_twitch(100)
        assert state is not None
        assert "Hydrate" in state.config.redemption_names

    def test_add_duplicate_leaves_list_unchanged(self) -> None:
        cfg = TwitchWatcherConfig(redemption_names=["Hydrate"])
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(redemptions_add_cmd(ctx, name="Hydrate"))
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.redemption_names.count("Hydrate") == 1

    def test_add_duplicate_responds_already_present(self) -> None:
        cfg = TwitchWatcherConfig(redemption_names=["Hydrate"])
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(redemptions_add_cmd(ctx, name="Hydrate"))
        assert "already" in _respond_text(ctx).lower()


# ---------------------------------------------------------------------------
# /twitch redemptions remove
# ---------------------------------------------------------------------------


class TestRedemptionsRemoveCmd:
    def test_remove_not_connected(self) -> None:
        ctx = _make_ctx()
        asyncio.run(redemptions_remove_cmd(ctx, name="Hydrate"))
        assert "not connected" in _respond_text(ctx).lower()

    def test_remove_existing_name(self) -> None:
        cfg = TwitchWatcherConfig(redemption_names=["Hydrate", "Other"])
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(redemptions_remove_cmd(ctx, name="Hydrate"))
        state = get_guild_twitch(100)
        assert state is not None
        assert "Hydrate" not in state.config.redemption_names

    def test_remove_nonexistent_name(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(redemptions_remove_cmd(ctx, name="Ghost"))
        text = _respond_text(ctx).lower()
        assert "not found" in text or "not in" in text


# ---------------------------------------------------------------------------
# /twitch redemptions list
# ---------------------------------------------------------------------------


class TestRedemptionsListCmd:
    def test_list_not_connected(self) -> None:
        ctx = _make_ctx()
        asyncio.run(redemptions_list_cmd(ctx))
        assert "not connected" in _respond_text(ctx).lower()

    def test_list_shows_names(self) -> None:
        cfg = TwitchWatcherConfig(redemption_names=["Hydrate", "Snack"])
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(redemptions_list_cmd(ctx))
        text = _respond_text(ctx)
        assert "Hydrate" in text
        assert "Snack" in text

    def test_list_empty(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(redemptions_list_cmd(ctx))
        text = _respond_text(ctx).lower()
        assert "no redemption" in text or "none" in text


# ---------------------------------------------------------------------------
# /twitch redemptions clear
# ---------------------------------------------------------------------------


class TestRedemptionsClearCmd:
    def test_clear_not_connected(self) -> None:
        ctx = _make_ctx()
        asyncio.run(redemptions_clear_cmd(ctx))
        assert "not connected" in _respond_text(ctx).lower()

    def test_clear_empties_list(self) -> None:
        cfg = TwitchWatcherConfig(redemption_names=["Hydrate"])
        set_guild_twitch(100, _make_state(config=cfg))
        ctx = _make_ctx()
        asyncio.run(redemptions_clear_cmd(ctx))
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.redemption_names == []

    def test_clear_already_empty_confirms(self) -> None:
        set_guild_twitch(100, _make_state())
        ctx = _make_ctx()
        asyncio.run(redemptions_clear_cmd(ctx))
        ctx.respond.assert_called_once()
        state = get_guild_twitch(100)
        assert state is not None
        assert state.config.redemption_names == []
