"""Tests for the guild-scoped Twitch watcher registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from familiar_connect.twitch import TwitchWatcherConfig
from familiar_connect.twitch_registry import (
    GuildTwitchState,
    RegistryError,
    clear_guild_twitch,
    get_guild_twitch,
    set_guild_twitch,
)
from familiar_connect.twitch_watcher import TwitchWatcher


def _make_state(guild_id: int = 1) -> GuildTwitchState:
    config = TwitchWatcherConfig()
    watcher = TwitchWatcher(
        config=config,
        broadcaster_id="123",
        channel="coolstreamer",
    )
    task = MagicMock()
    queue = MagicMock()
    return GuildTwitchState(
        guild_id=guild_id,
        channel="coolstreamer",
        broadcaster_id="123",
        config=config,
        watcher=watcher,
        task=task,
        queue=queue,
    )


@pytest.fixture(autouse=True)
def _clear_registry():
    """Reset registry state before and after each test."""
    clear_guild_twitch(1)
    clear_guild_twitch(2)
    yield
    clear_guild_twitch(1)
    clear_guild_twitch(2)


# ---------------------------------------------------------------------------
# GuildTwitchState dataclass
# ---------------------------------------------------------------------------


class TestGuildTwitchState:
    def test_fields_stored(self) -> None:
        """All fields are accessible after instantiation."""
        state = _make_state(guild_id=42)
        assert state.guild_id == 42
        assert state.channel == "coolstreamer"
        assert state.broadcaster_id == "123"
        assert isinstance(state.config, TwitchWatcherConfig)
        assert isinstance(state.watcher, TwitchWatcher)
        assert state.task is not None
        assert state.queue is not None


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_returns_none_for_unknown_guild(self) -> None:
        assert get_guild_twitch(999) is None

    def test_set_then_get_returns_state(self) -> None:
        state = _make_state(guild_id=1)
        set_guild_twitch(1, state)
        assert get_guild_twitch(1) is state

    def test_set_twice_raises_registry_error(self) -> None:
        state1 = _make_state(guild_id=1)
        state2 = _make_state(guild_id=1)
        set_guild_twitch(1, state1)
        with pytest.raises(RegistryError):
            set_guild_twitch(1, state2)

    def test_clear_removes_entry(self) -> None:
        state = _make_state(guild_id=1)
        set_guild_twitch(1, state)
        clear_guild_twitch(1)
        assert get_guild_twitch(1) is None

    def test_clear_noop_when_absent(self) -> None:
        """clear_guild_twitch is idempotent."""
        clear_guild_twitch(999)  # should not raise

    def test_different_guilds_are_independent(self) -> None:
        s1 = _make_state(guild_id=1)
        s2 = _make_state(guild_id=2)
        set_guild_twitch(1, s1)
        set_guild_twitch(2, s2)
        assert get_guild_twitch(1) is s1
        assert get_guild_twitch(2) is s2
        clear_guild_twitch(1)
        assert get_guild_twitch(1) is None
        assert get_guild_twitch(2) is s2
