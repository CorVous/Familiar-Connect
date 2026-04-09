"""Red-first tests for the SubscriptionRegistry.

The registry replaces the single-slot ``text_session`` with a
persistent, multi-channel, multi-kind set of subscriptions. Each
``/subscribe-text`` or ``/subscribe-my-voice`` slash command adds a
row; the registry saves itself to TOML on every mutation so
subscriptions survive bot restarts.

Covers familiar_connect.subscriptions, which doesn't exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.subscriptions import (
    Subscription,
    SubscriptionKind,
    SubscriptionRegistry,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# SubscriptionKind
# ---------------------------------------------------------------------------


class TestSubscriptionKind:
    def test_has_text_and_voice(self) -> None:
        assert SubscriptionKind.text.value == "text"
        assert SubscriptionKind.voice.value == "voice"


# ---------------------------------------------------------------------------
# Registry — core CRUD
# ---------------------------------------------------------------------------


class TestRegistryCore:
    def test_empty_registry_has_nothing(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        assert list(reg.all()) == []
        assert reg.get(channel_id=1, kind=SubscriptionKind.text) is None

    def test_add_and_get(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=999)

        sub = reg.get(channel_id=42, kind=SubscriptionKind.text)
        assert isinstance(sub, Subscription)
        assert sub.channel_id == 42
        assert sub.kind is SubscriptionKind.text
        assert sub.guild_id == 999

    def test_add_is_idempotent(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=999)
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=999)
        assert len(list(reg.all())) == 1

    def test_remove(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=None)
        reg.remove(channel_id=42, kind=SubscriptionKind.text)
        assert reg.get(channel_id=42, kind=SubscriptionKind.text) is None

    def test_remove_of_unknown_is_noop(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.remove(channel_id=42, kind=SubscriptionKind.text)  # must not raise

    def test_text_and_voice_in_same_channel_coexist(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=1)
        reg.add(channel_id=42, kind=SubscriptionKind.voice, guild_id=1)
        assert len(list(reg.all())) == 2

    def test_all_returns_every_subscription(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=1, kind=SubscriptionKind.text, guild_id=None)
        reg.add(channel_id=2, kind=SubscriptionKind.text, guild_id=None)
        reg.add(channel_id=3, kind=SubscriptionKind.voice, guild_id=None)
        all_subs = list(reg.all())
        assert len(all_subs) == 3


# ---------------------------------------------------------------------------
# Registry — voice-specific helpers
# ---------------------------------------------------------------------------


class TestRegistryVoiceHelpers:
    def test_voice_in_guild_returns_sub(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        reg.add(channel_id=9000, kind=SubscriptionKind.voice, guild_id=123)
        assert reg.voice_in_guild(123) is not None

    def test_voice_in_guild_returns_none_when_absent(self, tmp_path: Path) -> None:
        reg = SubscriptionRegistry(tmp_path / "subs.toml")
        assert reg.voice_in_guild(123) is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_add_writes_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "subs.toml"
        reg = SubscriptionRegistry(path)
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=999)
        assert path.exists()

    def test_reload_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "subs.toml"
        reg = SubscriptionRegistry(path)
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=999)
        reg.add(channel_id=42, kind=SubscriptionKind.voice, guild_id=999)
        reg.add(channel_id=77, kind=SubscriptionKind.text, guild_id=None)

        reloaded = SubscriptionRegistry(path)
        assert reloaded.get(channel_id=42, kind=SubscriptionKind.text) is not None
        assert reloaded.get(channel_id=42, kind=SubscriptionKind.voice) is not None
        assert reloaded.get(channel_id=77, kind=SubscriptionKind.text) is not None
        assert len(list(reloaded.all())) == 3

    def test_remove_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "subs.toml"
        reg = SubscriptionRegistry(path)
        reg.add(channel_id=42, kind=SubscriptionKind.text, guild_id=999)
        reg.remove(channel_id=42, kind=SubscriptionKind.text)

        reloaded = SubscriptionRegistry(path)
        assert reloaded.get(channel_id=42, kind=SubscriptionKind.text) is None
