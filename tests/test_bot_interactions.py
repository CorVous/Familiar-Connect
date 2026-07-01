"""Slash-command interaction guards.

Handlers race Discord's 3s ACK window; a missed ACK surfaces as
``NotFound (10062)``. :func:`_defer_interaction` claims the window and
:func:`_reply` delivers the confirmation — both must treat a dead
interaction as benign (action already ran).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from familiar_connect.activities.config import ActivitiesConfig, ActivityType
from familiar_connect.activities.engine import ActivityEngine
from familiar_connect.bot import (
    DM_BOT_DISCLAIMER_DELETE_EMOJI,
    DM_BOT_DISCLAIMER_DISMISS_HINT,
    BotHandle,
    _defer_interaction,
    _register_events,
    _reply,
    build_activity_presence_cb,
    message_pings_bot,
)
from familiar_connect.bus.bus import InProcessEventBus
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.subscriptions import SubscriptionKind, SubscriptionRegistry

from .conftest import build_fake_llm_clients

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    from familiar_connect.familiar import Familiar


def _not_found() -> discord.NotFound:
    resp = SimpleNamespace(status=404, reason="Not Found")
    return discord.NotFound(resp, "Unknown interaction")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def _ctx(*, defer: AsyncMock | None = None, followup: AsyncMock | None = None):
    return SimpleNamespace(
        defer=defer or AsyncMock(),
        followup=SimpleNamespace(send=followup or AsyncMock()),
        command=SimpleNamespace(name="subscribe-text"),
    )


@pytest.mark.asyncio
async def test_defer_returns_true_on_success() -> None:
    ctx = _ctx()
    assert await _defer_interaction(ctx) is True
    ctx.defer.assert_awaited_once_with(ephemeral=True)


@pytest.mark.asyncio
async def test_defer_returns_false_on_dead_interaction() -> None:
    ctx = _ctx(defer=AsyncMock(side_effect=_not_found()))
    # must not raise — a stale interaction is benign
    assert await _defer_interaction(ctx) is False


@pytest.mark.asyncio
async def test_reply_sends_followup() -> None:
    ctx = _ctx()
    await _reply(ctx, "ok")
    ctx.followup.send.assert_awaited_once_with("ok", ephemeral=True)


@pytest.mark.asyncio
async def test_reply_swallows_dead_interaction() -> None:
    ctx = _ctx(followup=AsyncMock(side_effect=_not_found()))
    # must not raise
    await _reply(ctx, "ok")


class TestMessagePingsBot:
    """Ping detection over ``message.mentions``.

    Carries both ``<@id>`` mentions and reply-ping targets (py-cord);
    roles/@everyone live elsewhere and never count.
    """

    def test_true_when_bot_in_mentions(self) -> None:
        msg = SimpleNamespace(mentions=[SimpleNamespace(id=7), SimpleNamespace(id=99)])
        assert message_pings_bot(cast("discord.Message", msg), 99) is True

    def test_false_when_bot_absent(self) -> None:
        msg = SimpleNamespace(mentions=[SimpleNamespace(id=7)])
        assert message_pings_bot(cast("discord.Message", msg), 99) is False

    def test_false_when_bot_user_id_unknown(self) -> None:
        msg = SimpleNamespace(mentions=[SimpleNamespace(id=99)])
        assert message_pings_bot(cast("discord.Message", msg), None) is False

    def test_false_on_empty_mentions(self) -> None:
        msg = SimpleNamespace(mentions=[])
        assert message_pings_bot(cast("discord.Message", msg), 99) is False


def _presence_handle(*, ready: bool = True) -> tuple[BotHandle, MagicMock]:
    """Handle + raw bot mock (assertions on the mock keep ty happy)."""
    bot = MagicMock(name="bot")
    bot.is_ready.return_value = ready
    bot.change_presence = AsyncMock()
    fm = MagicMock(name="focus_manager")
    fm.presence_guild.return_value = "Guild"
    fm.presence_text.return_value = "general"
    return BotHandle(bot=bot, send_text=AsyncMock(), focus_manager=fm), bot


class TestActivityPresenceCb:
    """ActivityEngine presence callback — idle while out, focus restore back."""

    @pytest.mark.asyncio
    async def test_idle_sets_idle_status_with_label(self) -> None:
        handle, bot = _presence_handle()
        cb = build_activity_presence_cb(handle)
        await cb("idle", "creek walk")
        kwargs = bot.change_presence.await_args.kwargs
        assert kwargs["status"] is discord.Status.idle
        assert kwargs["activity"].name == "creek walk"

    @pytest.mark.asyncio
    async def test_dnd_sets_dnd_status_with_label(self) -> None:
        handle, bot = _presence_handle()
        cb = build_activity_presence_cb(handle)
        await cb("dnd", "hatbox tending")
        kwargs = bot.change_presence.await_args.kwargs
        assert kwargs["status"] is discord.Status.dnd
        assert kwargs["activity"].name == "hatbox tending"

    @pytest.mark.asyncio
    async def test_online_restores_focus_presence(self) -> None:
        handle, bot = _presence_handle()
        cb = build_activity_presence_cb(handle)
        await cb("online", None)
        kwargs = bot.change_presence.await_args.kwargs
        assert kwargs["status"] is discord.Status.online
        # _sync_presence path — custom status carries focus label
        assert "general" in kwargs["activity"].state

    @pytest.mark.asyncio
    async def test_noop_when_bot_not_ready(self) -> None:
        handle, bot = _presence_handle(ready=False)
        cb = build_activity_presence_cb(handle)
        await cb("idle", "creek walk")
        bot.change_presence.assert_not_awaited()


class _NoFocus:
    """FocusLike stand-in — no focused channel."""

    def get_focus(self, modality: str) -> int | None:
        del modality
        return None


class TestOnReadyPresenceResync:
    """on_ready must end with away presence when mid-activity.

    Boot order: engine.start() runs pre-login, its away call is
    dropped by the cb's ready guard, and ``on_ready``'s focus sync
    sets online — the post-sync resync has to win.
    """

    @staticmethod
    def _engine(handle: BotHandle, tmp_path: Path) -> ActivityEngine:
        store = AsyncHistoryStore(HistoryStore(tmp_path / "history.db"))
        hatbox = ActivityType(
            id="hatbox",
            label="hatbox tending",
            duration_minutes=(10, 20),
            reachable=False,
            seed="Tending the hatbox.",
        )
        now = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)
        store.sync.create_activity(
            familiar_id="aria",
            type_id="hatbox",
            label="hatbox tending",
            started_at=now - timedelta(minutes=5),
            planned_return_at=now + timedelta(minutes=15),
            note=None,
        )
        return ActivityEngine(
            store=store,
            config=ActivitiesConfig(catalog=(hatbox,)),
            llm_clients=build_fake_llm_clients(),
            bus=InProcessEventBus(),
            focus_manager=_NoFocus(),
            presence_cb=build_activity_presence_cb(handle),
            familiar_id="aria",
            display_tz="UTC",
            bot_user_id=lambda: 99,
            now_fn=lambda: now,
        )

    @pytest.mark.asyncio
    async def test_ready_after_reload_ends_with_away_presence(
        self, tmp_path: Path
    ) -> None:
        events: dict[str, Callable[..., Coroutine[None, None, None]]] = {}
        handle, bot = _presence_handle(ready=False)
        bot.user = SimpleNamespace(id=99)
        bot.guilds = []
        bot.event.side_effect = lambda coro: events.setdefault(coro.__name__, coro)
        engine = self._engine(handle, tmp_path)
        handle.activity_engine = engine
        familiar = cast("Familiar", SimpleNamespace(bot_user_id=None))
        _register_events(bot, familiar, MagicMock(), handle)
        # boot order: engine reload fires pre-ready — cb drops the call
        await engine.start()
        bot.change_presence.assert_not_awaited()
        bot.is_ready.return_value = True
        await events["on_ready"]()
        calls = bot.change_presence.await_args_list
        # _sync_presence first (online), resync wins with away (dnd)
        assert calls[0].kwargs["status"] is discord.Status.online
        assert calls[-1].kwargs["status"] is discord.Status.dnd
        assert calls[-1].kwargs["activity"].name == "hatbox tending"
        await engine.stop()

    @pytest.mark.asyncio
    async def test_ready_without_engine_skips_resync(self) -> None:
        events: dict[str, Callable[..., Coroutine[None, None, None]]] = {}
        handle, bot = _presence_handle(ready=True)
        bot.user = SimpleNamespace(id=99)
        bot.guilds = []
        bot.event.side_effect = lambda coro: events.setdefault(coro.__name__, coro)
        familiar = cast("Familiar", SimpleNamespace(bot_user_id=None))
        _register_events(bot, familiar, MagicMock(), handle)
        # activities disabled — clean no-op, only the focus sync runs
        await events["on_ready"]()
        calls = bot.change_presence.await_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["status"] is discord.Status.online


# Verbatim first-DM disclaimer the reviewer (PR #176) requested. The test
# pins the exact wording here; the bot's constant must match it byte-for-byte.
_DM_DISCLAIMER = (
    "⚠ This is a bot, and content may not be isolated solely to this channel- "
    "treat messages in this conversation as if they were public."
)


def _dm_message(
    *, author_id: int, channel_id: int, guild: object | None, bot: bool = False
) -> SimpleNamespace:
    """Fake ``discord.Message`` carrying only the fields ``on_message`` reads.

    ``channel.send`` returns a stub disclaimer message (id 777) whose
    ``add_reaction`` / ``delete`` are awaitable so the checkmark-to-dismiss
    flow can be exercised.
    """
    sent = SimpleNamespace(id=777, add_reaction=AsyncMock(), delete=AsyncMock())
    return SimpleNamespace(
        author=SimpleNamespace(id=author_id, bot=bot, name="x", display_name="X"),
        channel=SimpleNamespace(id=channel_id, send=AsyncMock(return_value=sent)),
        guild=guild,
        content="hi",
        mentions=[],
        attachments=[],
        embeds=[],
        reference=None,
        id=42,
    )


def _reaction_payload(
    *, user_id: int, message_id: int, channel_id: int, emoji_name: str
) -> SimpleNamespace:
    """Fake ``discord.RawReactionActionEvent`` for ``on_raw_reaction_add``."""
    return SimpleNamespace(
        user_id=user_id,
        message_id=message_id,
        channel_id=channel_id,
        emoji=discord.PartialEmoji(name=emoji_name),
    )


class TestOnMessageDmAllowlist:
    """DM allowlist gate in ``on_message``.

    Allowlisted DMs become ephemeral text subscriptions (never written
    to the sidecar) so the normal focus/respond machinery handles them;
    guild channels keep their existing subscription gate.
    """

    @staticmethod
    def _setup(
        tmp_path: Path, *, allowlist: tuple[int, ...] = (123,)
    ) -> tuple[
        dict[str, Callable[..., Coroutine[None, None, None]]],
        Familiar,
        MagicMock,
        MagicMock,
    ]:
        events: dict[str, Callable[..., Coroutine[None, None, None]]] = {}
        bot = MagicMock(name="bot")
        bot.event.side_effect = lambda coro: events.setdefault(coro.__name__, coro)
        text_source = MagicMock(name="text_source")
        text_source.publish_text = AsyncMock()
        fm = MagicMock(name="focus_manager")
        fm.guild_names = {}
        fm.get_focus.return_value = None
        handle = BotHandle(bot=bot, send_text=AsyncMock(), focus_manager=fm)
        familiar = cast(
            "Familiar",
            SimpleNamespace(
                bot_user_id=99,
                subscriptions=SubscriptionRegistry(tmp_path / "subs.toml"),
                config=SimpleNamespace(dm_allowlist=allowlist),
                id="fam",
                history_store=SimpleNamespace(sync=MagicMock()),
            ),
        )
        _register_events(bot, familiar, text_source, handle)
        return events, familiar, text_source, fm

    @pytest.mark.asyncio
    async def test_allowlisted_dm_registers_ephemeral_and_ingests(
        self, tmp_path: Path
    ) -> None:
        events, familiar, text_source, fm = self._setup(tmp_path)
        msg = _dm_message(author_id=123, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        text_source.publish_text.assert_awaited_once()
        assert (
            familiar.subscriptions.get(channel_id=555, kind=SubscriptionKind.text)
            is not None
        )
        assert fm.guild_names[555] == "Private Message"
        fm.set_focus_immediately.assert_called_once_with(555, "text")
        # ephemeral row must never touch the sidecar
        assert not (tmp_path / "subs.toml").exists()

    @pytest.mark.asyncio
    async def test_non_allowlisted_dm_ignored(self, tmp_path: Path) -> None:
        events, familiar, text_source, fm = self._setup(tmp_path)
        msg = _dm_message(author_id=999, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        text_source.publish_text.assert_not_awaited()
        assert (
            familiar.subscriptions.get(channel_id=555, kind=SubscriptionKind.text)
            is None
        )
        assert fm.guild_names == {}

    @pytest.mark.asyncio
    async def test_bot_authored_dm_ignored_even_if_allowlisted(
        self, tmp_path: Path
    ) -> None:
        # author.bot guard precedes the allowlist branch — a bot whose id
        # collides with the allowlist must never be admitted (no DM loops).
        events, familiar, text_source, fm = self._setup(tmp_path, allowlist=(123,))
        msg = _dm_message(author_id=123, channel_id=555, guild=None, bot=True)
        await events["on_message"](cast("discord.Message", msg))
        text_source.publish_text.assert_not_awaited()
        assert (
            familiar.subscriptions.get(channel_id=555, kind=SubscriptionKind.text)
            is None
        )
        assert fm.guild_names == {}

    @pytest.mark.asyncio
    async def test_own_dm_echo_ignored_even_if_allowlisted(
        self, tmp_path: Path
    ) -> None:
        # bot-self guard (author.id == bot_user_id, which is 99) precedes the
        # allowlist branch — the familiar must not answer its own DM echo.
        events, familiar, text_source, fm = self._setup(tmp_path, allowlist=(99,))
        msg = _dm_message(author_id=99, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        text_source.publish_text.assert_not_awaited()
        assert (
            familiar.subscriptions.get(channel_id=555, kind=SubscriptionKind.text)
            is None
        )
        assert fm.guild_names == {}

    @pytest.mark.asyncio
    async def test_allowlisted_dm_keeps_existing_focus(self, tmp_path: Path) -> None:
        events, _familiar, text_source, fm = self._setup(tmp_path)
        fm.get_focus.return_value = 777
        msg = _dm_message(author_id=123, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        fm.set_focus_immediately.assert_not_called()
        text_source.publish_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subscribed_guild_channel_ingests(self, tmp_path: Path) -> None:
        events, familiar, text_source, _fm = self._setup(tmp_path)
        familiar.subscriptions.add(
            channel_id=888, kind=SubscriptionKind.text, guild_id=7
        )
        msg = _dm_message(author_id=123, channel_id=888, guild=SimpleNamespace(id=7))
        await events["on_message"](cast("discord.Message", msg))
        text_source.publish_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unsubscribed_guild_channel_ignored(self, tmp_path: Path) -> None:
        events, _familiar, text_source, _fm = self._setup(tmp_path)
        msg = _dm_message(author_id=123, channel_id=888, guild=SimpleNamespace(id=7))
        await events["on_message"](cast("discord.Message", msg))
        text_source.publish_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_first_allowlisted_dm_sends_disclaimer_once(
        self, tmp_path: Path
    ) -> None:
        events, _familiar, _ts, _fm = self._setup(tmp_path)
        msg = _dm_message(author_id=123, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        # Verbatim core + dismissal hint, and the checkmark is pre-seeded so
        # dismissing is a single click.
        msg.channel.send.assert_awaited_once_with(
            _DM_DISCLAIMER + DM_BOT_DISCLAIMER_DISMISS_HINT
        )
        msg.channel.send.return_value.add_reaction.assert_awaited_once_with(
            DM_BOT_DISCLAIMER_DELETE_EMOJI
        )

    @pytest.mark.asyncio
    async def test_second_dm_same_user_does_not_resend_disclaimer(
        self, tmp_path: Path
    ) -> None:
        events, _familiar, _ts, _fm = self._setup(tmp_path)
        first = _dm_message(author_id=123, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", first))
        first.channel.send.assert_awaited_once_with(
            _DM_DISCLAIMER + DM_BOT_DISCLAIMER_DISMISS_HINT
        )
        second = _dm_message(author_id=123, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", second))
        second.channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_allowlisted_dm_never_sends_disclaimer(
        self, tmp_path: Path
    ) -> None:
        events, _familiar, _ts, _fm = self._setup(tmp_path)
        msg = _dm_message(author_id=999, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        msg.channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bot_authored_dm_does_not_send_disclaimer(
        self, tmp_path: Path
    ) -> None:
        # bot-author guard precedes admission — a bot whose id collides with
        # the allowlist must never trigger the disclaimer (no DM loop).
        events, _familiar, _ts, _fm = self._setup(tmp_path, allowlist=(123,))
        msg = _dm_message(author_id=123, channel_id=555, guild=None, bot=True)
        await events["on_message"](cast("discord.Message", msg))
        msg.channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_own_echo_dm_does_not_send_disclaimer(self, tmp_path: Path) -> None:
        # bot-self guard precedes admission — the familiar's own disclaimer
        # echo (author.id == bot_user_id) must not re-trigger a disclaimer.
        events, _familiar, _ts, _fm = self._setup(tmp_path, allowlist=(99,))
        msg = _dm_message(author_id=99, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        msg.channel.send.assert_not_awaited()

    async def _send_disclaimer(
        self, tmp_path: Path
    ) -> tuple[
        dict[str, Callable[..., Coroutine[None, None, None]]],
        MagicMock,
        SimpleNamespace,
    ]:
        """Trigger a first DM; return (events, bump-reaction mock, sent stub).

        The mock is ``familiar.history_store.sync.bump_reaction`` — the delta
        write the reaction handlers must *not* perform for a disclaimer.
        """
        events, familiar, _ts, _fm = self._setup(tmp_path)
        msg = _dm_message(author_id=123, channel_id=555, guild=None)
        await events["on_message"](cast("discord.Message", msg))
        store = cast("MagicMock", familiar.history_store.sync)
        return events, store.bump_reaction, msg.channel.send.return_value

    @pytest.mark.asyncio
    async def test_user_checkmark_deletes_disclaimer(self, tmp_path: Path) -> None:
        events, bump, sent = await self._send_disclaimer(tmp_path)
        payload = _reaction_payload(
            user_id=123,
            message_id=sent.id,
            channel_id=555,
            emoji_name=DM_BOT_DISCLAIMER_DELETE_EMOJI,
        )
        await events["on_raw_reaction_add"](
            cast("discord.RawReactionActionEvent", payload)
        )
        sent.delete.assert_awaited_once()
        # Dismissal is not a history reaction — never written to the store.
        bump.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_own_checkmark_does_not_delete_disclaimer(
        self, tmp_path: Path
    ) -> None:
        # The bot pre-seeds the checkmark (user_id == bot_user_id); that must
        # not self-delete the disclaimer, nor write an orphan reaction row for
        # a message id that was never ingested as a history turn.
        events, bump, sent = await self._send_disclaimer(tmp_path)
        payload = _reaction_payload(
            user_id=99,
            message_id=sent.id,
            channel_id=555,
            emoji_name=DM_BOT_DISCLAIMER_DELETE_EMOJI,
        )
        await events["on_raw_reaction_add"](
            cast("discord.RawReactionActionEvent", payload)
        )
        sent.delete.assert_not_awaited()
        bump.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_checkmark_reaction_does_not_delete_disclaimer(
        self, tmp_path: Path
    ) -> None:
        events, bump, sent = await self._send_disclaimer(tmp_path)
        payload = _reaction_payload(
            user_id=123, message_id=sent.id, channel_id=555, emoji_name="👍"
        )
        await events["on_raw_reaction_add"](
            cast("discord.RawReactionActionEvent", payload)
        )
        sent.delete.assert_not_awaited()
        # A non-checkmark on the disclaimer is ignored, not recorded.
        bump.assert_not_called()

    @pytest.mark.asyncio
    async def test_unreacting_disclaimer_writes_no_history(
        self, tmp_path: Path
    ) -> None:
        # Un-reacting the pre-seeded checkmark must not write a -1 orphan row.
        events, bump, sent = await self._send_disclaimer(tmp_path)
        payload = _reaction_payload(
            user_id=123,
            message_id=sent.id,
            channel_id=555,
            emoji_name=DM_BOT_DISCLAIMER_DELETE_EMOJI,
        )
        await events["on_raw_reaction_remove"](
            cast("discord.RawReactionActionEvent", payload)
        )
        bump.assert_not_called()

    @pytest.mark.asyncio
    async def test_checkmark_on_unknown_message_does_not_delete(
        self, tmp_path: Path
    ) -> None:
        # A checkmark on any other message must fall through to the normal
        # reaction-delta path: the disclaimer is untouched, the delta is recorded.
        events, bump, sent = await self._send_disclaimer(tmp_path)
        payload = _reaction_payload(
            user_id=123,
            message_id=999_999,
            channel_id=555,
            emoji_name=DM_BOT_DISCLAIMER_DELETE_EMOJI,
        )
        await events["on_raw_reaction_add"](
            cast("discord.RawReactionActionEvent", payload)
        )
        sent.delete.assert_not_awaited()
        bump.assert_called_once()
