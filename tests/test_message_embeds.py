"""Discord embed unfurls in text messages.

URL previews arrive as ``message.embeds`` — sometimes already attached
on the inbound ``on_message``, more often via a follow-up
``on_message_edit`` once Discord finishes unfurling. The bot merges
embed text into the message content so the LLM sees the same body
humans do.
"""

from __future__ import annotations

import discord

from familiar_connect.bot import (
    apply_message_edit,
    compose_content_with_embeds,
)
from familiar_connect.history.store import HistoryStore
from familiar_connect.identity import Author


def _author() -> Author:
    return Author(
        platform="discord",
        user_id="111",
        username="alice",
        display_name="Alice",
    )


def _store_with_turn(*, content: str, message_id: str = "m1") -> HistoryStore:
    store = HistoryStore(":memory:")
    store.append_turn(
        familiar_id="fam",
        channel_id=10,
        role="user",
        content=content,
        author=_author(),
        platform_message_id=message_id,
    )
    return store


class TestComposeContentWithEmbeds:
    def test_no_embeds_returns_original(self) -> None:
        assert compose_content_with_embeds("hi", ()) == "hi"

    def test_appends_embed_text_to_content(self) -> None:
        e = discord.Embed(description="body")
        out = compose_content_with_embeds("look at this", [e])
        assert out == "look at this\n\n[embed]\nbody"

    def test_url_only_message_yields_just_embed_text(self) -> None:
        # blank ``content`` shouldn't leave a leading "\n\n" prefix
        e = discord.Embed(description="body")
        out = compose_content_with_embeds("", [e])
        assert out == "[embed]\nbody"

    def test_blank_embed_keeps_content_only(self) -> None:
        # an embed that renders to "" must not introduce a trailing "\n\n"
        out = compose_content_with_embeds("hi", [discord.Embed()])
        assert out == "hi"


class TestApplyMessageEdit:
    def test_updates_stored_turn_when_embed_added(self) -> None:
        store = _store_with_turn(content="check this")
        e = discord.Embed(description="body")

        apply_message_edit(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _: True,
            channel_id=10,
            message_id="m1",
            content="check this",
            embeds=[e],
        )

        turn = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="m1"
        )
        assert turn is not None
        assert turn.content == "check this\n\n[embed]\nbody"

    def test_skips_when_channel_not_subscribed(self) -> None:
        store = _store_with_turn(content="check this")
        e = discord.Embed(description="body")

        apply_message_edit(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _: False,
            channel_id=10,
            message_id="m1",
            content="check this",
            embeds=[e],
        )

        turn = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="m1"
        )
        assert turn is not None
        assert turn.content == "check this"

    def test_no_op_when_no_embeds(self) -> None:
        # no embeds yet → don't bother updating; row stays as inserted
        store = _store_with_turn(content="check this")

        apply_message_edit(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _: True,
            channel_id=10,
            message_id="m1",
            content="check this (edited)",
            embeds=[],
        )

        turn = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="m1"
        )
        assert turn is not None
        # original content preserved — pure-text edits aren't tracked here
        assert turn.content == "check this"

    def test_no_row_for_message_is_silent(self) -> None:
        store = HistoryStore(":memory:")
        e = discord.Embed(description="body")

        # must not raise — bot may have started after the message landed
        apply_message_edit(
            store=store,
            familiar_id="fam",
            is_subscribed=lambda _: True,
            channel_id=10,
            message_id="m-unknown",
            content="hi",
            embeds=[e],
        )


class TestUpdateTurnContentByMessageId:
    def test_updates_existing_row(self) -> None:
        store = _store_with_turn(content="old")
        store.update_turn_content_by_message_id(
            familiar_id="fam",
            platform_message_id="m1",
            content="new",
        )
        turn = store.lookup_turn_by_platform_message_id(
            familiar_id="fam", platform_message_id="m1"
        )
        assert turn is not None
        assert turn.content == "new"

    def test_unknown_message_id_is_no_op(self) -> None:
        store = HistoryStore(":memory:")
        # must not raise
        store.update_turn_content_by_message_id(
            familiar_id="fam",
            platform_message_id="m-missing",
            content="x",
        )
