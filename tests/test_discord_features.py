"""Unit tests for :mod:`familiar_connect.discord_features`.

Pure-Python helpers for Discord mention / reply / channel-link
normalisation. No py-cord types; lookups are passed in as callables so
tests can exercise every branch without a real :class:`discord.Message`.
"""

from __future__ import annotations

from familiar_connect.discord_features import (
    ChannelLinkRef,
    MentionRosterEntry,
    ReplyContext,
    build_mention_roster,
    extract_channel_link_refs,
    format_channel_link,
    format_mention,
    normalise_inbound_content,
)
from familiar_connect.identity import Author


def _lookup(table: dict[int, str]):
    """Return a dict-backed lookup callable that yields ``None`` on miss."""
    return table.get


# ---------------------------------------------------------------------------
# normalise_inbound_content
# ---------------------------------------------------------------------------


class TestNormaliseInboundContent:
    def test_user_mention(self) -> None:
        text = "hey <@123> how are you"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({123: "Alice"}),
            channel_lookup=_lookup({}),
            role_lookup=_lookup({}),
        )
        assert out == "hey @Alice how are you"

    def test_user_mention_with_nickname_marker(self) -> None:
        # older Discord nickname form <@!id>
        text = "hi <@!456>"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({456: "Bob"}),
            channel_lookup=_lookup({}),
            role_lookup=_lookup({}),
        )
        assert out == "hi @Bob"

    def test_channel_mention(self) -> None:
        text = "see <#789>"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({}),
            channel_lookup=_lookup({789: "general"}),
            role_lookup=_lookup({}),
        )
        assert out == "see #general"

    def test_role_mention(self) -> None:
        text = "calling <@&42>"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({}),
            channel_lookup=_lookup({}),
            role_lookup=_lookup({42: "mods"}),
        )
        assert out == "calling @mods"

    def test_unknown_user_mention_falls_back(self) -> None:
        text = "who is <@999>"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({}),
            channel_lookup=_lookup({}),
            role_lookup=_lookup({}),
        )
        # unknown id stays human-readable, not raw markup
        assert "<@999>" not in out
        assert "@unknown-user" in out

    def test_unknown_channel_mention_falls_back(self) -> None:
        text = "where is <#111>"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({}),
            channel_lookup=_lookup({}),
            role_lookup=_lookup({}),
        )
        assert "<#111>" not in out
        assert "#unknown-channel" in out

    def test_mixed(self) -> None:
        text = "hey <@1> look at <#2> with <@&3>"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({1: "Alice"}),
            channel_lookup=_lookup({2: "general"}),
            role_lookup=_lookup({3: "mods"}),
        )
        assert out == "hey @Alice look at #general with @mods"

    def test_no_mentions_returns_unchanged(self) -> None:
        text = "plain message"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({}),
            channel_lookup=_lookup({}),
            role_lookup=_lookup({}),
        )
        assert out == "plain message"

    def test_everyone_and_here_preserved(self) -> None:
        # keep the literal text; outbound AllowedMentions handles safety.
        text = "hi @everyone and @here"
        out = normalise_inbound_content(
            text,
            user_lookup=_lookup({}),
            channel_lookup=_lookup({}),
            role_lookup=_lookup({}),
        )
        assert out == "hi @everyone and @here"


# ---------------------------------------------------------------------------
# extract_channel_link_refs
# ---------------------------------------------------------------------------


class TestExtractChannelLinkRefs:
    def test_full_message_link(self) -> None:
        text = "see https://discord.com/channels/10/20/30 please"
        refs = extract_channel_link_refs(text)
        assert refs == [
            ChannelLinkRef(
                guild_id=10,
                channel_id=20,
                message_id=30,
                raw="https://discord.com/channels/10/20/30",
            )
        ]

    def test_channel_only_link(self) -> None:
        text = "channel https://discord.com/channels/10/20"
        refs = extract_channel_link_refs(text)
        assert refs == [
            ChannelLinkRef(
                guild_id=10,
                channel_id=20,
                message_id=None,
                raw="https://discord.com/channels/10/20",
            )
        ]

    def test_multiple_links(self) -> None:
        text = "https://discord.com/channels/1/2/3 and https://discord.com/channels/1/4"
        refs = extract_channel_link_refs(text)
        assert [r.channel_id for r in refs] == [2, 4]
        assert [r.message_id for r in refs] == [3, None]

    def test_ptb_and_canary_subdomains(self) -> None:
        text = "https://ptb.discord.com/channels/1/2/3 https://canary.discord.com/channels/1/4/5"
        refs = extract_channel_link_refs(text)
        assert [r.channel_id for r in refs] == [2, 4]
        assert [r.message_id for r in refs] == [3, 5]

    def test_non_matching_url_ignored(self) -> None:
        text = "https://example.com/channels/1/2/3"
        assert extract_channel_link_refs(text) == []

    def test_no_links(self) -> None:
        assert extract_channel_link_refs("hi") == []


# ---------------------------------------------------------------------------
# build_mention_roster
# ---------------------------------------------------------------------------


def _author(name: str, user_id: str) -> Author:
    return Author(
        platform="discord",
        user_id=user_id,
        username=name.lower(),
        display_name=name,
    )


class TestBuildMentionRoster:
    def test_deduplicates_by_user_id(self) -> None:
        alice_a = _author("Alice", "1")
        alice_b = _author("Alice", "1")
        bob = _author("Bob", "2")
        roster = build_mention_roster([alice_a, alice_b, bob])
        assert roster == [
            MentionRosterEntry(user_id="1", label="Alice"),
            MentionRosterEntry(user_id="2", label="Bob"),
        ]

    def test_preserves_first_seen_order(self) -> None:
        bob = _author("Bob", "2")
        alice = _author("Alice", "1")
        roster = build_mention_roster([bob, alice])
        assert [e.user_id for e in roster] == ["2", "1"]

    def test_skips_non_discord_platform(self) -> None:
        twitch_user = Author(
            platform="twitch",
            user_id="99",
            username="streamer",
            display_name="Streamer",
        )
        discord_user = _author("Alice", "1")
        roster = build_mention_roster([twitch_user, discord_user])
        assert roster == [MentionRosterEntry(user_id="1", label="Alice")]


# ---------------------------------------------------------------------------
# formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_mention(self) -> None:
        assert format_mention("123") == "<@123>"

    def test_format_channel_link_with_message(self) -> None:
        assert (
            format_channel_link(10, 20, 30) == "https://discord.com/channels/10/20/30"
        )

    def test_format_channel_link_without_message(self) -> None:
        assert format_channel_link(10, 20) == "https://discord.com/channels/10/20"


# ---------------------------------------------------------------------------
# ReplyContext dataclass smoke
# ---------------------------------------------------------------------------


class TestReplyContext:
    def test_fields(self) -> None:
        rc = ReplyContext(author_label="Alice", content_preview="hi")
        assert rc.author_label == "Alice"
        assert rc.content_preview == "hi"
