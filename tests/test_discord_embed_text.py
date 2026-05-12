"""Tests for :mod:`familiar_connect.sources.discord_embed_text`.

The formatter renders ``discord.Embed`` (and any duck-typed equivalent)
into plain text the bot appends to a message's ``content``, so the LLM
sees URL unfurls the same way humans do.
"""

from __future__ import annotations

from types import SimpleNamespace

import discord

from familiar_connect.sources.discord_embed_text import format_embeds


class TestFormatEmbeds:
    def test_empty_iterable_returns_empty_string(self) -> None:
        assert not format_embeds([])

    def test_skips_blank_embeds(self) -> None:
        # an embed with no text fields contributes nothing.
        e = discord.Embed()
        assert not format_embeds([e])

    def test_description_only_embed(self) -> None:
        e = discord.Embed(description="hello world")
        out = format_embeds([e])
        assert out == "[embed]\nhello world"

    def test_renders_provider_author_title_description(self) -> None:
        # ``provider`` is read-only on Embed and only set when an embed
        # round-trips through ``from_dict`` — the same path the gateway
        # uses to deliver unfurls.
        e = discord.Embed.from_dict({
            "title": "The beach that makes you old",
            "description": "body",
            "type": "rich",
            "author": {"name": "dotshaft"},
            "provider": {"name": "Tumblr"},
        })
        out = format_embeds([e])
        assert out == (
            "[embed]\n(Tumblr) — dotshaft — The beach that makes you old\nbody"
        )

    def test_dedupes_title_when_equal_to_author(self) -> None:
        e = discord.Embed(title="dotshaft", description="body")
        e.set_author(name="dotshaft")
        out = format_embeds([e])
        # title repeats author -> drop title from header
        assert out == "[embed]\ndotshaft\nbody"

    def test_renders_fields(self) -> None:
        e = discord.Embed(title="match")
        e.add_field(name="Score", value="3-1")
        e.add_field(name="Stadium", value="Anfield")
        out = format_embeds([e])
        assert out == "[embed]\nmatch\nScore: 3-1\nStadium: Anfield"

    def test_renders_footer(self) -> None:
        e = discord.Embed(description="body")
        e.set_footer(text="via example.com")
        out = format_embeds([e])
        assert out == "[embed]\nbody\n— via example.com"

    def test_image_only_embed_falls_back_to_url(self) -> None:
        e = discord.Embed(url="https://example.com/x")
        e.set_image(url="https://example.com/x.png")
        out = format_embeds([e])
        assert out == "[embed]\n[link: https://example.com/x]"

    def test_image_only_embed_without_url_returns_empty(self) -> None:
        e = discord.Embed()
        e.set_image(url="https://example.com/x.png")
        assert not format_embeds([e])

    def test_multiple_embeds_separated_by_blank_line(self) -> None:
        a = discord.Embed(description="first")
        b = discord.Embed(description="second")
        out = format_embeds([a, b])
        assert out == "[embed]\nfirst\n\n[embed]\nsecond"

    def test_tumblr_reblog_chain_renders_verbatim(self) -> None:
        # Discord's Tumblr unfurl puts the entire reblog chain in the
        # description verbatim; the formatter must not reflow it.
        chain = (
            "🔁 dotshaft\n\n"
            "shittymoviedetails\n\n"
            "The beach that makes you old\n\n"
            "powerjock\n\n"
            "I can never seem to find her, but she always finds me"
        )
        e = discord.Embed.from_dict({
            "description": chain,
            "type": "rich",
            "author": {"name": "fratal"},
            "provider": {"name": "Tumblr"},
        })
        out = format_embeds([e])
        assert out == f"[embed]\n(Tumblr) — fratal\n{chain}"

    def test_duck_typed_input(self) -> None:
        # any object with the relevant attributes works — tests don't
        # need a live ``discord.Embed`` to exercise the formatter.
        e = SimpleNamespace(
            title="t",
            description="d",
            url=None,
            author=SimpleNamespace(name="a"),
            provider=SimpleNamespace(name="p"),
            fields=(SimpleNamespace(name="k", value="v"),),
            footer=SimpleNamespace(text="f"),
        )
        out = format_embeds([e])
        assert out == "[embed]\n(p) — a — t\nd\nk: v\n— f"
