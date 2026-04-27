"""Red-first tests for the Author identity dataclass.

Exercises construction, derived keys/labels, and platform factories
from ``familiar_connect.identity``, which does not yet exist.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from familiar_connect.identity import Author


class TestAuthorConstruction:
    def test_required_fields(self) -> None:
        a = Author(
            platform="discord",
            user_id="123",
            username="ada",
            display_name="Ada",
        )
        assert a.platform == "discord"
        assert a.user_id == "123"
        assert a.username == "ada"
        assert a.display_name == "Ada"
        assert a.aliases == frozenset()

    def test_aliases_default_empty(self) -> None:
        a = Author(platform="discord", user_id="1", username=None, display_name=None)
        assert a.aliases == frozenset()

    def test_aliases_accept_iterable(self) -> None:
        a = Author(
            platform="discord",
            user_id="1",
            username="ada",
            display_name="Ada",
            aliases=frozenset({"Addy", "A"}),
        )
        assert "Addy" in a.aliases
        assert "A" in a.aliases

    def test_frozen(self) -> None:
        """Author is immutable — identity should not mutate under our feet."""
        a = Author(platform="discord", user_id="1", username=None, display_name="Ada")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            a.display_name = "Eve"  # ty: ignore[invalid-assignment]


class TestCanonicalKeyAndSlug:
    def test_canonical_key_is_platform_colon_id(self) -> None:
        a = Author(platform="discord", user_id="42", username=None, display_name=None)
        assert a.canonical_key == "discord:42"

    def test_slug_replaces_colon_and_lowercases(self) -> None:
        a = Author(platform="Discord", user_id="42", username=None, display_name=None)
        assert a.slug == "discord-42"

    def test_slug_strips_surrounding_dashes(self) -> None:
        a = Author(platform=":x:", user_id=":7:", username=None, display_name=None)
        assert a.slug == "x-7"

    def test_slug_collapses_runs_of_non_alphanumerics(self) -> None:
        a = Author(
            platform="twitch",
            user_id="U__99!!",
            username=None,
            display_name=None,
        )
        assert a.slug == "twitch-u-99"


class TestLabel:
    def test_label_prefers_display_name(self) -> None:
        a = Author(
            platform="discord",
            user_id="1",
            username="ada_l",
            display_name="Ada Lovelace",
        )
        assert a.label == "Ada Lovelace"

    def test_label_falls_back_to_username(self) -> None:
        a = Author(
            platform="discord",
            user_id="1",
            username="ada_l",
            display_name=None,
        )
        assert a.label == "ada_l"

    def test_label_falls_back_to_user_id(self) -> None:
        a = Author(
            platform="discord",
            user_id="1",
            username=None,
            display_name=None,
        )
        assert a.label == "1"


class TestOpenAIName:
    def test_sanitizes_display_name(self) -> None:
        a = Author(
            platform="discord",
            user_id="1",
            username=None,
            display_name="Ada Lovelace!",
        )
        # spaces and punctuation replaced with underscores by sanitize_name
        assert a.openai_name == "Ada_Lovelace"

    def test_falls_back_to_user_id_when_label_sanitizes_to_empty(self) -> None:
        a = Author(
            platform="discord",
            user_id="42",
            username=None,
            display_name="！！！",  # noqa: RUF001
        )
        assert a.openai_name == "42"


class TestAllKnownNames:
    def test_includes_display_username_and_aliases(self) -> None:
        a = Author(
            platform="discord",
            user_id="1",
            username="ada_l",
            display_name="Ada",
            aliases=frozenset({"Addy"}),
        )
        assert a.all_known_names == {"Ada", "ada_l", "Addy"}

    def test_drops_none_values(self) -> None:
        a = Author(
            platform="discord",
            user_id="1",
            username=None,
            display_name="Ada",
        )
        assert a.all_known_names == {"Ada"}


class TestFromDiscordMember:
    def test_extracts_id_name_display_name(self) -> None:
        member = SimpleNamespace(id=987, name="ada_l", display_name="Ada")
        a = Author.from_discord_member(member)
        assert a.platform == "discord"
        assert a.user_id == "987"
        assert a.username == "ada_l"
        assert a.display_name == "Ada"

    def test_extracts_global_name_and_guild_nick(self) -> None:
        """Discord exposes 4 names: id, username (.name), global_name, nick.

        ``global_name`` is the global "real name" (introduced 2023) —
        distinct from username and from per-guild nick.
        ``nick`` is the per-guild override; ``None`` for users without
        a guild-scoped nickname or for DMs (where Member isn't bound).
        """
        member = SimpleNamespace(
            id=987,
            name="ada_l",
            display_name="Aria",  # py-cord's resolved view (nick wins)
            global_name="Ada Lovelace",
            nick="Aria",
        )
        a = Author.from_discord_member(member)
        assert a.global_name == "Ada Lovelace"
        assert a.guild_nick == "Aria"
        assert a.display_name == "Aria"

    def test_handles_member_without_global_name_or_nick(self) -> None:
        """Older py-cord shapes / DM Users may lack the new fields."""
        member = SimpleNamespace(id=987, name="ada_l", display_name="ada_l")
        a = Author.from_discord_member(member)
        assert a.global_name is None
        assert a.guild_nick is None


class TestFromTwitch:
    def test_builds_from_api_fields(self) -> None:
        a = Author.from_twitch(user_id="U99", user_login="adadev", user_name="AdaDev")
        assert a.platform == "twitch"
        assert a.user_id == "U99"
        assert a.username == "adadev"
        assert a.display_name == "AdaDev"
