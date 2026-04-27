"""Structured author identity for platform users.

Replaces the bare ``speaker: str`` that used to thread through history,
context, memory, and providers. An :class:`Author` carries both an
immutable platform-level key (``platform`` + ``user_id``) and the
human-readable variants (``username``, ``display_name``, ``aliases``)
so recall can resolve by any known name while storage pins to a stable
slug.

See ``docs/architecture/memory.md`` for how files under
``people/<slug>.md`` and the alias index use this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from familiar_connect.llm import sanitize_name


class _DiscordMemberLike(Protocol):
    """Minimal Discord Member/User surface for :meth:`Author.from_discord_member`.

    Real ``discord.Member`` carries four name fields: ``id`` (immutable
    snowflake), ``name`` (global username), ``global_name`` (global
    display name), and ``nick`` (per-guild override; ``None`` on
    DMs / Users without a guild-scoped nick). ``display_name`` is
    py-cord's resolved view: ``nick → global_name → name``. We read
    all four; the optional pair via ``getattr`` to tolerate older
    shapes and ``SimpleNamespace`` test doubles.
    """

    id: int
    name: str
    display_name: str


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Author:
    """Platform-scoped identity for one speaker.

    ``canonical_key`` is the storage-stable identifier; ``label`` is
    the human-readable string rendered in prompts. Multiple display
    names / nicknames map to one ``Author`` via ``aliases`` and the
    on-disk alias index.
    """

    platform: str
    user_id: str
    username: str | None
    display_name: str | None
    global_name: str | None = None
    guild_nick: str | None = None
    aliases: frozenset[str] = field(default_factory=frozenset)

    # ------------------------------------------------------------------
    # Derived keys
    # ------------------------------------------------------------------

    @property
    def canonical_key(self) -> str:
        """Stable identifier across renames: ``<platform>:<user_id>``."""
        return f"{self.platform}:{self.user_id}"

    @property
    def slug(self) -> str:
        """Filesystem-safe form of :attr:`canonical_key`.

        Lowercase, non-alphanumerics collapsed to single dashes,
        trimmed. Used as the basename for ``people/<slug>.md``.
        """
        return _SLUG_NON_ALNUM.sub("-", self.canonical_key.lower()).strip("-")

    # ------------------------------------------------------------------
    # Human-readable
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        """Preferred display string: display_name → username → user_id."""
        return self.display_name or self.username or self.user_id

    @property
    def openai_name(self) -> str | None:
        """Value for OpenAI-style ``Message.name``. Scrubbed per API rules.

        Falls back to a sanitized ``user_id`` if the label sanitizes to
        nothing (e.g. display name is all punctuation). ``None`` only
        if even the id scrubs empty — not expected in practice.
        """
        return sanitize_name(self.label) or sanitize_name(self.user_id)

    @property
    def all_known_names(self) -> set[str]:
        """Every name this author is known by. Used when rebuilding the alias index."""
        names: set[str] = set(self.aliases)
        if self.display_name:
            names.add(self.display_name)
        if self.username:
            names.add(self.username)
        return names

    # ------------------------------------------------------------------
    # Platform factories
    # ------------------------------------------------------------------

    @classmethod
    def from_discord_member(cls, member: _DiscordMemberLike) -> Author:
        """Build from a Discord ``Member`` / ``User``.

        Reads four name fields. ``id`` is immutable; ``name``,
        ``global_name``, ``nick`` may all change. ``display_name`` is
        py-cord's resolved view (``nick → global_name → name``).
        ``global_name`` and ``nick`` are read via ``getattr`` so
        ``User`` (DM, no guild → no ``.nick``) and older py-cord
        shapes still build cleanly.
        """
        return cls(
            platform="discord",
            user_id=str(member.id),
            username=member.name,
            display_name=member.display_name,
            global_name=getattr(member, "global_name", None),
            guild_nick=getattr(member, "nick", None),
        )

    @classmethod
    def from_twitch(
        cls,
        *,
        user_id: str,
        user_login: str | None,
        user_name: str | None,
    ) -> Author:
        """Build from Twitch Helix fields.

        ``user_login`` is the lowercase immutable login; ``user_name``
        is the mutable display case (often just cased differently).
        """
        return cls(
            platform="twitch",
            user_id=str(user_id),
            username=user_login,
            display_name=user_name,
        )


def format_turn_for_transcript(role: str, author: Author | None, content: str) -> str:
    """Render one turn as ``role (label): content`` / ``role: content``.

    Shared between the history-summary provider and memory writer so
    their transcript format stays in sync. User turns include the
    author label; assistant/system turns use role only.
    """
    if role == "user" and author is not None:
        return f"{role} ({author.label}): {content}"
    return f"{role}: {content}"
