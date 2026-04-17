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
    """Minimal Discord Member/User surface for :meth:`Author.from_discord_member`."""

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

        Uses ``.id``, ``.name``, ``.display_name`` — the first is
        immutable, the other two may change over time.
        """
        return cls(
            platform="discord",
            user_id=str(member.id),
            username=member.name,
            display_name=member.display_name,
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
