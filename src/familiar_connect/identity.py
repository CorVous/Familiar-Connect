"""Structured author identity for platform users.

Replaces bare ``speaker: str`` previously threading history, context,
memory, providers. :class:`Author` carries immutable platform key
(``platform`` + ``user_id``) plus human-readable variants
(``username``, ``display_name``, ``aliases``) — recall resolves by any
known name; storage pins to stable slug.

See ``docs/architecture/memory.md`` for ``people/<slug>.md`` + alias
index usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from familiar_connect.llm import sanitize_name


class _DiscordMemberLike(Protocol):
    """Minimal Discord Member/User surface for :meth:`Author.from_discord_member`.

    ``discord.Member`` carries four name fields: ``id`` (immutable
    snowflake), ``name`` (global username), ``global_name`` (global
    display name), ``nick`` (per-guild override; ``None`` on DMs).
    ``display_name`` is py-cord's resolved view (``nick → global_name → name``).
    Optional pair read via ``getattr`` for older shapes / ``SimpleNamespace``
    test doubles.
    """

    id: int
    name: str
    display_name: str


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Platform key namespace for an :class:`Author`: the real chat platforms
# (``discord`` / ``twitch``) plus the reserved ``ego`` platform for the
# familiar's own narrative subject (see :data:`EGO_PLATFORM`).
Platform = Literal["discord", "twitch", "ego"]

# reserved platform for the familiar's OWN narrative subject. ``ego:``
# can never collide with ``discord:`` / ``twitch:`` keys — those use a
# real platform name. one canonical_key per familiar: ``ego:<id>``.
# Named ``ego`` rather than ``self`` so it never shadows Python's ``self``
# parameter in the surrounding code (see issue #154).
EGO_PLATFORM: Platform = "ego"


def ego_canonical_key(familiar_id: str) -> str:
    """Reserved subject key for *familiar_id*'s own narrative: ``ego:<id>``."""
    return f"{EGO_PLATFORM}:{familiar_id}"


def is_ego_key(canonical_key: str) -> bool:
    """Test ``ego:<id>`` membership: ``ego`` platform + non-empty id."""
    platform, sep, rest = canonical_key.partition(":")
    return platform == EGO_PLATFORM and bool(sep) and bool(rest)


@dataclass(frozen=True)
class Author:
    """Platform-scoped identity for one speaker.

    canonical_key: storage-stable id.
    label: human-readable string for prompts.
    aliases + on-disk index: map many display names / nicknames → one Author.
    """

    platform: Platform
    user_id: str
    username: str | None
    display_name: str | None
    global_name: str | None = None
    guild_nick: str | None = None
    pronouns: str | None = None
    bio: str | None = None
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

        Lowercase; non-alphanumerics collapsed to single dashes; trimmed.
        Basename for ``people/<slug>.md``.
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
        """OpenAI-style ``Message.name``; scrubbed per API rules.

        Falls back to sanitized ``user_id`` when label scrubs empty
        (e.g. display name all punctuation). ``None`` only if id also
        scrubs empty — not expected.
        """
        return sanitize_name(self.label) or sanitize_name(self.user_id)

    @property
    def all_known_names(self) -> set[str]:
        """Every known name; used to rebuild alias index."""
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
        """Build from Discord ``Member`` / ``User``.

        ``id`` immutable; ``name`` / ``global_name`` / ``nick`` mutable.
        ``display_name`` is py-cord's resolved view (``nick → global_name
        → name``). ``global_name`` / ``nick`` via ``getattr`` so DM
        ``User`` (no guild → no ``.nick``) and older py-cord still build.
        ``pronouns`` / ``bio`` are profile-only; bot tokens rarely
        populate them, so getattr defaults to ``None``.
        """
        return cls(
            platform="discord",
            user_id=str(member.id),
            username=member.name,
            display_name=member.display_name,
            global_name=getattr(member, "global_name", None),
            guild_nick=getattr(member, "nick", None),
            pronouns=getattr(member, "pronouns", None),
            bio=getattr(member, "bio", None),
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

        user_login: lowercase immutable login.
        user_name: mutable display case (often just different casing).
        """
        return cls(
            platform="twitch",
            user_id=str(user_id),
            username=user_login,
            display_name=user_name,
        )


def format_turn_for_transcript(role: str, author: Author | None, content: str) -> str:
    """Render one turn as ``role (label): content`` / ``role: content``.

    Shared between history-summary provider and memory writer; keeps
    transcript format in sync. User turns include author label;
    assistant/system turns use role only.
    """
    if role == "user" and author is not None:
        return f"{role} ({author.label}): {content}"
    return f"{role}: {content}"
