"""Discord-feature normalisation helpers.

Pure helpers that translate raw Discord markup (``<@id>`` / ``<@!id>`` /
``<@&id>`` / ``<#id>``) into human-readable text, extract channel-link
references from free text, and build mention rosters from Authors.

All functions are synchronous and take callables for id-to-name lookups
so they're unit-testable without a live :class:`discord.Message` — the
bot's ``on_message`` path builds the lookups from py-cord's pre-resolved
caches.

See ``docs/architecture/overview.md`` for the subscribed-vs-accessible
channel scope rule governing link resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from familiar_connect.identity import Author


# <@id> or <@!id> (nickname form); id is digits
_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
# <@&id>
_ROLE_MENTION_RE = re.compile(r"<@&(\d+)>")
# <#id>
_CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")

# Discord message / channel link. Accepts the canary and ptb subdomains.
_CHANNEL_LINK_RE = re.compile(
    r"https?://(?:ptb\.|canary\.)?discord(?:app)?\.com"
    r"/channels/(\d+)/(\d+)(?:/(\d+))?",
)

_UNKNOWN_USER = "@unknown-user"
_UNKNOWN_CHANNEL = "#unknown-channel"
_UNKNOWN_ROLE = "@unknown-role"


@dataclass(frozen=True)
class ChannelLinkRef:
    """Parsed ``discord.com/channels/...`` link."""

    guild_id: int
    channel_id: int
    message_id: int | None
    raw: str


@dataclass(frozen=True)
class ReplyContext:
    """Minimal snapshot of a message being replied to."""

    author_label: str
    content_preview: str


@dataclass(frozen=True)
class MentionRosterEntry:
    """Single ``(user_id, label)`` pair the LLM may ping."""

    user_id: str
    label: str


def normalise_inbound_content(
    raw: str,
    *,
    user_lookup: Callable[[int], str | None],
    channel_lookup: Callable[[int], str | None],
    role_lookup: Callable[[int], str | None],
) -> str:
    """Replace Discord id-markup with human-readable forms.

    ``<@id>`` / ``<@!id>`` → ``@Label``; ``<@&id>`` → ``@RoleName``;
    ``<#id>`` → ``#channel-name``. Unknown ids fall through to stable
    placeholders — never raises.
    """

    def _user(match: re.Match[str]) -> str:
        label = user_lookup(int(match.group(1)))
        return f"@{label}" if label else _UNKNOWN_USER

    def _role(match: re.Match[str]) -> str:
        label = role_lookup(int(match.group(1)))
        return f"@{label}" if label else _UNKNOWN_ROLE

    def _channel(match: re.Match[str]) -> str:
        name = channel_lookup(int(match.group(1)))
        return f"#{name}" if name else _UNKNOWN_CHANNEL

    out = _ROLE_MENTION_RE.sub(_role, raw)
    out = _USER_MENTION_RE.sub(_user, out)
    return _CHANNEL_MENTION_RE.sub(_channel, out)


def extract_channel_link_refs(text: str) -> list[ChannelLinkRef]:
    """Return every Discord channel/message link in *text*.

    Leaves order preserved; duplicates are returned as-is (caller
    dedupes if needed).
    """
    refs: list[ChannelLinkRef] = []
    for match in _CHANNEL_LINK_RE.finditer(text):
        guild_id = int(match.group(1))
        channel_id = int(match.group(2))
        message_group = match.group(3)
        message_id = int(message_group) if message_group is not None else None
        refs.append(
            ChannelLinkRef(
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                raw=match.group(0),
            )
        )
    return refs


def build_mention_roster(participants: Iterable[Author]) -> list[MentionRosterEntry]:
    """Dedupe Discord participants into a ``<@id>`` roster.

    Preserves first-seen order; non-Discord authors are skipped (their
    ``user_id`` isn't a valid Discord snowflake).
    """
    seen: set[str] = set()
    roster: list[MentionRosterEntry] = []
    for author in participants:
        if author.platform != "discord":
            continue
        if author.user_id in seen:
            continue
        seen.add(author.user_id)
        roster.append(MentionRosterEntry(user_id=author.user_id, label=author.label))
    return roster


def format_mention(user_id: str) -> str:
    """Render a Discord user mention token."""
    return f"<@{user_id}>"


def format_channel_link(
    guild_id: int,
    channel_id: int,
    message_id: int | None = None,
) -> str:
    """Render a Discord channel or message link URL."""
    base = f"https://discord.com/channels/{guild_id}/{channel_id}"
    if message_id is not None:
        return f"{base}/{message_id}"
    return base
