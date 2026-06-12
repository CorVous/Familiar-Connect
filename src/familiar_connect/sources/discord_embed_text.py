"""Render Discord embeds as plain text.

Discord delivers URL unfurls as ``Embed`` objects on a message. Bot
only sees ``message.content`` by default — link previews vanish.
Flattens ``Iterable[Embed]`` into text block caller appends to
``content`` before publishing onto bus, so LLM sees same body humans
see in client.

Duck-typed: any object exposing relevant attributes works
(``discord.Embed``, ``SimpleNamespace`` in tests, etc.). No hard
dependency on ``discord.Embed`` — keeps formatter unit-testable
without a Discord stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


_EMBED_TAG = "[embed]"


def format_embeds(embeds: Iterable[Any]) -> str:
    """Flatten ``embeds`` into plain-text block.

    Empty input or all-blank embeds → ``""``. Multiple embeds
    separated by blank line so LLM can tell them apart.
    """
    blocks = [_format_one(e) for e in embeds]
    return "\n\n".join(b for b in blocks if b)


def _format_one(embed: Any) -> str:  # noqa: ANN401 — duck-typed
    """Render single embed; ``""`` when nothing meaningful."""
    provider_name = _attr_chain(embed, "provider", "name")
    author_name = _attr_chain(embed, "author", "name")
    title = _str(getattr(embed, "title", None))
    description = _str(getattr(embed, "description", None))
    footer_text = _attr_chain(embed, "footer", "text")
    url = _str(getattr(embed, "url", None))

    header_bits: list[str] = []
    if provider_name:
        header_bits.append(f"({provider_name})")
    if author_name:
        header_bits.append(author_name)
    # Avoid echoing same string twice when embed's title mirrors
    # author handle (common on Tumblr / Bluesky cards).
    if title and title != author_name:
        header_bits.append(title)
    header = " — ".join(header_bits)

    lines: list[str] = []
    if header:
        lines.append(header)
    if description:
        lines.append(description)

    for field in getattr(embed, "fields", None) or ():
        name = _str(getattr(field, "name", None))
        value = _str(getattr(field, "value", None))
        if name and value:
            lines.append(f"{name}: {value}")
        elif name or value:
            lines.append(name or value)

    if footer_text:
        lines.append(f"— {footer_text}")

    if not lines:
        # Image-only embed: surface link target so LLM at least knows
        # a media URL was attached. drop entirely when even url
        # missing — nothing to say.
        if url:
            return f"{_EMBED_TAG}\n[link: {url}]"
        return ""

    return f"{_EMBED_TAG}\n" + "\n".join(lines)


def _attr_chain(obj: Any, *path: str) -> str:  # noqa: ANN401 — duck-typed
    """Walk ``getattr`` chain; ``""`` on any miss."""
    cur: Any = obj
    for name in path:
        cur = getattr(cur, name, None)
        if cur is None:
            return ""
    return _str(cur)


def _str(value: Any) -> str:  # noqa: ANN401 — duck-typed
    """Coerce to non-empty stripped str, else ``""``."""
    if value is None:
        return ""
    return str(value).strip()
