"""Prompt layer implementations.

Each layer owns one segment of the system prompt with its own
invalidation signal. See plan § Design.4 *Prompt composition*.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol

from familiar_connect.llm import Message, sanitize_name

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.context.assembler import AssemblyContext
    from familiar_connect.history.store import HistoryStore
    from familiar_connect.identity import Author


class Layer(Protocol):
    """Protocol for a single prompt layer.

    ``build`` returns the layer's text contribution to the system
    prompt (empty string opts out). ``invalidation_key`` is a short
    string used for in-process caching.
    """

    name: str

    async def build(self, ctx: AssemblyContext) -> str: ...

    def invalidation_key(self, ctx: AssemblyContext) -> str: ...


# ---------------------------------------------------------------------------
# Static / file-sourced layers
# ---------------------------------------------------------------------------


def _content_hash(path: Path) -> str:
    """Short content hash, used as an invalidation key for file layers.

    Content hash (not mtime) so sub-second edits are caught —
    filesystem mtime resolution is sometimes coarser than test timing.
    """
    if not path.exists():
        return "missing"
    return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()


class CoreInstructionsLayer:
    """Baseline role + safety instructions from a checked-in markdown file."""

    name: str = "core_instructions"

    def __init__(self, *, path: Path) -> None:
        self._path = path

    async def build(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8").strip()

    def invalidation_key(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        return _content_hash(self._path)


class CharacterCardLayer:
    """Per-familiar persona text from a ``character.md`` sidecar."""

    name: str = "character_card"

    def __init__(self, *, card_path: Path) -> None:
        self._path = card_path

    async def build(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8").strip()

    def invalidation_key(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        return _content_hash(self._path)


class OperatingModeLayer:
    """Per-viewer-mode directive block.

    Voice channels want terse replies; text channels allow markdown.
    The ``modes`` mapping is keyed by :attr:`AssemblyContext.viewer_mode`.
    Unknown modes yield ``""`` (layer opts out).
    """

    name: str = "operating_mode"

    def __init__(self, *, modes: dict[str, str]) -> None:
        self._modes = dict(modes)

    async def build(self, ctx: AssemblyContext) -> str:
        return self._modes.get(ctx.viewer_mode, "")

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        return ctx.viewer_mode


# ---------------------------------------------------------------------------
# Dynamic layer: recent history
# ---------------------------------------------------------------------------


class RecentHistoryLayer:
    """Verbatim tail of ``turns`` for the active channel.

    Unlike the other layers, this contributes to the ``recent_history``
    message list rather than the system prompt. :meth:`build` returns
    ``""`` so the assembler's system-prompt composition skips it.
    """

    name: str = "recent_history"

    def __init__(self, *, store: HistoryStore, window_size: int = 20) -> None:
        self._store = store
        self._window_size = window_size

    async def build(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        return ""

    def invalidation_key(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        # Dynamic — always rebuild. Real caching would key on
        # ``(channel_id, latest_turn_id)`` but we'd still need the
        # turns themselves, so there's nothing to reuse.
        return "always-rebuild"

    async def recent_messages(self, ctx: AssemblyContext) -> list[Message]:
        """Return the last ``window_size`` turns as LLM messages.

        User turns gain a ``name`` (platform:user_id) and a
        ``[display_name]`` content prefix — critical for multi-user
        channels where the model has to distinguish speakers.
        """
        turns = self._store.recent(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id or 0,
            limit=self._window_size,
        )
        return [
            _turn_to_message(turn.role, turn.content, turn.author) for turn in turns
        ]


def _turn_to_message(role: str, content: str, author: Author | None) -> Message:
    """Render a :class:`HistoryTurn`-like tuple into an LLM :class:`Message`."""
    if role == "assistant" or author is None:
        return Message(role=role, content=content)
    name = sanitize_name(author.canonical_key)
    display = author.display_name or author.username or author.user_id
    # Belt-and-braces prefix so models that drop `name` still see attribution.
    prefixed = f"[{display}] {content}"
    return Message(role=role, content=prefixed, name=name)
