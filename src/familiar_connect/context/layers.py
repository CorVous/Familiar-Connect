"""Prompt layer implementations.

Each layer owns one segment of the system prompt with its own
invalidation signal. See plan § Design.4 *Prompt composition*.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
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


# ---------------------------------------------------------------------------
# Phase-3 layers: summaries + RAG
# ---------------------------------------------------------------------------


class ConversationSummaryLayer:
    """Read-only layer over :class:`HistoryStore` summaries.

    Produced by :class:`SummaryWorker`. Invalidation key keys on
    ``last_summarised_id`` so the assembler cache rebuilds only when
    the summary actually changes.
    """

    name: str = "conversation_summary"

    def __init__(self, *, store: HistoryStore) -> None:
        self._store = store

    async def build(self, ctx: AssemblyContext) -> str:
        entry = self._store.get_summary(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id or 0,
        )
        if entry is None or not entry.summary_text.strip():
            return ""
        return "## Conversation so far\n\n" + entry.summary_text.strip()

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        entry = self._store.get_summary(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id or 0,
        )
        if entry is None:
            return "none"
        return f"ch{ctx.channel_id}:wm{entry.last_summarised_id}"


class CrossChannelContextLayer:
    """Other-channel summary block rendered into the viewer's prompt.

    ``viewer_map`` maps viewer channel id → list of source channel
    ids whose cross-context summary should appear. Summaries older
    than ``ttl_seconds`` are suppressed (the layer opts out) — they
    are still present in SQLite; the next :class:`SummaryWorker` tick
    will replace them.
    """

    name: str = "cross_channel_context"

    def __init__(
        self,
        *,
        store: HistoryStore,
        viewer_map: dict[int, list[int]],
        ttl_seconds: int = 600,
    ) -> None:
        self._store = store
        self._viewer_map = {k: list(v) for k, v in viewer_map.items()}
        self._ttl_seconds = ttl_seconds

    def _viewer_key(self, ctx: AssemblyContext) -> str:
        return f"{ctx.viewer_mode}:{ctx.channel_id}"

    async def build(self, ctx: AssemblyContext) -> str:
        sources = self._viewer_map.get(ctx.channel_id or -1, [])
        if not sources:
            return ""

        now = datetime.now(tz=UTC)
        sections: list[str] = []
        for source_id in sources:
            entry = self._store.get_cross_context(
                familiar_id=ctx.familiar_id,
                viewer_mode=self._viewer_key(ctx),
                source_channel_id=source_id,
            )
            if entry is None or not entry.summary_text.strip():
                continue
            age = (now - entry.created_at).total_seconds()
            if age > self._ttl_seconds:
                continue
            sections.append(
                f"### From channel #{source_id}\n\n" + entry.summary_text.strip()
            )
        if not sections:
            return ""
        return "## Cross-channel context\n\n" + "\n\n".join(sections)

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        sources = self._viewer_map.get(ctx.channel_id or -1, [])
        if not sources:
            return "none"
        parts: list[str] = []
        for source_id in sources:
            entry = self._store.get_cross_context(
                familiar_id=ctx.familiar_id,
                viewer_mode=self._viewer_key(ctx),
                source_channel_id=source_id,
            )
            if entry is None:
                parts.append(f"{source_id}:none")
            else:
                parts.append(f"{source_id}:wm{entry.source_last_id}")
        return "|".join(parts)


class RagContextLayer:
    """FTS-backed retrieval of relevant historical turns.

    :meth:`set_current_cue` is called by the responder with the query
    derived from the inbound user turn. When unset (or empty), the
    layer opts out.

    Invalidation: ``(cue, latest_fts_id)``. The latest id moves when
    new turns are written, so the cache flips automatically after a
    fresh user message without needing the layer to re-run against
    the same cue twice unnecessarily.
    """

    name: str = "rag_context"

    def __init__(self, *, store: HistoryStore, max_results: int = 5) -> None:
        self._store = store
        self._max_results = max_results
        self._current_cue: str = ""

    def set_current_cue(self, cue: str) -> None:
        self._current_cue = (cue or "").strip()

    async def build(self, ctx: AssemblyContext) -> str:
        cue = self._current_cue
        if not cue:
            return ""
        results = self._store.search_turns(
            familiar_id=ctx.familiar_id,
            query=cue,
            limit=self._max_results,
        )
        if not results:
            return ""
        lines: list[str] = ["## Possibly relevant earlier turns\n"]
        for t in results:
            who = (
                t.author.display_name
                if t.author is not None and t.author.display_name
                else t.role
            )
            lines.append(f"- [{who}] {t.content}")
        return "\n".join(lines)

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        cue = self._current_cue
        latest = self._store.latest_fts_id(familiar_id=ctx.familiar_id)
        return f"{cue}|{latest}"
