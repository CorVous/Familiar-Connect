"""Prompt layer implementations.

Each layer owns one segment of the system prompt with its own
invalidation signal. See plan § Design.4 *Prompt composition*.
"""

from __future__ import annotations

import hashlib
import operator
import re
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from familiar_connect.budget import (
    estimate_message_tokens,
    estimate_tokens,
)
from familiar_connect.llm import Message, sanitize_name

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.context.assembler import AssemblyContext
    from familiar_connect.history.store import (
        AccountProfile,
        Fact,
        HistoryStore,
        HistoryTurn,
    )
    from familiar_connect.identity import Author


# Discord mention syntax: <@USER_ID>, optionally with ! for nick form.
_DISCORD_MENTION_RE = re.compile(r"<@!?(\d+)>")

# Soft-cap on a parent reply's full content when inlined into a child's prefix.
_REPLY_PARENT_FULL_CAP = 400
# Snippet cap when the parent is already in the recent window.
_REPLY_PARENT_SNIPPET_CAP = 80


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

    def __init__(
        self,
        *,
        store: HistoryStore,
        window_size: int = 20,
        max_tokens: int | None = None,
    ) -> None:
        self._store = store
        self._window_size = window_size
        self._max_tokens = max_tokens

    async def build(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        return ""

    def invalidation_key(self, ctx: AssemblyContext) -> str:  # noqa: ARG002
        # dynamic — always rebuild. real caching would key on
        # ``(channel_id, latest_turn_id)`` but turns are still needed,
        # so nothing to reuse.
        return "always-rebuild"

    async def recent_messages(self, ctx: AssemblyContext) -> list[Message]:
        """Last ``window_size`` turns as LLM messages.

        User turns get a ``name`` (platform:user_id) + ``[HH:MM
        display_name]`` content prefix — needed for multi-user channels
        to distinguish speakers and gauge rhythm.

        Reply marker. Turns whose ``reply_to_message_id`` resolves to a
        known parent get a ``↩`` prefix carrying parent's author + text.
        Depth adapts: in-window parents contribute a short snippet (full
        text rendering imminent); out-of-window parents contribute full
        content (capped) so the reply stays intelligible.

        Mention rewriting. Discord ``<@USER_ID>`` / ``<@!USER_ID>``
        become ``[@DisplayName]`` via :meth:`HistoryStore.resolve_label`
        — symmetric with the LLM's expected output form.

        Reactions. Each turn's ``platform_message_id`` is looked up in
        ``message_reactions`` (single batch query for the whole window)
        and rendered as a trailing ``[reactions: 👍 x3 ❤️ x1]`` suffix.
        """
        turns = self._store.recent(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id or 0,
            limit=self._window_size,
        )
        in_window_msg_ids = {
            t.platform_message_id for t in turns if t.platform_message_id
        }
        reactions = self._store.reactions_for_messages(
            familiar_id=ctx.familiar_id,
            platform_message_ids=[
                t.platform_message_id for t in turns if t.platform_message_id
            ],
        )
        rendered = [
            _turn_to_message_with_context(
                turn=turn,
                store=self._store,
                familiar_id=ctx.familiar_id,
                guild_id=ctx.guild_id,
                in_window_msg_ids=in_window_msg_ids,
                reactions=reactions.get(turn.platform_message_id or "", ()),
            )
            for turn in turns
        ]
        if self._max_tokens is None:
            return rendered
        return _trim_messages_to_token_cap(rendered, self._max_tokens)


def _render_fact_line(
    store: HistoryStore,
    familiar_id: str,
    fact: Fact,
    *,
    guild_id: int | None = None,
) -> str:
    """Render fact text with optional rename annotations.

    Original text preserved verbatim — that's what was observed. For
    each subject whose current display name differs from the baked-in
    one, append a soft hint: ``(Cass is now known as peeks)``.
    Unchanged names / unresolvable canonical keys add nothing.

    Subject → canonical_key is the extractor's best guess, not
    authoritative — mic-sharing and relays break 1:1 mapping. Render is
    intentionally additive (not substituted) to preserve the original.
    Resolution via :meth:`HistoryStore.resolve_label` so active-guild
    nicknames beat the snapshot name.
    """
    if not fact.subjects:
        return fact.text
    notes: list[str] = []
    seen_keys: set[str] = set()
    for subject in fact.subjects:
        if subject.canonical_key in seen_keys:
            continue
        seen_keys.add(subject.canonical_key)
        current_display = store.resolve_label(
            canonical_key=subject.canonical_key,
            guild_id=guild_id,
            familiar_id=familiar_id,
        )
        # No row + snapshot ⇒ resolve_label returned the raw user_id;
        # nothing meaningful to annotate against.
        if current_display == subject.canonical_key.partition(":")[2]:
            continue
        if current_display == subject.display_at_write:
            continue
        notes.append(f"{subject.display_at_write} is now known as {current_display}")
    if not notes:
        return fact.text
    return f"{fact.text} ({'; '.join(notes)})"


def _display_for(author: Author | None, role: str) -> str:
    if author is not None and author.display_name:
        return author.display_name
    return role


def _resolve_turn_label(
    store: HistoryStore, ctx: AssemblyContext, turn: HistoryTurn
) -> str:
    """Resolve a turn's speaker label using the per-guild preference order."""
    if turn.author is None:
        return turn.role
    return store.resolve_label(
        canonical_key=turn.author.canonical_key,
        guild_id=ctx.guild_id,
        familiar_id=ctx.familiar_id,
    )


def _rewrite_mentions(
    content: str,
    *,
    store: HistoryStore,
    familiar_id: str,
    guild_id: int | None,
) -> str:
    """Rewrite ``<@USER_ID>`` and ``<@!USER_ID>`` to ``[@DisplayName]``.

    Resolution goes through :meth:`HistoryStore.resolve_label` so the
    same per-guild preference order applies as for speaker names.
    Unknown ids fall back to the bare ``user_id`` — never raises.
    """

    def _sub(match: re.Match[str]) -> str:
        user_id = match.group(1)
        display = store.resolve_label(
            canonical_key=f"discord:{user_id}",
            guild_id=guild_id,
            familiar_id=familiar_id,
        )
        return f"[@{display}]"

    return _DISCORD_MENTION_RE.sub(_sub, content)


def _truncate(text: str, *, limit: int) -> str:
    """Hard cap on a string with an ellipsis suffix when truncated."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


# Bio cap for the prompt header — long bios bloat the context budget;
# the dossier carries the substance so 240 chars of free text is plenty.
_BIO_CHAR_CAP = 240


def _format_profile_header(display: str, profile: AccountProfile | None) -> str:
    """Render the per-person header block.

    Layout:

    ```
    ### <display>
    @<username> · <pronouns>
    Bio: <bio>
    ```

    Each metadata line is omitted when its source is missing or empty.
    The username row collapses to ``@<username>`` alone, or
    ``<pronouns>`` alone, when only one of the two is set. ``display``
    is :meth:`HistoryStore.resolve_label`'s output — guild nick wins
    over global name, so this stays consistent with the rest of the
    read path.
    """
    lines: list[str] = [f"### {display}"]
    if profile is not None:
        username = (profile.username or "").strip()
        pronouns = (profile.pronouns or "").strip()
        meta_parts: list[str] = []
        if username:
            meta_parts.append(f"@{username}")
        if pronouns:
            meta_parts.append(pronouns)
        if meta_parts:
            lines.append(" · ".join(meta_parts))
        bio = (profile.bio or "").strip()
        if bio:
            lines.append(f"Bio: {_truncate(bio, limit=_BIO_CHAR_CAP)}")
    return "\n".join(lines)


def _truncate_to_tokens(text: str, *, max_tokens: int) -> str:
    """Truncate ``text`` so its estimated token count fits ``max_tokens``.

    Uses the same char/4 heuristic as the budgeter — works on the
    char length proxy so this stays cheap.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    char_cap = max_tokens * 4
    return _truncate(text, limit=char_cap)


def _trim_rag_lines_to_tokens(
    *,
    fact_lines: list[str],
    turn_lines: list[str],
    max_tokens: int,
) -> tuple[list[str], list[str]]:
    """Cap RAG fact + turn lines together; facts win ties.

    Adds lines in order (facts first, then turns) until the next would
    cross the cap. Headers aren't counted — they're a flat ~10 tokens
    of overhead the budgeter can absorb.
    """
    used = 0
    kept_facts: list[str] = []
    for line in fact_lines:
        cost = estimate_tokens(line)
        if used + cost > max_tokens and kept_facts:
            break
        kept_facts.append(line)
        used += cost
    kept_turns: list[str] = []
    for line in turn_lines:
        cost = estimate_tokens(line)
        if used + cost > max_tokens and (kept_turns or kept_facts):
            break
        kept_turns.append(line)
        used += cost
    return kept_facts, kept_turns


def _rerank_fact_candidates(
    scored: list[tuple[Fact, float]],
    *,
    limit: int,
    bm25_weight: float,
    recency_weight: float,
    importance_weight: float,
) -> list[Fact]:
    """Fuse BM25 / recency / importance into a single rank, keep top N.

    All three signals are mapped to ``[0, 1]`` quality (higher = better)
    before weighting:

    * **BM25** — FTS5 returns negative numbers (lower = better). The
      best score in the batch maps to 1.0, the worst to 0.0; ties map
      to 0.5. Single-result batches always get 1.0.
    * **Recency** — fact id rank within the batch. Newest = 1.0.
    * **Importance** — ``importance / 10``. ``None`` → neutral 0.5 so
      legacy / unscored rows aren't penalised relative to a 5/10.

    Sort is stable on equal scores; ties fall back to BM25 order
    (the candidate list itself is BM25-ordered when handed in).
    """
    if not scored:
        return []
    bm25_scores = [s for _, s in scored]
    bm25_min = min(bm25_scores)
    bm25_max = max(bm25_scores)
    bm25_span = bm25_max - bm25_min

    fact_ids_sorted = sorted({f.id for f, _ in scored})
    recency_rank: dict[int, float] = {}
    if len(fact_ids_sorted) == 1:
        recency_rank[fact_ids_sorted[0]] = 1.0
    else:
        last = len(fact_ids_sorted) - 1
        for i, fid in enumerate(fact_ids_sorted):
            recency_rank[fid] = i / last

    ranked: list[tuple[float, int, Fact]] = []
    for idx, (fact, bm25) in enumerate(scored):
        # bm25 is negative; min = best. Normalise so best = 1, worst = 0.
        bm25_q = (bm25_max - bm25) / bm25_span if bm25_span > 0 else 1.0
        recency_q = recency_rank[fact.id]
        importance_q = (fact.importance / 10.0) if fact.importance is not None else 0.5
        score = (
            bm25_weight * bm25_q
            + recency_weight * recency_q
            + importance_weight * importance_q
        )
        # idx is the candidate position from the SQL BM25 sort; used as
        # a stable tiebreak so equal-scored facts keep deterministic order.
        ranked.append((score, -idx, fact))
    ranked.sort(key=operator.itemgetter(0, 1), reverse=True)
    return [f for _, _, f in ranked[:limit]]


def _format_clock_12h(ts: datetime) -> str:
    """Render UTC time as ``2:29PM`` (no leading zero on hour)."""
    return ts.astimezone(UTC).strftime("%I:%M%p").lstrip("0")


def _format_date_iso(ts: datetime) -> str:
    """Render UTC date as ``YYYY-MM-DD``."""
    return ts.astimezone(UTC).strftime("%Y-%m-%d")


def _trim_messages_to_token_cap(
    messages: list[Message], max_tokens: int
) -> list[Message]:
    """Drop oldest messages until total estimated tokens fit.

    Always keeps the newest message even if it alone exceeds the cap —
    the most recent turn is the model's primary cue.
    """
    if not messages:
        return messages
    kept_reversed: list[Message] = []
    used = 0
    for msg in reversed(messages):
        cost = estimate_message_tokens(msg)
        if used + cost > max_tokens and kept_reversed:
            break
        kept_reversed.append(msg)
        used += cost
    return list(reversed(kept_reversed))


def _reply_prefix(
    *,
    parent: HistoryTurn,
    parent_in_window: bool,
    store: HistoryStore,
    familiar_id: str,
    guild_id: int | None,
) -> str:
    """Build the ``[↩ Bob (HH:MM): …]`` prefix for a child reply turn.

    The parent's author label is resolved with the same
    ``resolve_label`` callsite as the speaker name, so a per-guild
    nick wins over the snapshot stored on the parent's ``Author``.
    """
    if parent.author is not None:
        parent_label = store.resolve_label(
            canonical_key=parent.author.canonical_key,
            guild_id=guild_id,
            familiar_id=familiar_id,
        )
    else:
        parent_label = parent.role
    parent_text = _rewrite_mentions(
        parent.content, store=store, familiar_id=familiar_id, guild_id=guild_id
    )
    if parent_in_window:
        snippet = _truncate(parent_text, limit=_REPLY_PARENT_SNIPPET_CAP)
        return f"↩ {parent_label}: {snippet}"
    parent_ts = parent.timestamp.astimezone(UTC).strftime("%H:%M")
    full = _truncate(parent_text, limit=_REPLY_PARENT_FULL_CAP)
    return f"↩ {parent_label} ({parent_ts}): {full}"


def _format_reactions(reactions: tuple[tuple[str, int], ...]) -> str:
    """Render ``[reactions: 👍 x3 ❤️ x1]``; empty input → empty string."""
    if not reactions:
        return ""
    parts = " ".join(f"{e} x{c}" for e, c in reactions)
    return f"[reactions: {parts}]"


def _turn_to_message_with_context(
    *,
    turn: HistoryTurn,
    store: HistoryStore,
    familiar_id: str,
    guild_id: int | None,
    in_window_msg_ids: set[str],
    reactions: tuple[tuple[str, int], ...] = (),
) -> Message:
    """Render a :class:`HistoryTurn` into an LLM :class:`Message`.

    Timestamp rendered as UTC ``HH:MM``. Date intentionally omitted —
    the recent-history window is short. Reply markers and mention
    rewriting are applied here so all enrichment lives in one place.
    The ``platform_message_id`` for *user* turns (when present) is
    surfaced as ``#<id>`` next to the speaker so the model can target
    a specific message via ``[↩ <id>]``. Assistant turns deliberately
    skip the id tag — otherwise the model sees its own past messages
    prefixed with ``[#…]`` and starts emitting that prefix in fresh
    replies (mimicry).

    ``reactions`` (emoji, count) pairs append a ``[reactions: …]``
    suffix on either role.
    """
    role = turn.role
    author = turn.author
    content = _rewrite_mentions(
        turn.content, store=store, familiar_id=familiar_id, guild_id=guild_id
    )
    reactions_suffix = _format_reactions(reactions)
    if role == "assistant" or author is None:
        body = f"{content} {reactions_suffix}" if reactions_suffix else content
        return Message(role=role, content=body)
    msg_id_tag = f" #{turn.platform_message_id}" if turn.platform_message_id else ""
    label = store.resolve_label(
        canonical_key=author.canonical_key,
        guild_id=guild_id,
        familiar_id=familiar_id,
    )
    name = sanitize_name(author.canonical_key)
    ts = turn.timestamp.astimezone(UTC).strftime("%H:%M")

    # Reply marker, depth-adaptive.
    reply_prefix = ""
    if turn.reply_to_message_id:
        parent = store.lookup_turn_by_platform_message_id(
            familiar_id=familiar_id,
            platform_message_id=turn.reply_to_message_id,
        )
        if parent is not None:
            parent_in_window = (
                parent.platform_message_id in in_window_msg_ids
                if parent.platform_message_id is not None
                else False
            )
            reply_prefix = (
                _reply_prefix(
                    parent=parent,
                    parent_in_window=parent_in_window,
                    store=store,
                    familiar_id=familiar_id,
                    guild_id=guild_id,
                )
                + " "
            )

    prefixed = f"[{ts} {label}{msg_id_tag}] {reply_prefix}{content}"
    if reactions_suffix:
        prefixed = f"{prefixed} {reactions_suffix}"
    return Message(role=role, content=prefixed, name=name)


def _turn_to_message(
    role: str,
    content: str,
    author: Author | None,
    timestamp: datetime | None = None,
) -> Message:
    """Legacy renderer kept for non-history-store callers (e.g. voice intake).

    Lacks reply-marker / mention-rewriting context — those need a
    store handle and a window. New code should prefer
    :func:`_turn_to_message_with_context`.
    """
    if role == "assistant" or author is None:
        return Message(role=role, content=content)
    name = sanitize_name(author.canonical_key)
    display = author.display_name or author.username or author.user_id
    if timestamp is not None:
        ts = timestamp.astimezone(UTC).strftime("%H:%M")
        prefixed = f"[{ts} {display}] {content}"
    else:
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

    def __init__(
        self,
        *,
        store: HistoryStore,
        max_tokens: int | None = None,
    ) -> None:
        self._store = store
        self._max_tokens = max_tokens

    async def build(self, ctx: AssemblyContext) -> str:
        entry = self._store.get_summary(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id or 0,
        )
        if entry is None or not entry.summary_text.strip():
            return ""
        body = entry.summary_text.strip()
        if self._max_tokens is not None:
            body = _truncate_to_tokens(body, max_tokens=self._max_tokens)
        return "## Conversation so far\n\n" + body

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
        max_tokens: int | None = None,
    ) -> None:
        self._store = store
        self._viewer_map = {k: list(v) for k, v in viewer_map.items()}
        self._ttl_seconds = ttl_seconds
        self._max_tokens = max_tokens

    def _viewer_key(self, ctx: AssemblyContext) -> str:
        return f"{ctx.viewer_mode}:{ctx.channel_id}"

    async def build(self, ctx: AssemblyContext) -> str:
        sources = self._viewer_map.get(ctx.channel_id or -1, [])
        if not sources:
            return ""

        now = datetime.now(tz=UTC)
        sections: list[str] = []
        # Token budget for the whole block; sections are added until
        # the next would push us over. Newest source first so the most
        # recently active channel always lands.
        remaining = self._max_tokens
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
            section = f"### From channel #{source_id}\n\n" + entry.summary_text.strip()
            if remaining is not None:
                cost = estimate_tokens(section)
                if cost > remaining and sections:
                    break
                section = _truncate_to_tokens(section, max_tokens=remaining)
                remaining -= estimate_tokens(section)
            sections.append(section)
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


class PeopleDossierLayer:
    """Per-person dossier block for people active in this channel.

    Reads ``people_dossiers`` (maintained by :class:`PeopleDossierWorker`)
    for canonical keys that show up as authors *or* mentions in the
    last ``window_size`` turns of the active channel. Candidates are
    de-duped most-recent-first and capped at ``max_people`` — same
    hard-count budgeting style as :class:`RecentHistoryLayer`'s
    ``window_size``. Subjects without a stored dossier are skipped
    silently; the worker fills them in within one tick.

    Invalidation: ``(latest_turn_id, max_people, [key:f<wm>, …])`` —
    any new channel turn flips ``latest_turn_id`` (changing the
    candidate set), and each kept candidate's ``last_fact_id`` flips
    when the worker refreshes its dossier.
    """

    name: str = "people_dossier"

    def __init__(
        self,
        *,
        store: HistoryStore,
        window_size: int = 20,
        max_people: int = 8,
        max_tokens: int | None = None,
    ) -> None:
        self._store = store
        self._window_size = window_size
        self._max_people = max_people
        self._max_tokens = max_tokens

    def _candidate_keys(self, ctx: AssemblyContext) -> list[str]:
        """Ordered, deduped canonical keys for the active channel.

        Newest turn first. Per turn, the author's key comes before
        the turn's mentions — a person speaking *now* outranks a
        person being talked about. Capped at ``max_people``.
        """
        if ctx.channel_id is None:
            return []
        turns = self._store.recent(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id,
            limit=self._window_size,
        )
        ordered: list[str] = []
        seen: set[str] = set()

        def _add(key: str) -> None:
            if key in seen:
                return
            seen.add(key)
            ordered.append(key)

        for turn in reversed(turns):  # newest first
            if turn.author is not None:
                _add(turn.author.canonical_key)
            for key in self._store.mentions_for_turn(turn_id=turn.id):
                _add(key)
            if len(ordered) >= self._max_people:
                break
        return ordered[: self._max_people]

    async def build(self, ctx: AssemblyContext) -> str:
        candidates = self._candidate_keys(ctx)
        if not candidates:
            return ""
        sections: list[str] = []
        remaining = self._max_tokens
        for key in candidates:
            entry = self._store.get_people_dossier(
                familiar_id=ctx.familiar_id, canonical_key=key
            )
            if entry is None or not entry.dossier_text.strip():
                continue
            display = self._store.resolve_label(
                canonical_key=key,
                guild_id=ctx.guild_id,
                familiar_id=ctx.familiar_id,
            )
            profile = self._store.get_account_profile(canonical_key=key)
            header = _format_profile_header(display, profile)
            section = f"{header}\n\n{entry.dossier_text.strip()}"
            if remaining is not None:
                cost = estimate_tokens(section)
                if cost > remaining and sections:
                    break
                section = _truncate_to_tokens(section, max_tokens=remaining)
                remaining -= estimate_tokens(section)
            sections.append(section)
        if not sections:
            return ""
        return "## People in this conversation\n\n" + "\n\n".join(sections)

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        candidates = self._candidate_keys(ctx)
        latest = (
            self._store.latest_id(
                familiar_id=ctx.familiar_id, channel_id=ctx.channel_id or 0
            )
            or 0
        )
        parts: list[str] = [f"t{latest}", f"cap{self._max_people}"]
        for key in candidates:
            entry = self._store.get_people_dossier(
                familiar_id=ctx.familiar_id, canonical_key=key
            )
            parts.append(
                f"{key}:none" if entry is None else f"{key}:f{entry.last_fact_id}"
            )
        return "|".join(parts)


class RagContextLayer:
    """FTS-backed retrieval of relevant historical turns *and* facts.

    :meth:`set_current_cue` is called by the responder with the query
    derived from the inbound user turn. When unset (or empty), the
    layer opts out.

    Invalidation: ``(cue, latest_fts_id, latest_fact_id)``. Both
    watermarks move when new turns or facts are written, so the
    cache flips automatically. Phase 4 adds facts alongside turns;
    Phase 3 only surfaced turns.

    :param recent_window_size: when >0, exclude turns from the
        current channel whose id falls within the last
        ``recent_window_size`` rows — those are already shown by
        :class:`RecentHistoryLayer` verbatim. Default 0 preserves
        the unfiltered behaviour for tests / callers that don't opt
        in.

    :param context_window: per FTS hit, expand to ``hit.id ±
        context_window`` so each retrieved turn keeps a small
        surrounding context. ``0`` = hit alone (legacy behaviour).

    :param bm25_weight: ranking weight on the BM25 quality of a fact
        match. Default 1.0 reproduces pre-M2 ordering.
    :param recency_weight: ranking weight on fact id rank (newer =
        higher score). 0 disables.
    :param importance_weight: ranking weight on the extractor's 1-10
        importance hint. 0 disables (default — opt in via TOML).
        ``None`` importance is treated as the neutral midpoint (5/10).
    :param embedding_weight: M6 placeholder; ignored today, accepted
        so the constructor's surface stabilises ahead of the
        embedding projector landing.
    :param fact_overfetch: when any non-default rank weight is set,
        over-fetch this many BM25 candidates before reranking. Hard
        cap on extra DB work — never exceeds 4x ``max_facts``.
    """

    name: str = "rag_context"

    def __init__(
        self,
        *,
        store: HistoryStore,
        max_results: int = 5,
        max_facts: int = 3,
        recent_window_size: int = 0,
        max_tokens: int | None = None,
        context_window: int = 1,
        bm25_weight: float = 1.0,
        recency_weight: float = 0.0,
        importance_weight: float = 0.0,
        embedding_weight: float = 0.0,
        fact_overfetch: int = 12,
    ) -> None:
        self._store = store
        self._max_results = max_results
        self._max_facts = max_facts
        self._recent_window_size = recent_window_size
        self._max_tokens = max_tokens
        self._context_window = max(0, context_window)
        self._bm25_weight = float(bm25_weight)
        self._recency_weight = float(recency_weight)
        self._importance_weight = float(importance_weight)
        self._embedding_weight = float(embedding_weight)
        self._fact_overfetch = max(int(fact_overfetch), 1)
        self._current_cue: str = ""

    @property
    def _rerank_facts(self) -> bool:
        """True when any non-BM25 weight is set; opts the rerank path on."""
        return self._recency_weight > 0.0 or self._importance_weight > 0.0

    def set_current_cue(self, cue: str) -> None:
        self._current_cue = (cue or "").strip()

    async def build(self, ctx: AssemblyContext) -> str:
        cue = self._current_cue
        if not cue:
            return ""
        # Window cutoff for the current channel: anything newer than
        # this id is already in RecentHistoryLayer's output.
        max_id: int | None = None
        if self._recent_window_size > 0 and ctx.channel_id is not None:
            latest = self._store.latest_id(
                familiar_id=ctx.familiar_id, channel_id=ctx.channel_id
            )
            if latest is not None:
                max_id = latest - self._recent_window_size
        turn_results = self._store.search_turns(
            familiar_id=ctx.familiar_id,
            query=cue,
            limit=self._max_results,
            max_id=max_id,
        )
        if self._rerank_facts:
            fetch = min(self._fact_overfetch, self._max_facts * 4)
            fetch = max(fetch, self._max_facts)
            scored = self._store.search_facts_scored(
                familiar_id=ctx.familiar_id,
                query=cue,
                limit=fetch,
            )
            fact_results = _rerank_fact_candidates(
                scored,
                limit=self._max_facts,
                bm25_weight=self._bm25_weight,
                recency_weight=self._recency_weight,
                importance_weight=self._importance_weight,
            )
        else:
            fact_results = self._store.search_facts(
                familiar_id=ctx.familiar_id,
                query=cue,
                limit=self._max_facts,
            )
        if not turn_results and not fact_results:
            return ""

        # Build candidate item lines first so we can apply a token cap
        # uniformly across facts + turns. Facts come first — usually
        # higher signal per token than retrieved turns.
        fact_lines = [
            f"- {_render_fact_line(self._store, ctx.familiar_id, f, guild_id=ctx.guild_id)}"  # noqa: E501
            for f in fact_results
        ]
        turn_lines = self._render_turn_lines(ctx, turn_results, max_id=max_id)

        if self._max_tokens is not None:
            fact_lines, turn_lines = _trim_rag_lines_to_tokens(
                fact_lines=fact_lines,
                turn_lines=turn_lines,
                max_tokens=self._max_tokens,
            )

        sections: list[str] = []
        if fact_lines:
            sections.append("\n".join(["## Possibly relevant facts\n", *fact_lines]))
        if turn_lines:
            sections.append(
                "\n".join(["## Possibly relevant earlier turns\n", *turn_lines])
            )
        return "\n\n".join(sections)

    def _render_turn_lines(
        self,
        ctx: AssemblyContext,
        hits: list[HistoryTurn],
        *,
        max_id: int | None,
    ) -> list[str]:
        """Group hits + neighbour context by date, render blockquote lines.

        For each FTS hit, expand to ``id ± context_window`` and
        de-duplicate, then bucket by UTC date. Each bucket emits a
        ``YYYY-MM-DD:`` header followed by ``> [H:MMpm Author]: text``
        lines so the model sees a hit with its surrounding chatter.

        ``max_id`` mirrors the FTS exclusion: neighbour turns from the
        active channel that fall inside the recent-history window are
        dropped — :class:`RecentHistoryLayer` already shows them.
        """
        if not hits:
            return []
        wanted_ids: set[int] = set()
        for h in hits:
            wanted_ids.update(
                h.id + d for d in range(-self._context_window, self._context_window + 1)
            )
        expanded = self._store.turns_by_ids(familiar_id=ctx.familiar_id, ids=wanted_ids)
        hit_channels = {h.channel_id for h in hits}
        kept: list[HistoryTurn] = []
        for t in expanded:
            if t.channel_id not in hit_channels:
                continue
            if (
                max_id is not None
                and ctx.channel_id is not None
                and t.channel_id == ctx.channel_id
                and t.id > max_id
            ):
                continue
            kept.append(t)

        by_date: dict[str, list[HistoryTurn]] = {}
        for turn in kept:
            by_date.setdefault(_format_date_iso(turn.timestamp), []).append(turn)

        lines: list[str] = []
        for date_label in sorted(by_date):
            lines.append(f"{date_label}:")
            for turn in by_date[date_label]:
                label = _resolve_turn_label(self._store, ctx, turn)
                rewritten = _rewrite_mentions(
                    turn.content,
                    store=self._store,
                    familiar_id=ctx.familiar_id,
                    guild_id=ctx.guild_id,
                )
                clock = _format_clock_12h(turn.timestamp)
                content_lines = rewritten.split("\n")
                lines.append(f"> [{clock} {label}]: {content_lines[0]}")
                # keep blockquote intact across newlines; bare `>` for
                # blank lines so spacing inside the quote is preserved.
                lines.extend(f"> {cont}" if cont else ">" for cont in content_lines[1:])
            lines.append("")  # blank line between date groups
        if lines and not lines[-1]:
            lines.pop()
        return lines

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        cue = self._current_cue
        latest_turn = self._store.latest_fts_id(familiar_id=ctx.familiar_id)
        latest_fact = self._store.latest_fact_id(familiar_id=ctx.familiar_id)
        return f"{cue}|t{latest_turn}|f{latest_fact}"


class ReflectionLayer:
    """Recent reflections block — higher-order syntheses with citations (M3).

    Reads ``reflections`` (maintained by :class:`ReflectionWorker`) and
    renders the most recent ``max_reflections`` rows scoped to the
    active channel (channel-agnostic rows always surface). Citations
    render as breadcrumbs ``[T#42, F#7]`` so the reader (the LLM) can
    map a synthesis back to its provenance. Reflections that cite at
    least one superseded fact are flagged ``(stale)`` — never deleted,
    per the supersession-over-overwrite rule.

    Invalidation: ``(latest_reflection_id, channel_id, max_reflections)``.
    A new reflection write or a fact supersession is detected by
    :class:`Assembler` cache walks already keyed on ``latest_*_id``;
    here we just key on the newest reflection's id.
    """

    name: str = "reflection"

    def __init__(
        self,
        *,
        store: HistoryStore,
        max_reflections: int = 3,
        max_tokens: int | None = None,
    ) -> None:
        self._store = store
        self._max_reflections = max(0, max_reflections)
        self._max_tokens = max_tokens

    async def build(self, ctx: AssemblyContext) -> str:
        if self._max_reflections <= 0:
            return ""
        rows = self._store.recent_reflections(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id,
            limit=self._max_reflections,
        )
        if not rows:
            return ""
        # Single batched query for stale citations across every row.
        all_cited_facts: list[int] = []
        for r in rows:
            all_cited_facts.extend(r.cited_fact_ids)
        stale = self._store.superseded_fact_ids(
            familiar_id=ctx.familiar_id, fact_ids=all_cited_facts
        )

        sections: list[str] = []
        remaining = self._max_tokens
        for r in rows:
            citations: list[str] = []
            citations.extend(f"T#{tid}" for tid in r.cited_turn_ids)
            citations.extend(f"F#{fid}" for fid in r.cited_fact_ids)
            cite_block = f" [{', '.join(citations)}]" if citations else ""
            stale_block = (
                " (stale)" if any(fid in stale for fid in r.cited_fact_ids) else ""
            )
            line = f"- {r.text.strip()}{cite_block}{stale_block}"
            if remaining is not None:
                cost = estimate_tokens(line)
                if cost > remaining and sections:
                    break
                line = _truncate_to_tokens(line, max_tokens=remaining)
                remaining -= estimate_tokens(line)
            sections.append(line)
        if not sections:
            return ""
        return "## Recent reflections\n\n" + "\n".join(sections)

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        rows = self._store.recent_reflections(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id,
            limit=1,
        )
        latest_id = rows[0].id if rows else 0
        return f"ch{ctx.channel_id}|r{latest_id}|cap{self._max_reflections}"


# ---------------------------------------------------------------------------
# Lorebook (M4) — hand-authored, keyword-activated canon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LorebookEntry:
    """One ``[[entries]]`` row from ``lorebook.toml``.

    :param keys: substrings that activate this entry; matched
        case-insensitively against the recent-history scan window.
    :param content: text inserted into the system prompt on hit.
    :param priority: higher renders first; ties keep file order.
    :param selective: when ``True``, *all* keys must match (AND); the
        default ``False`` matches if any key matches (OR).
    """

    keys: tuple[str, ...]
    content: str
    priority: int = 0
    selective: bool = False


class LorebookLayer:
    """Keyword-activated authored canon (M4).

    Source of truth: ``data/familiars/<id>/lorebook.toml``. Each
    ``[[entries]]`` row carries ``keys`` / ``content`` / ``priority``
    and an optional ``selective`` flag. ``build`` scans the active
    channel's last ``recent_window`` turns, fires entries whose
    keys hit, and renders the surviving content sorted by descending
    priority — file order breaks ties. No worker; the file is the
    sole source of truth.

    Cache key combines the file's content hash with the matched
    entry indices, so the layer flips when either changes.
    """

    name: str = "lorebook"

    def __init__(
        self,
        *,
        store: HistoryStore,
        path: Path,
        recent_window: int = 20,
        max_entries: int = 10,
        max_tokens: int | None = None,
    ) -> None:
        self._store = store
        self._path = path
        self._recent_window = max(1, recent_window)
        self._max_entries = max(0, max_entries)
        self._max_tokens = max_tokens

    def _load_entries(self) -> list[LorebookEntry]:
        if not self._path.exists():
            return []
        try:
            with self._path.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            return []
        raw_entries = data.get("entries", [])
        if not isinstance(raw_entries, list):
            return []
        out: list[LorebookEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            keys_raw = raw.get("keys", [])
            if not isinstance(keys_raw, list):
                continue
            keys = tuple(str(k) for k in keys_raw if isinstance(k, str) and k)
            if not keys:
                continue
            content = str(raw.get("content", "")).strip()
            if not content:
                continue
            priority_raw = raw.get("priority", 0)
            priority = (
                int(priority_raw)
                if isinstance(priority_raw, int) and not isinstance(priority_raw, bool)
                else 0
            )
            selective = bool(raw.get("selective", False))
            out.append(
                LorebookEntry(
                    keys=keys,
                    content=content,
                    priority=priority,
                    selective=selective,
                )
            )
        return out

    def _scan_text(self, ctx: AssemblyContext) -> str:
        if ctx.channel_id is None:
            return ""
        turns = self._store.recent(
            familiar_id=ctx.familiar_id,
            channel_id=ctx.channel_id,
            limit=self._recent_window,
        )
        return "\n".join(t.content for t in turns).lower()

    def _matched_indices(self, entries: list[LorebookEntry], scan: str) -> list[int]:
        """File-order indices of entries whose keys hit *scan*."""
        if not scan:
            return []
        out: list[int] = []
        for idx, entry in enumerate(entries):
            keys_lc = [k.lower() for k in entry.keys]
            if entry.selective:
                if all(k in scan for k in keys_lc):
                    out.append(idx)
            elif any(k in scan for k in keys_lc):
                out.append(idx)
        return out

    async def build(self, ctx: AssemblyContext) -> str:
        if self._max_entries <= 0:
            return ""
        entries = self._load_entries()
        if not entries:
            return ""
        scan = self._scan_text(ctx)
        idxs = self._matched_indices(entries, scan)
        if not idxs:
            return ""
        # Sort by priority desc; ties keep file order via the index.
        idxs.sort(key=lambda i: (-entries[i].priority, i))
        idxs = idxs[: self._max_entries]

        sections: list[str] = []
        remaining = self._max_tokens
        for i in idxs:
            section = entries[i].content
            if remaining is not None:
                cost = estimate_tokens(section)
                if cost > remaining and sections:
                    break
                section = _truncate_to_tokens(section, max_tokens=remaining)
                remaining -= estimate_tokens(section)
            sections.append(section)
        if not sections:
            return ""
        return "## Lorebook\n\n" + "\n\n".join(sections)

    def invalidation_key(self, ctx: AssemblyContext) -> str:
        file_hash = _content_hash(self._path)
        entries = self._load_entries()
        scan = self._scan_text(ctx)
        idxs = self._matched_indices(entries, scan)
        return (
            f"f{file_hash}|ch{ctx.channel_id}|"
            f"m{','.join(str(i) for i in idxs)}|cap{self._max_entries}"
        )
