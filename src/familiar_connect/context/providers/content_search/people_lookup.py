"""Deterministic people-file lookup — tier 1 of ContentSearchProvider.

No LLM. Always injects the speaker's ``people/<author.slug>.md`` and
any file whose canonical slug resolves from a name mentioned in the
utterance (via ``people/_aliases.json``). Correctness floor —
guarantees the familiar never forgets someone it has notes on.

Files are keyed by :attr:`Author.slug` (e.g. ``people/discord-123.md``)
so renames never orphan a file. The alias index maps lowercased
human-readable names — display names, usernames, nicknames — to those
canonical slugs and is rebuilt each memory-writer pass.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer
from familiar_connect.memory.store import MemoryStoreError

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest
    from familiar_connect.memory.store import MemoryStore


_logger = logging.getLogger(__name__)


PEOPLE_LOOKUP_PRIORITY = 85
"""Priority for deterministic people-file contributions.

Between CharacterProvider (100) and the agent-loop tier (70). Loses
to the familiar's own persona under budget pressure, but wins over
any retrieved/RAG context — a notes file on a person we're talking
to is more important than a fuzzy keyword match."""

PEOPLE_LOOKUP_SOURCE = "content_search.people"
"""Source tag on emitted Contributions for budget-tracing logs."""

DEFAULT_MAX_TOKENS_PER_FILE = 800
"""Per-file truncation cap. Applied before any layer-budget trim so
a single oversized bio can't starve other people files."""

_PEOPLE_DIR = "people"
_ALIAS_INDEX_PATH = f"{_PEOPLE_DIR}/_aliases.json"
_UTTERANCE_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_TRUNCATION_MARKER = "\n\n[…truncated]"


@dataclass(frozen=True)
class LookupResult:
    """Deterministic-tier output for the orchestrator.

    ``rel_paths`` lets the orchestrator hand the retriever an exclude
    set so embedding hits don't re-surface files already loaded
    verbatim here.
    """

    contributions: list[Contribution] = field(default_factory=list)
    rel_paths: list[str] = field(default_factory=list)


def lookup(
    store: MemoryStore,
    request: ContextRequest,
    *,
    content_cap_tokens: int | None = None,
    max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
) -> list[Contribution]:
    """Return Contributions for speaker + mentioned-name people files.

    Ordering: speaker first, then utterance order. Overflow against
    *content_cap_tokens* drops the tail, preserving the speaker.
    """
    return lookup_with_paths(
        store,
        request,
        content_cap_tokens=content_cap_tokens,
        max_tokens_per_file=max_tokens_per_file,
    ).contributions


def lookup_with_paths(
    store: MemoryStore,
    request: ContextRequest,
    *,
    content_cap_tokens: int | None = None,
    max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
) -> LookupResult:
    """Return contributions plus the loaded rel_paths for retriever exclusion."""
    aliases = _load_alias_index(store)
    ordered_slugs = _select_slugs(store, request, aliases)
    if not ordered_slugs:
        return LookupResult()

    contributions: list[Contribution] = []
    rel_paths: list[str] = []
    tokens_used = 0
    for slug in ordered_slugs:
        rel_path = f"{_PEOPLE_DIR}/{slug}.md"
        try:
            raw = store.read_file(rel_path)
        except MemoryStoreError:
            # alias points to a stale slug — degrade gracefully
            continue

        text = _truncate(raw, max_tokens_per_file)
        est = estimate_tokens(text)
        if content_cap_tokens is not None and tokens_used + est > content_cap_tokens:
            _logger.info(
                f"{ls.tag('👥 People', ls.M)} "
                f"{ls.kv('dropped', rel_path, vc=ls.LM)} "
                f"{ls.kv('used', str(tokens_used), vc=ls.LM)} "
                f"{ls.kv('cap', str(content_cap_tokens), vc=ls.LM)}"
            )
            break
        tokens_used += est
        contributions.append(
            Contribution(
                layer=Layer.content,
                priority=PEOPLE_LOOKUP_PRIORITY,
                text=text,
                estimated_tokens=est,
                source=PEOPLE_LOOKUP_SOURCE,
            )
        )
        rel_paths.append(rel_path)

    return LookupResult(contributions=contributions, rel_paths=rel_paths)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_alias_index(store: MemoryStore) -> dict[str, str]:
    """Return ``{lower_name: canonical_slug}`` from ``people/_aliases.json``.

    Silent fallback to an empty dict if the file is missing or malformed —
    deterministic tier degrades gracefully and mention-pass simply finds
    nothing.
    """
    try:
        raw = store.read_file(_ALIAS_INDEX_PATH)
    except MemoryStoreError:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        _logger.warning("alias index %s malformed; ignoring", _ALIAS_INDEX_PATH)
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            result[key.lower()] = value
    return result


def _file_exists(store: MemoryStore, rel_path: str) -> bool:
    try:
        store.read_file(rel_path)
    except MemoryStoreError:
        return False
    return True


def _select_slugs(
    store: MemoryStore,
    request: ContextRequest,
    aliases: dict[str, str],
) -> list[str]:
    """Pick which canonical slugs to load, in priority order."""
    seen: set[str] = set()
    ordered: list[str] = []

    # speaker file first — keyed directly by Author.slug
    if request.author is not None:
        slug = request.author.slug
        if slug and _file_exists(store, f"{_PEOPLE_DIR}/{slug}.md"):
            seen.add(slug)
            ordered.append(slug)

    utterance = request.utterance
    lower = utterance.lower()
    tokens = _UTTERANCE_TOKEN.findall(lower)

    # pass (a): capitalized mid-sentence words → alias index
    for word in _capitalized_candidates(utterance):
        slug = aliases.get(word.lower())
        if slug and slug not in seen:
            seen.add(slug)
            ordered.append(slug)

    # pass (b): single-word tokens → alias index
    for token in tokens:
        slug = aliases.get(token)
        if slug and slug not in seen:
            seen.add(slug)
            ordered.append(slug)

    # pass (c): multi-word aliases (phrases with spaces or hyphens)
    for alias_name, slug in aliases.items():
        if slug in seen:
            continue
        if " " in alias_name and alias_name in lower:
            seen.add(slug)
            ordered.append(slug)
            continue
        if "-" in alias_name and alias_name in lower:
            seen.add(slug)
            ordered.append(slug)

    return ordered


def _capitalized_candidates(utterance: str) -> list[str]:
    """Capitalized words not at sentence start.

    Naive sentence split on ``[.!?]`` — good enough: we only use the
    output to look up the alias index, and the miss there filters out
    false positives.
    """
    candidates: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(utterance):
        words = _WORD.findall(sentence)
        candidates.extend(word for word in words[1:] if word[0].isupper())
    return candidates


def _truncate(text: str, max_tokens: int) -> str:
    """Truncate to ~*max_tokens* tokens, preserving line boundaries.

    Uses the project's ``estimate_tokens`` convention (~4 chars per
    token) to pick a character cut-off. If the cut lands inside a
    line, back up to the previous newline when close to the cut.
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    # estimate_tokens ~ len/4 → 4*max_tokens chars per cap
    approx_chars = max_tokens * 4
    truncated = text[:approx_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > approx_chars * 0.8:
        truncated = truncated[:last_nl]
    return truncated.rstrip() + _TRUNCATION_MARKER
