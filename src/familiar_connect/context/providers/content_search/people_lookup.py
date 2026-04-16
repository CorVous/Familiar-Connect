"""Deterministic people-file lookup — tier 1 of ContentSearchProvider.

No LLM. Always injects the speaker's ``people/<slug>.md`` and any
file whose stem matches a name mentioned in the utterance. The
correctness floor — guarantees the familiar never forgets someone it
has notes on, regardless of LLM behaviour in the later tiers.

Slug convention mirrors ``memory/writer.py`` exactly: lowercase the
name, replace any run of non-alphanumeric chars with a single dash,
strip leading/trailing dashes.
"""

from __future__ import annotations

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
_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_UTTERANCE_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_TRUNCATION_MARKER = "\n\n[…truncated]"


def slug(name: str) -> str:
    """Slugify *name* to match ``memory/writer.py`` exactly.

    Mirrors the write-side convention so a file created as
    ``people/<slug(speaker)>.md`` is guaranteed to round-trip here.
    """
    return _SLUG_NON_ALNUM.sub("-", name.lower()).strip("-")


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

    Ordering: speaker first, then utterance order (pass-a candidates
    before pass-b reverse matches). Overflow against
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
    stems = _list_people_stems(store)
    if not stems:
        return LookupResult()

    ordered_stems = _select_stems(stems, request)
    if not ordered_stems:
        return LookupResult()

    contributions: list[Contribution] = []
    rel_paths: list[str] = []
    tokens_used = 0
    for stem in ordered_stems:
        rel_path = f"{_PEOPLE_DIR}/{stem}.md"
        try:
            raw = store.read_file(rel_path)
        except MemoryStoreError:
            # File listed by glob but unreadable (size cap, permissions)
            # — skip; deterministic tier degrades gracefully.
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


def _list_people_stems(store: MemoryStore) -> set[str]:
    """Return the set of ``people/<stem>.md`` stems present in the store."""
    try:
        matches = store.glob(f"{_PEOPLE_DIR}/*.md")
    except MemoryStoreError:
        return set()
    stems: set[str] = set()
    prefix = f"{_PEOPLE_DIR}/"
    for rel in matches:
        if rel.startswith(prefix) and rel.endswith(".md"):
            stem = rel[len(prefix) : -len(".md")]
            if stem:
                stems.add(stem)
    return stems


def _select_stems(stems: set[str], request: ContextRequest) -> list[str]:
    """Pick which stems to load, in priority order (speaker → mentions)."""
    seen: set[str] = set()
    ordered: list[str] = []

    # speaker file first
    if request.speaker:
        s = slug(request.speaker)
        if s and s in stems:
            seen.add(s)
            ordered.append(s)

    utterance = request.utterance
    lower = utterance.lower()
    tokens = _UTTERANCE_TOKEN.findall(lower)
    token_set = set(tokens)

    # pass (a): capitalized mid-sentence words → slug → check exists
    for word in _capitalized_candidates(utterance):
        candidate = slug(word)
        if candidate and candidate in stems and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)

    # pass (b): reverse match each stem against utterance tokens/phrases
    for stem in sorted(stems):
        if stem in seen:
            continue
        if _stem_matches(stem, token_set, lower):
            seen.add(stem)
            ordered.append(stem)

    return ordered


def _capitalized_candidates(utterance: str) -> list[str]:
    """Capitalized words not at sentence start.

    Naive sentence split on ``[.!?]`` — good enough: we only use the
    output as a set of slug candidates, and the stem-existence check
    filters out false positives.
    """
    candidates: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(utterance):
        words = _WORD.findall(sentence)
        candidates.extend(word for word in words[1:] if word[0].isupper())
    return candidates


def _stem_matches(stem: str, token_set: set[str], utterance_lower: str) -> bool:
    """Reverse-match a file stem against the utterance.

    - single-token stem (``alice``): match as whole word
    - hyphenated stem (``bob-the-builder``): match as space-separated
      phrase ("bob the builder") OR as the literal hyphenated form.
    """
    parts = stem.split("-")
    if len(parts) == 1:
        return parts[0] in token_set
    if stem in utterance_lower:
        return True
    phrase = " ".join(parts)
    return phrase in utterance_lower


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
