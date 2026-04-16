"""Single-shot filter tier — tier 3 of ContentSearchProvider.

Replaces the tool-using agent loop. One LLM call with the utterance
plus the top-K embedding-retrieved chunks plus a hint about which
files the deterministic tier already included. The model either:

- ``ANSWER: <text>`` — wrap as a ``Layer.content`` Contribution
- ``ANSWER:`` (empty) — return nothing; retrieved context wasn't useful
- ``ESCALATE: <reason>; GREP: <pattern>`` — trigger one store.grep,
  then a single forced-answer follow-up call

Total LLM cap = 2 calls. Malformed responses are surfaced as the
answer so a noisy model still produces something usable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer
from familiar_connect.llm import Message
from familiar_connect.memory.store import MemoryStoreError

if TYPE_CHECKING:
    from familiar_connect.context.providers.content_search.retrieval import (
        RetrievedChunk,
    )
    from familiar_connect.context.types import ContextRequest
    from familiar_connect.llm import LLMClient
    from familiar_connect.memory.store import MemoryStore


_logger = logging.getLogger(__name__)


FILTER_MAX_ITERATIONS = 2
"""Total LLM calls per ``run`` — first turn + optional forced second turn."""

FILTER_PRIORITY = 70
"""Priority on emitted Contributions. Lower than CharacterProvider
(100), the deterministic people-lookup tier (85), and
HistoryProvider's recent window (80); higher than the rolling
history summary (60)."""

FILTER_SOURCE = "content_search.rag"
"""Source tag on emitted Contributions — distinguishes filter output
from the deterministic ``content_search.people`` tier."""

MAX_SNIPPET_TOKENS = 150
"""Per-chunk snippet cap for the filter prompt — keep the prompt
cheap even when the retriever returns long oversized-section chunks."""

_ANSWER_PREFIX = "ANSWER:"
_ESCALATE_PREFIX = "ESCALATE:"
_GREP_SEPARATOR = "; GREP:"
_MAX_GREP_HITS = 10

_FIRST_PROMPT_TEMPLATE = """\
You are assembling memory context for a reply to the user. Relevant
snippets from the familiar's memory directory have already been
retrieved; your job is to pick out what's actually worth forwarding
to the main model, or to ask for ONE follow-up grep if the snippets
don't cover it.

The current speaker is {speaker}. The user said:

    {utterance}

Retrieved snippets (highest relevance first):
{retrieved}

{already_included}

Reply with EXACTLY ONE LINE in ONE of these shapes:

    ANSWER: <the context worth forwarding, or empty if nothing>
    ESCALATE: <reason>; GREP: <regex pattern>

Prefer ANSWER. Use ESCALATE only if the snippets miss something
specific you can name with a short regex pattern."""


_FORCED_PROMPT_TEMPLATE = """\
You asked to escalate; the grep results are below. This is your
FINAL turn — emit ANSWER now, or an empty ANSWER if nothing fits.
NO MORE ESCALATIONS.

The current speaker is {speaker}. The user said:

    {utterance}

Retrieved snippets (from earlier):
{retrieved}

{already_included}

Your earlier escalate reason: {reason}
Grep pattern: {pattern}
Grep results:
{grep_results}

Reply with EXACTLY ONE LINE:

    ANSWER: <the context worth forwarding, or empty if nothing>"""


async def run(
    *,
    llm_client: LLMClient,
    store: MemoryStore,
    request: ContextRequest,
    retrieved: list[RetrievedChunk],
    deterministic: list[Contribution],
) -> list[Contribution]:
    """Run the filter and return at most one Contribution."""
    first_prompt = _first_prompt(request, retrieved, deterministic)
    reply = await llm_client.chat([Message(role="user", content=first_prompt)])
    parsed = _parse(reply.content)

    if parsed.kind == "answer":
        return _answer_to_contribution(parsed.text)
    if parsed.kind == "fallback":
        return _answer_to_contribution(parsed.text)

    # escalate path
    grep_results = _safe_grep(store, parsed.grep_pattern)
    forced_prompt = _forced_prompt(
        request,
        retrieved,
        deterministic,
        parsed.reason,
        parsed.grep_pattern,
        grep_results,
    )
    reply2 = await llm_client.chat([Message(role="user", content=forced_prompt)])
    parsed2 = _parse(reply2.content)
    if parsed2.kind == "answer":
        return _answer_to_contribution(parsed2.text)
    if parsed2.kind == "fallback":
        return _answer_to_contribution(parsed2.text)
    # second escalate — forced turn ignores it
    _logger.info(
        f"{ls.tag('🔍 Filter', ls.M)} "
        f"{ls.kv('event', 'escalate-exhausted', vc=ls.LY)} "
        f"{ls.word('giving up cleanly', ls.LW)}"
    )
    return []


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _first_prompt(
    request: ContextRequest,
    retrieved: list[RetrievedChunk],
    deterministic: list[Contribution],
) -> str:
    return _FIRST_PROMPT_TEMPLATE.format(
        speaker=request.speaker or "(unknown)",
        utterance=request.utterance,
        retrieved=_format_retrieved(retrieved),
        already_included=_format_deterministic(deterministic),
    )


def _forced_prompt(
    request: ContextRequest,
    retrieved: list[RetrievedChunk],
    deterministic: list[Contribution],
    reason: str,
    pattern: str,
    grep_results: str,
) -> str:
    return _FORCED_PROMPT_TEMPLATE.format(
        speaker=request.speaker or "(unknown)",
        utterance=request.utterance,
        retrieved=_format_retrieved(retrieved),
        already_included=_format_deterministic(deterministic),
        reason=reason or "(none)",
        pattern=pattern or "(none)",
        grep_results=grep_results or "(no matches)",
    )


def _format_retrieved(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(none)"
    parts: list[str] = []
    for c in chunks:
        snippet = _truncate_snippet(c.text)
        heading = c.heading_path or "(root)"
        parts.append(f"- [{c.score:.2f}] {c.rel_path} — {heading}\n  {snippet}")
    return "\n".join(parts)


def _format_deterministic(contributions: list[Contribution]) -> str:
    if not contributions:
        return ""
    return (
        "Already included verbatim by the people-lookup tier "
        "(do NOT repeat their content): "
        f"{len(contributions)} file(s)."
    )


def _truncate_snippet(text: str) -> str:
    if estimate_tokens(text) <= MAX_SNIPPET_TOKENS:
        return text
    # 4 chars per token, per project convention
    approx_chars = MAX_SNIPPET_TOKENS * 4
    return text[:approx_chars].rstrip() + "…"


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------


def _safe_grep(store: MemoryStore, pattern: str) -> str:
    if not pattern.strip():
        return "(no pattern provided)"
    try:
        hits = store.grep(pattern)
    except MemoryStoreError as exc:
        return f"(grep error: {type(exc).__name__}: {exc})"
    except Exception as exc:  # noqa: BLE001 — fold into prompt
        _logger.warning(
            f"{ls.tag('🔍 Filter', ls.M)} "
            f"{ls.kv('grep', 'raised')} "
            f"{ls.kv('type', type(exc).__name__)} "
            f"{ls.kv('msg', str(exc), vc=ls.LW)}"
        )
        return f"(grep error: {type(exc).__name__})"
    if not hits:
        return "(no matches)"
    lines = [
        f"{h.rel_path}:{h.line_number}: {h.line_text}" for h in hits[:_MAX_GREP_HITS]
    ]
    if len(hits) > _MAX_GREP_HITS:
        lines.append(f"... ({len(hits) - _MAX_GREP_HITS} more matches suppressed)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Parsed:
    """Discriminated parse: ``answer`` / ``escalate`` / ``fallback``."""

    kind: str
    text: str = ""
    reason: str = ""
    grep_pattern: str = ""


def _parse(response: str) -> _Parsed:
    stripped = response.strip()
    if stripped.startswith(_ANSWER_PREFIX):
        text = stripped[len(_ANSWER_PREFIX) :]
        return _Parsed(kind="answer", text=text)
    if stripped.startswith(_ESCALATE_PREFIX):
        body = stripped[len(_ESCALATE_PREFIX) :]
        if _GREP_SEPARATOR in body:
            reason, pattern = body.split(_GREP_SEPARATOR, 1)
            return _Parsed(
                kind="escalate",
                reason=reason.strip(),
                grep_pattern=pattern.strip(),
            )
        # ESCALATE without a GREP: still escalates; empty pattern triggers
        # the "no pattern" grep result in the forced turn.
        return _Parsed(kind="escalate", reason=body.strip(), grep_pattern="")
    return _Parsed(kind="fallback", text=stripped)


# ---------------------------------------------------------------------------
# Contribution assembly
# ---------------------------------------------------------------------------


def _answer_to_contribution(text: str) -> list[Contribution]:
    clean = text.strip()
    if not clean:
        return []
    return [
        Contribution(
            layer=Layer.content,
            priority=FILTER_PRIORITY,
            text=clean,
            estimated_tokens=estimate_tokens(clean),
            source=FILTER_SOURCE,
        )
    ]
