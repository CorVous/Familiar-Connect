"""Cold-cache signals — research-phase instrumentation.

Plan treats "is our retrieval stale?" as open question with multiple
candidate signals. Phase 3 ships these as *spans* only — they log
when they fire, but don't drive cache invalidation. After collecting
data correlating signals with "had to redo work" outcomes, the
most-useful ones get wired into layer invalidation.

Signals shipped:

- :func:`detect_topic_shift` — tiny shared-vocabulary test between
  inbound turn and current summary.
- :func:`detect_unknown_proper_noun` — capitalized token in inbound
  turn absent from prior context.
- :func:`detect_silence_gap` — wall-clock gap between turns above a
  threshold.

``log_signals`` wraps all three, emits one span per firing signal,
tagged with inbound turn's channel id for traceability.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls

if TYPE_CHECKING:
    from datetime import datetime

_logger = logging.getLogger("familiar_connect.diagnostics.cold_cache")

_WORD_RE = re.compile(r"[\w']{3,}", re.UNICODE)
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")

# capitalized discourse markers / sentence-starters that the regex
# above would otherwise flag on every short voice fragment ("But.",
# "Okay.", "Which means…"). stored lowercase; matched case-insensitively.
# incomplete by design — additions are cheap, full NER out of scope.
_SENTENCE_STARTER_STOPWORDS: frozenset[str] = frozenset({
    "actually",
    "also",
    "and",
    "but",
    "for",
    "however",
    "just",
    "like",
    "maybe",
    "now",
    "oh",
    "okay",
    "really",
    "right",
    "since",
    "something",
    "sometimes",
    "still",
    "that",
    "the",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "well",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "yeah",
    "yes",
    "you",
    "your",
})


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def detect_topic_shift(
    *,
    new_text: str,
    prior_context: str,
    min_overlap: float = 0.15,
    min_tokens: int = 4,
) -> bool:
    """Detect when ``new_text`` shares too few content words with prior.

    Jaccard overlap below ``min_overlap`` treated as a shift. Default
    0.15 intentionally permissive — want the signal to fire often so
    we can see how it correlates with retrieval failures.

    Voice fragments often reduce to 0-1 content tokens after the 3+
    char filter, guaranteeing near-zero Jaccard regardless of actual
    topic continuity. ``min_tokens`` floors input length; below it
    we return ``False`` rather than emit noise.
    """
    new_tokens = _tokens(new_text)
    old_tokens = _tokens(prior_context)
    if not new_tokens or not old_tokens:
        return False
    if len(new_tokens) < min_tokens:
        return False
    overlap = len(new_tokens & old_tokens) / len(new_tokens | old_tokens)
    return overlap < min_overlap


def detect_unknown_proper_noun(
    *,
    new_text: str,
    prior_context: str,
    stopwords: frozenset[str] = _SENTENCE_STARTER_STOPWORDS,
) -> list[str]:
    """Return proper nouns in ``new_text`` absent from ``prior_context``.

    Proper noun here = capitalized word of 3+ letters. Sentence-start
    capitalisation inevitably leaks in (``"Which"``, ``"But"``);
    ``stopwords`` set filters most common offenders. Tradeoff: real
    names matching a stopword (rare) get suppressed.
    """
    prior_lower = prior_context.lower()
    unknowns: list[str] = []
    seen: set[str] = set()
    for match in _PROPER_NOUN_RE.findall(new_text or ""):
        lowered = match.lower()
        if lowered in stopwords:
            continue
        if match in seen:
            continue
        seen.add(match)
        if lowered not in prior_lower:
            unknowns.append(match)
    return unknowns


def detect_silence_gap(
    *,
    prev_turn_at: datetime | None,
    current_turn_at: datetime,
    threshold_seconds: float = 300.0,
) -> float | None:
    """Return gap in seconds if it exceeds ``threshold_seconds``.

    Returns ``None`` if no prior turn exists or gap is below threshold.
    """
    if prev_turn_at is None:
        return None
    gap = (current_turn_at - prev_turn_at).total_seconds()
    if gap < threshold_seconds:
        return None
    return gap


# ---------------------------------------------------------------------------
# Combined emission
# ---------------------------------------------------------------------------


def log_signals(
    *,
    channel_id: int,
    turn_id: str,
    new_text: str,
    prior_context: str,
    prev_turn_at: datetime | None,
    current_turn_at: datetime,
    topic_shift_threshold: float = 0.15,
    topic_shift_min_tokens: int = 4,
    silence_gap_threshold_s: float = 300.0,
) -> dict[str, object]:
    """Run all detectors; emit one span per firing signal.

    Returns dict describing which signals fired. Callers that
    eventually wire signals into cache invalidation can inspect the
    return value; today it's informational only.
    """
    fired: dict[str, object] = {}
    if detect_topic_shift(
        new_text=new_text,
        prior_context=prior_context,
        min_overlap=topic_shift_threshold,
        min_tokens=topic_shift_min_tokens,
    ):
        fired["topic_shift"] = True
        _logger.info(
            f"{ls.tag('ColdCache', ls.LY)} "
            f"{ls.kv('signal', 'topic_shift', vc=ls.LY)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LC)} "
            f"{ls.kv('turn', turn_id, vc=ls.LC)}"
        )
    unknowns = detect_unknown_proper_noun(
        new_text=new_text, prior_context=prior_context
    )
    if unknowns:
        fired["unknown_proper_nouns"] = unknowns
        _logger.info(
            f"{ls.tag('ColdCache', ls.LY)} "
            f"{ls.kv('signal', 'unknown_proper_noun', vc=ls.LY)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LC)} "
            f"{ls.kv('turn', turn_id, vc=ls.LC)} "
            f"{ls.kv('nouns', ','.join(unknowns[:5]), vc=ls.LW)}"
        )
    gap = detect_silence_gap(
        prev_turn_at=prev_turn_at,
        current_turn_at=current_turn_at,
        threshold_seconds=silence_gap_threshold_s,
    )
    if gap is not None:
        fired["silence_gap_s"] = gap
        _logger.info(
            f"{ls.tag('ColdCache', ls.LY)} "
            f"{ls.kv('signal', 'silence_gap', vc=ls.LY)} "
            f"{ls.kv('channel', str(channel_id), vc=ls.LC)} "
            f"{ls.kv('turn', turn_id, vc=ls.LC)} "
            f"{ls.kv('gap_s', f'{gap:.1f}', vc=ls.LW)}"
        )
    return fired
