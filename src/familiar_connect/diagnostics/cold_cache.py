"""Cold-cache signals — research-phase instrumentation.

The plan treats "is our retrieval stale?" as an open question with
multiple candidate signals. Phase 3 ships these as *spans* only —
they log when they fire, but do not drive cache invalidation. After
collecting enough data to correlate signals with "had to redo work"
outcomes, the most-useful ones will get wired into the layer
invalidation logic.

Signals shipped:

- :func:`detect_topic_shift` — tiny shared-vocabulary test between
  the inbound turn and the current summary.
- :func:`detect_unknown_proper_noun` — a capitalized token in the
  inbound turn absent from the prior context.
- :func:`detect_silence_gap` — wall-clock gap between turns above a
  threshold.

``log_signals`` wraps all three and emits one span per firing
signal, tagged with the inbound turn's channel id for traceability.
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


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def detect_topic_shift(
    *, new_text: str, prior_context: str, min_overlap: float = 0.15
) -> bool:
    """Detect when ``new_text`` shares too few content words with prior.

    Jaccard overlap below ``min_overlap`` is treated as a shift. The
    default 0.15 is intentionally permissive — we want the signal to
    fire often so we can see how it correlates with retrieval
    failures.
    """
    new_tokens = _tokens(new_text)
    old_tokens = _tokens(prior_context)
    if not new_tokens or not old_tokens:
        return False
    overlap = len(new_tokens & old_tokens) / len(new_tokens | old_tokens)
    return overlap < min_overlap


def detect_unknown_proper_noun(*, new_text: str, prior_context: str) -> list[str]:
    """Return proper nouns in ``new_text`` absent from ``prior_context``.

    Proper noun here = capitalized word of 3+ letters. Sentence-start
    capitalisation inevitably leaks in (``"Tomorrow"``) — that's fine
    for Phase-3 data collection; we'd refine with an NER pass if the
    signal proves useful.
    """
    prior_lower = prior_context.lower()
    unknowns: list[str] = []
    seen: set[str] = set()
    for match in _PROPER_NOUN_RE.findall(new_text or ""):
        if match in seen:
            continue
        seen.add(match)
        if match.lower() not in prior_lower:
            unknowns.append(match)
    return unknowns


def detect_silence_gap(
    *,
    prev_turn_at: datetime | None,
    current_turn_at: datetime,
    threshold_seconds: float = 300.0,
) -> float | None:
    """Return the gap in seconds if it exceeds ``threshold_seconds``.

    Returns ``None`` if no prior turn exists or the gap is below the
    threshold.
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
    silence_gap_threshold_s: float = 300.0,
) -> dict[str, object]:
    """Run all detectors; emit one span per firing signal.

    Returns a dict describing which signals fired. Callers that
    eventually wire signals into cache invalidation can inspect the
    return value; today it's informational only.
    """
    fired: dict[str, object] = {}
    if detect_topic_shift(
        new_text=new_text,
        prior_context=prior_context,
        min_overlap=topic_shift_threshold,
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
