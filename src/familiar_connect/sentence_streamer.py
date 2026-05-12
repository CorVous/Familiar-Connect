"""Sentence-boundary aggregator for streamed LLM output.

Sits between :meth:`LLMClient.chat_stream` and the TTS player. Buffers
deltas, emits whole sentences as soon as a terminator
(``.``/``!``/``?``) is followed by whitespace. Falls back to draining
the partial buffer on :meth:`flush` when the stream ends without a
final terminator. Drops time-to-first-audio from "after the LLM
finishes" to "after the first sentence". See ``voice-pipeline.md``
under ``docs/architecture/`` for the latency breakdown.

Abbreviation-aware: ``Mr.``/``Dr.``/``etc.``/initials don't trip a
boundary. Decision is local — no language model lookahead.
"""

from __future__ import annotations

# titles + low-risk Latin abbreviations. lowercase, no trailing dot.
# kept tight: false negatives (one extra sentence) are cheaper than
# false positives (mid-sentence flush of "Mr.").
_ABBREVIATIONS: frozenset[str] = frozenset({
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "sr",
    "jr",
    "prof",
    "rev",
    "fr",
    "etc",
    "vs",
    "no",
    "vol",
    "pg",
    "ft",
    "e.g",
    "i.e",
})

_TERMINATORS: frozenset[str] = frozenset({".", "!", "?"})


class SentenceStreamer:
    """Buffer streamed text, emit on sentence boundaries."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, delta: str) -> list[str]:
        """Append ``delta``; return zero or more completed sentences."""
        if not delta:
            return []
        self._buf += delta
        out: list[str] = []
        while True:
            split = self._try_split()
            if split is None:
                break
            head, rest = split
            out.append(head)
            self._buf = rest
        return out

    def flush(self) -> str:
        """Drain remaining buffer verbatim. Resets internal state."""
        out = self._buf
        self._buf = ""
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _try_split(self) -> tuple[str, str] | None:
        """Find earliest non-abbrev terminator + whitespace; split there."""
        buf = self._buf
        i = 0
        while i < len(buf):
            ch = buf[i]
            if ch not in _TERMINATORS:
                i += 1
                continue
            # eat consecutive terminators ("?!" → one boundary)
            end = i + 1
            while end < len(buf) and buf[end] in _TERMINATORS:
                end += 1
            if end >= len(buf):
                # punctuation at buffer end — wait for next delta
                return None
            if not buf[end].isspace():
                # ".5" / "1.0" / "?<tag>" — not a boundary
                i = end
                continue
            if "." in buf[i:end] and self._looks_like_abbreviation(i):
                i = end
                continue
            head = buf[:end]
            # consume one separator char so the next sentence starts clean
            rest_start = end
            while rest_start < len(buf) and buf[rest_start].isspace():
                rest_start += 1
            return head, buf[rest_start:]
        return None

    def _looks_like_abbreviation(self, dot_index: int) -> bool:
        """``dot_index`` indexes a ``.`` in ``self._buf``. Walk back the token."""
        buf = self._buf
        # collect the token immediately preceding the dot, allowing inner
        # dots so "e.g" / "i.e" round-trip ("e.g." ends at second dot).
        j = dot_index - 1
        while j >= 0 and (buf[j].isalpha() or buf[j] == "."):
            j -= 1
        token = buf[j + 1 : dot_index].lower()
        if not token:
            return False
        if token in _ABBREVIATIONS:
            return True
        # single-letter initial: "J. K. Rowling"
        return len(token) == 1 and token.isalpha()
