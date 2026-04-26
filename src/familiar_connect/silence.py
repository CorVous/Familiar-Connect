"""Silent-sentinel detection for LLM reply streams.

The system prompt instructs the model to emit ``<silent>`` as its
entire reply when it shouldn't speak. :class:`SilentDetector` watches
a streamed reply delta-by-delta and decides as early as possible
whether the model is staying silent — so the responder can abort
before paying for downstream costs (Discord post, TTS synthesis).

Decision is *prefix-only*: a stray ``<silent>`` mid-reply is treated
as content, not a gate.
"""

from __future__ import annotations

SILENT_TOKEN = "<silent>"  # noqa: S105 — model output sentinel, not a credential


class SilentDetector:
    """Streaming inspector for the silent sentinel.

    Feed deltas with :meth:`feed`; it returns ``True`` once the
    leading non-whitespace content matches :data:`SILENT_TOKEN`,
    ``False`` once a mismatch is certain, or ``None`` while still
    undecided. Decision sticks — once final, further calls return
    the same value without inspecting their argument.
    """

    __slots__ = ("_buf", "_decided")

    def __init__(self) -> None:
        self._buf: str = ""
        self._decided: bool | None = None

    @property
    def decided(self) -> bool | None:
        """Latched decision: ``True`` silent, ``False`` speak, ``None`` pending."""
        return self._decided

    def feed(self, delta: str) -> bool | None:
        if self._decided is not None:
            return self._decided
        self._buf += delta
        stripped = self._buf.lstrip()
        if stripped.startswith(SILENT_TOKEN):
            self._decided = True
            return True
        if len(stripped) >= len(SILENT_TOKEN):
            # enough non-whitespace seen to rule out the sentinel
            self._decided = False
            return False
        if stripped and not SILENT_TOKEN.startswith(stripped):
            # diverged before reaching full length
            self._decided = False
            return False
        return None
