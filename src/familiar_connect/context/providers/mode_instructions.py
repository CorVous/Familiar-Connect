"""ModeInstructionProvider — per-mode static instruction loader.

Reads ``data/familiars/<id>/modes/<mode_value>.md`` and emits its
contents as a ``Layer.author_note`` Contribution. Each ChannelMode can
ship its own static instruction file without editing Python code or
the per-character persona.

Constructed per turn with the active mode baked in. Precedence (high → low):

1. ``channel_backdrop_override`` — per-channel text set via
   ``/channel-backdrop``; replaces the mode file entirely.
2. ``<modes_root>/<mode>.md`` — familiar-specific instruction file.
3. ``<defaults_modes_root>/<mode>.md`` — repo-wide fallback from
   ``data/familiars/_default/modes/``.
4. Nothing → no contribution.

Missing file, missing directory, and whitespace-only values all yield no
contribution at their tier and cause the next tier to be tried.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from familiar_connect import log_style as ls
from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.config import ChannelMode
    from familiar_connect.context.types import ContextRequest


_logger = logging.getLogger(__name__)


def _read_mode_file(root: Path, mode: ChannelMode) -> str | None:
    """Read ``<root>/<mode>.md``; return stripped text or ``None``."""
    path = root / f"{mode.value}.md"
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    stripped = raw.strip()
    return stripped or None


MODE_INSTRUCTION_PRIORITY = 80
"""Priority assigned to mode-instruction contributions.

Below ``CHARACTER_PRIORITY`` (100) so the persona always wins under
budget pressure, but high enough that the instruction survives ahead
of retrieved/searched content."""

_DEADLINE_S = 0.25
"""Deadline for the provider's contribute() call.

A single filesystem read against a small Markdown file should
complete in milliseconds; this is a generous upper bound."""


class ModeInstructionProvider:
    """ContextProvider that loads the mode's instruction file."""

    id = "mode_instructions"
    deadline_s = _DEADLINE_S

    def __init__(
        self,
        *,
        modes_root: Path,
        mode: ChannelMode,
        channel_backdrop_override: str | None = None,
        defaults_modes_root: Path | None = None,
    ) -> None:
        self._modes_root = modes_root
        self._mode = mode
        self._backdrop = channel_backdrop_override
        self._defaults_root = defaults_modes_root

    async def contribute(
        self,
        request: ContextRequest,  # noqa: ARG002
    ) -> list[Contribution]:
        """Return one Contribution per non-empty mode instruction.

        Walks the precedence chain: channel backdrop → familiar modes file
        → default modes file. Returns [] if nothing is set.
        """
        # 1. per-channel backdrop override
        if self._backdrop is not None:
            stripped = self._backdrop.strip()
            if stripped:
                self._log_source("channel-backdrop", self._mode.value)
                return [self._make(stripped, f"channel_backdrop:{self._mode.value}")]

        # 2. familiar-specific modes/<mode>.md
        text = _read_mode_file(self._modes_root, self._mode)
        if text is not None:
            path = self._modes_root / f"{self._mode.value}.md"
            self._log_source("familiar", str(path))
            return [self._make(text, f"mode_instructions:{self._mode.value}")]

        # 3. _default/modes/<mode>.md fallback
        if self._defaults_root is not None:
            text = _read_mode_file(self._defaults_root, self._mode)
            if text is not None:
                path = self._defaults_root / f"{self._mode.value}.md"
                self._log_source("default", str(path))
                return [
                    self._make(text, f"mode_instructions_default:{self._mode.value}")
                ]

        self._log_source("none", "(no file)")
        return []

    # ------------------------------------------------------------------

    def _log_source(self, tier: str, file: str) -> None:
        _logger.info(
            f"{ls.tag('🎬 Mode', ls.B)} "
            f"{ls.kv('mode', self._mode.value, vc=ls.LB)} "
            f"{ls.kv('tier', tier, vc=ls.LB)} "
            f"{ls.kv('file', file, vc=ls.LW)}"
        )

    def _make(self, text: str, source: str) -> Contribution:
        return Contribution(
            layer=Layer.author_note,
            priority=MODE_INSTRUCTION_PRIORITY,
            text=text,
            estimated_tokens=estimate_tokens(text),
            source=source,
        )


def resolve_mode_default(
    *,
    modes_root: Path,
    mode: ChannelMode,
    defaults_modes_root: Path | None = None,
) -> str | None:
    """Resolve the effective default-backdrop text for *mode*.

    Mirrors the provider's precedence chain but ignores any per-channel
    override: familiar's ``modes/<mode>.md`` first, then
    ``defaults_modes_root/<mode>.md``. Returns stripped text or ``None``.
    """
    for root in (modes_root, defaults_modes_root):
        if root is None:
            continue
        text = _read_mode_file(root, mode)
        if text is not None:
            return text
    return None
