"""ModeInstructionProvider — per-mode static instruction loader.

Reads ``data/familiars/<id>/modes/<mode_value>.md`` and emits its
contents as a ``Layer.author_note`` Contribution. Each ChannelMode can
ship its own static instruction file without editing Python code or
the per-character persona.

Constructed per turn with the active mode baked in. Missing file,
missing directory, and whitespace-only file all return no
contributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer

if TYPE_CHECKING:
    from pathlib import Path

    from familiar_connect.config import ChannelMode
    from familiar_connect.context.types import ContextRequest


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
    ) -> None:
        self._modes_root = modes_root
        self._mode = mode

    async def contribute(
        self,
        request: ContextRequest,  # noqa: ARG002
    ) -> list[Contribution]:
        """Return one Contribution per non-empty mode instruction file.

        Missing directory, missing file, and whitespace-only file
        all return an empty list.
        """
        path = self._modes_root / f"{self._mode.value}.md"
        if not path.is_file():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        text = raw.strip()
        if not text:
            return []

        return [
            Contribution(
                layer=Layer.author_note,
                priority=MODE_INSTRUCTION_PRIORITY,
                text=text,
                estimated_tokens=estimate_tokens(text),
                source=f"mode_instructions:{self._mode.value}",
            )
        ]
