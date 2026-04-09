"""ModeInstructionProvider — per-mode static instruction loader.

Reads ``data/familiars/<id>/modes/<mode_value>.md`` and emits its
contents as a :class:`Layer.author_note` Contribution. Each
:class:`ChannelMode` can ship its own static "how to write a reply
in this mode" instruction file — *this channel is in
``text_conversation_rp`` mode, so keep replies short and
chat-room-ish* — without editing Python code or the per-character
persona.

The provider is constructed per turn by
:meth:`Familiar.build_pipeline` with the active mode baked in at
construction time. This keeps the :class:`ContextProvider` protocol
unchanged (the request still doesn't need to carry a mode) at the
cost of a trivial per-turn allocation.

Missing file, missing ``modes/`` directory, and empty-or-whitespace
file all collapse to returning no contributions, so users can opt
in per mode by just dropping a file in.
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
    """ContextProvider that loads the mode's instruction file.

    Conforms structurally to the :class:`ContextProvider` protocol.

    :param modes_root: Directory that holds one ``<mode>.md`` file
        per :class:`ChannelMode`. Typically
        ``data/familiars/<familiar_id>/modes/``.
    :param mode: The :class:`ChannelMode` this provider instance is
        scoped to. One provider is constructed per turn inside
        :meth:`Familiar.build_pipeline`, so the mode is fixed for
        the lifetime of the provider.
    """

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
