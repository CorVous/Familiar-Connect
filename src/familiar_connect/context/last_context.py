"""Per-channel cache of the last LLM context window.

Refreshed on every response (text, voice, voice-regen); consumed by
``/context``. Sibling markdown file next to ``channel_config.py``'s TOML
sidecars under ``data/familiars/<id>/channels/``. The on-disk file is
byte-identical to the ``context.md`` attachment that ``/context`` posts.

Read/write errors are swallowed — debug artifact, never raise into the
request path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from familiar_connect.llm import Message

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Entry:
    """Internal helper passed to :func:`_render_markdown`."""

    messages: tuple[Message, ...]
    captured_at: datetime
    modality: str


class LastContextCache:
    """File-backed cache keyed by channel id."""

    def __init__(self, *, channels_root: Path) -> None:
        self._root = channels_root

    def put(
        self,
        *,
        channel_id: int,
        messages: Sequence[Message],
        modality: str,
    ) -> None:
        """Atomically write *messages* as rendered markdown for *channel_id*.

        Swallows write errors — never breaks the response path.
        """
        path = self._path(channel_id)
        entry = _Entry(
            messages=tuple(messages),
            captured_at=datetime.now(UTC),
            modality=modality,
        )
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(_render_markdown(entry), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            _logger.warning("last_context.put failed: %s: %s", type(exc).__name__, exc)

    def get(self, *, channel_id: int) -> bytes | None:
        """Return cached markdown bytes for *channel_id*, or ``None`` if missing."""
        path = self._path(channel_id)
        if not path.exists():
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            _logger.warning("last_context.get failed: %s: %s", type(exc).__name__, exc)
            return None

    def _path(self, channel_id: int) -> Path:
        return self._root / f"{channel_id}.last-context.md"


def _render_markdown(entry: _Entry) -> str:
    """Format *entry* as a ``context.md`` attachment body."""
    header = (
        f"# Last context "
        f"(captured {entry.captured_at.isoformat()}, "
        f"modality={entry.modality}, "
        f"{len(entry.messages)} messages)"
    )
    parts: list[str] = [header]
    for i, msg in enumerate(entry.messages):
        suffix = f" ({msg.name})" if msg.name else ""
        parts.append(f"## [{i}] {msg.role}{suffix}\n\n{msg.content}")
    return "\n\n---\n\n".join(parts) + "\n"
