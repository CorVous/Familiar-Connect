"""Per-channel cache of the last LLM context window.

Refreshed on every response (text, voice, voice-regen); consumed by
``/context``. Sibling JSON next to ``channel_config.py``'s TOML sidecars
under ``data/familiars/<id>/channels/``.

Read/write errors are swallowed — debug artifact, never raise into the
request path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect.llm import Message

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LastContextEntry:
    """One cached render of ``assemble_chat_messages`` output."""

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
        """Atomically write *messages* for *channel_id*.

        Swallows write errors — never breaks the response path.
        """
        path = self._path(channel_id)
        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "modality": modality,
            "messages": [m.to_dict() for m in messages],
        }
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            tmp.replace(path)
        except OSError as exc:
            _logger.warning("last_context.put failed: %s: %s", type(exc).__name__, exc)

    def get(self, *, channel_id: int) -> LastContextEntry | None:
        """Return cached entry for *channel_id*, or ``None`` if missing/corrupt."""
        path = self._path(channel_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            messages = tuple(
                Message(
                    role=m["role"],
                    content=m["content"],
                    name=m.get("name"),
                )
                for m in raw["messages"]
            )
            return LastContextEntry(
                messages=messages,
                captured_at=datetime.fromisoformat(raw["captured_at"]),
                modality=raw["modality"],
            )
        except (OSError, ValueError, KeyError) as exc:
            _logger.warning("last_context.get failed: %s: %s", type(exc).__name__, exc)
            return None

    def _path(self, channel_id: int) -> Path:
        return self._root / f"{channel_id}.last-context.json"


def render_markdown(entry: LastContextEntry) -> str:
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
