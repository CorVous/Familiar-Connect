"""In-memory text-channel session registry.

The bot can be in at most one active session at a time (text or voice).
This module provides a single-slot registry so any part of the bot can
check or modify the current text session.

TODO: This registry is global and shared across all Discord guilds.
      Cross-guild isolation is out of scope for the prototype — add per-guild
      sessions when multi-server support is needed.

TODO: History is unbounded.  Add sliding-window trimming / summarisation per
      plan.md's context-management design before deploying to production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from familiar_connect.llm import Message


class SessionError(Exception):
    """Raised when a session operation is invalid (e.g. double-set)."""


@dataclass
class TextSession:
    """State for an active text-channel session."""

    channel_id: int
    system_prompt: str
    history: list[Message] = field(default_factory=list)


# Module-level single-slot registry.
_active: TextSession | None = None


def get_session() -> TextSession | None:
    """Return the current active session, or None."""
    return _active


def set_session(session: TextSession) -> None:
    """Register *session* as the active session.

    :raises SessionError: If a session is already active.
    """
    global _active  # noqa: PLW0603
    if _active is not None:
        msg = "A session is already active. Call clear_session() first."
        raise SessionError(msg)
    _active = session


def clear_session() -> None:
    """Clear the active session (no-op if none is set)."""
    global _active  # noqa: PLW0603
    _active = None
