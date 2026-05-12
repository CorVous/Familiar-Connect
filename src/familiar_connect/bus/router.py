"""Per-session turn routing + cancel-prior-scope semantics.

Owns ``active_turn_by_session``. On :meth:`begin_turn` for a session
that has an active scope, the prior scope is cancelled before the new
one is registered. See plan § Design.3 *Turn identity and
interruption*.
"""

from __future__ import annotations

import asyncio

from familiar_connect.bus.envelope import TurnScope


class TurnRouter:
    """Cancels prior turn in same session; isolates across sessions."""

    def __init__(self) -> None:
        self._active: dict[str, TurnScope] = {}

    def begin_turn(self, *, session_id: str, turn_id: str) -> TurnScope:
        """Cancel any active turn in ``session_id``; register a new one."""
        prior = self._active.get(session_id)
        if prior is not None:
            prior.cancel()
        scope = TurnScope(
            turn_id=turn_id,
            session_id=session_id,
            started_at=asyncio.get_event_loop().time()
            if self._has_running_loop()
            else 0.0,
        )
        self._active[session_id] = scope
        return scope

    def end_turn(self, scope: TurnScope) -> None:
        """Clear ``scope`` from active. No-op if already superseded."""
        active = self._active.get(scope.session_id)
        if active is scope:
            del self._active[scope.session_id]

    def active_scope(self, session_id: str) -> TurnScope | None:
        return self._active.get(session_id)

    def shutdown(self) -> None:
        """Cancel every active turn.

        Does not clear the map — caller inspects post-shutdown state
        for diagnostics.
        """
        for scope in self._active.values():
            scope.cancel()

    @staticmethod
    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True
