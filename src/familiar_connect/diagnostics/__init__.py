"""Diagnostics: timing spans + (Phase 5) aggregation surface.

Logs-first: spans emit via :mod:`familiar_connect.log_style` so
``grep span=llm.chat`` is the initial aggregator. A collector and the
``/diagnostics`` slash command come in Phase 5. See plan § Design.9.
"""

from __future__ import annotations

from familiar_connect.diagnostics.spans import span

__all__ = ["span"]
