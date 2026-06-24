"""Sleep cycle — nightly memory consolidation + opinion-formation pass.

Consolidation proposes fact retirements/rewrites over the day's window
and validates every proposal against safety rails in code; the
opinion-formation pass forms the familiar's opinions. Rail-blocked
proposals log as warnings. See ``docs/architecture/sleep.md``.
"""
