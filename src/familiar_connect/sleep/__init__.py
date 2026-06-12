"""Sleep cycle — nightly memory consolidation + (future) dream pass.

V1 ships memory hygiene only: a dry-run-first CLI pass that proposes
fact retirements/rewrites over the day's window, validates every
proposal against safety rails in code, and emits an audit artifact.
See ``docs/architecture/sleep.md``.
"""
