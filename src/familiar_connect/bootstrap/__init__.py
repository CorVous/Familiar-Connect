"""One-shot bootstrapping utilities for operators.

Nothing in this package is imported by the runtime reply pipeline
(``bot.py``, ``familiar.py``, ``commands/run.py``). Modules here
translate external assets — character cards, SillyTavern lorebooks,
etc. — into files under a familiar's :class:`MemoryStore`. They are
expected to be invoked once per familiar by a human operator and
then never touched again.

The dependency direction is one-way: this package may import from
:mod:`familiar_connect.memory.store`, but the reverse is forbidden.
A ruff ``flake8-tidy-imports`` ``banned-api`` rule in
``pyproject.toml`` enforces the invariant — runtime modules that
accidentally import from :mod:`familiar_connect.bootstrap` will
fail lint. See ``bootstrapping.md`` at the repo root for the
operator-facing how-to.
"""
