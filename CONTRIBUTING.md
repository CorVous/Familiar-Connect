# Contributing

This file is the working contributor reference — hot commands, the
post-change checklist, and code conventions. The pages under
[`docs/`](docs/index.md) are the authoritative deep dive:
[`docs/contributing.md`](docs/contributing.md) for the full human workflow,
[`docs/architecture/overview.md`](docs/architecture/overview.md) for the system.

## Project specifics

- This is a **`uv`** project. Run everything through `uv run` — there is no bare
  `python` on `PATH`. Update `uv` first: `uv self update`.
- Test collection imports the `local-turn` (numpy, huggingface_hub) and
  `local-embed` (fastembed) extras unconditionally, so run tests with both:
  `uv run --extra local-turn --extra local-embed pytest -q`. The live bot also
  runs from this repo's `.venv` and crashes at startup without `local-embed`, so
  keep both extras synced: `uv sync --dev --extra local-turn --extra local-embed`.

## After every change

Run `scripts/ci-local.sh` — it mirrors the CI `lint-and-test` job exactly (sync,
`ruff check`, `ruff format`, `ty check`, `pytest`), puts `uv` on `PATH`, and
re-syncs disk exec bits to git's modes first so `ruff` doesn't fire spurious
`EXE002`. Run the steps by hand when you need finer control.

Red / green TDD. One project nuance: **import errors don't count as red** — a
test failing on `ImportError` / `ModuleNotFoundError` is not a valid red; the
module or function must exist before the test can fail for the right reason.

If the change touched **env vars / config keys**, **on-disk layout under
`data/familiars/`**, or **architecture** (providers, processors, pipeline,
memory, history), update the matching page under `docs/` **in the same commit**,
then run `uv run mkdocs build --strict`. `tests/test_docs.py` fails CI if
documented env vars have drifted from code.

CLI flags and slash-command descriptions don't need a manual doc edit — the
`docs/hooks/cli_reference.py` mkdocs hook inlines them from `bot.py` / `cli.py`
at build time. Still run `mkdocs build --strict` after touching those; if you add
a *new* slash command or CLI subcommand, check it renders in the
`getting-started/slash-commands.md` / `getting-started/installation.md` pages.

## Conventions — technical writing

Comments, docstrings, and documentation:

- Be concise. Prefer **telegraphic** style.
    - Omit: articles ("the", "a"), auxiliary verbs, unnecessary prepositions, filler.
    - Keep: nouns, verbs, adjectives, key modifiers.
- Avoid restating the obvious — don't restate types already in signatures; don't
  summarize a function when its name says it.
- Document what's close and stable; avoid "far away" references likely to change
  (exception: ok if lints/tests/jobs catch the breakage).
- Capitalize the first word of a comment (PEP 8), unless it's an identifier that
  begins lowercase (`os.environ`, `self._x`, backticked code); continuation lines
  of a multi-line sentence stay lowercase. Periods only for full sentences. Full
  sentences only when needed; lean on context.

**Scope:** telegraphic style applies strictly to docstrings and inline comments.
Wiki pages (`docs/*.md`) keep full sentences for readability but stay concise —
trim wordiness, filler, restating.

## Conventions — logging

Adding a log call. Match existing style — don't invent a new one.

- Per module: `_logger = logging.getLogger(__name__)` at top. Never root.
- Compose with `from familiar_connect import log_style as ls`:
    - `ls.tag(label, color)` — leading `[label]`
    - `ls.kv(key, val, vc=color)` — `key=value` chunk
    - `ls.trunc(text, limit=200)` — ellipsis-truncate payloads
- Layout: one line, leading `ls.tag(...)` then space-separated `ls.kv(...)` pairs.
  `StyledFormatter` repaints the leading tag for `WARNING`/`ERROR` — keep the tag
  first. Example (`mood.py`):
  ```python
  _logger.info(
      f"{ls.tag('Mood', ls.M)} "
      f"{ls.kv('modifier', f'{modifier:+.2f}', vc=ls.LM)}"
  )
  ```
- Emoji: reserve for notable transitions (✨ summon, 🎙️ stream).
- One color per subsystem; stay consistent across that subsystem's logs.
