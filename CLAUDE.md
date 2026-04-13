# Claude Notes

## First Step

Update uv before anything else: `uv self update`

## Key Documents

- **[`docs/index.md`](./docs/index.md)** — full documentation landing page.
- **[`docs/architecture/overview.md`](./docs/architecture/overview.md)** — architecture, context pipeline, memory directory, configuration model, security, design decisions.
- **[`docs/contributing.md`](./docs/contributing.md)** — full dev workflow (the bullets below are the TL;DR).
- **[`tests/test_docs.py`](./tests/test_docs.py)** — automated doc-drift checks. If a test here fails in CI, documentation and implementation have diverged and must be reconciled in the same PR.

Read the docs when evaluating current state of implementation and roadmap.
Always update docs after feature implemented.
Update relevant doc after bugfix where necessary.

## TDD Workflow

Always follow red/green TDD:
1. Write a failing test first (red)
2. Write the minimum code to make it pass (green)
3. Refactor if needed

**Import errors do not count as red.** A test that fails due to an `ImportError` or `ModuleNotFoundError` is not a valid red test — the module/function must exist before the test can legitimately fail for the right reason.

## After Every Code Assignment

1. Run `uv sync --dev` to keep dependencies up to date
2. Run `uv run ruff check` to lint
3. Run `uv run ruff format` to format
4. Run `uv run ty check` to type-check
5. Run `uv run pytest` to run tests
6. If the change touched any of:
   - environment variables or config keys
   - on-disk layout under `data/familiars/`
   - architecture (providers, processors, pipeline, memory, history)

   update the matching page under `docs/` **in the same commit**, then run
   `uv run mkdocs build --strict` locally. `tests/test_docs.py` will fail
   CI if documented env vars have drifted from code.

   CLI flags and slash-command descriptions don't need a manual doc
   edit: the `docs/hooks/cli_reference.py` mkdocs hook inlines them
   from `bot.py` and `cli.py` at build time. Still run
   `uv run mkdocs build --strict` after touching those, and if you
   add a *new* slash command or CLI subcommand, check that it shows
   up in the rendered `getting-started/slash-commands.md` /
   `getting-started/installation.md` pages.

## Technical Writing Style

All comments, docstrings, and documentation must follow this style:

* Be concise
* Prefer **telegraphic** style
    * Omit: articles ("the", "a"), auxiliary verbs, unnecessary prepositions, filler words
    * Keep: nouns, verbs, adjectives, key modifiers
* Avoid restating the obvious
    * Don't restate types already in signatures
    * Don't summarize functions when the name is self-explanatory
* Document what's close and stable
    * Avoid "far away" references likely to change
    * Exception: ok if lints/tests/jobs catch breakage
* Start inline comments lowercase
* Use periods only for full sentences
* Use full sentences only when needed; lean on context

**Scope:** telegraphic style applies strictly to docstrings and inline
comments. Wiki pages (`docs/*.md`) keep full sentences for readability
but should still be concise — trim wordiness, filler, and restating.
