# Claude Notes

## First Step

Update uv before anything else: `uv self update`

## Key Documents

- **[`docs/index.md`](./docs/index.md)** — full documentation landing page.
- **[`docs/architecture/overview.md`](./docs/architecture/overview.md)** — architecture, context pipeline, memory directory, configuration model, security, design decisions.
- **[`docs/guides/bootstrapping.md`](./docs/guides/bootstrapping.md)** — one-shot operator utilities (character-card unpacker, lorebook importer) that are never invoked by the runtime reply pipeline.
- **[`docs/contributing.md`](./docs/contributing.md)** — full dev workflow (the bullets below are the TL;DR).

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

## Banned-API Rule

Runtime modules (`bot.py`, `familiar.py`, `commands/run.py`, anything in the context pipeline) must never import from `familiar_connect.bootstrap`. The bootstrap subpackage is operator-only. Ruff enforces this via `[tool.ruff.lint.flake8-tidy-imports.banned-api]`. See [`docs/guides/bootstrapping.md`](./docs/guides/bootstrapping.md) for the rationale.
