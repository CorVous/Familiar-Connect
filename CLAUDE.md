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
   - CLI flags or slash commands
   - on-disk layout under `data/familiars/`
   - architecture (providers, processors, pipeline, memory, history)

   update the matching page under `docs/` **in the same commit**, then run
   `uv run mkdocs build --strict` locally. `tests/test_docs.py` will fail
   CI if documented env vars or slash commands have drifted from code.
