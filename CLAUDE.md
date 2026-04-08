# Claude Notes

## First Step

Update uv before anything else: `uv self update`

## Key Documents

- **`plan.md`** — contains the project plan and feature roadmap
- **`README.md`** — contains project overview and development commands

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