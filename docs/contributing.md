# Contributing

Dev workflow and expectations for Familiar-Connect.

## Environment setup

Dependencies via [uv](https://docs.astral.sh/uv/). Update `uv` first:

```bash
uv self update
```

Install project + dev + docs groups:

```bash
uv sync --dev --group docs
```

See [Installation](getting-started/installation.md) for runtime prerequisites (libopus, Discord token, OpenRouter key, Cartesia key, etc.).

## TDD workflow

Red / green TDD:

1. Failing test first (red).
2. Minimum code to pass (green).
3. Refactor if needed.

**Import errors don't count as red.** A test failing on `ImportError` / `ModuleNotFoundError` is not a valid red — the module or function must exist before the test can fail for the right reason.

## After every code change

Run the same four checks CI runs:

```bash
uv sync --dev                 # sync deps
uv run ruff check             # lint
uv run ruff format            # format
uv run ty check               # type-check
uv run pytest                 # run suite
```

Cheap and fast on a clean tree. Local failures will fail CI the same way — fix root cause before pushing.

Tests marked `@pytest.mark.integration` hit live services (e.g. OpenRouter) and skip by default. Run explicitly:

```bash
uv run pytest -m integration  # needs OPENROUTER_API_KEY etc. in env
```

## Docs build & preview

Built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/). Local preview:

```bash
uv run mkdocs serve
```

Strict build (fails on broken internal links — what CI runs):

```bash
uv run mkdocs build --strict
```

## Commit style

- Conventional prefix (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) + imperative description.
- Body explains *why*, not *what* — the diff shows the what.
- Keep commits focused. Bug fix shouldn't drag a refactor; refactor shouldn't drag a bug fix.

## Scope discipline

- No features, refactors, or "improvements" beyond the task. Bug fixes don't need surrounding cleanup. Simple features don't need extra configurability.
- No error handling, fallbacks, or validation for scenarios that can't happen. Trust internal guarantees. Validate only at system boundaries (user input, external APIs).
- No helpers / abstractions for one-time ops. Three similar lines beats a premature abstraction.
- No design for hypothetical future requirements. [Design decisions](architecture/decisions.md) holds rejected ideas — check there first.

## Where things live

- [Architecture overview](architecture/overview.md) — big picture, component map.
- [Configuration model](architecture/configuration-model.md) — two-level config split, on-disk layout.
- [Security](architecture/security.md) — credential storage, logging rules.
- [Design decisions](architecture/decisions.md) — rejected ideas.
