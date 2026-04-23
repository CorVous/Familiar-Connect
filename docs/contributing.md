# Contributing

Development workflow and expectations for anyone working on Familiar-Connect.

## Environment setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Make sure `uv` is current before anything else:

```bash
uv self update
```

Install the project and its dev + docs dependency groups:

```bash
uv sync --dev --group docs
```

See [Installation](getting-started/installation.md) for the runtime prerequisites (libopus, Discord token, OpenRouter key, Cartesia key, etc.).

## TDD workflow

Always follow red / green TDD:

1. Write a failing test first (red).
2. Write the minimum code to make it pass (green).
3. Refactor if needed.

**Import errors do not count as red.** A test that fails due to an `ImportError` or `ModuleNotFoundError` is not a valid red test — the module or function under test must exist before the test can legitimately fail for the right reason.

## After every code change

Run the same four checks CI runs:

```bash
uv sync --dev                 # keep dependencies in sync
uv run ruff check             # lint
uv run ruff format            # format
uv run ty check               # type-check
uv run pytest                 # run the suite
```

These are cheap and fast on a clean working tree. If any fail locally, CI will fail the same way — fix the root cause before pushing.

## Docs build & preview

The docs site is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/). Local preview:

```bash
uv run mkdocs serve
```

Strict build (fails on broken internal links — this is what CI runs):

```bash
uv run mkdocs build --strict
```

## Commit style

- Conventional prefix (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) followed by an imperative description.
- Body explains the *why*, not the *what* — the diff already shows the what.
- Keep commits focused. A bug fix shouldn't drag a refactor with it; a refactor shouldn't drag a bug fix with it.

## Scope discipline

- Don't add features, refactor code, or "improve" things beyond what the task asked for. A bug fix doesn't need surrounding cleanup. A simple feature doesn't need extra configurability.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal guarantees. Validate only at system boundaries (user input, external APIs).
- Don't create helpers or abstractions for one-time operations. Three similar lines is better than a premature abstraction.
- Don't design for hypothetical future requirements. [Design decisions](architecture/decisions.md) is where rejected ideas live; before designing something new, check whether it's already there.

## Where things live

- [Architecture overview](architecture/overview.md) — the big picture, the pipeline diagram, the component map.
- Context pipeline — how providers, processors, and the budgeter fit together.
- Memory — the per-familiar memory directory and `MemoryStore`.
- [Configuration model](architecture/configuration-model.md) — the two-level config split, slash commands, on-disk layout.
- [Security](architecture/security.md) — credential storage, path-traversal defences, logging rules.
- [Design decisions](architecture/decisions.md) — ideas considered and rejected.
- Roadmap — planned work with per-item rationale.
