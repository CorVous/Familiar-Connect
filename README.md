# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users, understands speech, and talks back using real AI voices.

**Full documentation lives at [`docs/`](./docs/index.md)** — or browse the rendered site via `uv run mkdocs serve`.

## Quickstart

```bash
uv sync --dev
cp .env.example .env                           # fill in your tokens
FAMILIAR_ID=aria uv run familiar-connect run
```

See [Installation](./docs/getting-started/installation.md) for prerequisites (libopus, Discord bot token, OpenRouter key, optional TTS / STT keys) and the env var reference. See [On-disk layout](./docs/getting-started/on-disk-layout.md) for the minimum `data/familiars/<id>/` shape.

## Where things are

- **[Getting started](./docs/getting-started/installation.md)** — install, run, troubleshoot.
- **[Architecture](./docs/architecture/overview.md)** — the bot shell, configuration model, security, design decisions.
- **[Contributing](./docs/contributing.md)** — dev workflow, TDD expectations, docs build.

## Development commands

```bash
uv sync --dev                 # install + dev deps
uv run ruff check             # lint
uv run ruff format            # format
uv run ty check               # type-check
uv run pytest                 # run tests
uv run mkdocs serve           # preview the docs site
```
