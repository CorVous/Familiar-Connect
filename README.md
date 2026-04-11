# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users, understands speech, and talks back using real AI voices.

**Full documentation lives at [`docs/`](./docs/index.md)** — or browse the rendered site via `uv run mkdocs serve`.

## Quickstart

```bash
uv sync --dev
cp .env.example .env                           # fill in your tokens
FAMILIAR_ID=aria uv run familiar-connect run
```

See [Installation](./docs/getting-started/installation.md) for the full prerequisites (libopus, Discord bot token, OpenRouter key, optional Cartesia key), env var reference, and side-model picking advice. See [On-disk layout](./docs/getting-started/on-disk-layout.md) for the minimum `data/familiars/<id>/` shape and `character.toml` examples.

## Where things are

- **[Getting started](./docs/getting-started/installation.md)** — install, run, smoke-test, troubleshoot.
- **[Architecture](./docs/architecture/overview.md)** — the context pipeline, memory directory, configuration model, security, design decisions.
- **[Guides](./docs/guides/bootstrapping.md)** — bootstrapping from SillyTavern character cards and lorebooks, Twitch integration.
- **[Roadmap](./docs/roadmap/index.md)** — planned features (chattiness, voice input, session logging, web search, …).
- **[Contributing](./docs/contributing.md)** — dev workflow, TDD expectations, the ruff `banned-api` bootstrap-to-runtime rule, docs build.

## Development commands

```bash
uv sync --dev                 # install + dev deps
uv run ruff check             # lint
uv run ruff format            # format
uv run ty check               # type-check
uv run pytest                 # run tests
uv run mkdocs serve           # preview the docs site
```
