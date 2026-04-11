# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- `libopus` for Discord voice (`brew install opus`, `apt install libopus0`,
  `dnf install opus`, or `pacman -S opus`)
- A Discord application + bot token
  ([portal](https://discord.com/developers/applications)) with the
  `message_content`, `messages`, and `voice_states` intents enabled
- An OpenRouter API key
- *(optional, voice only)* a Cartesia API key

## Environment variables

Create a `.env` in the repo root (loaded automatically on startup):

```bash
# required
DISCORD_BOT=<discord bot token>
OPENROUTER_API_KEY=<openrouter key>

# pick the familiar to load (or pass --familiar on the CLI)
FAMILIAR_ID=aria

# optional overrides for the main reply-path model
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_TEMPERATURE=0.8

# optional — cheaper model used for side-model work.
# Stepped thinking, recast, history summary, and content search all
# run through the side model slot. If you leave this unset, those
# calls reuse OPENROUTER_MODEL — which works but can be slow and
# expensive, especially because the content-search agent runs up to
# 5 side-model calls per turn. Set it to a fast, cheap model to blunt
# the cost hit.
OPENROUTER_SIDE_MODEL=openai/gpt-4o-mini
OPENROUTER_SIDE_TEMPERATURE=0.5

# optional — voice output
CARTESIA_API_KEY=<cartesia key>
CARTESIA_VOICE_ID=<voice id>
CARTESIA_MODEL=sonic-english
```

### Picking a side model

The side model is used for focused sub-tasks where accuracy matters
less than latency and cost. Good starting points:

- `openai/gpt-4o-mini` — cheapest OpenAI, fast, honours the `name`
  field, strong structured-output for the content-search TOOL/ANSWER
  protocol.
- `anthropic/claude-3.5-haiku` — similar tier / price on the
  Anthropic side.
- `meta-llama/llama-3.1-8b-instruct` — very cheap via OpenRouter,
  decent for simple summarisation.

The startup log prints which side model is in use on every launch, or
warns if you left it unset.

## Start

```bash
uv sync --dev
uv run familiar-connect run
uv run familiar-connect run --familiar aria
uv run familiar-connect -vv run --familiar aria
```

The `run` subcommand resolves the active familiar via `--familiar`
first, then `FAMILIAR_ID`. `-v` / `-vv` / `-vvv` tune logging
verbosity — `-vv` is the sweet spot for smoke tests.

## CLI reference

The help text below is generated at build time from the argparse
parser in `src/familiar_connect/cli.py`. Run `uv run familiar-connect
--help` locally to get the same output.

<!-- @cli-help: familiar-connect -->

<!-- @cli-help: familiar-connect run -->

Once the bot is online, see [Slash commands](slash-commands.md) for
the smoke-test surface.
