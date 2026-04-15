# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- `libopus` for Discord voice (`brew install opus`, `apt install libopus0`,
  `dnf install opus`, or `pacman -S opus`)
- A Discord application + bot token
  ([portal](https://discord.com/developers/applications)) with the
  `message_content`, `messages`, and `voice_states` intents enabled
- An OpenRouter API key
- *(optional, voice only)* Azure Cognitive Services key + region **or** a Cartesia API key

## Environment variables

Create a `.env` in the repo root (loaded automatically on startup).
Only install-wide secrets and the active-familiar selector live here;
everything tunable about the familiar — LLM model per call site, TTS
voice, chattiness — lives in `data/familiars/<id>/character.toml`.

```bash
# required
DISCORD_BOT=<discord bot token>
OPENROUTER_API_KEY=<openrouter key>

# pick the familiar to load (or pass --familiar on the CLI)
FAMILIAR_ID=aria

# optional — TTS provider selection (overrides [tts].provider in character.toml)
# Valid values: azure (default), cartesia
TTS_PROVIDER=azure

# Azure Speech credentials (required when TTS_PROVIDER=azure or provider="azure"):
AZURE_SPEECH_KEY=<azure cognitive services key>
AZURE_SPEECH_REGION=<azure region, e.g. eastus>

# Cartesia credentials (required when TTS_PROVIDER=cartesia or provider="cartesia"):
CARTESIA_API_KEY=<cartesia key>

# optional — Deepgram transcription secret (voice channels only)
DEEPGRAM_API_KEY=<deepgram key>
```

### Per-familiar model choice

LLM model selection is per-call-site and lives in the familiar's
`character.toml` under one `[llm.<slot>]` table per call site:
`main_prose`, `post_process_style`, `reasoning_context`,
`history_summary`, `memory_search`, and `interjection_decision`.

The checked-in reference profile at
`data/familiars/_default/character.toml` fills in every slot with
sensible defaults. A user's own `character.toml` only needs to
override the fields it wants to change — missing slots inherit from
the default profile on load. Copy the default to start a new
familiar:

```bash
cp -r data/familiars/_default data/familiars/my-familiar
# then edit data/familiars/my-familiar/character.toml
```

Good slot-level starting points for cheap / fast models (everything
except `main_prose`):

- `openai/gpt-4o-mini` — cheapest OpenAI, fast, honours the `name`
  field, strong structured-output for the content-search TOOL/ANSWER
  protocol.
- `anthropic/claude-3.5-haiku` — similar tier / price on the
  Anthropic side.
- `meta-llama/llama-3.1-8b-instruct` — very cheap via OpenRouter,
  decent for simple summarisation.

The startup log prints each slot's resolved model on every launch.

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
