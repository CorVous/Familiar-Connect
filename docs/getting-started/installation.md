# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- `libopus` for Discord voice (`brew install opus`, `apt install libopus0`,
  `dnf install opus`, or `pacman -S opus`)
- A Discord application + bot token
  ([portal](https://discord.com/developers/applications)) with the
  `message_content`, `messages`, and `voice_states` intents enabled
- An OpenRouter API key
- *(optional, voice only)* One of: Azure Cognitive Services key + region, a Cartesia API key, or a Google Gemini API key
- *(optional, voice only)* A Deepgram API key for speech transcription

### Windows: C++ build tools required for `pyturso`

`pyturso` (the Rust-backed SQLite replacement that backs the history
store) does **not** ship Windows wheels — `uv sync` falls back to a
source build, which needs a Rust toolchain *and* the MSVC linker
(`link.exe`). uv's bootstrap installs Rust automatically; the linker
is on you.

Install the **Visual Studio Build Tools** with the *"Desktop
development with C++"* workload (this provides `link.exe` and the
Windows SDK):

- Download:
  <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
- In the installer, check **Desktop development with C++**. The
  default selection (MSVC, Windows SDK) is sufficient.
- Open a fresh PowerShell after install so `link.exe` is on `PATH`,
  then re-run `uv sync --dev`.

VS Code alone is *not* sufficient — the installer is the standalone
Build Tools package, no full Visual Studio required. The first
`uv sync` after installing the toolchain will compile pyturso from
source (one-time, a few minutes); subsequent syncs reuse the cached
wheel.

WSL/Ubuntu is the supported alternative if you'd rather not install
the MSVC toolchain — pyturso ships a Linux x86_64 wheel.

## Environment variables

Create a `.env` in the repo root (loaded automatically on startup).
Only install-wide secrets and the active-familiar selector live here;
everything tunable about the familiar — LLM model, TTS voice — lives
in `data/familiars/<id>/character.toml`.

```bash
# required
DISCORD_BOT=<discord bot token>
OPENROUTER_API_KEY=<openrouter key>

# pick the familiar to load (or pass --familiar on the CLI)
FAMILIAR_ID=aria

# TTS credentials — set the one matching [tts].provider in character.toml

# Azure Speech (default provider):
AZURE_SPEECH_KEY=<azure cognitive services key>
AZURE_SPEECH_REGION=<azure region, e.g. eastus>

# Cartesia (provider="cartesia"):
CARTESIA_API_KEY=<cartesia key>

# Google Gemini TTS (provider="gemini"):
GOOGLE_API_KEY=<google ai studio key>

# optional — Deepgram speech transcription (voice channels only)
DEEPGRAM_API_KEY=<deepgram key>
```

### Per-familiar model choice

LLM model selection is per-call-site and lives in the familiar's
`character.toml` under `[llm.<slot>]` tables. Three tiered slots
ship today: `fast` (voice), `prose` (text replies), and `background`
(summaries / fact extraction / dossiers). See
[Tuning — LLM slots](../architecture/tuning.md#llm-slots) for the
schema.

The checked-in reference profile at
`data/familiars/_default/character.toml` fills in the slot with a
sensible default. A user's own `character.toml` only needs to
override the fields it wants to change. Copy the default to start a
new familiar:

```bash
cp -r data/familiars/_default data/familiars/my-familiar
# then edit data/familiars/my-familiar/character.toml
```

## Start

```bash
uv sync --dev
uv run familiar-connect run
uv run familiar-connect run --familiar aria
uv run familiar-connect -vv run --familiar aria
```

The `run` subcommand resolves the active familiar via `--familiar`
first, then `FAMILIAR_ID`. `-v` / `-vv` / `-vvv` tune logging
verbosity.

## CLI reference

The help text below is generated at build time from the argparse
parser in `src/familiar_connect/cli.py`. Run `uv run familiar-connect
--help` locally to get the same output.

<!-- @cli-help: familiar-connect -->

<!-- @cli-help: familiar-connect run -->

Once the bot is online, see [Slash commands](slash-commands.md) for
the subscription surface.
