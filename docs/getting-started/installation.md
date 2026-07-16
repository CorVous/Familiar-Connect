# Installation

## Prerequisites

- A Rust toolchain — install via [rustup](https://rustup.rs/); the pinned
  stable version in `rust-toolchain.toml` is fetched automatically on first
  build.
- A Discord application + bot token
  ([portal](https://discord.com/developers/applications)) with the
  `message_content`, `messages`, and `voice_states` intents enabled
- An OpenRouter API key
- *(optional, voice only)* One of: Azure Cognitive Services key + region, a Cartesia API key, or a Google Gemini API key
- *(optional, voice only)* A Deepgram API key for speech transcription
- *(voice builds only)* CMake — the `discord-voice` feature compiles libopus
  from source (`songbird → opus2 → libopus_sys`). Windows especially needs it
  installed; text-only builds (`--features discord`) never need it. See the
  [DAVE runbook](../rust-port/DAVE-RUNBOOK.md) for the full voice
  prerequisites and platform notes.

## Environment variables

Create a `.env` in the repo root (loaded on startup). Only install-wide
secrets and the active-familiar selector live here; everything tunable
about the familiar — LLM model, TTS voice — lives in
`data/familiars/<id>/character.toml`.

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

LLM model selection is per-call-site, in the familiar's `character.toml`
under `[llm.<slot>]` tables. Three tiered slots ship today: `fast`
(voice), `prose` (text replies), and `background` (summaries / fact
extraction / dossiers). See [Tuning — LLM
slots](../architecture/tuning.md#llm-slots) for the schema.

The reference profile at `data/familiars/_default/character.toml`
fills each slot with a sensible default. A user's own `character.toml`
only overrides the fields it wants to change. Copy the default to
start a new familiar:

```bash
cp -r data/familiars/_default data/familiars/my-familiar
# then edit data/familiars/my-familiar/character.toml
```

## Start

Build with the feature flags for the surfaces you want (see below), then
run:

```bash
cargo run --release --features discord -- run --familiar aria
cargo run --release --features discord -- run --familiar aria -v
```

`run` resolves the active familiar via `--familiar` first, then
`FAMILIAR_ID`. `-v` / `-vv` / `-vvv` tune logging verbosity — a global flag,
accepted before or after `run`.

### Feature flags

Feature flags select the integration surface; the defaults cover storage,
HTTP, and images:

```bash
# Text-only Discord bot
cargo build --release --features discord

# Voice (DAVE E2EE via songbird) + Deepgram streaming STT
cargo build --release --features discord,discord-voice,stt-deepgram

# Local ML extras (ONNX turn detection, local embeddings)
cargo build --release --features local-turn,local-embed
```

## CLI reference

Run `familiar-connect --help` (or `cargo run -- --help`) locally for the
same output.

```text
familiar-connect CLI tool

Usage: familiar-connect [OPTIONS] [COMMAND]

Commands:
  run       Start the Discord bot
  diagnose  Aggregate span timings from log files
  version   Display package version
  help      Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...  Increase verbosity (can be repeated: -v, -vv, -vvv)
  -h, --help        Print help
  -V, --version     Print version
```

```text
Start the Discord bot

Usage: familiar-connect run [OPTIONS]

Options:
      --familiar <ID>  Folder name of the character to run (under `data/familiars/`). Overrides `FAMILIAR_ID`
  -v, --verbose...     Increase verbosity (can be repeated: -v, -vv, -vvv)
  -h, --help           Print help
```

Once the bot is online, see [Slash commands](slash-commands.md) for the
subscription surface.
