# Familiar-Connect

An AI "familiar" that joins Discord voice channels, listens to users, understands speech, and talks back using real AI voices.

## Running the bot

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- `libopus` for Discord voice (`brew install opus`, `apt install libopus0`, `dnf install opus`, or `pacman -S opus`)
- A Discord application + bot token ([portal](https://discord.com/developers/applications)) with the `message_content`, `messages`, and `voice_states` intents enabled
- An OpenRouter API key
- *(optional, voice only)* a Cartesia API key

### Environment variables

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

**Picking a side model.** The side model is used for focused sub-tasks where accuracy matters less than latency and cost. Good starting points:

- `openai/gpt-4o-mini` — cheapest OpenAI, fast, honours the `name` field (see below), strong structured-output for the content-search TOOL/ANSWER protocol.
- `anthropic/claude-3.5-haiku` — similar tier / price on the Anthropic side.
- `meta-llama/llama-3.1-8b-instruct` — very cheap via OpenRouter, decent for simple summarisation.

The startup log prints which side model is in use on every launch, or warns if you left it unset.

### Minimum on-disk layout

The bot loads exactly one character per process, picked by `FAMILIAR_ID` (or `--familiar <id>`), from `data/familiars/<id>/`. The smallest layout that boots is:

```
data/familiars/aria/
└── character.toml          # optional — defaults apply if missing
```

Everything else (`memory/`, `history.db`, `subscriptions.toml`, `channels/`, `modes/`) is created on first launch. Multiple character folders can sit side-by-side under `data/familiars/`; only the one you point `FAMILIAR_ID` at is loaded per process.

To give the familiar a persona, drop Markdown files into `data/familiars/<id>/memory/self/` (e.g. `description.md`, `personality.md`, `scenario.md`) — the `CharacterProvider` concatenates whatever is present. A character-card unpacker exists at `familiar_connect.memory.unpack_character.unpack_character` if you want to bootstrap from a V3 PNG programmatically.

**Per-mode instructions.** To tune *how* the familiar writes in a given channel mode (e.g. "keep it short, reply like a chat-room message"), drop a Markdown file into `data/familiars/<id>/modes/<mode>.md`. The filename must match the mode value: `text_conversation_rp.md`, `full_rp.md`, or `imitate_voice.md`. Missing file = no per-mode instruction; empty file = no-op. The text lands in `Layer.author_note` of the system prompt on every turn whose channel is in that mode.

```
data/familiars/aria/modes/
├── text_conversation_rp.md    # "Reply as if in an internet chat room. A few lines, max."
├── full_rp.md                 # "Prose style. Describe actions in italics. Stay in character."
└── imitate_voice.md           # "Speak naturally. One or two sentences."
```

Example `character.toml`:

```toml
default_mode = "text_conversation_rp"   # full_rp | text_conversation_rp | imitate_voice

[providers.history]
window_size = 20

[layers.depth_inject]
position = 0   # SillyTavern @D 0 — immediately before the final user turn
role = "system"
```

### Start

```bash
uv sync --dev
uv run familiar-connect run                      # uses $FAMILIAR_ID
uv run familiar-connect run --familiar aria      # or pick one explicitly
uv run familiar-connect -vv run --familiar aria  # verbose logging, good for smoke-tests
```

### Smoke-test slash commands

Once the bot is online and invited to your test guild, the new subscription surface is:

| Command | What it does |
|---|---|
| `/subscribe-text` | Listen for messages in the current text channel |
| `/unsubscribe-text` | Stop listening in the current text channel |
| `/subscribe-my-voice` | Join your voice channel (if you're in one) and enable TTS replies |
| `/unsubscribe-voice` | Leave the voice channel |
| `/channel-full-rp` | Put the current channel into `full_rp` mode (all providers on) |
| `/channel-text-conversation-rp` | Put the current channel into `text_conversation_rp` mode |
| `/channel-imitate-voice` | Put the current channel into `imitate_voice` mode (tight budget, low TTFB) |

A full end-to-end smoke test:

1. `uv run familiar-connect -vv run --familiar aria`
2. In a test text channel, run `/subscribe-text`. Send a message — the bot should reply.
3. Run `/channel-full-rp` and send another — the log line `pipeline channel=… provider=… status=ok duration=…` should now show `content_search` running alongside `character` and `history`.
4. Join a voice channel and run `/subscribe-my-voice`. The bot should join and greet. Post another message in the subscribed text channel — the reply should also be spoken.
5. `/unsubscribe-voice` then `/unsubscribe-text` to tear down cleanly.

Subscriptions and channel modes persist across restarts under `data/familiars/<id>/subscriptions.toml` and `data/familiars/<id>/channels/<channel_id>.toml`; delete those files to reset.

### Troubleshooting

- **`DISCORD_BOT environment variable is not set`** — missing bot token.
- **`No familiar selected`** — neither `FAMILIAR_ID` nor `--familiar` was given.
- **`Familiar folder does not exist`** — create `data/familiars/<id>/`.
- **`LLM client unavailable: OPENROUTER_API_KEY environment variable is required`** — missing OpenRouter key.
- **`Opus library not found — voice playback will not work`** — voice commands still run, but no audio; install libopus.
- **Bot joins but doesn't reply** — confirm `/subscribe-text` was issued in that exact channel, and check the log for `pipeline channel=… status=…` lines.

## Development

Requires [uv](https://docs.astral.sh/uv/)

Setup: `uv sync --dev`

Lint: `uv run ruff check`

Format: `uv run ruff format`

Type-check: `uv run ty check`

Test: `uv run pytest`

Security audit: `uv audit --preview-features audit`
