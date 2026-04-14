# Configuration Model

The bot's configuration surface is split into two levels.

!!! success "Status: Implemented"
    The two-level model, per-channel TOML sidecars, `SubscriptionRegistry`, and all `/subscribe-*` / `/channel-*` slash commands ship today.

**Who runs this bot.** Familiar-Connect targets a single admin running the bot on their own machine. "Single operator" does *not* mean "single character" — the same install can hold multiple character folders under `data/familiars/<id>/`, and the operator flips between them by changing `FAMILIAR_ID` and restarting (or by running multiple processes in parallel, each with its own `FAMILIAR_ID`).

## The two levels

### 1. Bot instance config

The secrets and install selector the host machine needs to run the bot at all. Set by the **admin**, never exposed through Discord.

- `DISCORD_BOT` — Discord bot token
- `OPENROUTER_API_KEY` — single key shared across every LLM call site
- `CARTESIA_API_KEY` — Cartesia TTS key
- `DEEPGRAM_API_KEY` — Deepgram STT key (optional; required for voice input — see [Voice input](voice-input.md))
- Twitch client ID and OAuth token (optional)
- `FAMILIAR_ID` — selects which character folder under `data/familiars/` this process runs

Model choices, temperatures, and TTS voice are **not** here — those live on the per-familiar `character.toml` so swapping `FAMILIAR_ID` swaps the full configuration profile, not just the persona.

**Where it lives:** environment variables and / or a `.env` file. Never checked into git. Never editable from inside Discord. See [Installation](../getting-started/installation.md) for the full env var list.

### 2. Character config

Per-familiar configuration. The persona, behaviour knobs, pluggable component selection, and **every LLM model / temperature choice** for the single familiar this install runs.

- **Persona** — character card fields, unpacked into `memory/self/*.md` by `familiar_connect.bootstrap.unpack_character` (see [Bootstrapping](../guides/bootstrapping.md)).
- **Memory directory** — `memory/`, owned by `MemoryStore`. See [Memory](memory.md).
- **Tuning parameters** — history window size, depth-inject position and role, default channel mode.
- **Per-call-site LLM slots** — `[llm.<slot>]` sections, one per call site, each with its own `model` and `temperature`. See the [Per-call-site LLM slots](#per-call-site-llm-slots) section below.
- **TTS voice** — `[tts]` section carries `voice_id` and `model` for the Cartesia client.
- **Memory writer** — `[memory_writer]` section with `turn_threshold` (default 50) and `idle_timeout` (default 1800.0s / 30 min). Controls when the post-session writer pass runs to summarise conversation history into long-term memory files.
- **Per-channel overrides** — via `channels/<channel_id>.toml` sidecars written by the `/channel-*` slash commands. Each sidecar selects a `ChannelMode` (`full_rp`, `text_conversation_rp`, or `imitate_voice`); modes drive the provider / processor / budget table in `familiar_connect.config.channel_config_for_mode`. Channels may also carry a `[typing_simulation]` table — see [Typing simulation](typing-simulation.md).
- **Typing simulation** — `[typing_simulation]` section on `character.toml` and/or per-channel sidecar, layered over per-mode defaults. Controls chunk-based delivery and mid-flight cancellation on text channels. See [Typing simulation](typing-simulation.md).
- **Subscriptions** — which Discord channels the bot listens in, written to `subscriptions.toml` by `/subscribe-text` and `/subscribe-my-voice`.

**Where it lives:** `data/familiars/<familiar_id>/`. One folder per character. Only one runs at a time in any given process.

### Per-call-site LLM slots

Every LLM call site in the bot has its own slot, each independently configurable:

| Slot | Purpose |
|---|---|
| `main_prose` | The familiar's spoken reply to a Discord message |
| `post_process_style` | Rewrites a reply for tone / voice delivery (`RecastPostProcessor`) |
| `reasoning_context` | Hidden chain-of-thought before replying (`SteppedThinkingPreProcessor`) |
| `history_summary` | Summarizes own-channel and cross-channel history (`HistoryProvider`) |
| `memory_search` | Agentic memory/lore retrieval loop (`ContentSearchProvider`) |
| `memory_writer` | Post-session summarisation into long-term memory files (`MemoryWriter`) |
| `interjection_decision` | Decides whether to proactively join a conversation (`ConversationMonitor`) |

Each slot takes the same shape in `character.toml`:

```toml
[llm.main_prose]
model       = "z-ai/glm-5.1"
temperature = 0.7

[llm.post_process_style]
model       = "mistralai/mistral-small-2603"
temperature = 0.5
```

All six clients share the same `OPENROUTER_API_KEY` and the process-wide rate-limit semaphore in `familiar_connect.llm.get_request_semaphore`, so splitting a single "side" pool into six call sites does not multiply concurrency against the OpenRouter rate limit.

### Default profile

A reference familiar lives at `data/familiars/_default/` and is checked into the repo. It contains a `character.toml` with every configurable option filled in and serves two purposes:

1. **Fallback source.** When loading a user's familiar, any `[llm.<slot>]` section (or `[tts]` field) missing from the user's `character.toml` falls back to the corresponding value in `_default/character.toml`. No hardcoded defaults live in Python — the default profile is the single source of truth for fallback values.
2. **Documentation-by-example.** A new operator copies `_default/` to `data/familiars/my-familiar/` and edits from there.

The leading underscore is a convention to keep `FAMILIAR_ID=_default` from being a meaningful selection. `.gitignore` contains a `!data/familiars/_default/` exception so the default profile is checked in while user data stays ignored.

## One active familiar per process

A Familiar-Connect process runs exactly one familiar at a time, selected by `FAMILIAR_ID` at startup. That's a *process*-level constraint, not a *data*-level one: `data/familiars/` is free to hold any number of character folders, and nothing stops the operator from running a second process against a different `FAMILIAR_ID` in parallel. The per-process single-active-character rule exists because:

- Discord's voice API only lets a bot hold one voice connection per gateway session, so multi-character in one process would need per-character gateways anyway.
- The bot's runtime state (subscriptions, channel configs, TTS output queue) is simpler to reason about when it's scoped to one persona at a time.

The single-active constraint is enforced by:

- `FAMILIAR_ID` (env var) or `--familiar` (CLI flag) selecting exactly one folder.
- `Familiar.load_from_disk` returning a single runtime bundle the bot holds for its lifetime.
- `HistoryStore` partitioning by `familiar_id`, so a shared database (if one ever ends up shared across processes) still keeps turns separated.

If two-character-per-process ever becomes desirable, the data model already carries `familiar_id` through every store call — the refactor would be confined to the entry-point and subscription registry.

## On-disk layout

```
data/
└── familiars/
    └── <familiar_id>/               # one per alternate character
        ├── character.toml           # tuning + depth-inject config
        ├── subscriptions.toml       # persistent /subscribe-* state
        ├── channels/
        │   └── <channel_id>.toml    # per-channel mode + overrides
        ├── memory/                  # MemoryStore root
        │   ├── self/                # unpacked from card on creation
        │   ├── people/
        │   ├── topics/
        │   ├── sessions/
        │   └── lore/
        └── history.db               # SQLite HistoryStore
```

**Why TOML:** human-and-machine-readable, matching the memory directory's plain-text principle. A user can edit their character's config in any text editor; the bot loads it on startup or on the next mutation. No schema migrations, no opaque blob format.

**Why filesystem and not SQLite:** see [Design decisions](decisions.md) for the broader local-first principle. Per-character config is the kind of thing a user might want to back up, share, or version-control on their own; a filesystem layout makes that trivial.

## Identity model

- **`familiar_id`** is the folder name under `data/familiars/`. Kebab-case by convention (`aria`, `my-cat-familiar`). Used as the partition key for `HistoryStore` entries, though in practice there's only one active familiar per install.
- **Display name** is not stored separately — the character card carries it, and the bot reads it from the unpacked `memory/self/` files.

The speaker in a conversation (the Discord user who triggered a turn) is tracked separately as `ContextRequest.speaker`, and is the Discord display name (a string), not a user id.

## Trust model

The user explicitly trusts the admin. There is no per-user sandboxing, no resource limits enforced against the user, no isolation between users beyond filesystem path separation. The admin can read, edit, or delete any character's memory directly on disk. This is intentional — Familiar-Connect is a single-host bot for a small trusted group, not a multi-tenant SaaS. See [Security](security.md) for the defensive lines that *do* exist.

## Slash command surface

| Command | What |
|---|---|
| `/subscribe-text` | Register the current text channel as a text subscription |
| `/unsubscribe-text` | Drop the text subscription for the current channel |
| `/subscribe-my-voice` | Join the caller's voice channel and register a voice subscription |
| `/unsubscribe-voice` | Leave the current voice channel and drop the voice subscription |
| `/channel-full-rp` | Set the current channel's mode to `full_rp` (all providers, both processors, high budget) |
| `/channel-text-conversation-rp` | Set to `text_conversation_rp` (character + history, `stepped_thinking` on, `recast` off, medium budget) |
| `/channel-imitate-voice` | Set to `imitate_voice` (latency-tuned; `recast` on with voice flavour, `stepped_thinking` off) |

The old `/awaken` / `/sleep` commands have been removed. Their role is now split between the subscription commands and the channel-mode commands — `/subscribe-my-voice` in a voice channel does what `/awaken` did, while `/subscribe-text` in a text channel replaces the text-channel branch of `/awaken`.

Per-guild slash-command settings are still **not** in scope. A familiar travels with its install; which guild the user is in doesn't change its behaviour.

## Voice channel single-connection limitation

The bot can hold at most one voice connection at a time (a Discord `discord.VoiceClient` constraint, plus the practical fact that voice is the most resource-expensive path). When a second `/subscribe-my-voice` is issued anywhere, it refuses until the existing connection is dropped.

Spawning one bot process per familiar (so multiple familiars can hold voice sessions in parallel) is the documented scaling path. That's a deployment-shape change and is out of scope for this spec.

## Per-guild configuration (deferred)

Per-guild configuration is **not in scope**. The data model is partitioned by `familiar_id`, not `guild_id`, so a familiar's behaviour is identical regardless of which guild it's invoked in.

A future extension could add per-guild *overrides* on top of this — for example, "Aria is more reserved in the work guild and chattier in the personal guild." The natural place would be a sibling directory `data/guilds/<guild_id>/` containing TOML files that the runtime layers on top of the base character config. The current data model already carries `guild_id` as an observability field on `ContextRequest`, so this extension would not require schema changes.

## Relationship to context management

- `MemoryStore` is rooted at `data/familiars/<familiar_id>/memory/`.
- `HistoryStore` lives at `data/familiars/<familiar_id>/history.db` and partitions turns by `(familiar_id, channel_id)` for the per-channel recent window and `(familiar_id,)` for the global rolling summary.
- `ContextRequest` carries `familiar_id`, `channel_id`, `speaker`, `modality`, and `guild_id` (observability only).
- `ChannelConfigStore` lives at `data/familiars/<familiar_id>/channels/`.
- `SubscriptionRegistry` lives at `data/familiars/<familiar_id>/subscriptions.toml`.
