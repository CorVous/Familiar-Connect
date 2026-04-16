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
- `CARTESIA_API_KEY` — Cartesia TTS key (required when `[tts].provider="cartesia"`)
- `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` — Azure Speech credentials (required when `[tts].provider="azure"`)
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) — Gemini TTS key (required when `[tts].provider="gemini"`)
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
- **TTS provider** — `[tts]` section selects the active provider (`azure` / `cartesia` / `gemini`) and carries the provider-specific voice and model fields. See [TTS providers](#tts-providers) below.
- **Memory writer** — `[memory_writer]` section with `turn_threshold` (default 50) and `idle_timeout` (default 1800.0s / 30 min). Controls when the post-session writer pass runs to summarise conversation history into long-term memory files.
- **Per-channel overrides** — via `channels/<channel_id>.toml` sidecars written by the `/channel-*` slash commands. Each sidecar selects a `ChannelMode` (`full_rp`, `text_conversation_rp`, or `imitate_voice`); modes drive the provider / processor / budget table in `familiar_connect.config.channel_config_for_mode`. Channels may also carry a `backdrop` field (per-channel author note) and a `[typing_simulation]` table — see [Typing simulation](typing-simulation.md) and [Per-channel backdrop](#per-channel-backdrop).
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

A reference familiar lives at `data/familiars/_default/` and is checked into the repo. It serves two purposes:

1. **Fallback source.** When loading a user's familiar:
   - Any `[llm.<slot>]` section (or `[tts]` field) missing from the user's `character.toml` falls back to the corresponding value in `_default/character.toml`. No hardcoded defaults live in Python — the default profile is the single source of truth for fallback values.
   - Per-mode instruction files (`_default/modes/full_rp.md`, `text_conversation_rp.md`, `imitate_voice.md`) are loaded as runtime fallbacks when a familiar's own `modes/<mode>.md` is absent. Drop a file into `data/familiars/<id>/modes/` to override the default for that familiar.
2. **Documentation-by-example.** A new operator copies `_default/` to `data/familiars/my-familiar/` and edits from there.

The leading underscore is a convention to keep `FAMILIAR_ID=_default` from being a meaningful selection. `.gitignore` contains a `!data/familiars/_default/` exception so the default profile is checked in while user data stays ignored.

### TTS providers

Three text-to-speech providers are available. Select one per familiar in `character.toml` under `[tts].provider`.

| Provider | Key in `.env` | Character fields | Notes |
|---|---|---|---|
| `azure` (default) | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | `azure_voice` | Full word-level timestamps via Speech SDK word-boundary events |
| `cartesia` | `CARTESIA_API_KEY` | `cartesia_voice_id`, `cartesia_model` | WebSocket streaming; word-level timestamps in API response |
| `gemini` | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | `gemini_voice`, `gemini_model` | 24 kHz PCM upsampled to 48 kHz; timestamps estimated uniformly from duration |

#### Gemini-specific fields

Gemini TTS supports natural-language performance direction. Six optional fields in `[tts]` compose into an Audio Profile / Scene / Director's Notes prompt prepended to every synthesis request:

| Field | Purpose | Example |
|---|---|---|
| `gemini_audio_profile` | Core voice identity | `"warm, curious 20-something"` |
| `gemini_scene` | Physical environment / vibe | `"quiet late-night voice chat"` |
| `gemini_context` | Narrative context | `"roleplay as a tavern keeper"` |
| `gemini_style` | Delivery style | `"playful, conversational"` |
| `gemini_pace` | Pacing direction | `"relaxed with thoughtful pauses"` |
| `gemini_accent` | Accent direction | `"soft Irish lilt"` |

Any subset may be set — unset fields are omitted from the composed prompt. All six default to unset (no prompt prefix).

**Audio tags.** Gemini 3.1 Flash TTS supports 200+ inline audio tags (`[laughs]`, `[whispers]`, `[short pause]`, `[gasp]`, etc.). These flow through automatically if the LLM includes them in its reply — no config needed.

**Available voices.** Prebuilt voice names (select with `gemini_voice`):
Achernar, Achird, Algenib, Algieba, Alnilam, Aoede, Autonoe, Callirrhoe, Charon, Despina, Enceladus, Erinome, Fenrir (Excitable), Gacrux, Iapetus, Kore (Firm, default), Laomedeia, Leda, Orus, Puck (Upbeat), Pulcherrima, Rasalgethi, Sadachbia, Sadaltager, Schedar, Sulafat, Umbriel, Vindemiatrix, Zephyr (Bright), Zubenelgenubi.

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
    ├── _default/                    # checked-in reference profile
    │   ├── character.toml           # fallback LLM/TTS defaults
    │   └── modes/                   # fallback mode instruction files
    │       ├── full_rp.md
    │       ├── text_conversation_rp.md
    │       └── imitate_voice.md
    └── <familiar_id>/               # one per alternate character
        ├── character.toml           # tuning + depth-inject config
        ├── subscriptions.toml       # persistent /subscribe-* state
        ├── channels/
        │   └── <channel_id>.toml    # per-channel mode + backdrop + overrides
        ├── modes/                   # familiar-specific mode instructions
        │   └── <mode>.md            # overrides _default/modes/<mode>.md
        ├── memory/                  # MemoryStore root
        │   ├── self/                # unpacked from card on creation
        │   ├── people/
        │   ├── topics/
        │   ├── sessions/
        │   └── lore/
        └── history.db               # SQLite HistoryStore
```

## Per-channel backdrop

A **backdrop** is an author-note text injected into `Layer.author_note` of the system prompt on every turn in a specific channel. It replaces the mode-level instruction (from `modes/<mode>.md` or `_default/modes/<mode>.md`) entirely for that channel.

**Precedence** (high → low):

1. Per-channel `backdrop` in `channels/<channel_id>.toml` — set via `/channel-backdrop`.
2. Familiar's own `modes/<mode>.md`.
3. `_default/modes/<mode>.md` (repo default).
4. Nothing → no author-note contribution.

**Setting a backdrop:** run `/channel-backdrop` in the target channel. A Discord modal opens with a multi-line text field pre-filled with the current backdrop (if any). Submit to save.

**Clearing a backdrop:** run `/channel-backdrop` and submit the modal with the text field blank. The backdrop is removed and the mode default resumes on the next turn.

**TOML layout:**

```toml
channel_name = "general"   # informational; written by the slash command
mode = "full_rp"

backdrop = """
Reply as a stern tavern keeper. Call the user "traveler."
Keep it to two sentences.
"""

[typing_simulation]
enabled = false
```

Existing sidecars without `backdrop` or `channel_name` continue to work unchanged. Switching modes with `/channel-full-rp` (and siblings) now preserves any `backdrop` and `channel_name` that was already set.

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
| `/channel-backdrop` | Open a modal to set a custom author-note for this channel (replaces the mode default). Submit the modal blank to clear. |

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
