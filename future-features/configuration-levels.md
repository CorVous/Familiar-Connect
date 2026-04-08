# Configuration Levels

The bot's configuration surface is split into three concentric levels. This is a feature *spec*, not an implementation plan — context-management work has aligned its data model to it (memory and history are partitioned by `(owner_user_id, familiar_id)`), but the actual user-facing config tooling is a separate piece of work.

> **Open question — config surface tension.** There is an unresolved tension between two desirable properties of user config (level 2) and character config (level 3):
>
> - **"I want to connect a bunch of stuff on my computer to the app"** — argues for filesystem TOML the admin can edit directly, point at local files, point at local model endpoints, etc.
> - **"I want to share things like keys between instances and do things through Discord"** — argues for in-Discord configuration so users without shell access can manage their own familiars.
>
> The choices in this spec (TOML on disk, slash-command surface as a sketch) are a **development stop-gap** — they're the simplest thing that lets context-management work proceed. The real config-surface design is an open planning thread; revisiting any of the choices below should not invalidate the *data model* the context layer commits to. The data model — `(owner_user_id, familiar_id)` partition keys, the on-disk path for `memory/`, the per-familiar isolation rules — is independent of the config-mechanism question and stable regardless of how that thread resolves.

## The three levels

### 1. Bot instance config

The secrets and global knobs the host machine needs to run the bot at all. Set by the **admin**, never exposed through Discord.

- Discord bot token
- OpenRouter API key (and any other LLM provider tokens)
- Cartesia / Azure / Fish API keys for TTS
- Deepgram API key for STT
- Twitch client ID and OAuth token
- Default model overrides, default temperature, default budget caps
- Storage roots (e.g. `data/`)

**Where it lives:** environment variables and / or a `.env` file. Never checked into git. Never editable from inside Discord. The existing `Security Guidelines` section in `plan.md` already covers credential handling for this layer; nothing needs to change there.

### 2. User config

Per-Discord-user configuration. A user owns one or more familiars and can add, delete, and tune each one. Set by the **user**, exposed through Discord slash commands.

- Which familiars this user owns (a list of `familiar_id` strings)
- Per-familiar slash-command shortcuts (e.g. `/awaken aria` summons "their Aria")
- Default familiar (which one `/awaken` picks if no name is given)
- Possibly some user-level preferences down the line (preferred TTS voice family, default chattiness)

**Where it lives:** the filesystem, under `data/users/<owner_user_id>/`. One TOML file per user (`data/users/<owner_user_id>/user.toml`) plus per-familiar subdirectories beneath it. See [§ On-disk layout](#on-disk-layout) below.

### 3. Character config

Per-familiar configuration. The persona, behaviour knobs, and pluggable component selection for one specific familiar. Set by the **user** who owns that familiar.

- Persona — character card fields, unpacked into `memory/self/*.md` (already done by `unpack_character` in step 4 of `context-management.md`).
- Memory directory — `memory/`, owned by `MemoryStore` (step 3).
- Tuning parameters — temperature, top_p, max output tokens, chattiness, voice id, model selection, side-model selection.
- Provider/processor enablement — which `ContextProvider`s and `Pre/PostProcessor`s are active for this familiar, per modality (voice vs text).
- Per-layer token budgets for the budgeter.

**Where it lives:** `data/users/<owner_user_id>/familiars/<familiar_id>/character.toml`, alongside the `memory/` directory the familiar already owns.

## On-disk layout

```
data/
├── users/
│   ├── <owner_user_id>/                  # one Discord user
│   │   ├── user.toml                     # user-level config (level 2)
│   │   └── familiars/
│   │       ├── aria/                     # one familiar (user-chosen id)
│   │       │   ├── character.toml        # character-level config (level 3)
│   │       │   └── memory/               # MemoryStore root (already exists)
│   │       │       ├── self/             # unpacked from card on creation
│   │       │       ├── people/
│   │       │       ├── topics/
│   │       │       ├── sessions/
│   │       │       └── lore/
│   │       └── bob/
│   │           └── ...
│   └── <other_owner_user_id>/
│       └── ...
└── history.db                            # single SQLite, partitioned by owner+familiar
```

**Why TOML:** human-and-machine-readable matches the same principle the memory directory commits to. A user can edit their character's config in any text editor; the bot loads it on startup or on a `/reload`. No schema migrations, no opaque blob format.

**Why filesystem and not SQLite:** see § Decisions Considered and Rejected in `plan.md` for the broader local-first principle. Per-character config is the kind of thing a user might want to back up, share, or version-control on their own; a filesystem layout makes that trivial.

## Identity model

- **`owner_user_id`** is the **Discord user id** of the person who created and owns the familiar. It's the primary partition key for everything in `data/users/`. It's an `int`.
- **`familiar_id`** is a user-chosen string that identifies one specific familiar within a single owner. Probably kebab-case (`"aria"`, `"my-cat-familiar"`); the bot doesn't enforce uniqueness across owners, only within an owner.
- **Display name** is *not* stored — it's queryable from the Discord API given the `owner_user_id`, and can be cached at the call site if API rate-limiting becomes a concern.

A speaker in a conversation (the Discord user who triggered a turn) is a separate identity from the familiar's owner. The current `ContextRequest` carries `speaker` (a display-name string); a future expansion may add `speaker_user_id: int | None` for the same Discord-id treatment, but it's not needed yet.

## Trust model

The user explicitly trusts the admin. There is no per-user sandboxing, no resource limits enforced against the user, no isolation between users beyond filesystem path separation. The admin can read, edit, or delete any user's config and any familiar's memory directly on disk. This is intentional — Familiar-Connect is a single-host bot for a small trusted group, not a multi-tenant SaaS.

## Slash command surface (sketch, not committed)

A near-future user-config feature would expose something like:

| Command | Who | What |
|---|---|---|
| `/familiar create <id>` | user | Create a new familiar owned by you |
| `/familiar import <id>` | user | Upload a Character Card V3 PNG; the bot unpacks it into `memory/self/` |
| `/familiar list` | user | List the familiars you own |
| `/familiar delete <id>` | user | Delete one of your familiars (memory and config both) |
| `/familiar tune <id> <key> <value>` | user | Set a tuning parameter on a familiar (e.g. temperature) |
| `/awaken <id>` | user | Summon the named familiar to your current voice channel. Existing `/awaken` becomes "summon the user's default familiar" |
| `/sleep` | user | Disconnect the currently active voice familiar |

Per-guild slash-command settings are explicitly **not** in scope for the first cut. A familiar travels with its owner; if a user is in two guilds, the same familiar can be summoned in either.

## Voice channel single-connection limitation

Acknowledged for the record. The bot can hold at most one voice connection at a time (a Discord `discord.VoiceClient` constraint, plus the practical fact that voice is the most resource-expensive path). When a second `/awaken` is issued anywhere — even by a different user, even in a different guild — it disconnects the existing voice session first.

A future scaling path is to spawn one bot process per familiar, each with its own gateway connection, so multiple familiars can hold voice sessions in parallel. That's a deployment-shape change and is out of scope for this spec.

## Per-guild configuration (deferred)

Per-guild configuration is **not in scope** for the first cut. The data model is partitioned by `(owner_user_id, familiar_id)`, so a familiar's behaviour is identical regardless of which guild it's invoked in.

A future extension could add per-guild *overrides* on top of this — for example, "Aria is more reserved in the work guild and chattier in the personal guild." The natural place would be a sibling directory `data/guilds/<guild_id>/` containing TOML files that the runtime layers on top of the base character config. The current data model already carries `guild_id` as an observability field on `ContextRequest`, so this extension would not require schema changes.

But none of that is built or planned for the first cut. The point of mentioning it here is so future contributors don't paint themselves into a corner that precludes it.

## Relationship to context management

The context-management feature treats this spec as authoritative for partition keys and on-disk paths. Specifically:

- `MemoryStore` is rooted at `data/users/<owner_user_id>/familiars/<familiar_id>/memory/`.
- `HistoryStore` partitions turns by `(owner_user_id, familiar_id)` for the global view and `(owner_user_id, familiar_id, channel_id)` for the per-channel recent window.
- Rolling summaries are cached per `(owner_user_id, familiar_id)` — global per familiar, regardless of which channel a particular older turn happened in.
- `ContextRequest` carries `owner_user_id`, `familiar_id`, `channel_id`, and `guild_id` (observability only).

When this spec eventually gets implemented, the only piece of context management that will need to change is the wiring step (step 7 of `context-management.md`), which has to load and apply the per-character TOML config to enable the right providers and set the right budgets. The pipeline and provider modules themselves are all already shaped for it.
