# Twitch Slash Commands — Implementation Plan

## Overview

The Twitch event watcher backend (`twitch.py`, `twitch_watcher.py`) is fully built and tested. This plan covers the Discord slash command layer that lets users control it at runtime.

All commands live under a `/twitch` slash command group. They require the **Manage Server** permission (or a configured admin role).

---

## Existing Backend Surface

What we already have and can wire directly into commands:

| Component | Location | What it does |
|---|---|---|
| `TwitchWatcherConfig` | `twitch.py:50` | Dataclass holding all per-watcher toggles and the redemption allow-list |
| `TwitchWatcher` | `twitch_watcher.py:64` | Accepts a config + broadcaster info, registers EventSub listeners, runs as a trio task |
| `TwitchWatcher.run()` | `twitch_watcher.py:292` | Starts the EventSub connection and sleeps until cancelled |
| Event builders | `twitch.py:128–277` | Build `TwitchEvent` objects gated by config flags |

The slash commands need to:
1. Manage `TwitchWatcher` lifecycle (create, start, stop)
2. Mutate `TwitchWatcherConfig` at runtime
3. Store config per guild so it survives restarts (deferred — in-memory first, persistence later)

---

## Command Specifications

### `/twitch connect`

| Field | Value |
|---|---|
| Option | `channel` (string, required) — Twitch channel name |
| Permission | Manage Server |
| Behavior | Look up the broadcaster ID via Twitch API, create a `TwitchWatcher`, spawn it as a trio task, confirm to the user |
| Error cases | Already connected (respond with current channel), invalid channel name, missing Twitch credentials |

**Implementation notes:**
- Use `twitchAPI.Twitch` client to resolve channel name → broadcaster ID
- Store the watcher + its cancel scope in a per-guild registry so `/twitch disconnect` can stop it
- Twitch credentials come from env vars (`TWITCH_CLIENT_ID`, `TWITCH_ACCESS_TOKEN`) — never from command args

### `/twitch disconnect`

| Field | Value |
|---|---|
| Options | None |
| Permission | Manage Server |
| Behavior | Cancel the running watcher's trio task, remove from registry, confirm |
| Error cases | Not connected |

### `/twitch status`

| Field | Value |
|---|---|
| Options | None |
| Permission | Manage Server |
| Behavior | Show connected channel name, and a summary of enabled/disabled event types and redemption allow-list |
| Error cases | Not connected |

**Response format** (embed or code block):
```
Channel: coolstreamer
Subscriptions: enabled
Cheers: enabled
Follows: disabled
Ads: enabled (immediate)
Redemptions: Talk to Familiar, Hydrate Check
```

### `/twitch events`

| Field | Value |
|---|---|
| Options | `subscriptions` (bool, optional), `cheers` (bool, optional), `follows` (bool, optional), `ads` (bool, optional) |
| Permission | Manage Server |
| Behavior | Update the corresponding flags on the active `TwitchWatcherConfig`. Omitted options are unchanged. Respond with the new state. |
| Error cases | Not connected |

**Implementation note:** Toggling event types only updates the config object — the event builders already gate on these flags, so no listener re-registration is needed.

### `/twitch ads-immediate`

| Field | Value |
|---|---|
| Options | `enabled` (bool, required) |
| Permission | Manage Server |
| Behavior | Set `config.ads_immediate`. Confirm new value. |
| Error cases | Not connected |

### `/twitch redemptions add`

| Field | Value |
|---|---|
| Options | `name` (string, required) — redemption title to add |
| Permission | Manage Server |
| Behavior | Append to `config.redemption_names` if not already present. Confirm. |
| Error cases | Not connected, name already in list |

**Note:** If this is the first redemption added and the watcher is running, we need to register the `listen_channel_points_custom_reward_redemption_add` listener (since `register_listeners` skips it when the list is empty). This is the one case where a runtime config change requires re-registration. Options:
- **Option A:** Always register the redemption listener regardless of list contents. The builder returns `None` for unlisted names anyway.
- **Option B:** Track whether it's registered and add it on first redemption add.
- **Recommendation:** Option A — simpler, no runtime re-registration needed. Costs one idle subscription.

### `/twitch redemptions remove`

| Field | Value |
|---|---|
| Options | `name` (string, required) |
| Permission | Manage Server |
| Behavior | Remove from `config.redemption_names`. Confirm. |
| Error cases | Not connected, name not in list |

### `/twitch redemptions list`

| Field | Value |
|---|---|
| Options | None |
| Permission | Manage Server |
| Behavior | Show all names in the allow-list, or "No redemptions configured" |
| Error cases | Not connected |

### `/twitch redemptions clear`

| Field | Value |
|---|---|
| Options | None |
| Permission | Manage Server |
| Behavior | Clear the entire allow-list. Confirm. |
| Error cases | Not connected |

---

## Architecture

### Guild Registry

A new module (`twitch_registry.py` or added to an existing module) that maps guild IDs to active watcher state:

```python
@dataclass
class GuildTwitchState:
    channel: str
    broadcaster_id: str
    config: TwitchWatcherConfig
    watcher: TwitchWatcher
    cancel_scope: trio.CancelScope
```

- One entry per guild (a guild can watch at most one Twitch channel)
- The registry is in-memory for now; persistence comes later via SQLite

### Command Group Registration

Use py-cord's `SlashCommandGroup`:

```python
twitch_group = bot.create_group("twitch", "Twitch channel event watcher")
```

Subcommands and sub-groups register under this. The `redemptions` subcommands use a nested group:

```python
redemptions_group = twitch_group.create_subgroup("redemptions", "Manage redemption allow-list")
```

### Trio ↔ asyncio Bridge

The bot runs under `trio-asyncio`. The watcher's `run()` is a trio task. When `/twitch connect` is invoked from a py-cord (asyncio) callback, we need to spawn the trio task. Options:
- Use the existing nursery reference passed into `create_bot`
- Or use `trio_asyncio.trio_as_aio` to bridge

The cleanest approach: pass a reference to the root `trio.Nursery` into the bot factory, and store it so command handlers can call `nursery.start_soon(watcher.run, ...)`.

### Permission Check

Use py-cord's `@commands.has_permissions(manage_guild=True)` decorator, or a custom check that also accepts a configured admin role.

---

## Implementation Order (TDD)

Each step follows red/green TDD per CLAUDE.md.

### Phase 1: Guild Registry
1. `GuildTwitchState` dataclass and in-memory registry
2. Tests for set/get/clear per guild, error on double-connect

### Phase 2: Core Commands
3. `/twitch connect` — resolve channel, create watcher, register in guild state
4. `/twitch disconnect` — cancel scope, remove from registry
5. `/twitch status` — read config and format response

### Phase 3: Event Toggle Commands
6. `/twitch events` — mutate config flags
7. `/twitch ads-immediate` — mutate `ads_immediate` flag

### Phase 4: Redemption Commands
8. `/twitch redemptions add`
9. `/twitch redemptions remove`
10. `/twitch redemptions list`
11. `/twitch redemptions clear`

### Phase 5: Integration
12. Wire command group into `create_bot` / `commands/run.py`
13. Integration test: connect → toggle events → disconnect lifecycle

---

## Open Questions

1. **Nursery threading:** How should the root nursery be passed to command handlers? Via bot instance attribute, or a closure in create_bot?
2. **Reconnection:** If the Twitch WebSocket drops, should `/twitch status` reflect that and allow re-connect? The twitchAPI library may handle reconnection internally.
3. **Channel validation:** Should `/twitch connect` validate the channel exists before spawning the watcher, or let the EventSub connection fail?
4. **Autocomplete:** Should `/twitch redemptions remove` offer autocomplete from the current allow-list?
