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

## Implementation Order (Red/Green TDD)

Every step follows the red/green TDD workflow from CLAUDE.md:

1. **Red** — Write a failing test that describes the desired behavior. The test must fail for the *right reason* (an assertion failure, not an `ImportError`). This means the module/function/class under test must exist as a stub before the test can count as a valid red test.
2. **Green** — Write the minimum code to make the test pass. No more.
3. **Refactor** — Clean up if needed, re-run tests to confirm they still pass.

Import errors do not count as red. If a test fails because the module doesn't exist yet, create the empty module/stub first, then confirm the test fails on an assertion.

---

### Phase 1: Guild Registry

#### Step 1 — `GuildTwitchState` dataclass

**Red:** Create `tests/test_twitch_registry.py`. Write a test that imports `GuildTwitchState` from `familiar_connect.twitch_registry` and instantiates it with all required fields (`guild_id`, `channel`, `broadcaster_id`, `config`, `watcher`, `cancel_scope`). Assert the fields are stored correctly.

*Before running:* Create an empty `src/familiar_connect/twitch_registry.py` so the import doesn't fail — the test should fail on `AttributeError` or similar, not `ImportError`.

**Green:** Define the `GuildTwitchState` dataclass in `twitch_registry.py` with the required fields.

#### Step 2 — Registry set/get/clear

**Red:** Write tests for three functions:
- `set_guild_twitch(guild_id, state)` — stores state; raises `RegistryError` if already set for that guild
- `get_guild_twitch(guild_id)` → `GuildTwitchState | None`
- `clear_guild_twitch(guild_id)` — removes entry, no-op if absent

Test cases:
- `get` returns `None` for unknown guild
- `set` then `get` returns the state
- `set` twice for the same guild raises `RegistryError`
- `clear` then `get` returns `None`

**Green:** Implement registry as a module-level `dict[int, GuildTwitchState]` with the three functions.

---

### Phase 2: Core Commands

#### Step 3 — `/twitch connect`

**Red:** Write a test that simulates the `/twitch connect` slash command with a mock `ApplicationContext`. Assert:
- The guild registry has an entry after the command runs
- The response message includes the channel name
- A second connect in the same guild responds with an "already connected" error

*Stub:* Create the command function signature in a new module (e.g. `twitch_commands.py`) so the import works.

**Green:** Implement the connect handler:
- Look up broadcaster ID via Twitch API (mock in tests)
- Create `TwitchWatcherConfig` with defaults
- Create `TwitchWatcher`
- Register in guild state
- Respond with confirmation

**Red (error case):** Test that missing Twitch credentials (no env vars) responds with a clear error.

**Green:** Add the env var check.

#### Step 4 — `/twitch disconnect`

**Red:** Write tests:
- Disconnect when connected → clears registry, responds "Disconnected"
- Disconnect when not connected → responds "Not connected"

**Green:** Implement: look up guild state, cancel the scope, clear registry, respond.

#### Step 5 — `/twitch status`

**Red:** Write tests:
- Status when connected → response includes channel name and all toggle states
- Status when not connected → responds "Not connected"

**Green:** Implement: read config from registry, format as a readable string/embed.

**Red (formatting):** Test that the status output correctly shows "enabled"/"disabled" for each toggle and lists redemption names.

**Green:** Build the formatter.

---

### Phase 3: Event Toggle Commands

#### Step 6 — `/twitch events`

**Red:** Write tests:
- Set `subscriptions=False` → config updated, response confirms new state
- Set multiple flags at once → all updated
- Omit a flag → that flag unchanged
- Not connected → error response

**Green:** Implement: read guild state, update only the provided flags on `TwitchWatcherConfig`, respond with new state.

#### Step 7 — `/twitch ads-immediate`

**Red:** Write tests:
- Set `enabled=True` → `config.ads_immediate` is `True`, response confirms
- Set `enabled=False` → `config.ads_immediate` is `False`
- Not connected → error response

**Green:** Implement: one-field update on config.

---

### Phase 4: Redemption Commands

#### Step 8 — `/twitch redemptions add`

**Red:** Write tests:
- Add "Hydrate" → appears in `config.redemption_names`, response confirms
- Add duplicate → response says "already in list", list unchanged
- Not connected → error response

**Green:** Implement: append to list if not present.

#### Step 9 — `/twitch redemptions remove`

**Red:** Write tests:
- Remove existing name → removed from list, response confirms
- Remove non-existent name → response says "not found", list unchanged
- Not connected → error response

**Green:** Implement: remove from list if present.

#### Step 10 — `/twitch redemptions list`

**Red:** Write tests:
- With items → response lists all names
- Empty list → response says "No redemptions configured"
- Not connected → error response

**Green:** Implement: format list or empty message.

#### Step 11 — `/twitch redemptions clear`

**Red:** Write tests:
- Clear populated list → list is empty, response confirms
- Clear already-empty list → response still confirms (idempotent)
- Not connected → error response

**Green:** Implement: `config.redemption_names.clear()`.

---

### Phase 5: Integration

#### Step 12 — Wire into bot

**Red:** Write a test that `create_bot` returns a bot with the `/twitch` command group registered (check `bot.pending_application_commands` or similar).

**Green:** Register the `SlashCommandGroup` and all subcommands in `create_bot` or a dedicated setup function called from `commands/run.py`.

#### Step 13 — End-to-end lifecycle test

**Red:** Write an integration test that exercises the full lifecycle:
1. `/twitch connect coolstreamer` → success
2. `/twitch status` → shows "coolstreamer" with default config
3. `/twitch events subscriptions:false` → subscriptions disabled
4. `/twitch redemptions add "Hydrate"` → added
5. `/twitch redemptions list` → shows "Hydrate"
6. `/twitch disconnect` → success
7. `/twitch status` → "Not connected"

Mock the Twitch API and EventSub WebSocket. Assert each response message and the final registry state.

**Green:** Fix any issues discovered during integration.

---

## Open Questions

1. **Nursery threading:** How should the root nursery be passed to command handlers? Via bot instance attribute, or a closure in create_bot?
2. **Reconnection:** If the Twitch WebSocket drops, should `/twitch status` reflect that and allow re-connect? The twitchAPI library may handle reconnection internally.
3. **Channel validation:** Should `/twitch connect` validate the channel exists before spawning the watcher, or let the EventSub connection fail?
4. **Autocomplete:** Should `/twitch redemptions remove` offer autocomplete from the current allow-list?
