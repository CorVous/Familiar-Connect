## Summary

- Implements the full `/twitch` Discord slash command group across three new modules (`twitch_commands.py`, `twitch_registry.py`) wired into the existing `bot.py`
- Replaces the original trio nursery/cancel-scope design with `asyncio.create_task` and `asyncio.Task.cancel()` to match the asyncio migration on main
- Wires up the real `TwitchWatcher.run()` inside `_run_watcher`: creates an authenticated `Twitch` client, instantiates `EventSubWebsocket`, and suspends until the task is cancelled
- All `/twitch` commands are gated behind `default_member_permissions=manage_guild`

## Commands implemented

| Command | Behaviour |
|---|---|
| `/twitch connect <channel>` | Resolves broadcaster ID via Twitch API, creates `TwitchWatcher` + `asyncio.Queue[TwitchEvent]`, spawns background task |
| `/twitch disconnect` | Cancels the watcher task, removes guild from registry |
| `/twitch status` | Shows connected channel and all event-toggle states |
| `/twitch events [subscriptions] [cheers] [follows] [ads]` | Toggles event types on the live `TwitchWatcherConfig`; omitted options unchanged |
| `/twitch ads-immediate <enabled>` | Sets `config.ads_immediate` |
| `/twitch redemptions add/remove/list/clear` | Manages the channel-point redemption allow-list |

## Architecture

**`twitch_registry.py`** — per-guild in-memory state store:
```python
@dataclass
class GuildTwitchState:
    guild_id: int
    channel: str
    broadcaster_id: str
    config: TwitchWatcherConfig
    watcher: TwitchWatcher
    task: Any                          # asyncio.Task[None] at runtime
    queue: asyncio.Queue[TwitchEvent]  # event sink for future consumer
```

**`twitch_commands.py`** — pure async command handlers with no Discord coupling (all Discord types are `TYPE_CHECKING`-only imports), making them straightforward to unit test.

**`_run_watcher`** — background task that owns the Twitch connection lifetime:
```python
async def _run_watcher(watcher, client_id, token, send):
    async with await Twitch(client_id, token) as api:
        eventsub = EventSubWebsocket(api)
        await watcher.run(send, eventsub)  # suspends until cancelled
```

## Test coverage

808 new lines of tests across three test files, following red/green TDD throughout:

- `tests/test_twitch_registry.py` — dataclass fields, set/get/clear, duplicate-guild error
- `tests/test_twitch_commands.py` — all 9 command handlers, background task creation, queue storage, `_run_watcher` wiring via mock `Twitch` + `EventSubWebsocket`
- `tests/test_twitch_bot_integration.py` — full lifecycle (connect → status → events toggle → redemptions → disconnect), permission guard assertion

## Notes

- Twitch credentials are read from `TWITCH_CLIENT_ID` / `TWITCH_ACCESS_TOKEN` env vars at connect time; never accepted as command arguments
- The `asyncio.Queue[TwitchEvent]` stored in `GuildTwitchState` is ready for a future consumer task to drain and forward events to the LLM pipeline
- Guild state is in-memory only; persistence via SQLite is tracked in `future-features/persistence.md`
