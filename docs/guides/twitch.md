# Twitch integration

The bot connects to Twitch EventSub as a task in the root
`asyncio.TaskGroup` and feeds channel events directly into the
internal text queue.

!!! success "Status: Implemented"
    Event subscription, event parsing, and the `/twitch *` slash
    command surface all ship in `familiar_connect.twitch` and
    `familiar_connect.twitch_watcher`.

## Events

### Channel point redemptions
- Watch for viewers redeeming channel points.
- Filter by a configurable list of redemption names (only specific
  redemptions trigger a message).
- If the viewer included a text input with the redemption, include
  it in the message.
- Example: `"Alice has redeemed Talk to Sapphire and says: hello!"`

### New subscriptions
- Notify when a viewer subscribes for the first time (non-gift).
- Include subscription tier (1, 2, or 3).
- Example: `"Alice has subscribed at tier 1"`

### Gift subscriptions
- Notify when a viewer gifts subscriptions to the channel.
- Include gifter name (or "An anonymous gifter"), number gifted, and
  tier.
- Example: `"Bob has gifted 5 tier 1 subscriptions"`

### Resubscriptions
- Notify when a viewer resubscribes and includes a message.
- Include how many months they've been subscribed, tier, and their
  message.
- Example: `"Alice has subscribed for 6 months at tier 2 and says: love this stream"`

### Cheers / bits
- Notify when a viewer cheers with bits.
- Include viewer name (or "An anonymous cheerer"), bit amount, and
  their message.
- Example: `"Bob has cheered with 100 bits and says: poggers"`

### Follows
- Example: `"Alice has followed the channel"`

### Ad breaks
- Notify when an ad break starts on the channel, and again when it
  ends (duration is known at start time).
- Ad messages support an **immediate** mode — sent directly to the
  LLM without waiting for the normal batch cycle.
- Example: `"An ad has begun on the channel"` →
  (after N seconds) → `"Ads have ended"`

## Configuration per watcher instance

Each watcher is configurable independently:

- Toggle subscriptions on/off
- Toggle cheers on/off
- Toggle follows on/off
- Toggle ads on/off
- Toggle ads immediate mode on/off
- Set a list of redemption names to listen for (empty = none)

## Message shape

All events produce a message with:

- The channel/context they belong to (which familiar/session)
- Plain-text description of the event
- A priority hint: `normal` (batch with others) or `immediate` (feed
  to LLM right away)
- A UTC timestamp

## Slash commands

| Command | Options | Description |
|---------|---------|-------------|
| `/twitch connect` | `channel` (string) | Connect the familiar to a Twitch channel and begin watching for events |
| `/twitch disconnect` | — | Stop watching the current Twitch channel |
| `/twitch status` | — | Show the currently connected channel and which event types are enabled |
| `/twitch events` | `subscriptions` (bool) `cheers` (bool) `follows` (bool) `ads` (bool) | Toggle which event categories produce messages; omitted options are unchanged |
| `/twitch ads-immediate` | `enabled` (bool) | Toggle whether ad break events are sent to the LLM immediately rather than batched with the normal cycle |
| `/twitch redemptions add` | `name` (string) | Add a channel point redemption name to the allow-list |
| `/twitch redemptions remove` | `name` (string) | Remove a channel point redemption name from the allow-list |
| `/twitch redemptions list` | — | Show all redemption names currently on the allow-list |
| `/twitch redemptions clear` | — | Remove all redemption names from the allow-list |

**Notes:**

- All `/twitch` commands require a role that has the "Manage Server"
  permission or a configured admin role.
- Twitch credentials (OAuth token, client ID) are set via `/setup` or
  in `.env` as `TWITCH_CLIENT_ID` and `TWITCH_ACCESS_TOKEN`; they are
  never accepted as slash command arguments.
- Settings are persisted per Discord guild so they survive bot
  restarts.
