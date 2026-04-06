# Twitch Channel Event Watcher — Feature Spec

## Core Concept
Watch a Twitch channel for real-time events → translate each event into a plain-text message → feed into LLM message batch.

---

## Features

### 1. Channel Point Redemptions
- Watch for viewers redeeming channel points
- Filter by a configurable list of redemption names (only specific redemptions trigger a message)
- If the viewer included a text input with the redemption, include it in the message
- Example output: `"Alice has redeemed Talk to Sapphire and says: hello!"`

### 2. New Subscriptions
- Notify when a viewer subscribes for the first time (non-gift)
- Include subscription tier (1, 2, or 3)
- Example output: `"Alice has subscribed at tier 1"`

### 3. Gift Subscriptions
- Notify when a viewer gifts subscriptions to the channel
- Include gifter name (or "An anonymous gifter"), number gifted, and tier
- Example output: `"Bob has gifted 5 tier 1 subscriptions"`

### 4. Resubscriptions
- Notify when a viewer resubscribes and includes a message
- Include how many months they've been subscribed, tier, and their message
- Example output: `"Alice has subscribed for 6 months at tier 2 and says: love this stream"`

### 5. Cheers / Bits
- Notify when a viewer cheers with bits
- Include viewer name (or "An anonymous cheerer"), bit amount, and their message
- Example output: `"Bob has cheered with 100 bits and says: poggers"`

### 6. Follows
- Notify when a viewer follows the channel
- Example output: `"Alice has followed the channel"`

### 7. Ad Breaks
- Notify when an ad break starts on the channel
- Notify again when the ad break ends (duration is known at start time)
- Ad messages support an **immediate** mode — sent directly to LLM without waiting for the normal batch cycle
- Example output: `"An ad has begun on the channel"` → (after N seconds) → `"Ads have ended"`

---

## Configuration Per Watcher Instance
Each watcher is configurable independently:
- Toggle subscriptions on/off
- Toggle cheers on/off
- Toggle follows on/off
- Toggle ads on/off
- Toggle ads immediate mode on/off
- Set a list of redemption names to listen for (empty = none)

---

## Message Shape
All events produce a message with:
- The channel/context they belong to (which familiar/session)
- Plain-text description of the event
- A priority hint: `normal` (batch with others) or `immediate` (feed to LLM right away)
- A UTC timestamp
