# Threads and Forum Posts

## Overview

The familiar cannot currently be summoned to Discord Threads or Forum Posts — the existing `/awaken` and `/sleep` commands only work in standard text channels and voice channels. This feature would extend summoning support to those channel subtypes.

## Affected Channel Types

- **Threads** — public and private threads created within a text channel
- **Forum Post threads** — the individual posts inside a Forum channel (each post is itself a thread in Discord's API)

## Why It Doesn't Work Out of the Box

Discord treats threads and forum posts as a distinct channel type (`GUILD_PUBLIC_THREAD`, `GUILD_PRIVATE_THREAD`, `GUILD_NEWS_THREAD`, and forum post threads). The bot must:

1. Check that it has the `SEND_MESSAGES_IN_THREADS` permission (separate from `SEND_MESSAGES` in a regular channel)
2. Join the thread before it can send messages to it (threads are not automatically joined by bots)
3. Resolve the parent channel correctly when looking up context — the thread's `parent_id` points to the text/forum channel, not the guild root

Currently the channel-type guard in the summoning logic rejects non-text, non-voice channels, which blocks threads and forum posts.

## Desired Behavior

- `/awaken` issued inside a thread or forum post should summon the familiar to that thread
- The familiar reads messages in that thread only (not the parent channel)
- Responses are sent into the thread, keeping conversation contained
- The one-location-at-a-time rule still applies — summoning to a thread while active elsewhere requires an explicit `/sleep` first
- Logging follows the same anchor-message pattern as text-channel sessions, anchored at the top of the thread if possible, or as the first bot message if not

## Session Handling

Session handling is identical to text channel sessions:

- A session summary is written at the end
- Lorebook entries (people, topics) are updated as normal
- The thread context (thread name, parent forum/channel name) should be included in the session metadata for reference
