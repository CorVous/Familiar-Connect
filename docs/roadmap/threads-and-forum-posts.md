# Threads and Forum Posts

!!! info "Status: Design"
    Extend the existing subscription model to cover Discord thread
    and forum-post channel types. No code exists for this yet.

## Overview

The familiar cannot currently be subscribed to Discord threads or forum posts — the `/subscribe-text` command only accepts standard text channels and voice channels. This feature would extend subscription support to those channel subtypes.

## Affected channel types

- **Threads** — public and private threads created within a text channel.
- **Forum post threads** — the individual posts inside a Forum channel (each post is itself a thread in Discord's API).

## Why it doesn't work out of the box

Discord treats threads and forum posts as a distinct channel type (`GUILD_PUBLIC_THREAD`, `GUILD_PRIVATE_THREAD`, `GUILD_NEWS_THREAD`, and forum post threads). The bot must:

1. Check that it has the `SEND_MESSAGES_IN_THREADS` permission (separate from `SEND_MESSAGES` in a regular channel).
2. Join the thread before it can send messages to it (threads are not automatically joined by bots).
3. Resolve the parent channel correctly when looking up context — the thread's `parent_id` points to the text/forum channel, not the guild root.

Currently the channel-type guard in `/subscribe-text` rejects non-text, non-voice channels, which blocks threads and forum posts.

## Desired behaviour

- `/subscribe-text` issued inside a thread or forum post should subscribe the familiar to that thread.
- The familiar reads messages in that thread only (not the parent channel).
- Responses are sent into the thread, keeping conversation contained.
- The subscription model's normal rules apply — a thread subscription is independent of a text-channel subscription, and can be removed via `/unsubscribe-text` from within the thread.
- Logging follows the same pattern as text-channel sessions, anchored at the top of the thread if possible, or as the first bot message if not.

## Session handling

Session handling is identical to text-channel sessions (see [Session logging](session-logging.md)):

- A session summary is written at the end.
- Lorebook entries (people, topics) are updated as normal.
- The thread context (thread name, parent forum/channel name) should be included in the session metadata for reference.
