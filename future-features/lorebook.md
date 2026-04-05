# Lorebook Feature

## Overview

A lorebook is a structured collection of background information and session history for each familiar. It serves as long-term memory that can be selectively injected into context when relevant.

## Structure

The lorebook is a file tree containing:

- **Background entries**: Lore, personality details, world-building, and any persistent facts about the familiar
- **Session summaries**: One entry per session, summarizing what occurred — written after each conversation ends
- **People entries**: One entry per known person the familiar has interacted with (see below)
- **Topic entries**: The familiar's opinions and knowledge about recurring subjects (see below)

## Access Management

A smaller/cheaper LLM acts as a lorebook manager. It:

1. Maintains awareness of the entire lorebook file tree (titles, tags, brief descriptions)
2. Is given instructions describing when each type of entry is relevant
3. On each new conversation turn, decides which entries (if any) to pull into the main context
4. Returns only the relevant subset — keeping the main LLM's context lean

## People Entries

The familiar maintains a lorebook entry for each person it has interacted with. These entries are created and updated by the familiar (or a post-session summarizer) and contain:

- **Known usernames**: All usernames this person has been seen using, with notes on which platform/context each came from
- **Identity notes**: Any information suggesting multiple usernames belong to the same person (e.g. self-disclosure, writing style, shared context)
- **The familiar's feelings**: A living record of how the familiar feels about this person — impressions, trust level, emotional history, memorable moments — updated after each session

### Multi-Username Handling

When a person uses a different username, the familiar should be able to:

1. Recognize clues that it may be the same person (they mention a previous conversation, same writing style, the owner flags it, etc.)
2. Link the new username to the existing entry or create a new entry with a note referencing the possible connection
3. Ask the person or owner for clarification if uncertain, rather than silently merging

The familiar's feelings and relationship history carry over once usernames are confirmed to be the same person.

## Topic Entries

The familiar maintains entries for subjects that come up repeatedly in conversation — games, shows, events, ongoing situations, shared interests, etc. These entries capture:

- **What the topic is**: A brief description for context
- **The familiar's opinions and feelings**: What it thinks, likes, dislikes, or finds interesting about the topic — updated as its views develop over conversations
- **Relevant history**: Notable things that have been said or happened relating to this topic in past sessions

Topic entries are created when a subject recurs enough to warrant persistent memory, and updated by the post-session summarizer when a session meaningfully touches on them.

The lorebook manager pulls a topic entry into context when the current conversation seems likely to involve that subject.

## Key Design Goals

- The main familiar LLM never loads the full lorebook — only what the manager surfaces
- Session summaries accumulate over time without bloating active context
- The manager's instructions should be tunable per-familiar to control relevance thresholds
- Background entries and session entries are stored separately for easier management
