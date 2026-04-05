# Lorebook Feature

## Overview

A lorebook is a structured collection of background information and session history for each familiar. It serves as long-term memory that can be selectively injected into context when relevant.

## Structure

The lorebook is a file tree containing:

- **Background entries**: Lore, personality details, world-building, and any persistent facts about the familiar
- **Session summaries**: One entry per session, summarizing what occurred — written after each conversation ends

## Access Management

A smaller/cheaper LLM acts as a lorebook manager. It:

1. Maintains awareness of the entire lorebook file tree (titles, tags, brief descriptions)
2. Is given instructions describing when each type of entry is relevant
3. On each new conversation turn, decides which entries (if any) to pull into the main context
4. Returns only the relevant subset — keeping the main LLM's context lean

## Key Design Goals

- The main familiar LLM never loads the full lorebook — only what the manager surfaces
- Session summaries accumulate over time without bloating active context
- The manager's instructions should be tunable per-familiar to control relevance thresholds
- Background entries and session entries are stored separately for easier management
