# Voice Channel Text & Image Input

## Overview

While the bot is active in a voice channel, the associated text channel also serves as an input source. Text messages and images sent there are ingested as conversation input alongside speech.

## Text Messages

While the bot is awake in a voice channel, it listens for messages posted in the associated text channel and feeds them into the same conversation pipeline as transcribed speech.

- Messages are attributed to the sender's username (linked to their people entry in the lorebook)
- Text input and voice input are interleaved in the conversation history so the LLM sees a unified stream
- Messages posted by the bot itself (status embeds, thread entries) are ignored as input

## Image Input

Images attached to messages in the voice text channel are passed to the LLM as vision input.

- Supported: images attached directly to a message or posted as Discord image links
- The image is downloaded and passed alongside any text in that message
- The LLM should be a vision-capable model (Claude supports this natively)
- If the model cannot handle vision, log a warning and describe the attachment by filename/type only
- Multiple images in one message are all passed in order
- Images are ephemeral by default — used for the current turn only. Persistence strategy to be decided later.

## Attribution

All input — whether spoken or typed — is treated identically by the LLM. It is attributed to the sender's username and processed in arrival order. The input method is not surfaced in the conversation history.
