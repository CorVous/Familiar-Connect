# Roadmap

Everything on this page is *planned, not shipped*. The shipped surface is described in [Architecture](../architecture/overview.md) and the [Getting started](../getting-started/installation.md) guides. Items here live in their own pages so each one carries its own rationale and open questions.

## Status at a glance

| Feature | State | Page |
|---|---|---|
| Voice channel logging (passive listeners) | Design | [Voice logging](voice-logging.md) |
| Web search provider | Design | [Web search](web-search.md) |
| Gemini TTS: dynamic audio tags + situational style prompts | Design | [Gemini TTS expressivity](gemini-tts-expressivity.md) |
| Proactive cross-channel reading (follow links outside subscribed channels) | Deferred | [Proactive cross-channel reading](proactive-cross-channel-reading.md) |
| Per-turn monitoring dashboard | Partial — data exists, no UI | [Context pipeline § deferred](../architecture/context-pipeline.md) |
| `/context` slash command | Deferred | [Context pipeline § deferred](../architecture/context-pipeline.md) |
| `familiar init --from-card` subcommand | Deferred | [Context pipeline § deferred](../architecture/context-pipeline.md) |

Previously listed here and now shipped (moved to Architecture):
[Conversation flow](../architecture/conversation-flow.md),
[Voice input](../architecture/voice-input.md),
[Voice interruption](../architecture/interruption.md),
[Session logging](../architecture/session-logging.md),
threads & forum-post subscriptions (see
[slash commands](../getting-started/slash-commands.md)).

## Scope rules for roadmap items

Every roadmap page follows the same shape so they stay comparable:

- **Motivation** — what problem this solves and why it belongs in the pipeline rather than in an ad-hoc script.
- **Sketch** — the rough design, enough to decide whether it fits the existing architecture.
- **Open questions** — the specific things that still need a decision before implementation starts.
- **Non-goals** — features that look adjacent and are deliberately out of scope.

Items are promoted out of this section when they ship. At that point the page either moves under [Architecture](../architecture/overview.md) (if it describes a permanent subsystem) or is deleted (if the content was purely about the decision to build it).

## Not on the roadmap

These have been considered and rejected. See [Design decisions](../architecture/decisions.md) for the full list:

- Bridging to a running SillyTavern instance.
- Embedding SillyTavern extensions via headless browser.
- Adopting a large LLM orchestration framework (LangChain, LlamaIndex, Haystack) as a runtime.
- Third-party managed memory services (mem0, Zep, etc.).
- Any third-party vector database. Local embeddings ship behind `ContentSearchProvider`'s retriever tier (see [Memory → Derived indices](../architecture/memory.md#derived-indices)) — a SQLite `.index/embeddings.sqlite` populated by `fastembed`/ONNX, no external service.
