# Roadmap

Everything on this page is *planned, not shipped*. The shipped surface is described in [Architecture](../architecture/overview.md) and the [Getting started](../getting-started/installation.md) guides. Items here live in their own pages so each one carries its own rationale and open questions.

## Status at a glance

| Feature | State | Page |
|---|---|---|
| Chattiness & interjection (proactive replies) | Design | [Conversation flow](conversation-flow.md) |
| Voice input (STT → pipeline) | Design | [Voice input](voice-input.md) |
| Barge-in / interruption handling | In progress — plumbing shipped, dispatch pending | [Interruption flow](interruption-flow.md) |
| Voice channel logging (passive listeners) | Design | [Voice logging](voice-logging.md) |
| Session logging & post-session writer pass | Design | [Session logging](session-logging.md) |
| Thread & forum-post subscriptions | Design | [Threads and forum posts](threads-and-forum-posts.md) |
| Web search provider | Design | [Web search](web-search.md) |
| Per-turn monitoring dashboard | Partial — data exists, no UI | [Context pipeline § deferred](../architecture/context-pipeline.md) |
| `/context` slash command | Deferred | [Context pipeline § deferred](../architecture/context-pipeline.md) |
| `familiar init --from-card` subcommand | Deferred | [Context pipeline § deferred](../architecture/context-pipeline.md) |

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
- Any vector database for memory storage in the first pass. Vector search may return later as a *tool* the content search agent can call.
