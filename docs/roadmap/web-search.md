# Web search

Let the familiar search the web to answer questions about current events, look up facts, fetch patch notes, and anything else that requires knowledge that isn't in its memory directory or in the model's training data.

!!! info "Status: Research"
    Not yet shipped and mostly in the research phase. The design goal is that web search lands as *another context provider* in the pipeline — not as a special code path.

## Motivation

The context pipeline is already provider-shaped: `CharacterProvider`, `HistoryProvider`, and `ContentSearchProvider` compose concurrently under a per-turn deadline. A web search provider slots into the same shape — it takes a `ContextRequest`, does its work, and returns `Contribution`s. Everything downstream (the budgeter, the renderer, the post-processors) is agnostic to where the content came from.

The hard problems for this feature aren't orchestration — they're *which API to use*, *how to prevent prompt injection from the open web*, and *how much control the operator needs over what the familiar is allowed to look up*.

## Things to research

### Search API choice

- **Hosted search APIs** (Brave Search, Kagi, Serper, Tavily, Exa). Different cost structures, different quality, different TOS.
- **LLM-integrated search** (OpenRouter-routed search-enabled models, Claude's native web search when available). Lower latency but couples the search path to a specific model.
- **Self-hosted** (SearXNG, Whoogle). Zero API cost, full control, but extra moving parts on the host.

Rule of thumb matches the project's local-first principle: avoid solutions that force every query off to a service that sees both the question *and* the familiar's persona. A hosted search API that only sees the query is fine; a service that sees the assembled prompt is not.

### Prompt-injection defence

Web content is adversarial — a page can contain instructions aimed at the familiar's prompt layer, hidden in whitespace or zero-width characters, or framed as "system messages." The defence needs more than "ask nicely":

- **Summarisation intermediary.** A cheap side model reads the raw search results (optionally after a trafilatura-style HTML-to-text clean-up) and emits a short factual summary. The main LLM sees only the summary, not the raw page text. This removes almost all injection-via-markup vectors at the cost of one extra model call per turn.
- **Content-type whitelisting.** Don't follow PDFs, don't render images, don't execute JavaScript. Only plain text / Markdown / HTML body.
- **Quoting discipline.** The summary is wrapped in a clearly-delimited `<web-search-results>...</web-search-results>` block in the system prompt so the main LLM can tell "this is external content" from "this is my instructions." Not a bulletproof defence on its own, but combined with the summariser it closes most of the realistic surface.
- **Size caps.** A hard ceiling on how many bytes of web content ever reach the main LLM (before or after summarisation). Prevents a single search burning the entire context budget.

### Operator control surface

- **Allow/deny lists.** Per-familiar `character.toml` options for which domains or topics the search provider may touch. Default is "everything"; operators who want stricter scoping opt in.
- **Toggle off entirely.** The provider is registered in code but the per-channel `ChannelMode` can disable it (same shape as how `imitate_voice` turns off `stepped_thinking` to protect TTFB).
- **Audit log.** Every search query, the URLs fetched, and the final summary all go into the per-turn trace log that `ContentSearchProvider` already pioneered. "Why did the bot say X?" stays answerable.

### Provider shape

Probable module: `familiar_connect.context.providers.web_search`.

- Called with the `ContextRequest`. Decides whether to search based on a cheap heuristic (keywords in the utterance) or on a side-model classifier call — TBD.
- If it decides to search: builds a query, hits the search API, optionally fetches the top N pages, runs the summariser, and emits one `Contribution(layer=Layer.content, priority=...)` with the summary and the source URLs in `source`.
- Deadline-bounded; falls back to emitting nothing if the deadline hits, never stalls the pipeline. Matches the same pattern `HistoryProvider` and `ContentSearchProvider` use.

## Non-goals

- **A general web-browsing agent** that clicks around the site like a human. Out of scope — one query, top-N results, extract, summarise, done.
- **Search-as-tool-call** inside the main LLM loop. Nice later, but it conflicts with the per-turn deadline model and makes caching harder. The provider pattern runs the search *before* the main LLM call so latency is predictable.
- **Vector-indexed web cache.** Same rationale as the first-pass rejection of vector memory — add a caching layer only if measurements show the search call is the bottleneck.

## Open questions

- **Does the provider always run, or only on classifier hit?** Always-run is simpler but wastes budget and latency on "hi, how are you?" turns. Start with a cheap keyword heuristic; escalate to a side-model classifier if the heuristic is too blunt.
- **Who pays for the search API?** Per-operator API key in `.env`, same as every other third-party credential.
- **How much of the per-turn trace log is safe to show in the `/context` slash command?** URLs are fine; search API keys must never leak. Same sanitisation rules as logging — see [Security § logging & error handling](../architecture/security.md#logging-error-handling).
- **Fallback behaviour when search fails.** Emit nothing and let the main LLM admit it doesn't know, or emit a marker contribution saying "web search failed" so the LLM knows *why* it's uninformed? The latter is more honest but longer-winded.
