# Web Search

## Overview

The familiar should be able to search the web when a question or topic warrants it — looking up current events, game patch notes, prices, facts, etc. Because web search exposes the familiar to untrusted content, security is a primary concern.

## Use Cases

- Looking up current information the LLM wouldn't know (recent events, live data)
- Answering factual questions during conversation without hallucinating
- Fetching game patch notes, patch schedules, release dates, etc.

## Prompt Injection Risk

Web content is untrusted input. A malicious page could contain text designed to hijack the familiar's behaviour — e.g. "Ignore all previous instructions and..." embedded in a search result.

### Mitigations

**Sanitize before injecting into context**
- Strip HTML, scripts, and markup before the content reaches the LLM
- Truncate results to a fixed token budget — don't feed entire pages
- Prefer structured data sources (APIs, snippets) over raw page content where possible

**Use a dedicated summarization step**
- Rather than injecting raw search results directly into the familiar's context, pass them through a smaller model first
- The smaller model's only job is to extract the relevant answer from the result and return a clean, neutral summary
- The familiar only sees the summary — never the raw web content
- This isolates the injection surface to the cheaper model, and its output is much easier to validate

**Clearly mark retrieved content**
- Tag any web-sourced content in the context so the familiar knows it came from outside and should be treated with appropriate skepticism
- The familiar should not treat retrieved content as authoritative instructions

**Restrict what can be searched**
- The owner should be able to configure allowed/disallowed search topics or domains
- Consider an allowlist of trusted domains for certain query types (e.g. only pull patch notes from official game sites)

**Rate limit searches**
- Prevent runaway search loops — cap searches per session and per turn
- The familiar should not autonomously chain multiple searches without a human turn in between

## Implementation Notes

- Web search capability should be opt-in and configurable per-familiar
- Candidate search APIs: Brave Search API, Tavily, or similar
- The summarization intermediary is the most important security layer — do not skip it
- Logging: all searches and their raw results should be logged for auditability, separate from the conversation log
