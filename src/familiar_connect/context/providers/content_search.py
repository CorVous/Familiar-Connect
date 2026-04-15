"""ContentSearchProvider — memory-search agent loop.

Hands the ``memory_search`` slot's LLMClient a small toolset
(``list_dir``, ``glob``, ``grep``, ``read_file``) scoped to one
familiar's MemoryStore and runs a bounded loop. Ends when the model
emits ``ANSWER:``, the iteration cap is hit, or the pipeline deadline
expires.

Currently uses structured prompting (``TOOL:``/``ANSWER:`` line
prefixes) rather than a real tool-call API; the loop logic is
wire-format-agnostic.

Store errors are fed back as the tool's result string so the model
can recover. See docs/architecture/context-pipeline.md.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from familiar_connect.context.budget import estimate_tokens
from familiar_connect.context.types import Contribution, Layer
from familiar_connect.llm import Message
from familiar_connect.memory.store import MemoryStoreError

if TYPE_CHECKING:
    from familiar_connect.context.types import ContextRequest
    from familiar_connect.llm import LLMClient
    from familiar_connect.memory.store import MemoryStore


_logger = logging.getLogger(__name__)


CONTENT_SEARCH_PRIORITY = 70
"""Priority assigned to content-search Contributions.

Lower than CharacterProvider (100) and HistoryProvider's recent
window (80), higher than the rolling history summary (60). The
familiar's persona and the active conversation always win against
retrieved context under budget pressure."""

DEFAULT_DEADLINE_S = 15.0
"""Hard cap the pipeline enforces on this provider's contribute() call.

The provider's internal loop has its own iteration cap; the deadline
exists so a runaway agent loop never blocks the reply.
"""

DEFAULT_MAX_ITERATIONS = 3
"""Maximum number of LLM calls per contribute() invocation.

Lowered from 5: with forced-answer on the final iteration plus
redundant-call bail-out, three richer iterations beat five flailing
ones. If even the forced-answer turn doesn't emit ``ANSWER:``, the
provider gives up and returns no contribution.
"""

FORCED_ANSWER_MARKER = "FORCED_ANSWER"
"""Sentinel substring in the forced-answer prompt.

Tests assert on this to distinguish the forced-final prompt from the
normal iterative one without depending on the full prose template.
"""


_SYSTEM_PROMPT_TEMPLATE = """\
You are searching the familiar's memory directory to find context relevant
to the current conversation. The memory directory is a tree of plain-text
(usually Markdown) files. Use the tools below to look up only what is
actually relevant; do not dump the whole directory.

Available tools:
- list_dir(path): list files and subdirectories at path. The "path" argument
  is optional and defaults to the root.
- glob(pattern): match files by glob pattern. Use "**/*.md" for a recursive
  Markdown listing.
- grep(pattern, path): regex-search files for pattern. The "path" argument
  is optional and defaults to the root. The search is case-insensitive.
- read_file(path): read the full text contents of a single file.

Reply with EXACTLY ONE LINE in one of these two shapes:

    TOOL: {{"tool": "name", "args": {{"key": "value"}}}}
    ANSWER: <the relevant context text>

If you choose TOOL, the result of the tool call will be appended to the
conversation and you will be asked again. If you choose ANSWER, your text
will be handed to the main model as additional context — keep it short,
factual, and only include things the user's message would actually benefit
from.

The current speaker is {speaker}. The user said:

    {utterance}

Conversation so far:
{scratchpad}

What's next?"""


_FORCED_ANSWER_PROMPT_TEMPLATE = """\
You are searching the familiar's memory directory to find context relevant
to the current conversation. You have already used your tool-call budget.
This is your FINAL turn — emit ANSWER now, based on whatever you have seen
so far. NO MORE TOOL CALLS.

Reply with EXACTLY ONE LINE of the shape:

    ANSWER: <relevant context, or an empty string if nothing useful was found>

The current speaker is {speaker}. The user said:

    {utterance}

Conversation so far:
{scratchpad}

FORCED_ANSWER: emit ANSWER now."""


_TOOL_LINE_PREFIX = "TOOL:"
_ANSWER_LINE_PREFIX = "ANSWER:"
_TOOL_RESULT_PREFIX = "TOOL_RESULT"


class ContentSearchProvider:
    """ContextProvider that runs a tool-using agent loop over a MemoryStore."""

    id = "content_search"
    deadline_s = DEFAULT_DEADLINE_S

    def __init__(
        self,
        *,
        store: MemoryStore,
        llm_client: LLMClient,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if max_iterations <= 0:
            msg = f"max_iterations must be > 0, got {max_iterations}"
            raise ValueError(msg)
        self._store = store
        self._llm_client = llm_client
        self._max_iterations = max_iterations

    async def contribute(
        self,
        request: ContextRequest,
    ) -> list[Contribution]:
        """Run the tool-using loop and return up to one Contribution.

        - The model emits ANSWER with non-empty text → one
          Contribution at ``Layer.content``.
        - The model emits ANSWER with empty text → no contribution.
        - Final iteration swaps to a forced-answer prompt so a stalling
          model gets one last chance to emit best-effort ANSWER.
        - Repeating a (tool, args) pair triggers early forced-answer —
          the redundant call is NOT executed.
        - The model emits a malformed line → that text is treated
          as the final answer (graceful fallback).
        """
        scratchpad: list[str] = []
        seen_tool_calls: set[tuple[str, str]] = set()
        redundancy_triggered = False

        for iteration in range(self._max_iterations):
            is_final = iteration == self._max_iterations - 1
            use_forced_prompt = is_final or redundancy_triggered

            template = (
                _FORCED_ANSWER_PROMPT_TEMPLATE
                if use_forced_prompt
                else _SYSTEM_PROMPT_TEMPLATE
            )
            prompt = template.format(
                speaker=request.speaker or "(unknown)",
                utterance=request.utterance,
                scratchpad="\n".join(scratchpad) if scratchpad else "(none yet)",
            )

            reply = await self._llm_client.chat(
                [Message(role="user", content=prompt)],
            )
            parsed = _parse_response(reply.content)

            if parsed.kind == "answer":
                return self._answer_to_contribution(parsed.text)

            if use_forced_prompt:
                # forced turn: no more tools. A fallback (malformed) is
                # still best-effort surfaced; a TOOL response means the
                # model ignored the instruction — give up cleanly.
                if parsed.kind == "fallback":
                    return self._answer_to_contribution(parsed.text)
                break

            if parsed.kind == "tool":
                call_key = (parsed.tool, _canonicalise_args(parsed.args))
                if call_key in seen_tool_calls:
                    _logger.info(
                        "content_search: redundant tool call %s; "
                        "skipping execution, forcing answer",
                        parsed.tool,
                    )
                    redundancy_triggered = True
                    continue
                seen_tool_calls.add(call_key)
                tool_result = self._execute_tool(parsed.tool, parsed.args)
                scratchpad.extend((
                    f"{_TOOL_LINE_PREFIX} {parsed.raw}",
                    f"{_TOOL_RESULT_PREFIX}: {tool_result}",
                ))
                continue

            # parsed.kind == "fallback" — treat the whole response as the
            # final answer.
            _logger.debug(
                "content_search: malformed response on iteration %d, "
                "treating as final answer",
                iteration,
            )
            return self._answer_to_contribution(parsed.text)

        _logger.warning(
            "content_search: hit max_iterations=%d without an ANSWER",
            self._max_iterations,
        )
        return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _answer_to_contribution(self, text: str) -> list[Contribution]:
        clean = text.strip()
        if not clean:
            return []
        return [
            Contribution(
                layer=Layer.content,
                priority=CONTENT_SEARCH_PRIORITY,
                text=clean,
                estimated_tokens=estimate_tokens(clean),
                source="content_search",
            )
        ]

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Run *tool_name* against the store; return a result string, never raises.

        Errors become model-readable result strings so the agent loop
        can recover.
        """
        try:
            if tool_name == "list_dir":
                path = str(args.get("path", "") or "")
                entries = self._store.list_dir(path)
                rendered = ", ".join(
                    f"{e.name}/" if e.is_dir else e.name for e in entries
                )
                return rendered or "(empty)"

            if tool_name == "glob":
                pattern = str(args.get("pattern", ""))
                if not pattern:
                    return "ERROR: glob requires a 'pattern' argument"
                matches = self._store.glob(pattern)
                return ", ".join(matches) if matches else "(no matches)"

            if tool_name == "grep":
                pattern = str(args.get("pattern", ""))
                if not pattern:
                    return "ERROR: grep requires a 'pattern' argument"
                path = str(args.get("path", "") or "")
                hits = self._store.grep(pattern, rel_path=path)
                if not hits:
                    return "(no matches)"
                lines = [f"{h.rel_path}:{h.line_number}: {h.line_text}" for h in hits]
                return "\n".join(lines)

            if tool_name == "read_file":
                path = str(args.get("path", ""))
                if not path:
                    return "ERROR: read_file requires a 'path' argument"
                return self._store.read_file(path)
        except MemoryStoreError as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 — feed any failure back to the model
            _logger.warning(
                "content_search tool %s raised %s: %s",
                tool_name,
                type(exc).__name__,
                exc,
            )
            return f"ERROR: {type(exc).__name__}: {exc}"
        else:
            return f"ERROR: unknown tool '{tool_name}'"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class _ParsedResponse:
    """Discriminated parse of a single LLM response.

    ``kind``: ``"tool"`` | ``"answer"`` | ``"fallback"``.
    """

    __slots__ = ("args", "kind", "raw", "text", "tool")

    def __init__(
        self,
        kind: str,
        *,
        text: str = "",
        tool: str = "",
        args: dict[str, Any] | None = None,
        raw: str = "",
    ) -> None:
        self.kind = kind
        self.text = text
        self.tool = tool
        self.args = args or {}
        self.raw = raw


def _canonicalise_args(args: dict[str, Any]) -> str:
    """Stable string key for (tool, args) dedup — sorted JSON."""
    return json.dumps(args, sort_keys=True, default=str)


def _parse_response(response: str) -> _ParsedResponse:
    stripped = response.strip()

    # ANSWER first because it's also the fallback shape
    if stripped.startswith(_ANSWER_LINE_PREFIX):
        text = stripped[len(_ANSWER_LINE_PREFIX) :]
        return _ParsedResponse(kind="answer", text=text)

    if stripped.startswith(_TOOL_LINE_PREFIX):
        payload = stripped[len(_TOOL_LINE_PREFIX) :].strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return _ParsedResponse(kind="fallback", text=stripped)

        if not isinstance(obj, dict):
            return _ParsedResponse(kind="fallback", text=stripped)

        tool = obj.get("tool")
        args = obj.get("args", {})
        if not isinstance(tool, str) or not isinstance(args, dict):
            return _ParsedResponse(kind="fallback", text=stripped)

        return _ParsedResponse(kind="tool", tool=tool, args=args, raw=payload)

    return _ParsedResponse(kind="fallback", text=stripped)
