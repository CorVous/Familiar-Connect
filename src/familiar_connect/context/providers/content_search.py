"""ContentSearchProvider — the memory-search agent loop.

Step 8 of docs/architecture/context-pipeline.md. The interesting one.

Each :meth:`contribute` call hands the ``memory_search`` slot's
:class:`LLMClient` a small toolset scoped to one familiar's
:class:`MemoryStore` — ``list_dir``, ``glob``, ``grep``,
``read_file`` — and runs a bounded loop. The model decides which
tools to call (or that it has enough context); the provider
executes the calls against the store; the loop ends when the
model emits an ``ANSWER:`` line, the iteration cap is hit, or the
pipeline-level deadline expires.

The first cut uses **structured prompting** rather than a real
tool-call API: the prompt asks the model to reply with one of two
line shapes,

    TOOL: {"tool": "name", "args": {...}}
    ANSWER: <relevant context text>

and the provider parses by line prefix. A real
``chat_with_tools``-style API is a future drop-in; the provider's
loop logic doesn't depend on which wire format the model speaks,
only on what comes back as text.

Errors from the store (path traversal, size cap, missing file, …)
are caught and fed back to the model as the tool's "result" string,
so the model can recover by trying a different path. The pipeline-
level deadline is enforced upstream by ``ContextPipeline``.
"""

from __future__ import annotations

import asyncio
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

Hit when the model keeps emitting tool calls without ever returning
an ANSWER. After the cap, the provider gives up and returns no
contribution; the pipeline gets one fewer Contribution rather than
a stalled reply.

Three iterations covers the realistic shape of a memory-search
agent (one to plan, one to execute, one to answer) without paying
the full 15 s outer deadline every time the model is chatty.
"""

_PER_ITERATION_BUDGET_S = 4.0
"""Estimated cost of a single agent iteration.

Used to decide whether to start a fresh iteration when the outer
deadline is close. If less than this remains, we stop early instead
of letting ``asyncio.timeout`` hard-cancel mid-LLM-call (which
discards the partial scratchpad entirely).
"""

_DEADLINE_SAFETY_MARGIN_S = 1.0
"""Wall-clock breathing room kept between the last iteration's start
and the outer deadline; prevents a just-barely-in-time iteration
from blowing past the pipeline's timeout."""


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


_TOOL_LINE_PREFIX = "TOOL:"
_ANSWER_LINE_PREFIX = "ANSWER:"
_TOOL_RESULT_PREFIX = "TOOL_RESULT"


class ContentSearchProvider:
    """ContextProvider that runs a tool-using agent loop over a MemoryStore.

    Conforms to the ContextProvider Protocol structurally — no
    inheritance required.

    :param store: The familiar's :class:`MemoryStore`.
    :param llm_client: The :class:`LLMClient` for the
        ``memory_search`` slot.
    :param max_iterations: Maximum number of LLM calls per
        contribute() invocation. Defaults to
        :data:`DEFAULT_MAX_ITERATIONS`.
    """

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
        - The model never emits ANSWER within ``max_iterations`` →
          no contribution; logged.
        - The model emits a malformed line → that text is treated
          as the final answer (graceful fallback).
        """
        scratchpad: list[str] = []
        loop = asyncio.get_event_loop()
        start = loop.time()
        budget_remaining = self.deadline_s - _DEADLINE_SAFETY_MARGIN_S

        for iteration in range(self._max_iterations):
            # Bail early if we don't have time for another full LLM
            # round-trip. Starting an iteration we know we can't finish
            # just means ``asyncio.timeout`` hard-cancels mid-call and
            # discards the scratchpad we already built.
            elapsed = loop.time() - start
            if elapsed + _PER_ITERATION_BUDGET_S > budget_remaining:
                _logger.info(
                    "content_search: stopping early on iteration %d "
                    "(elapsed=%.2fs, budget_remaining=%.2fs)",
                    iteration,
                    elapsed,
                    budget_remaining,
                )
                break

            prompt = _SYSTEM_PROMPT_TEMPLATE.format(
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

            if parsed.kind == "tool":
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
        """Run *tool_name* with *args* against the store, return a result string.

        Returns a short, model-readable serialisation of either the
        tool's output or the error message — never raises. The
        provider's loop feeds the returned string back into the next
        prompt as the tool's "result," so any error becomes a piece
        of context the model can react to.
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

    ``kind`` is one of:

    - ``"tool"`` — the model emitted a TOOL line; ``tool``, ``args``,
      and ``raw`` are populated.
    - ``"answer"`` — the model emitted an ANSWER line; ``text`` is
      populated.
    - ``"fallback"`` — the model emitted neither prefix; ``text`` is
      the entire response, treated as a final answer.
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


def _parse_response(response: str) -> _ParsedResponse:
    stripped = response.strip()

    # ANSWER comes first because it's also the fallback shape.
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
