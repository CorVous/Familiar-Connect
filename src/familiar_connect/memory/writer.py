"""Post-session memory writer — summarises history into long-term memory files.

After a configurable number of turns or an idle timeout, the writer
reads unsummarized turns from the :class:`HistoryStore`, calls a cheap
side-model to produce a structured summary, and writes session, people,
and topic files into the familiar's :class:`MemoryStore`.

See ``docs/architecture/memory.md`` for the content conventions the
writer follows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from familiar_connect.llm import LLMClient, Message
from familiar_connect.memory.store import MemoryStore, MemoryStoreError

if TYPE_CHECKING:
    from familiar_connect.history.store import HistoryStore, HistoryTurn

_logger = logging.getLogger(__name__)

_AUDIT_SOURCE = "memory_writer"

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryWriterResult:
    """Outcome of a single writer-pass run."""

    session_file: str | None = None
    people_files: list[str] = field(default_factory=list)
    topic_files: list[str] = field(default_factory=list)
    turns_summarized: int = 0
    watermark_id: int = 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_WRITER_SYSTEM_PROMPT = """\
You are a memory-management assistant for a character called {familiar_name}. \
Your job is to read a conversation transcript and produce structured updates \
for the character's long-term memory files.

Produce your output in the EXACT format shown below. Every section marker \
must appear on its own line. If a section has no content, leave it empty \
but still include the markers.

===SESSION_SUMMARY===
<A concise markdown summary of this conversation session. Focus on key \
events, decisions, emotional beats, and anything worth remembering. \
1-3 short paragraphs.>
===END_SESSION_SUMMARY===

===PEOPLE===
For each person worth remembering or updating, write:
---FILE: <filename>.md---
<Full markdown content for this person's file. Start with an H1 of their \
name. Include who they are, notable interactions, the character's feelings \
about them, and any relevant details. If updating an existing file, preserve \
and extend its content — do not drop existing information.>
---END_FILE---
===END_PEOPLE===

===TOPICS===
For each topic worth remembering or updating, write:
---FILE: <filename>.md---
<Full markdown content for this topic file. Start with an H1 of the topic \
name. Include what the topic is, notable discussions, opinions formed, and \
relevant participants. If updating an existing file, preserve and extend \
its content — do not drop existing information.>
---END_FILE---
===END_TOPICS===

Rules:
- Filenames must be lowercase, use hyphens for spaces, and end in .md
- Filenames must be a basename only — never include a directory prefix \
  (write ``alice.md``, not ``people/alice.md``)
- Only create/update people and topic files that are genuinely worth \
  remembering long-term
- Write from {familiar_name}'s perspective
- Be concise but capture the important details
- If existing file content is provided, incorporate and extend it — never \
  discard existing information"""


_WRITER_USER_PROMPT = """\
{existing_files_section}\
----- conversation transcript -----
{transcript}
----- end transcript -----

Produce the structured memory update now."""


_EXISTING_FILES_HEADER = """\
Here are the existing memory files that may be relevant. Preserve and \
extend their content if you update them.

"""

_EXISTING_FILE_TEMPLATE = """\
--- Existing {category} file — {filename} ---
{content}
--- End existing file ---

"""


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------


def _parse_writer_output(
    text: str,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Parse the structured LLM output into components.

    Returns ``(session_summary, people_files, topic_files)`` where
    the file dicts map ``filename -> content``.
    """
    session_summary = ""
    people_files: dict[str, str] = {}
    topic_files: dict[str, str] = {}

    # Extract session summary
    session_match = re.search(
        r"===SESSION_SUMMARY===\s*\n(.*?)\n\s*===END_SESSION_SUMMARY===",
        text,
        re.DOTALL,
    )
    if session_match:
        session_summary = session_match.group(1).strip()

    # Extract people files
    people_match = re.search(
        r"===PEOPLE===\s*\n(.*?)\n\s*===END_PEOPLE===",
        text,
        re.DOTALL,
    )
    if people_match:
        people_files = _parse_file_blocks(people_match.group(1))

    # Extract topic files
    topics_match = re.search(
        r"===TOPICS===\s*\n(.*?)\n\s*===END_TOPICS===",
        text,
        re.DOTALL,
    )
    if topics_match:
        topic_files = _parse_file_blocks(topics_match.group(1))

    return session_summary, people_files, topic_files


def _parse_file_blocks(section: str) -> dict[str, str]:
    """Extract ``filename -> content`` pairs from ``---FILE: x---`` blocks.

    Normalises filenames to a basename so a model echoing back a
    directory-prefixed path (e.g. ``people/alice.md``) doesn't cause
    the writer to nest directories (``people/people/alice.md``).
    """
    files: dict[str, str] = {}
    for match in re.finditer(
        r"---FILE:\s*(.+?)---\s*\n(.*?)\n\s*---END_FILE---",
        section,
        re.DOTALL,
    ):
        raw = match.group(1).strip()
        content = match.group(2).strip()
        # keep only the basename; reject empty or traversal artefacts
        filename = raw.replace("\\", "/").rstrip("/").split("/")[-1]
        if filename in {"", ".", ".."}:
            continue
        if filename and content:
            files[filename] = content
    return files


# ---------------------------------------------------------------------------
# Turn renderer (reuses the format from history provider)
# ---------------------------------------------------------------------------


def _render_turns(turns: list[HistoryTurn]) -> str:
    """Render turns as a text transcript."""
    lines: list[str] = []
    for t in turns:
        if t.role == "user" and t.speaker:
            lines.append(f"{t.role} ({t.speaker}): {t.content}")
        else:
            lines.append(f"{t.role}: {t.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session filename helpers
# ---------------------------------------------------------------------------


def _time_slot(dt: datetime) -> str:
    """Return a human-readable time-of-day slot for a datetime."""
    hour = dt.hour
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _session_filename(dt: datetime) -> str:
    """Return the session filename for a given datetime.

    One file per date+slot. If one already exists the writer
    overwrites it with a merged summary; it does not fork a
    ``-2``/``-3`` sibling — that used to fragment a single
    conversation across many files each time the turn-count
    threshold re-fired.
    """
    date_str = dt.strftime("%Y-%m-%d")
    slot = _time_slot(dt)
    return f"sessions/{date_str}-{slot}.md"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class MemoryWriter:
    """Summarises unsummarized conversation history into long-term memory.

    :param memory_store: The :class:`MemoryStore` to write files into.
    :param history_store: The :class:`HistoryStore` to read turns from
        and track the watermark in.
    :param llm_client: The :class:`LLMClient` for the ``memory_writer``
        slot.
    :param familiar_id: The active familiar's id.
    """

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        history_store: HistoryStore,
        llm_client: LLMClient,
        familiar_id: str,
    ) -> None:
        self._memory_store = memory_store
        self._history_store = history_store
        self._llm_client = llm_client
        self._familiar_id = familiar_id

    async def run(self) -> MemoryWriterResult:
        """Execute the memory-writer pass.

        Reads turns since the last watermark, summarises them via the
        LLM, writes session/people/topic files, and advances the
        watermark. Returns a result describing what was written.

        On LLM failure, the watermark is not advanced and no files are
        written — the next invocation will retry the same turns.
        """
        turns = self._history_store.turns_since_watermark(
            familiar_id=self._familiar_id,
        )
        if not turns:
            return MemoryWriterResult()

        new_watermark = turns[-1].id
        now = datetime.now(tz=UTC)
        session_path = _session_filename(now)

        # Build the prompt with existing file context
        transcript = _render_turns(turns)
        existing_section = self._build_existing_files_section(
            turns,
            session_path=session_path,
        )

        system_prompt = _WRITER_SYSTEM_PROMPT.format(
            familiar_name=self._familiar_id,
        )
        user_prompt = _WRITER_USER_PROMPT.format(
            existing_files_section=existing_section,
            transcript=transcript,
        )

        reply = await self._llm_client.chat(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
        )

        if not reply.content:
            return MemoryWriterResult()

        session_summary, people_files, topic_files = _parse_writer_output(
            reply.content,
        )

        result_session: str | None = None
        result_people: list[str] = []
        result_topics: list[str] = []

        # Write session summary — overwrites same date+slot file
        if session_summary:
            self._memory_store.write_file(
                session_path,
                session_summary,
                source=_AUDIT_SOURCE,
            )
            result_session = session_path

        # Write people files
        for filename, content in people_files.items():
            rel_path = f"people/{filename}"
            self._memory_store.write_file(
                rel_path,
                content,
                source=_AUDIT_SOURCE,
            )
            result_people.append(rel_path)

        # Write topic files
        for filename, content in topic_files.items():
            rel_path = f"topics/{filename}"
            self._memory_store.write_file(
                rel_path,
                content,
                source=_AUDIT_SOURCE,
            )
            result_topics.append(rel_path)

        # Advance the watermark only after all writes succeed
        self._history_store.put_writer_watermark(
            familiar_id=self._familiar_id,
            last_written_id=new_watermark,
        )

        _logger.info(
            "memory writer: session=%s people=%d topics=%d turns=%d watermark=%d",
            result_session,
            len(result_people),
            len(result_topics),
            len(turns),
            new_watermark,
        )

        return MemoryWriterResult(
            session_file=result_session,
            people_files=result_people,
            topic_files=result_topics,
            turns_summarized=len(turns),
            watermark_id=new_watermark,
        )

    def _build_existing_files_section(
        self,
        turns: list[HistoryTurn],
        *,
        session_path: str | None = None,
    ) -> str:
        """Load existing session/people/topic files relevant to the transcript.

        ``session_path`` points at the session file the writer will
        overwrite this pass. If it already exists, its prior summary
        is fed back so the model produces a merged update instead of
        restating only the new turns.
        """
        section_parts: list[str] = []

        # load current slot's prior session summary (merge, don't restate)
        if session_path:
            try:
                prior_session = self._memory_store.read_file(session_path)
            except MemoryStoreError:
                prior_session = ""
            if prior_session:
                filename = session_path.split("/")[-1]
                section_parts.append(
                    _EXISTING_FILE_TEMPLATE.format(
                        category="session",
                        filename=filename,
                        content=prior_session,
                    )
                )

        # Collect speaker names from user turns
        speakers: set[str] = set()
        for t in turns:
            if t.role == "user" and t.speaker:
                speakers.add(t.speaker)

        # Load existing people files matching speaker names
        people_loaded = 0
        for speaker in sorted(speakers):
            slug = re.sub(r"[^a-z0-9]+", "-", speaker.lower()).strip("-")
            if not slug:
                continue
            rel_path = f"people/{slug}.md"
            try:
                content = self._memory_store.read_file(rel_path)
                section_parts.append(
                    _EXISTING_FILE_TEMPLATE.format(
                        category="people",
                        filename=f"{slug}.md",
                        content=content,
                    )
                )
                people_loaded += 1
            except MemoryStoreError:
                pass
            if people_loaded >= 4:
                break

        # List existing topic files for model awareness
        topics_loaded = 0
        try:
            topic_files = self._memory_store.glob("topics/*.md")
        except MemoryStoreError:
            topic_files = []

        for topic_path in topic_files[:8]:
            try:
                content = self._memory_store.read_file(topic_path)
                filename = topic_path.split("/")[-1]
                section_parts.append(
                    _EXISTING_FILE_TEMPLATE.format(
                        category="topics",
                        filename=filename,
                        content=content,
                    )
                )
                topics_loaded += 1
            except MemoryStoreError:
                pass
            if topics_loaded >= 4:
                break

        if not section_parts:
            return ""

        return _EXISTING_FILES_HEADER + "".join(section_parts) + "\n"
