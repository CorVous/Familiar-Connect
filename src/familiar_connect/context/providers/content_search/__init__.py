"""ContentSearchProvider package — markdown-memory context retrieval.

Three-tier design (see docs/architecture/context-pipeline.md §8):

1. ``people_lookup`` — deterministic. Speaker + mentioned-name files
   loaded verbatim. Correctness floor.
2. *(reserved for embedding retrieval — arrives in a follow-up PR)*
3. ``_agent_loop`` — cheap LLM with ``list_dir``/``glob``/``grep``/
   ``read_file`` tools. Transitional fallback; replaced by a
   single-shot filter over embedding hits once tier 2 lands.
"""

from familiar_connect.context.providers.content_search._agent_loop import (
    CONTENT_SEARCH_PRIORITY,
    DEFAULT_DEADLINE_S,
    DEFAULT_MAX_ITERATIONS,
    FORCED_ANSWER_MARKER,
)
from familiar_connect.context.providers.content_search.people_lookup import (
    DEFAULT_MAX_TOKENS_PER_FILE,
    PEOPLE_LOOKUP_PRIORITY,
    PEOPLE_LOOKUP_SOURCE,
)
from familiar_connect.context.providers.content_search.provider import (
    ContentSearchProvider,
)

__all__ = [
    "CONTENT_SEARCH_PRIORITY",
    "DEFAULT_DEADLINE_S",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOKENS_PER_FILE",
    "FORCED_ANSWER_MARKER",
    "PEOPLE_LOOKUP_PRIORITY",
    "PEOPLE_LOOKUP_SOURCE",
    "ContentSearchProvider",
]
