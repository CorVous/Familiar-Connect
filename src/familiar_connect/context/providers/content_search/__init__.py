"""ContentSearchProvider package — markdown-memory context retrieval.

Three-tier design (see docs/architecture/context-pipeline.md §8):

1. ``people_lookup`` — deterministic. Speaker + mentioned-name files
   loaded verbatim. Correctness floor.
2. ``retrieval.EmbeddingRetriever`` — local fastembed/ONNX against
   the SQLite embedding cache under ``.index/embeddings.sqlite``.
3. ``filter`` — one cheap-LLM call (with optional single grep
   escalation) picking what to forward to the main model.
"""

from familiar_connect.context.providers.content_search.filter import (
    FILTER_MAX_ITERATIONS,
    FILTER_PRIORITY,
    FILTER_SOURCE,
)
from familiar_connect.context.providers.content_search.people_lookup import (
    DEFAULT_MAX_TOKENS_PER_FILE,
    PEOPLE_LOOKUP_PRIORITY,
    PEOPLE_LOOKUP_SOURCE,
    LookupResult,
)
from familiar_connect.context.providers.content_search.provider import (
    DEFAULT_DEADLINE_S,
    DEFAULT_TOP_K,
    ContentSearchProvider,
)
from familiar_connect.context.providers.content_search.retrieval import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModel,
    EmbeddingRetriever,
    FastEmbedModel,
    PreparedChunk,
    RetrievedChunk,
    Retriever,
    chunk_markdown,
)

__all__ = [
    "DEFAULT_DEADLINE_S",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MAX_TOKENS_PER_FILE",
    "DEFAULT_TOP_K",
    "FILTER_MAX_ITERATIONS",
    "FILTER_PRIORITY",
    "FILTER_SOURCE",
    "PEOPLE_LOOKUP_PRIORITY",
    "PEOPLE_LOOKUP_SOURCE",
    "ContentSearchProvider",
    "EmbeddingModel",
    "EmbeddingRetriever",
    "FastEmbedModel",
    "LookupResult",
    "PreparedChunk",
    "RetrievedChunk",
    "Retriever",
    "chunk_markdown",
]
