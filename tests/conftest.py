"""Shared test fixtures and configuration."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from familiar_connect.config import LLM_SLOT_NAMES
from familiar_connect.context.providers.content_search import retrieval
from familiar_connect.context.types import ContextRequest, Modality
from familiar_connect.identity import Author
from familiar_connect.llm import LLMClient, Message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROFILE_PATH = (
    _REPO_ROOT / "data" / "familiars" / "_default" / "character.toml"
)


@pytest.fixture(autouse=True)
def _stub_fastembed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block real HuggingFace downloads from :class:`FastEmbedModel`.

    Production ``FastEmbedModel.embed`` lazy-loads a ~68 MB ONNX model
    from HuggingFace on first call. In sandboxed CI that host isn't
    allowlisted; fastembed retries with 3s + 9s + 27s backoff, costing
    ~39 s per test that indirectly triggers a retrieval (e.g. any
    ``_run_text_response`` pathway through ``ContentSearchProvider``).

    Stub ``embed`` → zero vectors, ``_ensure_loaded`` → no-op. Tests
    that care about retrieval relevance inject their own model via
    :class:`EmbeddingRetriever` (see ``test_embedding_retriever.py``)
    and aren't touched by this patch.
    """

    def _fake_embed(self: retrieval.FastEmbedModel, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def _fake_ensure_loaded(self: retrieval.FastEmbedModel) -> None:
        # sentinel so any assert ``self._impl is not None`` still holds
        self._impl = object()

    monkeypatch.setattr(retrieval.FastEmbedModel, "embed", _fake_embed)
    monkeypatch.setattr(retrieval.FastEmbedModel, "_ensure_loaded", _fake_ensure_loaded)


@pytest.fixture
def package_name() -> str:
    """Return the package name (snake_case for imports)."""
    return __package__.split(".")[0] if __package__ else "familiar_connect"


@pytest.fixture
def cli_name() -> str:
    """Return the CLI command name (kebab-case)."""
    # Dynamically get from package metadata if possible
    try:
        return importlib.metadata.metadata("familiar-connect")["Name"]
    except (importlib.metadata.PackageNotFoundError, KeyError):
        return "familiar-connect"


@pytest.fixture
def default_profile_path() -> Path:
    """Return the path to the checked-in ``_default/character.toml``.

    Tests pass this to :func:`load_character_config` so the merge
    step has a real defaults file to read. The content of that file
    is the single source of truth for fallback values in the whole
    codebase.
    """
    return _DEFAULT_PROFILE_PATH


@pytest.fixture
def make_context_request() -> Callable[..., ContextRequest]:
    """Return a factory for ContextRequest with optional keyword overrides.

    Canonical defaults match the original ``_make_request`` helper
    from ``test_context_pipeline.py``. Tests that need a non-standard
    utterance or deadline pass overrides directly:
    ``make_context_request(utterance="hi", deadline_s=10.0)``.
    """
    alice = Author(
        platform="discord", user_id="1", username="alice", display_name="Alice"
    )

    def _factory(**overrides: Any) -> ContextRequest:  # noqa: ANN401
        defaults: dict[str, Any] = {
            "familiar_id": "aria",
            "channel_id": 100,
            "guild_id": 1,
            "author": alice,
            "utterance": "hello",
            "modality": Modality.text,
            "budget_tokens": 2048,
            "deadline_s": 5.0,
        }
        defaults.update(overrides)
        return ContextRequest(**defaults)  # type: ignore[arg-type]

    return _factory


class FakeLLMClient(LLMClient):
    """In-memory :class:`LLMClient` stand-in for tests.

    Yields scripted :class:`Message` replies from a queue; records
    every call for assertions. Inherits from :class:`LLMClient` so
    it's a drop-in replacement at call sites typed as ``LLMClient``;
    overrides ``chat`` to skip the HTTP layer entirely.

    :param replies: An iterable of canned strings. Each chat call
        pops the next string and wraps it in an assistant
        :class:`Message`. An empty queue returns an empty message.
    """

    def __init__(
        self,
        replies: Iterable[str] | None = None,
    ) -> None:
        super().__init__(api_key="fake-test-key", model="fake/test-model")
        self.calls: list[list[Message]] = []
        self._replies: list[str] = list(replies or [])

    async def chat(self, messages: list[Message]) -> Message:
        self.calls.append(list(messages))
        if not self._replies:
            return Message(role="assistant", content="")
        content = self._replies.pop(0)
        return Message(role="assistant", content=content)

    async def close(self) -> None:  # pragma: no cover — no resources
        return


def build_fake_llm_clients(
    *,
    per_slot_replies: dict[str, Iterable[str]] | None = None,
) -> dict[str, LLMClient]:
    """Return a ``slot_name -> FakeLLMClient`` dict covering every slot.

    Convenience builder for :meth:`Familiar.load_from_disk` tests.
    Unknown slot names in ``per_slot_replies`` are ignored; missing
    slots get an empty FakeLLMClient so every call site has a valid
    client to call.
    """
    per_slot_replies = per_slot_replies or {}
    clients: dict[str, LLMClient] = {
        slot: FakeLLMClient(per_slot_replies.get(slot)) for slot in LLM_SLOT_NAMES
    }
    return clients
