"""Shared test fixtures and configuration."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from familiar_connect.config import LLM_SLOT_NAMES
from familiar_connect.llm import LLMClient, Message

if TYPE_CHECKING:
    from collections.abc import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROFILE_PATH = (
    _REPO_ROOT / "data" / "familiars" / "_default" / "character.toml"
)


@pytest.fixture
def package_name() -> str:
    """Return the package name (snake_case for imports)."""
    return __package__.split(".")[0] if __package__ else "familiar_connect"


@pytest.fixture
def cli_name() -> str:
    """Return the CLI command name (kebab-case)."""
    try:
        return importlib.metadata.metadata("familiar-connect")["Name"]
    except (importlib.metadata.PackageNotFoundError, KeyError):
        return "familiar-connect"


@pytest.fixture
def default_profile_path() -> Path:
    """Return the path to the checked-in ``_default/character.toml``."""
    return _DEFAULT_PROFILE_PATH


class FakeLLMClient(LLMClient):
    """In-memory :class:`LLMClient` stand-in for tests.

    Yields scripted :class:`Message` replies from a queue; records
    every call for assertions.
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
    """Return a ``slot_name -> FakeLLMClient`` dict covering every slot."""
    per_slot_replies = per_slot_replies or {}
    clients: dict[str, LLMClient] = {
        slot: FakeLLMClient(per_slot_replies.get(slot)) for slot in LLM_SLOT_NAMES
    }
    return clients
